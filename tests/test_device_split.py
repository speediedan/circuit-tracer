"""A model split across devices for attribution: the placement checks, and parity with a single-device graph.

The CPU tests cover the placement checks, including accelerate's real disk offload, which leaves parameters on the
meta device outside their own forward (the failure a split must refuse). The parity test needs two CUDA devices: it
builds the same graph with the model on one GPU and with its layers split across two, and compares them.
"""

import accelerate
import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, Gemma3ForCausalLM, Gemma3TextConfig

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.utils.device_placement import check_materialized, placement_summary
from tests.test_attributions_gemma3_nnsight import (
    initialize_transcoder_weights,
    set_l0_via_thresholds,
)

two_cuda_devices = torch.cuda.is_available() and torch.cuda.device_count() >= 2


class _Mapped(nn.Module):
    def __init__(self, device_map: dict | None = None):
        super().__init__()
        self.a = nn.Linear(2, 2)
        self.b = nn.Linear(2, 2)
        if device_map is not None:
            self.hf_device_map = device_map


@pytest.mark.parametrize(
    "device_map",
    [None, {"": "cpu"}, {"a": "cpu", "b": "cpu"}, {"a": 0, "b": 1}],
    ids=["no-map", "single-cpu", "all-cpu", "two-gpus"],
)
def test_check_materialized_accepts(device_map):
    check_materialized(_Mapped(device_map))


@pytest.mark.parametrize(
    "device_map, offloaded",
    [({"a": 0, "b": "cpu"}, "b"), ({"a": 0, "b": "disk"}, "b"), ({"a": "disk", "b": "cpu"}, "a")],
    ids=["cpu-beside-gpu", "disk", "disk-without-gpu"],
)
def test_check_materialized_refuses_offload(device_map, offloaded):
    with pytest.raises(
        ValueError, match=rf"offloaded off the accelerators: {offloaded}\b.*sequential"
    ):
        check_materialized(_Mapped(device_map))


def test_check_materialized_refuses_meta_tensor():
    model = _Mapped()
    model.b.weight = nn.Parameter(torch.empty(2, 2, device="meta"))
    with pytest.raises(ValueError, match=r"tensors on the meta device: b\.weight"):
        check_materialized(model)


def _tiny_gemma3() -> Gemma3ForCausalLM:
    cfg = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    return Gemma3ForCausalLM(cfg).eval()


def test_check_materialized_catches_accelerate_disk_offload(tmp_path):
    """The September failure's mechanism on CPU: a module placed on disk keeps meta parameters outside its forward."""
    model = _tiny_gemma3()
    device_map: dict[str, str | int | torch.device] = {
        "model.embed_tokens": "cpu",
        "model.layers": "cpu",
        "model.rotary_emb": "cpu",
        "model.norm": "disk",
        "lm_head": "cpu",
    }
    accelerate.dispatch_model(model, device_map=device_map, offload_dir=str(tmp_path))
    # positive control: the planted defect is present, and the model still runs (the hook materializes the norm)
    assert model.model.norm.weight.is_meta
    model(torch.tensor([[1, 2, 3]]))
    assert model.model.norm.weight.is_meta

    with pytest.raises(ValueError, match=r"model\.norm\b") as err:
        check_materialized(model)
    assert "disk: 1" in str(err.value)


def test_placement_summary_counts_per_device():
    assert (
        placement_summary({"a": 0, "b": 0, "c": 1, "d": "disk"}) == "cuda:0: 2, cuda:1: 1, disk: 1"
    )


def _dummy_transcoders(cfg) -> TranscoderSet:
    """Skip transcoders with the fork's test initialization; thresholds are calibrated per model afterwards."""
    transcoders = {}
    for layer in range(cfg.num_hidden_layers):
        tc = SingleLayerTranscoder(
            cfg.hidden_size,
            cfg.hidden_size * 4,
            JumpReLU(torch.zeros(cfg.hidden_size * 4), 0.1),
            layer,
            skip_connection=True,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        initialize_transcoder_weights(tc.W_enc, tc.W_dec, tc.b_enc, tc.b_dec)
        with torch.no_grad():
            nn.init.uniform_(tc.W_skip, a=-0.01, b=0.01)  # type: ignore[arg-type]
        transcoders[layer] = tc
    return TranscoderSet(
        transcoders, feature_input_hook="hook_resid_mid", feature_output_hook="hook_mlp_out"
    )


def test_from_pretrained_records_attribution_device():
    """A caller device map reaches the load, and the transcoders follow the attribution device."""
    cfg = AutoConfig.from_pretrained("google/gemma-3-270m")
    model = ReplacementModel.from_pretrained_and_transcoders(
        "google/gemma-3-270m",
        _dummy_transcoders(cfg),
        backend="nnsight",
        device=torch.device("cpu"),
        device_map={"": "cpu"},
    )
    assert model.attribution_device == torch.device("cpu")
    assert model.split_devices is False
    assert {p.device for p in model.transcoders.parameters()} == {torch.device("cpu")}


def _split_map(n_layers: int, first_on_1: int) -> dict:
    """Layers from ``first_on_1`` onward, and the final norm, on cuda:1; the rest (embedding, tied head) on cuda:0."""
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "lm_head": 0, "model.norm": 1}
    for layer in range(n_layers):
        device_map[f"model.layers.{layer}"] = 0 if layer < first_on_1 else 1
    return device_map


@pytest.mark.skipif(not two_cuda_devices, reason="needs two CUDA devices")
def test_split_graph_matches_single_device():
    model_name = "google/gemma-3-270m"
    cfg = AutoConfig.from_pretrained(model_name)
    prompt = "The National Digital Analytics Group (ND"

    # one transcoder set for both loads: calibrated once (about 16 active features per position, as the fork's
    # other dummy-transcoder tests do), so the two graphs differ only in where the model's layers live
    transcoders = _dummy_transcoders(cfg)
    single = ReplacementModel.from_pretrained_and_transcoders(
        model_name, transcoders, backend="nnsight", device=torch.device("cuda:0")
    )
    _, calibration_acts = single.get_activations(prompt)
    for layer, transcoder in enumerate(transcoders):
        set_l0_via_thresholds(
            calibration_acts[layer : layer + 1],
            transcoder.activation_function.threshold,
            target_l0=16,
        )
    graph_single = attribute(prompt, single, batch_size=64)
    logits_single, acts_single = single.get_activations(prompt)
    del single
    torch.cuda.empty_cache()

    split = ReplacementModel.from_pretrained_and_transcoders(
        model_name,
        transcoders,
        backend="nnsight",
        device=torch.device("cuda:0"),
        device_map=_split_map(cfg.num_hidden_layers, cfg.num_hidden_layers // 2),
    )
    assert split.split_devices
    assert {p.device.index for p in split._module.parameters()} == {0, 1}
    graph_split = attribute(prompt, split, batch_size=64)
    logits_split, acts_split = split.get_activations(prompt)

    torch.testing.assert_close(
        logits_split.to(logits_single.device), logits_single, rtol=1e-4, atol=1e-4
    )
    torch.testing.assert_close(acts_split.to(acts_single.device), acts_single, rtol=1e-4, atol=1e-4)
    assert torch.equal(graph_split.active_features, graph_single.active_features)
    assert torch.equal(graph_split.logit_token_ids, graph_single.logit_token_ids)
    torch.testing.assert_close(
        graph_split.adjacency_matrix, graph_single.adjacency_matrix, rtol=1e-3, atol=1e-4
    )
