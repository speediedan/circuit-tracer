"""The TransformerLens backend on a ``TransformerBridge`` attributes what the model computes.

TransformerLens 4 removed ``HookedTransformer``; the backend now wraps the HuggingFace model in a
bridge and reads the bridge's TransformerLens-named hooks. ``test_attributions_gemma`` and
``test_attributions_llama`` check that every edge predicts the measured effect of an intervention,
which holds for any token vectors the graph and the intervention agree on. These tests pin what
that check cannot see:

- the token vectors are the embedding the model actually uses: with nothing left unattributed, the
  edges into a raw logit sum to the logit, on Gemma 2 (whose embedding is scaled by
  $\\sqrt{d_{model}}$) and on Llama;
- on the same weights, the bridge's graph is the nnsight backend's graph;
- generating with interventions through the bridge's KV cache gives, at every step, the
  intervened forward pass of the prefix so far.

Hermetic apart from the tokenizers, which come from the cache or the Hub. CPU only.
"""

import copy
from typing import Any

import pytest
import torch
import torch.nn as nn
from transformer_lens.model_bridge import TransformerBridge
from transformers import AutoModelForCausalLM, AutoTokenizer

from circuit_tracer import attribute
from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel
from circuit_tracer.replacement_model.replacement_model_transformerlens import (
    TransformerLensReplacementModel,
)
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU
from tests.tl_bridge_configs import gpt2_tokenizer, hf_config_from_tl_dict

TOKENS = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
SMALL = dict(n_layers=2, d_model=8, d_head=4, n_heads=2, d_mlp=16, d_vocab=16)


def _config(architecture: str):
    return hf_config_from_tl_dict({**SMALL, "original_architecture": architecture})


def _transcoders(d_model: int, n_layers: int, input_hook: str, output_hook: str) -> TranscoderSet:
    transcoders = {
        layer: SingleLayerTranscoder(d_model, d_model * 4, JumpReLU(torch.tensor(0.0), 0.1), layer)
        for layer in range(n_layers)
    }
    for transcoder in transcoders.values():
        for _, param in transcoder.named_parameters():
            nn.init.uniform_(param, a=-1, b=1)
        nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)
    return TranscoderSet(
        transcoders, feature_input_hook=input_hook, feature_output_hook=output_hook
    )


def _random_hf_model(config) -> nn.Module:
    config = copy.deepcopy(config)
    config._attn_implementation = "eager"
    hf_model = AutoModelForCausalLM.from_config(config)
    for _, param in hf_model.named_parameters():
        nn.init.uniform_(param, a=-1, b=1)
    return hf_model.eval()


def _tl_model(hf_model, tokenizer, transcoders) -> TransformerLensReplacementModel:
    bridge = TransformerBridge.boot_transformers(
        hf_model.config._name_or_path, hf_model=hf_model, tokenizer=tokenizer, device="cpu"
    )
    return TransformerLensReplacementModel.from_bridge(bridge, transcoders)


class _SpecialIds:
    """Treat only token 0 as special, so every other position is attributed."""

    def __init__(self, tokenizer):
        self.cls = type(tokenizer)

    def __enter__(self):
        self.original = self.cls.all_special_ids
        self.cls.all_special_ids = property(lambda self: [0])  # type: ignore[assignment]

    def __exit__(self, *exc):
        self.cls.all_special_ids = self.original  # type: ignore[assignment]


@pytest.mark.parametrize("architecture", ["Gemma2ForCausalLM", "LlamaForCausalLM"])
def test_edges_into_a_logit_sum_to_it(architecture):
    torch.manual_seed(1)
    config = _config(architecture)
    transcoders = _transcoders(8, 2, "mlp.hook_in", "mlp.hook_out")
    model = _tl_model(_random_hf_model(config), gpt2_tokenizer(), transcoders)
    if architecture == "Gemma2ForCausalLM":
        normalizer = float(model.original_model.model.config.hidden_size) ** 0.5
        assert normalizer > 1.0, "the embedding is not scaled; the test would be vacuous"
    with torch.no_grad():
        for transcoder in model.transcoders:  # nothing outside the nodes: no decoder bias
            transcoder.b_dec.zero_()
    model.zero_positions = slice(0, 0)

    with _SpecialIds(model.tokenizer):
        target = CustomTarget("logit_3", 1.0, model.unembed_weight.detach()[3])
        graph = attribute(TOKENS, model, attribution_targets=[target], batch_size=16)
        logits, _ = model.get_activations(TOKENS)

    n_sources = graph.adjacency_matrix.shape[1] - 1
    edges = graph.adjacency_matrix[-1, :n_sources].double()
    token_edges = edges[-len(TOKENS) :]
    assert token_edges.abs().max() > 1e-3 * edges.abs().max(), "token nodes carry nothing"
    assert float(edges.sum()) == pytest.approx(float(logits[0, -1, 3]), rel=1e-4, abs=1e-4)


def test_the_graph_is_the_nnsight_backends_on_the_same_weights():
    torch.manual_seed(2)
    config = _config("Gemma2ForCausalLM")
    transcoders = _transcoders(8, 2, "hook_resid_mid", "hook_mlp_out")
    hf_model = _random_hf_model(config)
    tokenizer = AutoTokenizer.from_pretrained(config._name_or_path)

    nnsight_model: Any = NNSightReplacementModel.from_config(
        copy.deepcopy(config), copy.deepcopy(transcoders)
    )
    # nnsight attaches the transcoders to the HF module, so they are the only keys the copy lacks
    missing, unexpected = nnsight_model._module.load_state_dict(hf_model.state_dict(), strict=False)
    assert not unexpected and all(key.startswith("transcoders.") for key in missing), (
        missing,
        unexpected,
    )
    tl_model = _tl_model(hf_model, tokenizer, transcoders)

    with torch.no_grad():
        expected = nnsight_model._module(TOKENS.unsqueeze(0)).logits
        logits = tl_model(TOKENS.unsqueeze(0))
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)

    graphs = []
    for model in (nnsight_model, tl_model):
        with _SpecialIds(model.tokenizer):
            graphs.append(attribute(TOKENS, model, batch_size=16))
    nnsight_graph, tl_graph = graphs
    assert torch.equal(tl_graph.active_features, nnsight_graph.active_features)
    assert torch.equal(tl_graph.logit_token_ids, nnsight_graph.logit_token_ids)
    assert tl_graph.adjacency_matrix.abs().max() > 1e-3, "an empty graph would match trivially"
    torch.testing.assert_close(
        tl_graph.adjacency_matrix, nnsight_graph.adjacency_matrix, atol=1e-4, rtol=1e-3
    )


def test_each_generated_step_is_the_intervened_forward_of_its_prefix():
    """Generation reuses the bridge's KV cache, with open-ended interventions moved to each new
    token; every step's logits must still be the intervened forward pass of the prefix so far."""
    torch.manual_seed(3)
    config = _config("Gemma2ForCausalLM")
    model = _tl_model(
        _random_hf_model(config),
        gpt2_tokenizer(),
        _transcoders(8, 2, "mlp.hook_in", "mlp.hook_out"),
    )
    _, activations = model.get_activations(TOKENS)
    layer, pos, feature = (int(x) for x in activations.nonzero()[-1])
    interventions = [(layer, slice(pos, None), feature, 0.0)]

    generated, logits, _ = model.feature_intervention_generate(
        TOKENS,
        interventions,
        freeze_attention=False,
        do_sample=False,
        max_new_tokens=3,
        verbose=False,
    )
    new_tokens = generated[0, len(TOKENS) :]
    assert logits.shape == (3, config.vocab_size)
    assert torch.equal(logits.argmax(-1), new_tokens)
    plain = model.generate(TOKENS.unsqueeze(0), max_new_tokens=3, do_sample=False, verbose=False)
    assert not torch.equal(plain[0, len(TOKENS) :], new_tokens), "the intervention changed nothing"

    prefix = TOKENS
    for step in range(3):
        expected, _ = model.feature_intervention(prefix, interventions, freeze_attention=False)
        torch.testing.assert_close(logits[step], expected[0, -1], atol=1e-5, rtol=1e-5)
        prefix = torch.cat([prefix, new_tokens[step : step + 1]])


@pytest.mark.requires_disk
@pytest.mark.parametrize("tied", [True, False])
def test_disk_offload_gives_the_graph_of_no_offload(tied):
    """A bridge lists its MLP weights twice and shares a tied embedding: disk offload handles both."""
    torch.manual_seed(1)
    config = _config("Gemma2ForCausalLM")
    config.tie_word_embeddings = tied
    model = _tl_model(
        _random_hf_model(config),
        gpt2_tokenizer(),
        _transcoders(8, 2, "mlp.hook_in", "mlp.hook_out"),
    )
    names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    assert len(names) > len(dict(model.named_parameters())), "no weight is listed twice"

    with _SpecialIds(model.tokenizer):
        graph = attribute(TOKENS, model, batch_size=16, offload=None)
        offloaded = attribute(TOKENS, model, batch_size=16, offload="disk")
        again = attribute(TOKENS, model, batch_size=16, offload=None)

    torch.testing.assert_close(offloaded.adjacency_matrix, graph.adjacency_matrix)
    torch.testing.assert_close(again.adjacency_matrix, graph.adjacency_matrix)
    assert all(param.device.type == "cpu" for param in model.parameters())
    hf_model = model.original_model
    assert (hf_model.lm_head.weight is hf_model.model.embed_tokens.weight) == tied
