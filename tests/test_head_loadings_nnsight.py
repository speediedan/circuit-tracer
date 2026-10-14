"""Head loadings on the nnsight backend: the parts sum to the edge, the edge is the graph's own, and each part belongs
to its head (a head silenced in the model carries nothing, and the others do not move)."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from nnsight import save

from circuit_tracer import attribute
from circuit_tracer.attribution.head_loadings_nnsight import head_loadings
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU
from tests.test_attention_modes import TOKENS, small_model  # noqa: F401  (fixture)
from tests.test_attributions_gemma3_nnsight import gemma_3_config


@pytest.fixture(scope="module")
def small_skip_model():
    """The same toy Gemma3, with skip transcoders: their linear MLP path is part of every bypass."""
    cfg = copy.deepcopy(gemma_3_config)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 8
    cfg.intermediate_size = 16
    cfg.head_dim = 4
    cfg.vocab_size = 16
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.final_logit_softcapping = None
    torch.manual_seed(7)
    transcoders = {
        layer: SingleLayerTranscoder(8, 32, JumpReLU(0.0, 0.1), layer, skip_connection=True)
        for layer in range(cfg.num_hidden_layers)
    }
    for transcoder in transcoders.values():
        for param in transcoder.parameters():
            nn.init.uniform_(param, a=-1, b=1)
    model = ReplacementModel.from_config(
        cfg, TranscoderSet(transcoders, "hook_resid_mid", "hook_mlp_out"), backend="nnsight"
    )
    for param in model.parameters():
        nn.init.uniform_(param, a=-1, b=1)
    for transcoder in transcoders.values():
        nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)
    model._module.to(torch.float64)
    model.transcoders.to(torch.float64)  # type: ignore[union-attr]
    model.tokenizer.pad_token = model.tokenizer.eos_token  # type: ignore[union-attr]
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type: ignore[attr-defined]
    tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore[attr-defined]
    try:
        yield model
    finally:
        tokenizer_class.all_special_ids = original_all_special_ids  # type: ignore[attr-defined]


def _graph_row(graph, target) -> torch.Tensor:
    """The graph's adjacency row for ``target`` over its source columns (features, errors, tokens)."""
    n_sources = graph.adjacency_matrix.shape[1] - len(graph.logit_targets)
    if isinstance(target, tuple):
        index = int(
            torch.where((graph.active_features == torch.tensor(target)).all(dim=1))[0][0]
        )  # every feature is selected here, in active order
        return graph.adjacency_matrix[index, :n_sources].double()
    return graph.adjacency_matrix[-len(graph.logit_targets), :n_sources].double()


def _targets(graph):
    """A logit target and a layer-1 feature target."""
    layer1 = graph.active_features[graph.active_features[:, 0] == 1]
    feature = tuple(int(v) for v in layer1[0])
    return [torch.tensor([graph.logit_targets[0].vocab_idx]), feature]


@pytest.mark.parametrize("model_name", ["small_model", "small_skip_model"])
@pytest.mark.parametrize("freeze_attention", [True, False], ids=["frozen", "patterns_live"])
def test_parts_sum_to_the_graph_edge(request, model_name, freeze_attention):
    model = request.getfixturevalue(model_name)
    graph = attribute(TOKENS, model, batch_size=32, freeze_attention=freeze_attention)
    for target in _targets(graph):
        for layer in (0, 1):
            loads = head_loadings(model, TOKENS, target, layer, freeze_attention=freeze_attention)
            # Measured on these models: partition errors up to 9e-8 and graph gaps up to 4e-8 (the copies run Gemma3's
            # float32-internal norms, and the graph stores edges in float32), against head parts of 3% to 143% of the
            # largest edge; a misassigned head would miss by order one.
            assert loads.partition_error < 1e-6, (target, layer)
            expected = _graph_row(graph, target)
            gap = float((loads.total - expected).abs().max() / expected.abs().max())
            assert gap < 1e-6, (target, layer, gap)
            assert loads.per_head.abs().max() > 1e-3 * expected.abs().max(), (
                "no edge passes these heads"
            )


def test_head_parts_match_an_explicit_per_head_propagation(small_model):  # noqa: F811
    """Each head's part, recomputed by hand through that head alone.

    A layer-0 feature's write reaches a layer-1 feature's pre-activation (read before layer 1's MLP) through layer 1's
    attention only: the input norm and the post-attention norm with their scales frozen at the clean values, the
    head's value projection, its frozen pattern and its slice of the output projection. That chain, written out per
    head, must give the head loadings, so a misassigned head or slice cannot pass.
    """
    model = small_model
    graph = attribute(TOKENS, model, batch_size=32)
    layer1 = graph.active_features[graph.active_features[:, 0] == 1]
    target = tuple(
        int(v) for v in layer1[layer1[:, 1] == layer1[:, 1].max()][0]
    )  # latest position: most sources
    loads = head_loadings(model, TOKENS, target, layer=1)

    with torch.no_grad(), model.trace(TOKENS):  # read in forward order, as nnsight requires
        block = model.model.layers[1]
        x_in = save(block.input_layernorm.input)
        pattern = save(model._resolve_attr(model, model._attention_pattern.format(layer=1)).output)
        x_attn = save(block.post_attention_layernorm.input)
    hf_block = model._module.model.layers[1]
    eps = hf_block.input_layernorm.eps

    def frozen_norm(norm, x_clean, delta):
        # Gemma3 RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w), with the rsqrt held at its clean value
        scale = torch.rsqrt(x_clean.double().pow(2).mean(-1, keepdim=True) + eps)
        return delta * scale * (1.0 + norm.weight.double())

    n_heads, d_head = model.cfg.n_heads, model.cfg.d_head
    n_kv = model.cfg.n_key_value_heads or n_heads
    transcoders = model.transcoders._module  # the TranscoderSet the nnsight envoy wraps
    enc = transcoders[1].W_enc[target[2]].double()  # type: ignore[index]
    v_weight = hf_block.self_attn.v_proj.weight.double()
    o_weight = hf_block.self_attn.o_proj.weight.double()
    q = target[1]
    checked = swapped_misses = 0
    for row, (layer, pos, feature) in enumerate(loads.active_features.tolist()):
        if layer != 0 or pos > q:
            continue
        activation = graph.activation_values[row].double()
        write = torch.zeros(len(TOKENS), model.cfg.d_model, dtype=torch.float64)
        write[pos] = transcoders[0].W_dec[feature].double() * activation  # type: ignore[index]
        normed = frozen_norm(hf_block.input_layernorm, x_in[0], write)
        for head in range(n_heads):
            kv = head // (n_heads // n_kv)
            values = normed @ v_weight[kv * d_head : (kv + 1) * d_head].T
            moved = pattern[0, head].double() @ values
            written = moved @ o_weight[:, head * d_head : (head + 1) * d_head].T
            arriving = frozen_norm(hf_block.post_attention_layernorm, x_attn[0], written)
            expected = float(arriving[q] @ enc)
            got = float(loads.per_head[head, row])
            assert abs(got - expected) <= 1e-6 * max(1.0, abs(expected)), (row, head, got, expected)
            # the control: the other head's part does not fit this head's chain, so a swap would be caught
            other = float(loads.per_head[(head + 1) % n_heads, row])
            swapped_misses += abs(other - expected) > 1e-3 * max(1.0, abs(expected))
            checked += 1
    assert checked >= 2 * n_heads, f"only {checked} head parts checked"
    assert swapped_misses > 0, (
        "every head's part also fits the other head's chain; the check cannot tell them apart"
    )


def test_sources_written_after_the_layer_bypass_it(small_model):  # noqa: F811
    graph = attribute(TOKENS, small_model, batch_size=32)
    loads = head_loadings(
        small_model, TOKENS, torch.tensor([graph.logit_targets[0].vocab_idx]), layer=1
    )
    written_after = (
        loads.active_features[:, 0] >= 1
    )  # a layer-1 feature writes after layer 1's attention
    assert written_after.any()
    assert loads.per_head[:, loads.feature_rows()][:, written_after].abs().max() == 0


def test_refusals(small_model):  # noqa: F811
    graph = attribute(TOKENS, small_model, batch_size=32)
    layer0 = tuple(int(v) for v in graph.active_features[graph.active_features[:, 0] == 0][0])
    with pytest.raises(ValueError, match="passes attention layer 1"):
        head_loadings(small_model, TOKENS, layer0, layer=1)
    with pytest.raises(ValueError, match="not active on this prompt"):
        head_loadings(small_model, TOKENS, (1, 0, 10**6), layer=0)
    with pytest.raises(ValueError, match="nnsight backend only"):
        head_loadings(SimpleNamespace(backend="transformerlens"), TOKENS, layer0, layer=0)  # type: ignore[arg-type]
