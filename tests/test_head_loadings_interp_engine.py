"""Attention-aware attribution on the interp_engine backend, checked against the nnsight backend on the same weights.

The nnsight backend's regimes and head loadings have their own first-principles tests (``test_attention_modes.py``,
``test_head_loadings_nnsight.py``). Here the interp_engine backend runs the same toy Gemma3 with the same transcoders,
so its graphs, head loadings and head conductance must equal nnsight's, regime by regime. Each regime is also checked to
change the graph, since two backends that both ignored a regime would agree.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from circuit_tracer import attribute
from circuit_tracer.attribution.head_conductance import head_conductance
from circuit_tracer.attribution.head_loadings import head_loadings
from circuit_tracer.attribution.head_loadings_interp_engine import (
    head_loadings as head_loadings_interp_engine,
)
from circuit_tracer.attribution.targets import CustomTarget
from circuit_tracer.replacement_model import ReplacementModel
from circuit_tracer.transcoder import SingleLayerTranscoder, TranscoderSet
from circuit_tracer.transcoder.activation_functions import JumpReLU
from tests.test_attributions_gemma3_nnsight import gemma_3_config

TOKENS = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
REGIMES = {
    "frozen": dict(freeze_attention=True, freeze_norms=True),
    "patterns_live": dict(freeze_attention=False, freeze_norms=True),
    "norms_live": dict(freeze_attention=True, freeze_norms=False),
    "both_live": dict(freeze_attention=False, freeze_norms=False),
}
# Both backends run the same float64 arithmetic in a different order; measured gaps here are at float32 storage noise
# (graphs store edges in float32), against edges of order one.
TOL = 1e-6


def _rel_gap(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return float((a - b).abs().max() / b.abs().max())


@pytest.fixture(scope="module", params=[False, True], ids=["plain", "skip"])
def models(request):
    """The same toy Gemma3 on both backends: nnsight first, then interp_engine loaded with its weights."""
    cfg = copy.deepcopy(gemma_3_config)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 8
    cfg.intermediate_size = 16
    cfg.head_dim = 4
    # the toy config borrows GPT-2's tokenizer, and interp_engine's load-time probe runs its BOS token (50256)
    cfg.vocab_size = 50257
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.final_logit_softcapping = None
    torch.manual_seed(11)
    transcoders = {
        layer: SingleLayerTranscoder(
            8, 32, JumpReLU(0.0, 0.1), layer, skip_connection=request.param
        )
        for layer in range(cfg.num_hidden_layers)
    }
    nnsight = ReplacementModel.from_config(
        cfg, TranscoderSet(transcoders, "hook_resid_mid", "hook_mlp_out"), backend="nnsight"
    )
    for param in nnsight.parameters():
        nn.init.uniform_(param, a=-1, b=1)
    for transcoder in nnsight.transcoders._module:  # type: ignore[union-attr]
        nn.init.uniform_(transcoder.activation_function.threshold, a=0, b=1)

    interp_engine = ReplacementModel.from_config(
        cfg,
        copy.deepcopy(nnsight.transcoders._module),
        backend="interp_engine",  # type: ignore[union-attr]
    )
    # nnsight registers the transcoders under its causal LM; the interp_engine model holds its own copy above
    causal_lm = {
        k: v for k, v in nnsight._module.state_dict().items() if not k.startswith("transcoders.")
    }
    interp_engine.hf_model.load_state_dict(causal_lm)
    for model, module in ((nnsight, nnsight._module), (interp_engine, interp_engine.hf_model)):
        module.to(torch.float64)
        model.transcoders.to(torch.float64)  # type: ignore[union-attr]

    tokenizer_class = type(nnsight.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type: ignore[attr-defined]
    tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore[attr-defined]
    try:
        yield nnsight, interp_engine
    finally:
        tokenizer_class.all_special_ids = original_all_special_ids  # type: ignore[attr-defined]


def _targets(graph):
    """A logit target and a layer-1 feature target."""
    layer1 = graph.active_features[graph.active_features[:, 0] == 1]
    feature = tuple(int(v) for v in layer1[layer1[:, 1] == layer1[:, 1].max()][0])
    return [torch.tensor([graph.logit_targets[0].vocab_idx]), feature]


@pytest.mark.parametrize("regime", list(REGIMES))
def test_graphs_match_nnsight_in_every_regime(models, regime):
    nnsight, interp_engine = models
    expected = attribute(TOKENS, nnsight, batch_size=32, **REGIMES[regime])
    graph = attribute(TOKENS, interp_engine, batch_size=32, **REGIMES[regime])
    assert graph.attribution_regime == REGIMES[regime]
    assert torch.equal(graph.active_features, expected.active_features)
    assert _rel_gap(graph.adjacency_matrix, expected.adjacency_matrix) < TOL
    if regime != "frozen":
        # the control: the regime changes the graph, so agreeing above is not two backends that both ignore it
        frozen = attribute(TOKENS, interp_engine, batch_size=32)
        assert _rel_gap(graph.adjacency_matrix, frozen.adjacency_matrix) > 1e-3, regime


@pytest.mark.parametrize("regime", list(REGIMES))
def test_head_loadings_match_nnsight(models, regime):
    nnsight, interp_engine = models
    graph = attribute(TOKENS, nnsight, batch_size=32)
    largest_head_share = 0.0
    for target in _targets(graph):
        for layer in (0, 1):
            expected = head_loadings(nnsight, TOKENS, target, layer, **REGIMES[regime])
            loads = head_loadings(interp_engine, TOKENS, target, layer, **REGIMES[regime])
            assert loads.partition_error < TOL, (target, layer)
            for name in ("per_head", "bypass", "total"):
                gap = _rel_gap(getattr(loads, name), getattr(expected, name))
                assert gap < TOL, (target, layer, name, gap)
            assert torch.equal(loads.active_features, expected.active_features)
            largest_head_share = max(
                largest_head_share, float(loads.per_head.abs().max() / loads.total.abs().max())
            )
    # the comparison is not vacuous: some split carries real head paths (a split of zeros would also agree)
    assert largest_head_share > 1e-2, largest_head_share


def test_head_loadings_refusals(models):
    _, interp_engine = models
    graph = attribute(TOKENS, interp_engine, batch_size=32)
    layer0 = tuple(int(v) for v in graph.active_features[graph.active_features[:, 0] == 0][0])
    with pytest.raises(ValueError, match="passes attention layer 1"):
        head_loadings(interp_engine, TOKENS, layer0, layer=1)
    with pytest.raises(ValueError, match="not active on this prompt"):
        head_loadings(interp_engine, TOKENS, (1, 0, 10**6), layer=0)
    read_mid_stack = CustomTarget(
        token_str="mid", prob=1.0, vec=torch.ones(8, dtype=torch.float64), layer=1, position=3
    )
    with pytest.raises(NotImplementedError, match="nnsight backend only"):
        head_loadings(interp_engine, TOKENS, read_mid_stack, layer=0)
    with pytest.raises(ValueError, match="nnsight and interp_engine backends only"):
        head_loadings(SimpleNamespace(backend="transformerlens"), TOKENS, layer0, layer=0)
    with pytest.raises(ValueError, match="for the interp_engine backend"):
        head_loadings_interp_engine(SimpleNamespace(backend="nnsight"), TOKENS, layer0, layer=0)  # type: ignore[arg-type]


def test_head_conductance_runs_on_the_real_model(models):
    """Through ``real_model`` the conductance is the plain causal LM's, and the replacement model is intact after."""
    nnsight, interp_engine = models
    ids_to = TOKENS.clone()
    ids_to[3] = 6
    weights = torch.zeros(interp_engine.cfg.d_vocab, dtype=torch.float64)
    weights[2], weights[5] = 1.0, -1.0
    before = attribute(TOKENS, interp_engine, batch_size=32)

    expected = head_conductance(nnsight._module, TOKENS, ids_to, weights, n_steps=8)
    with interp_engine.real_model() as causal_lm:
        report = head_conductance(causal_lm, TOKENS, ids_to, weights, n_steps=8)
    for name in ("qk", "ov", "resid"):
        assert _rel_gap(getattr(report, name), getattr(expected, name)) < TOL, name
    assert report.t_to - report.t_from == pytest.approx(expected.t_to - expected.t_from, rel=TOL)
    assert report.head.abs().max() > 1e-3 * abs(report.t_to - report.t_from), (
        "no head conducts anything"
    )

    # outside it the model is the replacement model again: frozen attention, and the same graph as before
    with pytest.raises(ValueError, match="eager"):
        head_conductance(interp_engine.hf_model, TOKENS, ids_to, weights, n_steps=8)
    after = attribute(TOKENS, interp_engine, batch_size=32)
    assert torch.equal(after.adjacency_matrix, before.adjacency_matrix)
