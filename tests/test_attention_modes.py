"""Attribution with attention patterns or norm scales left live (nnsight backend).

A frozen graph's edges are exact direct effects. With the patterns (or the norm scales) live, an edge is the first
derivative of the same direct effect, so each one is checked against a central difference of direct-effect
interventions run in the matching regime. The check is shown to discriminate: a live graph disagrees with the frozen
measurement, because it credits what moves the attention patterns (or the norm scales).

Precision: the model runs in float64, but Gemma3's RMSNorm and eager softmax compute in float32 internally, so each
forward carries about 1e-7 relative rounding that a central difference divides by its step. A step of 1e-2 balances
that against the truncation error. Measured on this model at that step, edges agree with their own regime's
measurement to within 1.4e-5 (patterns live) and 1.0e-4 (norms free) of the largest edge, and miss the frozen
measurement by 4.6e-2 to 0.41: the tolerance below sits between the two with a margin of about five on each side.
"""

import copy
from types import SimpleNamespace

import pytest
import torch

from circuit_tracer import attribute
from circuit_tracer.graph import Graph
from tests.test_attributions_gemma3_nnsight import gemma_3_config, load_dummy_gemma_model

TOKENS = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])
EPS = 1e-2
TOL = 5e-4  # of the largest edge


@pytest.fixture(scope="module")
def small_model():
    cfg = copy.deepcopy(gemma_3_config)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 8
    cfg.intermediate_size = 16
    cfg.head_dim = 4
    cfg.vocab_size = 16
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.final_logit_softcapping = None
    cfg.torch_dtype = "float32"
    torch.manual_seed(42)
    model = load_dummy_gemma_model(cfg)  # type: ignore[arg-type]
    # float64, so a central difference resolves an edge far below the tolerances used here
    model._module.to(torch.float64)
    model.transcoders.to(torch.float64)  # type: ignore[union-attr]
    tokenizer_class = type(model.tokenizer)
    original_all_special_ids = tokenizer_class.all_special_ids  # type: ignore[attr-defined]
    tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore[attr-defined]
    try:
        yield model
    finally:
        tokenizer_class.all_special_ids = original_all_special_ids  # type: ignore[attr-defined]


def _edges_from(graph: Graph, source: int) -> torch.Tensor:
    """A source's edges into every feature and every logit target, in the order ``_measured_effects`` returns."""
    n_features = graph.active_features.shape[0]
    n_logits = len(graph.logit_targets)
    column = graph.adjacency_matrix[:, source].double()
    return torch.cat([column[:n_features], column[-n_logits:]])


def _measured_effects(
    model, graph: Graph, source: int, freeze_attention: bool, freeze_norms: bool, eps=EPS
):
    """Central difference of every feature's pre-activation and every demeaned target logit with respect to
    the source's activation, in the edge's units (a relative change times the activation), under direct-effect
    interventions in the given regime."""
    features = graph.active_features.to(model.device)
    layer, pos, feature = (int(v) for v in features[source])
    _, acts = model.get_activations(TOKENS, apply_activation_function=False)
    activation = acts[layer, pos, feature]

    def run(scale: float) -> torch.Tensor:
        logits, cache = model.feature_intervention(
            TOKENS,
            [(layer, pos, feature, activation * (1 + scale))],
            constrained_layers=range(model.cfg.n_layers),
            freeze_attention=freeze_attention,
            freeze_norms=freeze_norms,
            apply_activation_function=False,
        )
        last = logits.squeeze(0)[-1]
        values = cache[features[:, 0], features[:, 1], features[:, 2]]
        return (
            torch.cat([values, last[graph.logit_token_ids.to(last.device)] - last.mean()])
            .double()
            .cpu()
        )

    return (run(eps) - run(-eps)) / (2 * eps)


def _layer0_sources(graph: Graph, n: int = 6) -> list[int]:
    """Layer-0 features before the last position: their effects reach the targets through layer 1's attention."""
    feats = graph.active_features
    candidates = torch.where((feats[:, 0] == 0) & (feats[:, 1] < len(TOKENS) - 1))[0]
    return candidates[:n].tolist()


def _gap(measured: torch.Tensor, expected: torch.Tensor) -> float:
    """Largest disagreement, as a fraction of the largest edge."""
    return float((measured - expected).abs().max() / expected.abs().max())


def test_patterns_live_edges_are_first_order_direct_effects(small_model):
    live = attribute(TOKENS, small_model, freeze_attention=False, batch_size=32)
    frozen = attribute(TOKENS, small_model, batch_size=32)
    sources = _layer0_sources(live)
    assert sources, "no layer-0 source before the last position to test"

    for source in sources:
        live_ref = _measured_effects(
            small_model, live, source, freeze_attention=False, freeze_norms=True
        )
        frozen_ref = _measured_effects(
            small_model, frozen, source, freeze_attention=True, freeze_norms=True
        )
        assert _gap(live_ref, _edges_from(live, source)) < TOL, f"live edges of source {source}"
        assert _gap(frozen_ref, _edges_from(frozen, source)) < TOL, (
            f"frozen edges of source {source}"
        )
        # the positive control: live edges carry QK credit the frozen measurement does not contain
        assert _gap(frozen_ref, _edges_from(live, source)) > 20 * TOL, (
            f"source {source} carries no QK credit"
        )


def test_free_norm_edges_are_first_order_direct_effects(small_model):
    free = attribute(TOKENS, small_model, freeze_norms=False, batch_size=32)
    sources = _layer0_sources(free)
    assert sources, "no layer-0 source before the last position to test"
    for source in sources:
        free_ref = _measured_effects(
            small_model, free, source, freeze_attention=True, freeze_norms=False
        )
        frozen_ref = _measured_effects(
            small_model, free, source, freeze_attention=True, freeze_norms=True
        )
        assert _gap(free_ref, _edges_from(free, source)) < TOL, (
            f"free-norm edges of source {source}"
        )
        assert _gap(frozen_ref, _edges_from(free, source)) > 20 * TOL, (
            f"source {source} carries no norm credit"
        )


def test_the_regime_is_recorded_and_survives_a_round_trip(small_model, tmp_path):
    live = attribute(TOKENS, small_model, freeze_attention=False, batch_size=32)
    assert live.attribution_regime == {"freeze_attention": False, "freeze_norms": True}
    live.to_pt(str(tmp_path / "live.pt"))
    assert Graph.from_pt(str(tmp_path / "live.pt")).attribution_regime == live.attribution_regime
    assert attribute(TOKENS, small_model, batch_size=32).attribution_regime == {
        "freeze_attention": True,
        "freeze_norms": True,
    }


@pytest.mark.parametrize("backend", ["transformerlens", "interp_engine"])
def test_live_regimes_are_refused_on_other_backends(backend):
    with pytest.raises(ValueError, match="nnsight backend only"):
        attribute(TOKENS, SimpleNamespace(backend=backend), freeze_attention=False)  # type: ignore[arg-type]
