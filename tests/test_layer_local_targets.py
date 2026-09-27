"""Layer-local custom targets: a CustomTarget with ``layer`` set is read at the output of that block.

Validated as the rest of circuit-tracer is: under the full freeze (attention patterns, norm denominators and every MLP
output fixed at clean values), perturbing a source node changes the target's read by exactly the graph edge.
"""

import copy

import pytest
import torch

from circuit_tracer import attribute
from circuit_tracer.attribution.targets import AttributionTargets, CustomTarget
from tests.test_attributions_gemma_nnsight import gemma_2_config, load_dummy_gemma_model

PROMPT = torch.tensor([0, 3, 4, 3, 2, 5, 3, 8])


def _small_model():
    cfg = copy.deepcopy(gemma_2_config)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 8
    cfg.intermediate_size = 16
    cfg.head_dim = 4
    cfg.vocab_size = 16
    cfg.num_attention_heads = 2
    cfg.num_key_value_heads = 2
    cfg.final_logit_softcapping = None  # type: ignore
    cfg.torch_dtype = "float32"
    torch.manual_seed(661)
    return load_dummy_gemma_model(cfg)


@pytest.fixture
def model():
    m = _small_model()
    tokenizer_class = type(m.tokenizer)
    original = tokenizer_class.all_special_ids  # type: ignore
    tokenizer_class.all_special_ids = property(lambda self: [0])  # type: ignore
    try:
        yield m
    finally:
        tokenizer_class.all_special_ids = original  # type: ignore


def _block_module(model, layer):
    envoy = getattr(model.pre_logit_location, "layers")[layer]
    return getattr(envoy, "_module", envoy)


def _block_output_under_intervention(model, layer, interventions):
    """Block ``layer``'s output (all positions) during a fully frozen feature intervention."""
    store = {}

    def hook(mod, inp, out):
        store["h"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    handle = _block_module(model, layer).register_forward_hook(hook)
    try:
        if interventions:
            model.feature_intervention(
                PROMPT, interventions, constrained_layers=range(model.cfg.n_layers), apply_activation_function=False
            )
        else:
            model.get_activations(PROMPT, apply_activation_function=False)
    finally:
        handle.remove()
    return store["h"].squeeze(0)


def test_layer_local_target_edges_match_frozen_interventions(model):
    layer, pos = 0, 5
    torch.manual_seed(0)
    v = torch.randn(model.cfg.d_model)
    graph = attribute(PROMPT, model, attribution_targets=[CustomTarget("jl@0", 1.0, v, layer=layer, position=pos)])
    row = graph.adjacency_matrix[-1]  # the single target is the last node
    feats = graph.active_features
    _, acts = model.get_activations(PROMPT, apply_activation_function=False)
    base = _block_output_under_intervention(model, layer, [])[pos] @ v

    upstream = [i for i, (l, p, _) in enumerate(feats.tolist()) if l <= layer and p <= pos]
    downstream = [i for i, (l, p, _) in enumerate(feats.tolist()) if l > layer or p > pos]
    assert upstream and downstream, "the toy prompt must exercise both cases"
    # nothing downstream of the read site can reach it
    assert torch.all(row[downstream] == 0)
    checked = 0
    for i in upstream[:40]:
        l, p, f = feats[i].tolist()
        old = acts[l, p, f]
        h = _block_output_under_intervention(model, layer, [(l, p, f, old * 2)])
        measured = float(h[pos] @ v - base)
        assert measured == pytest.approx(float(row[i]), abs=5e-4, rel=1e-3), (l, p, f)
        checked += 1
    assert checked > 0


def test_last_block_target_matches_final_residual_target(model):
    """At the last block, a block-output read equals a final-residual read through the frozen final norm."""
    n = model.cfg.n_layers
    pos = len(PROMPT) - 1
    torch.manual_seed(1)
    u = torch.randn(model.cfg.d_model)
    final = attribute(PROMPT, model, attribution_targets=[CustomTarget("final", 1.0, u)])
    h = _block_output_under_intervention(model, n - 1, [])[pos]
    norm = model.pre_logit_location.norm
    norm_mod = getattr(norm, "_module", norm)
    # gemma RMSNorm: y = x * rsqrt(mean(x^2) + eps) * (1 + w); the denominator is frozen during attribution
    scale = torch.rsqrt(h.pow(2).mean() + norm_mod.eps)
    v = u * (1 + norm_mod.weight.detach()) * scale
    local = attribute(PROMPT, model, attribution_targets=[CustomTarget("local", 1.0, v, layer=n - 1, position=pos)])
    assert torch.equal(final.active_features, local.active_features)
    assert torch.allclose(final.adjacency_matrix[-1], local.adjacency_matrix[-1], atol=1e-5, rtol=1e-4)


def test_default_custom_target_unchanged(model):
    """A CustomTarget with no layer or position behaves exactly as before."""
    torch.manual_seed(2)
    u = torch.randn(model.cfg.d_model)
    plain = attribute(PROMPT, model, attribution_targets=[("t", 1.0, u)])
    named = attribute(PROMPT, model, attribution_targets=[CustomTarget("t", 1.0, u)])
    assert torch.equal(plain.adjacency_matrix, named.adjacency_matrix)


@pytest.mark.parametrize("kwargs, message", [({"layer": 2}, "not a block"), ({"layer": 0, "position": 99}, "outside the prompt")])
def test_out_of_range_sites_are_refused(model, kwargs, message):
    with pytest.raises(ValueError, match=message):
        attribute(PROMPT, model, attribution_targets=[CustomTarget("bad", 1.0, torch.ones(model.cfg.d_model), **kwargs)])


def test_site_fields_are_validated():
    with pytest.raises(TypeError, match="non-negative int"):
        AttributionTargets._validate_custom_target(CustomTarget("x", 1.0, torch.ones(4), layer=-1))
    with pytest.raises(ValueError, match="CustomTarget with layer="):
        AttributionTargets._validate_custom_target(("x", 1.0, torch.ones(4), 0, 1))  # type: ignore[arg-type]
