"""Per-head conductance: the parts of every layer's cut add up to the measured change, the endpoints are the real
model's, and a head whose pattern cannot move gets no pattern credit."""

from typing import Any

import pytest
import torch
from transformers import Gemma3ForCausalLM, Gemma3TextConfig

from circuit_tracer.attribution.head_conductance import head_conductance


def _tiny(attn_implementation: str = "eager") -> Gemma3ForCausalLM:
    torch.manual_seed(0)
    cfg = Gemma3TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=64,
    )
    cfg._attn_implementation = attn_implementation
    model = Gemma3ForCausalLM(cfg).eval()
    with torch.no_grad():
        for param in (
            model.parameters()
        ):  # larger than the default init, so patterns move visibly along the path
            param.mul_(8.0) if param.ndim > 1 else None
    return model


IDS_FROM = torch.tensor([2, 5, 7, 9, 11, 13, 4, 6])
IDS_TO = torch.tensor([2, 5, 19, 23, 11, 13, 4, 6])  # two tokens swapped, as a name swap is
WEIGHTS = torch.zeros(32)
WEIGHTS[3], WEIGHTS[8] = 1.0, -1.0  # a logit difference


def test_every_cut_adds_up_to_the_measured_change():
    """The completeness residual is the trapezoid rule's error alone: it falls as 1/K^2 with the step count K.

    A cut missing a path would leave a residual that does not shrink. Measured on this model: residuals of 1.4e-3,
    6.5e-4 and 4.4e-4 at 64 steps, each a quarter of its value at 32 steps, the same in float32 and float64.
    """
    model = _tiny()
    coarse = head_conductance(model, IDS_FROM, IDS_TO, WEIGHTS, n_steps=32, points_per_batch=5)
    fine = head_conductance(model, IDS_FROM, IDS_TO, WEIGHTS, n_steps=64, points_per_batch=17)
    with torch.no_grad():
        t = [
            float(model(input_ids=i.unsqueeze(0)).logits[0, -1] @ WEIGHTS)
            for i in (IDS_FROM, IDS_TO)
        ]
    assert fine.t_from == pytest.approx(t[0], rel=1e-5) and fine.t_to == pytest.approx(
        t[1], rel=1e-5
    )
    assert abs(fine.delta) > 1e-2, "the swap barely moves the target; the test would be vacuous"
    ratio = coarse.completeness_residual / fine.completeness_residual
    assert ((ratio > 3.6) & (ratio < 4.4)).all(), (
        ratio
    )  # second-order convergence, nothing left over
    assert fine.completeness_residual.abs().max() < 2e-3
    assert fine.qk.abs().max() > 1e-3 * abs(fine.delta), (
        "no pattern credit anywhere; the split is untested"
    )
    # the path derivative integrates to the same change (the input end of the cut), and the path is recorded
    integral = float(torch.trapezoid(fine.dt_dalpha, dx=1.0 / fine.n_steps))
    assert integral == pytest.approx(fine.delta, rel=2e-3)
    assert fine.t_path[0] == pytest.approx(t[0], rel=1e-5)
    assert fine.t_path[-1] == pytest.approx(t[1], rel=1e-5)
    assert len(fine.t_path) == len(fine.dt_dalpha) == fine.n_steps + 1


def test_a_pattern_that_cannot_move_gets_no_pattern_credit():
    model = _tiny()
    with (
        torch.no_grad()
    ):  # zero queries: every score is 0, so layer 1's pattern is uniform whatever the input
        layer: Any = model.model.layers[1]
        layer.self_attn.q_proj.weight.zero_()
    result = head_conductance(model, IDS_FROM, IDS_TO, WEIGHTS, n_steps=128, points_per_batch=33)
    assert result.qk[1].abs().max() < 1e-9 * abs(result.delta)
    assert result.ov[1].abs().max() > 1e-4 * abs(result.delta), (
        "layer 1 carries nothing; the control is vacuous"
    )
    assert result.qk[2].abs().max() > 1e-4 * abs(result.delta)
    assert result.completeness_residual.abs().max() < 5e-3


def test_refusals():
    with pytest.raises(ValueError, match="eager"):
        head_conductance(_tiny("sdpa"), IDS_FROM, IDS_TO, WEIGHTS)
    with pytest.raises(ValueError, match="same length"):
        head_conductance(_tiny(), IDS_FROM, IDS_TO[:-1], WEIGHTS)
