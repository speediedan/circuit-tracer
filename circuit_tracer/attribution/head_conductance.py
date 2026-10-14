r"""Per-head conductance along a discrete input change, split into its attention-pattern and value parts.

Attribution graphs are local: an edge is exact for a linear replacement model, or first-order around one prompt. A
discrete change, such as swapping one name for another, needs credit that adds up to the change actually measured.
This module gives that at the level of attention heads, on the real model rather than a replacement model.

The input embeddings are interpolated, $e(\alpha) = e_0 + \alpha (e_1 - e_0)$, at the output of the embedding module
(for Gemma that is after its $\sqrt{d}$ scale), and the target $T$ (a fixed weighting of the last position's logits) is
followed along the path. At attention layer $L$, the residual after attention depends on the input only through the
residual entering the layer, $r_L$, and each head's attention output, $z_{L,h}$ (before the output projection). That
set is a complete cut, so the conductances

$$c_y = \int_0^1 \nabla_y T \cdot \frac{dy}{d\alpha} d\alpha, \quad y \in \{r_L, z_{L,1}, \ldots, z_{L,H}\}$$

sum to $T(1) - T(0)$ at every layer, exactly up to the discretization of the integral, which is reported. A head's
output is its pattern times its values, $z_h = A_h V_h$, so its conductance splits exactly into a pattern (QK) part,
$\int \nabla_{z_h} T \cdot \frac{dA_h}{d\alpha} V_h$, and a value (OV) part, $\int \nabla_{z_h} T \cdot A_h \frac{dV_h}{d\alpha}$.
Each step of the trapezoid rule uses the identity
$A_1 V_1 - A_0 V_0 = (A_1 - A_0) \bar V + \bar A (V_1 - V_0)$, with bars for the two endpoints' mean, so the two parts
sum to the head's conductance exactly at every step.

Every gradient comes from one backward pass per batch of path points: the gradient of $T$ with respect to an
intermediate activation is the derivative of everything downstream of it, which is the partial a cut needs. The
residual's partial is read at the residual after attention, $r_L + \mathrm{attn}_L$, whose gradient equals the
partial of $T$ with respect to $r_L$ holding the heads fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

# where the residual after attention (the cut's other side) is read, per architecture: the input of the norm that
# opens the MLP block
_RESID_MID_MODULE = {
    "Gemma3ForCausalLM": "pre_feedforward_layernorm",
    "Gemma3ForConditionalGeneration": "pre_feedforward_layernorm",
    "Gemma2ForCausalLM": "pre_feedforward_layernorm",
    "LlamaForCausalLM": "post_attention_layernorm",
    "Qwen2ForCausalLM": "post_attention_layernorm",
    "Qwen3ForCausalLM": "post_attention_layernorm",
}


@dataclass(frozen=True)
class HeadConductance:
    """Conductance of every layer's cut along one path, in units of the target.

    Attributes:
        t_from: The target at the path's start (the first prompt).
        t_to: The target at its end (the second prompt).
        resid: ``(n_layers,)``: the residual entering each layer.
        qk: ``(n_layers, n_heads)``: each head's pattern part.
        ov: ``(n_layers, n_heads)``: each head's value part.
        n_steps: The number of trapezoid intervals.
    """

    t_from: float
    t_to: float
    resid: torch.Tensor
    qk: torch.Tensor
    ov: torch.Tensor
    n_steps: int

    @property
    def delta(self) -> float:
        """The measured change of the target."""
        return self.t_to - self.t_from

    @property
    def head(self) -> torch.Tensor:
        """``(n_layers, n_heads)``: each head's conductance (pattern plus value part)."""
        return self.qk + self.ov

    @property
    def completeness_residual(self) -> torch.Tensor:
        """``(n_layers,)``: the measured change minus each layer's parts, as a fraction of the change."""
        parts = self.resid + self.head.sum(-1)
        return (self.delta - parts) / abs(self.delta) if self.delta else self.delta - parts


def _decoder_layers(model: Any) -> nn.ModuleList:
    return model.get_decoder().layers


def head_conductance(
    model: Any,
    ids_from: torch.Tensor,
    ids_to: torch.Tensor,
    target_weights: torch.Tensor,
    n_steps: int = 32,
    points_per_batch: int = 17,
) -> HeadConductance:
    """Per-head conductance of every attention layer along the path from ``ids_from`` to ``ids_to``.

    Args:
        model: A Hugging Face causal LM with eager attention (its patterns must be readable), in eval mode.
        ids_from: ``(n_pos,)`` token ids of the first prompt.
        ids_to: ``(n_pos,)`` token ids of the second prompt, the same length.
        target_weights: ``(d_vocab,)``: the target is ``logits[-1] @ target_weights`` (for a logit difference,
            +1 and -1 at the two tokens).
        n_steps: Trapezoid intervals along the path.
        points_per_batch: Path points per forward and backward pass (memory).

    Returns:
        A :class:`HeadConductance`.
    """
    arch = type(model).__name__
    if arch not in _RESID_MID_MODULE:
        raise ValueError(
            f"{arch}: no residual-after-attention location is known for this architecture"
        )
    if getattr(model.config, "_attn_implementation", None) != "eager":
        raise ValueError(
            "head conductance reads attention patterns, which needs attn_implementation='eager'"
        )
    if ids_from.shape != ids_to.shape or ids_from.ndim != 1:
        raise ValueError(
            f"the two prompts must be 1-D and the same length, got {ids_from.shape}, {ids_to.shape}"
        )

    layers: Any = _decoder_layers(model)
    n_layers = len(layers)
    attn0 = layers[0].self_attn
    n_heads = model.config.get_text_config().num_attention_heads
    head_dim = attn0.head_dim
    n_kv = attn0.v_proj.out_features // head_dim
    device = next(model.parameters()).device
    embed = model.get_input_embeddings()
    with torch.no_grad():
        e0 = embed(ids_from.to(device).unsqueeze(0))
        e1 = embed(ids_to.to(device).unsqueeze(0))
    ids = ids_from.to(device).unsqueeze(0)
    w = target_weights.to(device=device, dtype=e0.dtype)

    def run_points(alphas: torch.Tensor):
        """Forward and backward at a batch of path points; per-layer values and gradients for the cut."""
        n = alphas.shape[0]
        store: dict[str, list] = {k: [None] * n_layers for k in ("r_in", "r_mid", "z", "v")}
        hooks = [
            embed.register_forward_hook(
                lambda mod, inp, out: (e0 + alphas.view(-1, 1, 1) * (e1 - e0)).requires_grad_(True)
            )
        ]
        for i, layer in enumerate(layers):

            def save_in(mod, args, kwargs, i=i):
                store["r_in"][i] = (args[0] if args else kwargs["hidden_states"]).detach()

            def save_mid(mod, args, i=i):
                args[0].retain_grad()
                store["r_mid"][i] = args[0]

            def save_z(mod, args, i=i):
                args[0].retain_grad()
                store["z"][i] = args[0]

            def save_v(mod, args, out, i=i):
                store["v"][i] = out.detach()

            hooks += [
                layer.register_forward_pre_hook(save_in, with_kwargs=True),
                getattr(layer, _RESID_MID_MODULE[arch]).register_forward_pre_hook(save_mid),
                layer.self_attn.o_proj.register_forward_pre_hook(save_z),
                layer.self_attn.v_proj.register_forward_hook(save_v),
            ]
        try:
            # the target reads the last position only; full logits would cost points x positions x vocabulary
            out = model(input_ids=ids.expand(n, -1), output_attentions=True, logits_to_keep=1)
            t = out.logits[:, -1] @ w
            t.sum().backward()
        finally:
            for h in hooks:
                h.remove()
        if out.attentions is None or len(out.attentions) != n_layers:
            raise ValueError("the model returned no attention patterns; it needs eager attention")
        per_layer = []
        for i in range(n_layers):
            a = out.attentions[i].detach()  # (n, heads, q, k)
            v = store["v"][i].view(n, -1, n_kv, head_dim).transpose(1, 2)  # (n, kv, k, d)
            v = v.repeat_interleave(n_heads // n_kv, dim=1)  # (n, heads, k, d)
            z = store["z"][i]
            zh = z.detach().view(n, -1, n_heads, head_dim).transpose(1, 2)  # (n, heads, q, d)
            recon = a @ v
            if not torch.allclose(recon, zh, rtol=1e-4, atol=1e-5 * float(zh.abs().max())):
                raise ValueError(
                    f"layer {i}: a head's output is not its pattern times its v_proj values "
                    f"(max gap {float((recon - zh).abs().max()):.3g}); this architecture transforms values or "
                    "outputs in a way the pattern/value split does not model"
                )
            gz = z.grad.view(n, -1, n_heads, head_dim).transpose(1, 2)
            per_layer.append((store["r_in"][i], store["r_mid"][i].grad, a, v, gz))
        model.zero_grad(set_to_none=True)
        return t.detach(), per_layer

    # only activations need gradients; a parameter gradient per pass would cost a model's worth of memory
    params = list(model.parameters())
    requires = [p.requires_grad for p in params]
    for p in params:
        p.requires_grad_(False)
    try:
        return _integrate(
            run_points, n_layers, n_heads, n_steps, points_per_batch, device, e0.dtype
        )
    finally:
        for p, r in zip(params, requires):
            p.requires_grad_(r)


def _integrate(
    run_points, n_layers, n_heads, n_steps, points_per_batch, device, dtype
) -> HeadConductance:
    """Trapezoid accumulation of every cut's conductance over consecutive path points."""
    alphas = torch.linspace(0.0, 1.0, n_steps + 1, device=device, dtype=dtype)
    resid = torch.zeros(n_layers, dtype=torch.float64, device=device)
    qk = torch.zeros(n_layers, n_heads, dtype=torch.float64, device=device)
    ov = torch.zeros(n_layers, n_heads, dtype=torch.float64, device=device)
    previous = None
    t_values = []
    start = 0
    while start <= n_steps:
        # each batch starts one point back, so consecutive batches share an endpoint and every interval is covered
        chunk = alphas[start : start + points_per_batch]
        t, per_layer = run_points(chunk)
        t_values.extend(t.tolist())
        points = [
            [(r[k], g[k], a[k], v[k], gz[k]) for (r, g, a, v, gz) in per_layer]
            for k in range(len(chunk))
        ]
        if previous is not None:
            points = [previous] + points
        for k in range(len(points) - 1):
            for i in range(n_layers):
                r0, g0, a0, v0, gz0 = points[k][i]
                r1, g1, a1, v1, gz1 = points[k + 1][i]
                resid[i] += (0.5 * (g0 + g1) * (r1 - r0)).sum().double()
                g_bar = 0.5 * (gz0 + gz1)
                qk[i] += (g_bar * ((a1 - a0) @ (0.5 * (v0 + v1)))).sum(dim=(-1, -2)).double()
                ov[i] += (g_bar * ((0.5 * (a0 + a1)) @ (v1 - v0))).sum(dim=(-1, -2)).double()
        previous = points[-1]
        start += len(chunk)
    return HeadConductance(
        t_from=t_values[0],
        t_to=t_values[-1],
        resid=resid.cpu(),
        qk=qk.cpu(),
        ov=ov.cpu(),
        n_steps=n_steps,
    )
