"""Split the attribution edges into one target across the attention heads of one layer (interp_engine backend).

The method is the one :mod:`~circuit_tracer.attribution.head_loadings` describes. Here the heads are read where
interp-engine resolves ``z``: the input of the attention output projection, found by inspecting the architecture rather
than from a per-family table. A forward pre-hook there masks each copy's gradient, and the layer's width is checked
against its own head count and value-head width, since a family may size its heads by layer type.

One limit carries over from this backend's ``attribute``: the attribution context has no block-output injection, so a
target read at an intermediate block output or a non-final position is refused. Feature targets and final-position
logit targets are accepted.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

from circuit_tracer.attribution.head_loadings import HeadLoadings, assemble, head_keep_mask
from circuit_tracer.attribution.targets import AttributionTargets
from circuit_tracer.replacement_model.replacement_model_interp_engine import (
    _input_tensor,
    _substitute_input,
)

if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_interp_engine import (
        InterpEngineReplacementModel,
    )

__all__ = ["HeadLoadings", "head_loadings"]


def _heads_site(model: InterpEngineReplacementModel, layer: int) -> tuple[nn.Module, int, int]:
    """The module whose input carries ``layer``'s heads, with this layer's head count and value-head width."""
    module, side = model.engine.resolve_point("z", layer)
    if side != "input":
        raise ValueError(
            f"interp-engine resolves layer {layer}'s 'z' to the {side} of {type(module).__name__}, not the input of "
            "the attention output projection; head loadings need the heads concatenated there"
        )
    d_head = int(model.arch.value_head_dim_for_layer(layer))
    weight = getattr(module, "weight", None)
    width = getattr(module, "in_features", None)
    if width is None and isinstance(weight, torch.Tensor):
        width = weight.shape[0]  # a Conv1D stores (in, out)
    n_heads = int(model.cfg.n_heads)
    if width != n_heads * d_head:
        raise ValueError(
            f"layer {layer}'s attention output projection takes {width} inputs, not n_heads * d_head = "
            f"{n_heads} * {d_head}; head loadings need one contiguous d_head slice per query head"
        )
    return module, n_heads, d_head


def _target_site(model, ctx, target, n_layers: int, n_pos: int) -> tuple[int, int, torch.Tensor]:
    """``(layer, position, vector)`` of the injection that reads ``target``."""
    if isinstance(target, tuple) and len(target) == 3 and all(isinstance(v, int) for v in target):
        layers, positions, features = ctx.activation_matrix.indices()
        hit = torch.where(
            (layers == target[0]) & (positions == target[1]) & (features == target[2])
        )[0]
        if hit.numel() != 1:
            raise ValueError(f"feature target {target} is not active on this prompt")
        return target[0], target[1], ctx.encoder_vecs[hit[0]]

    spec = target if isinstance(target, torch.Tensor) else [target]
    targets = AttributionTargets(
        attribution_targets=spec,
        logits=ctx.logits[0, -1],
        unembed_proj=model.unembed_weight.to(ctx.logits.device),
        tokenizer=model.tokenizer,
    )
    if len(targets) != 1:
        raise ValueError(f"head loadings take one target, got {len(targets)}")
    if targets.target_sites[0] != (None, None):
        raise NotImplementedError(
            "a target read at an intermediate block output or a non-final position is implemented for the nnsight "
            "backend only; the interp_engine backend's attribution context injects at the final residual"
        )
    return n_layers, n_pos - 1, targets.logit_vectors[0]


def head_loadings(
    model: InterpEngineReplacementModel,
    prompt: str | torch.Tensor | list[int],
    target,
    layer: int,
    *,
    freeze_attention: bool = True,
    freeze_norms: bool = True,
) -> HeadLoadings:
    """Split every edge into ``target`` across the heads of attention layer ``layer``.

    Args:
        model: An interp_engine-backend replacement model.
        prompt: The prompt the edges are attributed on.
        target: A feature ``(layer, position, feature)`` active on the prompt, or one final-position target in any
            form ``attribute``'s ``attribution_targets`` accepts (a token string, a ``CustomTarget`` without a layer
            or position, or a one-element tensor of token ids).
        layer: The attention layer to split over. A feature target reads its block's input after that block's
            attention, so ``layer`` may be at most the target's own layer.
        freeze_attention: As in ``attribute``: False splits first-order, patterns-live edges.
        freeze_norms: As in ``attribute``.

    Returns:
        A :class:`HeadLoadings` whose parts sum to each source's edge.
    """
    if model.backend != "interp_engine":
        raise ValueError(
            f"this head-loading split is for the interp_engine backend; got {model.backend!r}"
        )
    n_layers = model.cfg.n_layers
    if not 0 <= layer < n_layers:
        raise ValueError(
            f"layer {layer} is not an attention layer of this model (0..{n_layers - 1})"
        )

    input_ids = model.ensure_tokenized(prompt)
    n_pos = int(input_ids.shape[0])
    ctx = model.setup_attribution(input_ids)
    target_layer, target_position, vector = _target_site(model, ctx, target, n_layers, n_pos)
    # a feature reads its block's input after that block's attention and a logit reads the final residual (target
    # layer n_layers): either way attention up to target_layer is upstream
    if layer > target_layer:
        raise ValueError(
            f"no path into a target read at layer {target_layer} passes attention layer {layer}"
        )

    site, n_heads, d_head = _heads_site(model, layer)
    keep = head_keep_mask(n_heads, d_head)
    n_copies = keep.shape[0]

    def mask_heads(_module, args, kwargs):
        heads, key = _input_tensor(args, kwargs)
        mask = keep.to(device=heads.device, dtype=heads.dtype)
        # the forward value is unchanged; only the gradient through the masked heads stops
        return _substitute_input(
            args, kwargs, key, heads.detach() + mask * (heads - heads.detach())
        )

    handle = site.register_forward_pre_hook(mask_heads, with_kwargs=True)
    try:
        with (
            model.gradient_regime(freeze_attention=freeze_attention, freeze_norms=freeze_norms),
            ctx.install_hooks(model),
        ):
            ctx._resid_activations[-1] = model.forward_trunk(input_ids.expand(n_copies, -1))
    finally:
        handle.remove()

    try:
        rows = ctx.compute_batch(
            layers=torch.full((n_copies,), target_layer),
            positions=torch.full((n_copies,), target_position),
            inject_values=vector.unsqueeze(0).expand(n_copies, -1),
            retain_graph=False,
        ).detach()
    finally:
        ctx.remove_hooks()
    return assemble(rows, layer, n_heads, ctx, n_layers, n_pos)
