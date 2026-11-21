"""Split the attribution edges into one target across the attention heads of one layer (nnsight backend).

The method, and the :class:`HeadLoadings` it returns, are described in :mod:`circuit_tracer.attribution.head_loadings`.
Here the heads are read at the input of the attention layer's ``o_proj``, found beside the attention-pattern hook.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from nnsight import Envoy

from circuit_tracer.attribution.head_loadings import HeadLoadings, assemble, head_keep_mask
from circuit_tracer.attribution.targets import AttributionTargets

if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel

__all__ = ["HeadLoadings", "head_loadings"]


def _o_proj(model: NNSightReplacementModel, layer: int):
    """The output projection of ``layer``'s attention, found beside the attention-pattern hook."""
    block_attention = model._attention_pattern.split(".source", 1)[0]
    module = cast(
        Envoy, model._resolve_attr(model, f"{block_attention}.o_proj".format(layer=layer))
    )
    in_features = getattr(module._module, "in_features", None)
    expected = model.cfg.n_heads * model.cfg.d_head
    if in_features != expected:
        raise ValueError(
            f"layer {layer}'s o_proj takes {in_features} inputs, not n_heads * d_head = {expected}; "
            "head loadings need one contiguous d_head slice per query head"
        )
    return module


def _target_site(model, ctx, target, n_layers: int, n_pos: int):
    """``(layer, position, vector, block_output)`` of the injection that reads ``target``."""
    if isinstance(target, tuple) and len(target) == 3 and all(isinstance(v, int) for v in target):
        layers, positions, features = ctx.activation_matrix.indices()
        hit = torch.where(
            (layers == target[0]) & (positions == target[1]) & (features == target[2])
        )[0]
        if hit.numel() != 1:
            raise ValueError(f"feature target {target} is not active on this prompt")
        return target[0], target[1], ctx.encoder_vecs[hit[0]], False

    from circuit_tracer.attribution.attribute_nnsight import _target_injection_sites

    spec = target if isinstance(target, torch.Tensor) else [target]
    targets = AttributionTargets(
        attribution_targets=spec,
        logits=ctx.logits[0, -1],
        unembed_proj=model.unembed_weight.to(ctx.logits.device),
        tokenizer=model.tokenizer,
    )
    if len(targets) != 1:
        raise ValueError(f"head loadings take one target, got {len(targets)}")
    layers, positions, block_output = _target_injection_sites(targets, n_layers, n_pos)
    return int(layers[0]), int(positions[0]), targets.logit_vectors[0], bool(block_output[0])


def head_loadings(
    model: NNSightReplacementModel,
    prompt: str | torch.Tensor | list[int],
    target,
    layer: int,
    *,
    freeze_attention: bool = True,
    freeze_norms: bool = True,
) -> HeadLoadings:
    """Split every edge into ``target`` across the heads of attention layer ``layer``.

    Args:
        model: An nnsight-backend replacement model.
        prompt: The prompt the edges are attributed on.
        target: A feature ``(layer, position, feature)`` active on the prompt, or one target in any form
            ``attribute``'s ``attribution_targets`` accepts (a token string, a ``CustomTarget``, or a one-element
            tensor of token ids).
        layer: The attention layer to split over. A feature target reads its block's input after that block's
            attention, so ``layer`` may be at most the target's own layer.
        freeze_attention: As in ``attribute``: False splits first-order, patterns-live edges.
        freeze_norms: As in ``attribute``.

    Returns:
        A :class:`HeadLoadings` whose parts sum to each source's edge.
    """
    if model.backend != "nnsight":
        raise ValueError(
            f"head loadings are implemented for the nnsight backend only; got {model.backend!r}"
        )
    n_layers, n_heads, d_head = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_head
    if not 0 <= layer < n_layers:
        raise ValueError(
            f"layer {layer} is not an attention layer of this model (0..{n_layers - 1})"
        )

    input_ids = model.ensure_tokenized(prompt)
    n_pos = int(input_ids.shape[0])
    ctx = model.setup_attribution(input_ids)
    target_layer, target_position, vector, block_output = _target_site(
        model, ctx, target, n_layers, n_pos
    )
    # a feature reads its block's input after that block's attention, a layer-local target reads the block's output,
    # and a logit reads the final residual (target layer n_layers): either way attention up to target_layer is upstream
    if layer > target_layer:
        raise ValueError(
            f"no path into a target read at layer {target_layer} passes attention layer {layer}"
        )

    keep = head_keep_mask(n_heads, d_head)
    n_copies = keep.shape[0]

    with model.trace() as tracer:
        with tracer.invoke(input_ids.expand(n_copies, -1)):
            pass

        detach_barrier = tracer.barrier(2)
        model.configure_gradient_flow(
            tracer, freeze_attention=freeze_attention, freeze_norms=freeze_norms
        )
        model.configure_skip_connection(tracer, barrier=detach_barrier)
        ctx.cache_residual(model, tracer, barrier=detach_barrier, cache_block_outputs=block_output)

        with tracer.invoke():
            o_proj = _o_proj(model, layer)
            heads = o_proj.input
            mask = keep.to(device=heads.device, dtype=heads.dtype)
            # the forward value is unchanged; only the gradient through the masked heads stops
            o_proj.input = heads.detach() + mask * (heads - heads.detach())

    rows = ctx.compute_batch(
        layers=torch.full((n_copies,), target_layer),
        positions=torch.full((n_copies,), target_position),
        inject_values=vector.unsqueeze(0).expand(n_copies, -1),
        block_output=torch.full((n_copies,), block_output, dtype=torch.bool),
        retain_graph=False,
    ).detach()
    return assemble(rows, layer, n_heads, ctx, n_layers, n_pos)
