"""Split the attribution edges into one target across the attention heads of one layer (nnsight backend).

Every edge into a target sums the paths from its source. At any attention layer those paths divide into one part
per head plus a bypass that does not pass through that layer's attention: the residual stream and, with skip
transcoders, the MLP's linear skip. The split here comes from the same backward pass that builds a graph. The
attribution forward is run with one copy of the prompt per part, and in each copy a stop-gradient on the layer's
output-projection input leaves only that part's heads live:

- copy ``h`` keeps head ``h`` only, so its row is the bypass plus head ``h``'s part;
- one copy keeps no head (the bypass alone) and one keeps every head (the edge itself, computed independently).

In a frozen graph the map is linear, so the parts sum to the edge exactly. With attention patterns or norm scales
live the parts are first-order, and they still sum to the edge, because a gradient is linear in its paths.
:attr:`HeadLoadings.partition_error` reports how closely they do.

Rows run over every source node in the attribution context's order: each active feature (``active_features``), then
the error nodes layer-major, then the token embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from nnsight import Envoy

from circuit_tracer.attribution.targets import AttributionTargets

if TYPE_CHECKING:
    from circuit_tracer.replacement_model.replacement_model_nnsight import NNSightReplacementModel


@dataclass(frozen=True)
class HeadLoadings:
    """How the edges into one target divide across the heads of one attention layer.

    Attributes:
        layer: The attention layer split over.
        per_head: ``(n_heads, n_sources)``: each head's part of each source's edge.
        bypass: ``(n_sources,)``: each edge's part that does not pass through this layer's attention.
        total: ``(n_sources,)``: each edge, computed in its own copy with every head live.
        active_features: ``(n_active, 3)``: ``(layer, position, feature)`` of the feature sources, in row order.
        n_layers: Model layers, for decoding error-node rows.
        n_pos: Prompt positions, for decoding error- and token-node rows.
    """

    layer: int
    per_head: torch.Tensor
    bypass: torch.Tensor
    total: torch.Tensor
    active_features: torch.Tensor
    n_layers: int
    n_pos: int

    @property
    def partition_error(self) -> float:
        """Largest gap between the parts' sum and the edge, as a fraction of the largest edge."""
        gap = (self.per_head.sum(0) + self.bypass - self.total).abs().max()
        scale = self.total.abs().max()
        return float(gap / scale) if scale > 0 else float(gap)

    def feature_rows(self) -> slice:
        """The rows of the feature sources."""
        return slice(0, self.active_features.shape[0])

    def error_row(self, layer: int, position: int) -> int:
        """The row of the error node at ``(layer, position)``."""
        return self.active_features.shape[0] + layer * self.n_pos + position

    def token_row(self, position: int) -> int:
        """The row of the token-embedding node at ``position``."""
        return self.active_features.shape[0] + self.n_layers * self.n_pos + position


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

    # copies 0..n_heads-1 keep one head each, copy n_heads keeps none (the bypass), the last keeps all (the edge)
    n_copies = n_heads + 2
    keep = torch.zeros(n_copies, n_heads)
    keep[torch.arange(n_heads), torch.arange(n_heads)] = 1.0
    keep[-1] = 1.0
    keep = keep.repeat_interleave(d_head, dim=1).unsqueeze(1)  # (copies, 1, n_heads * d_head)

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
    bypass = rows[n_heads]
    return HeadLoadings(
        layer=layer,
        per_head=rows[:n_heads] - bypass,
        bypass=bypass,
        total=rows[-1],
        active_features=ctx.activation_matrix.indices().T,
        n_layers=n_layers,
        n_pos=n_pos,
    )
