"""Split the attribution edges into one target across the attention heads of one layer.

Every edge into a target sums the paths from its source. At any attention layer those paths divide into one part per
head plus a bypass that does not pass through that layer's attention: the residual stream and, with skip transcoders,
the MLP's linear skip. The split comes from the same backward pass that builds a graph. The attribution forward is run
with one copy of the prompt per part, and in each copy a stop-gradient on the layer's output-projection input leaves
only that part's heads live:

- copy ``h`` keeps head ``h`` only, so its row is the bypass plus head ``h``'s part;
- one copy keeps no head (the bypass alone) and one keeps every head (the edge itself, computed independently).

In a frozen graph the map is linear, so the parts sum to the edge exactly. With attention patterns or norm scales live
the parts are first-order, and they still sum to the edge, because a gradient is linear in its paths.
:attr:`HeadLoadings.partition_error` reports how closely they do.

Rows run over every source node in the attribution context's order: each active feature (``active_features``), then
the error nodes layer-major, then the token embeddings.

Each backend implements the split where it can reach the output projection's input:
:mod:`~circuit_tracer.attribution.head_loadings_nnsight` and
:mod:`~circuit_tracer.attribution.head_loadings_interp_engine`. :func:`head_loadings` here dispatches between them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

HEAD_LOADING_BACKENDS = ("nnsight", "interp_engine")


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


def head_keep_mask(n_heads: int, d_head: int) -> torch.Tensor:
    """``(n_heads + 2, 1, n_heads * d_head)``: which heads each copy keeps live.

    Copies ``0..n_heads-1`` keep one head each, copy ``n_heads`` keeps none (the bypass), the last keeps all (the edge).
    """
    keep = torch.zeros(n_heads + 2, n_heads)
    keep[torch.arange(n_heads), torch.arange(n_heads)] = 1.0
    keep[-1] = 1.0
    return keep.repeat_interleave(d_head, dim=1).unsqueeze(1)


def assemble(
    rows: torch.Tensor, layer: int, n_heads: int, ctx: Any, n_layers: int, n_pos: int
) -> HeadLoadings:
    """A :class:`HeadLoadings` from the ``(n_heads + 2, n_sources)`` rows of the masked copies."""
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


def head_loadings(
    model: Any,
    prompt: str | torch.Tensor | list[int],
    target: Any,
    layer: int,
    *,
    freeze_attention: bool = True,
    freeze_norms: bool = True,
) -> HeadLoadings:
    """Split every edge into ``target`` across the heads of attention layer ``layer``, on the model's own backend.

    Args and returns as in :func:`circuit_tracer.attribution.head_loadings_nnsight.head_loadings`. A backend that
    cannot split edges is refused by name.
    """
    if model.backend == "nnsight":
        from circuit_tracer.attribution.head_loadings_nnsight import head_loadings as split
    elif model.backend == "interp_engine":
        from circuit_tracer.attribution.head_loadings_interp_engine import head_loadings as split
    else:
        raise ValueError(
            f"head loadings are implemented for the {' and '.join(HEAD_LOADING_BACKENDS)} backends only; "
            f"got {model.backend!r}"
        )
    return split(
        model, prompt, target, layer, freeze_attention=freeze_attention, freeze_norms=freeze_norms
    )
