"""Placement checks for a model whose layers are split across devices.

Attribution reads weights directly (the embedding, the unembedding, the final norm) and runs a backward pass through
every layer. A device map that offloads a module to ``cpu`` or ``disk`` alongside accelerators leaves that module's
parameters on the ``meta`` device outside its own forward (accelerate materializes them only inside the forward), so a
direct read sees a tensor with no data. ``device_map="auto"`` produces exactly that on unequal GPUs: it means
"balanced", which caps every GPU but the last near an equal share of the model and sends the remainder to ``disk``.
A split for attribution must therefore place every module on a real device, and these checks refuse one that does not.
"""

from __future__ import annotations

from collections import Counter

import torch
from torch import nn

OFFLOAD_DEVICES = ("cpu", "disk", "meta")


def _device_key(device) -> str:
    """Normalize a device-map value (int, str or ``torch.device``) to a comparable string."""
    if isinstance(device, int):
        return f"cuda:{device}"
    return str(device)


def placement_summary(device_map: dict) -> str:
    """One-line count of device-map entries per device, for error messages and logs."""
    counts = Counter(_device_key(d) for d in device_map.values())
    return ", ".join(f"{dev}: {n}" for dev, n in sorted(counts.items()))


def check_materialized(model: nn.Module, max_listed: int = 8) -> None:
    """Refuse a model that has any module offloaded or any parameter or buffer left on ``meta``.

    Args:
        model: the loaded torch model (for nnsight, the wrapped ``_module``).
        max_listed: how many offending names to list in the error.

    Raises:
        ValueError: naming the offloaded modules or meta tensors, with the device map's per-device counts.
    """
    device_map = getattr(model, "hf_device_map", None) or {}
    accelerators = {
        _device_key(d) for d in device_map.values() if _device_key(d) not in OFFLOAD_DEVICES
    }
    # an all-cpu map is an ordinary CPU run; cpu is an offload target only beside an accelerator
    offloaded = [
        name
        for name, d in device_map.items()
        if _device_key(d) in ("disk", "meta") or (_device_key(d) == "cpu" and accelerators)
    ]
    meta = [name for name, t in (*model.named_parameters(), *model.named_buffers()) if t.is_meta]
    if not offloaded and not meta:
        return

    def listed(names: list[str]) -> str:
        more = len(names) - max_listed
        return ", ".join(names[:max_listed]) + (f" (+{more} more)" if more > 0 else "")

    parts = []
    if offloaded:
        parts.append(f"modules offloaded off the accelerators: {listed(offloaded)}")
    if meta:
        parts.append(f"tensors on the meta device: {listed(meta)}")
    summary = f" Device map: {placement_summary(device_map)}." if device_map else ""
    raise ValueError(
        "Attribution needs every module materialized on a real device, but the loaded model has "
        + "; ".join(parts)
        + "."
        + summary
        + " Offloaded parameters stay on the meta device outside their own forward, so weight reads and the "
        "backward pass would see tensors with no data. device_map='auto' (balanced) does this on unequal GPUs; "
        "use device_map='sequential' with max_memory, or an explicit device map, that fits the model on accelerators."
    )


def model_devices(model: nn.Module) -> set[torch.device]:
    """The set of devices holding the model's parameters."""
    return {p.device for p in model.parameters()}
