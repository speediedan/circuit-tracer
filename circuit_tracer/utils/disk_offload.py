import atexit
import os
import tempfile
from typing import Literal

import torch
from torch import nn
from safetensors.torch import load_file, save_file

_offload_files = set()

_TEMP_PREFIX = "safetensors-offload-YqKRr8m3-"


@atexit.register
def cleanup_offload_files():
    for f in _offload_files:
        if os.path.exists(f):
            os.remove(f)


def cleanup_all_offload_files():
    temp_dir = tempfile.gettempdir()
    n_removed = 0
    for f in os.listdir(temp_dir):
        if f.startswith(_TEMP_PREFIX):
            os.remove(os.path.join(temp_dir, f))
            n_removed += 1
    return n_removed


def _split_aliases(
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Split a state dict into its distinct tensors and the names that alias one of them.

    A module can list one parameter under several names (a TransformerBridge MLP lists its linear
    layers under its own and its wrapped component's names), and safetensors refuses to write tensors
    that share memory.
    """
    kept: dict[str, torch.Tensor] = {}
    aliases: dict[str, str] = {}
    owners: dict[tuple, str] = {}
    for name, tensor in state_dict.items():
        key = (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tensor.shape,
            tensor.stride(),
            tensor.dtype,
        )
        owner = owners.setdefault(key, name) if tensor.numel() else name
        if owner == name:
            kept[name] = tensor
        else:
            aliases[name] = owner
    return kept, aliases


def disk_offload_module(module, org_device=None):
    org_device = org_device or next(module.parameters()).device
    state_dict = module.state_dict()
    # A state dict leaves out non-persistent buffers (Gemma's embedding scale), and moving the module
    # to "meta" drops their values, so they are saved beside it.
    buffers = {
        name: buffer
        for name, buffer in module.named_buffers(remove_duplicate=False)
        if name not in state_dict
    }
    tensors, aliases = _split_aliases({**state_dict, **buffers})
    buffer_names = tuple(buffers)
    # safetensors opens the path itself, and Windows refuses a second open of a file still held
    # open, so the descriptor is closed before anything is written.
    fd, path = tempfile.mkstemp(prefix=_TEMP_PREFIX)
    os.close(fd)
    save_file(tensors, path)
    _offload_files.add(path)

    module.to(device="meta")

    def reload_handle(device=None):
        target_device = str(device or org_device)
        state_dict = load_file(path, device=target_device)
        # CPU tensors from load_file stay memory-mapped to the file, and Windows refuses to delete a
        # mapped file, so they are copied out and the mapping dropped first.
        if target_device.startswith("cpu"):
            state_dict = {name: tensor.clone() for name, tensor in state_dict.items()}
        state_dict.update({name: state_dict[owner] for name, owner in aliases.items()})
        for name in buffer_names:
            owner, _, leaf = name.rpartition(".")
            module.get_submodule(owner)._buffers[leaf] = state_dict.pop(name)
        module.load_state_dict(state_dict, assign=True)
        os.remove(path)
        _offload_files.remove(path)

    return reload_handle


def cpu_offload_module(module, org_device=None):
    org_device = org_device or next(module.parameters()).device
    module.to(device="cpu")

    def reload_handle():
        module.to(device=org_device)

    return reload_handle


def _tied_parameters(modules: list[nn.Module]) -> list[list[tuple[nn.Module, str]]]:
    """Where each parameter held by more than one of these modules (or names) is registered."""
    homes: dict[int, list[tuple[nn.Module, str]]] = {}
    for module in modules:
        if isinstance(module, nn.Module):
            for name, param in module.named_parameters(remove_duplicate=False):
                homes.setdefault(id(param), []).append((module, name))
    return [home for home in homes.values() if len(home) > 1]


def _retie(tied: list[list[tuple[nn.Module, str]]]) -> None:
    """Make each tied group one parameter again, once every member is back from disk."""
    for home in tied:
        members = [
            (module.get_submodule(name.rpartition(".")[0]), name.rpartition(".")[2])
            for module, name in home
        ]
        params = [owner._parameters[leaf] for owner, leaf in members]
        if all(param is not None and param.device.type != "meta" for param in params):
            for owner, leaf in members[1:]:
                owner._parameters[leaf] = params[0]


def offload_modules(
    modules: list | nn.Module | nn.ModuleList | nn.ModuleDict | nn.Sequential,
    offload_type: Literal["cpu", "disk"],
) -> list:
    """Offload one or more modules to CPU or disk.

    Args:
        modules: A single module, list of modules, or PyTorch module container
                 (ModuleList, ModuleDict, Sequential)
        offload_type: Type of offload - "cpu" or "disk"

    Returns:
        List of reload handles, one per module
    """
    offload_fn = disk_offload_module if offload_type == "disk" else cpu_offload_module

    if isinstance(modules, nn.ModuleDict):
        mods = list(modules.values())
    elif isinstance(modules, (list, nn.ModuleList, nn.Sequential)):
        mods = list(modules)
    else:
        mods = [modules]
    # Read every device before moving anything. A tied embedding and unembedding share one
    # parameter, so offloading the first moves the second's too; read afterwards, the second's
    # device would be "cpu", and reloading it would put the shared weight back on the CPU.
    org_devices = [next(module.parameters()).device for module in mods]
    tied = _tied_parameters(mods) if offload_type == "disk" else []
    handles = [offload_fn(module, device) for module, device in zip(mods, org_devices)]
    if not tied:
        return handles

    # Reloading from disk gives each module its own copy of a parameter it shared, so the copies are
    # replaced by one parameter when the last of them is back.
    def with_retie(handle):
        def reload_handle(*args, **kwargs):
            handle(*args, **kwargs)
            _retie(tied)

        return reload_handle

    return [with_retie(handle) for handle in handles]
