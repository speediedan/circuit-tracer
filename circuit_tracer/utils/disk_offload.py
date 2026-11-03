import atexit
import os
import tempfile
from typing import Literal

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


def disk_offload_module(module):
    org_device = next(module.parameters()).device
    # safetensors opens the path itself, and Windows refuses a second open of a file still held
    # open, so the descriptor is closed before anything is written.
    fd, path = tempfile.mkstemp(prefix=_TEMP_PREFIX)
    os.close(fd)
    save_file(module.state_dict(), path)
    _offload_files.add(path)

    module.to(device="meta")

    def reload_handle(device=None):
        target_device = str(device or org_device)
        module.load_state_dict(load_file(path, device=target_device), assign=True)
        os.remove(path)
        _offload_files.remove(path)

    return reload_handle


def cpu_offload_module(module):
    org_device = next(module.parameters()).device
    module.to(device="cpu")

    def reload_handle():
        module.to(device=org_device)

    return reload_handle


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
        mods = modules.values()
    elif isinstance(modules, (list, nn.ModuleList, nn.Sequential)):
        mods = modules
    else:
        mods = [modules]
    return [offload_fn(module) for module in mods]
