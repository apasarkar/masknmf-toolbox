import torch
import numpy as np
import torch


def torch_select_device(device: str | torch.device = "auto") -> str:
    if isinstance(device, torch.device):
        return device

    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"You specified a 'cuda' device but no cuda device is available.\n"
                             f"Do you have a device that supports CUDA? Are nvidia drivers installed? "
                             f"Are the nvidia drivers loaded in your OS kernel?  "
                             f"Is cuda installed? Google: 'torch no cuda device found'.")
    elif device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    return torch.device(device)

def is_cpu(device: torch.device | str | None) -> bool:
    return device is not None and torch.device(device).type == "cpu"


def to_numpy(x, dtype: np.dtype | type | str | None = None, copy: bool = False) -> np.ndarray:
    """
    Convert a torch tensor, numpy array, scalar or nested sequence to a numpy array on the host.

    Torch tensors are detached, made dense if sparse, and moved to the CPU (a device-to-host copy for GPU tensors).
    bfloat16 tensors become float32, since numpy has no bfloat16. Numpy arrays are returned as they are unless a
    dtype change or ``copy=True`` asks for a new array.

    Args:
        x: torch.Tensor, np.ndarray, python/numpy scalar, or anything np.asarray accepts (lists, tuples, ...).
        dtype: Optional numpy dtype for the result. Accepts anything np.dtype accepts.
        copy: If True, the result never shares memory with ``x``. If False (default), a CPU tensor or numpy array
            is returned without copying when no conversion is needed, so writing to the result can change ``x``.

    Returns:
        np.ndarray
    """
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.layout != torch.strided:
            x = x.to_dense()
        if x.dtype == torch.bfloat16:
            x = x.float()
        x = x.cpu().resolve_conj().resolve_neg()
        out = x.numpy()
    else:
        out = np.asarray(x)

    if dtype is not None:
        out = out.astype(dtype, copy=copy)
    elif copy:
        out = out.copy()
    return out
