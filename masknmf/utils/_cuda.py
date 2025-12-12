import torch
from warnings import warn

def torch_select_device(device: str = "auto", log_warning: bool = True) -> str:
    if device.startswith("cpu"):  # seems like "cpu:<index>" is also a valid device, maybe for multi-CPU setups
        if log_warning:
            warn(
                "You've explicitly selected to perform computations on the cpu, "
                "performance will be significantly slower"
            )

    elif device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(f"You specified a 'cuda' device but no cuda device is available.\n"
                             f"Do you have a device that supports CUDA? Are nvidia drivers installed? "
                             f"Are the nvidia drivers loaded in your OS kernel?  "
                             f"Is cuda installed? Google: 'torch no cuda device found'.")
    elif device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
            if log_warning:
                warn(
                    "No cuda device found, performing computations on the cpu, "
                    "performance will be significantly slower"
                )

    return torch.device(device)
