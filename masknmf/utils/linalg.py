import torch


def sparse_mm_async(coo: torch.Tensor, mat: torch.Tensor) -> torch.Tensor:
    """
    Async sparse matmul. coo must be sparse coo and coalesced
    
    Args:
        coo: 
        mat: 

    Returns:

    """
    rows = coo.indices()[0]
    cols = coo.indices()[1]
    vals = coo.values()

    weighted = vals.unsqueeze(1) * mat[cols]

    out = torch.zeros(coo.shape[0], mat.shape[1], device=mat.device, dtype=mat.dtype)
    out.scatter_add_(0, rows.unsqueeze(1).expand_as(weighted), weighted)

    return out
