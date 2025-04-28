import scipy.sparse
import torch
from typing import *


def baseline_update(uv_mean, a, c, to_torch=False):
    """
    Calculates baseline. Inputs:
        uv_mean. torch.Tensor of shape (d, 1), where d is the number of pixels in the FOV
        a: torch.sparse_coo_tensor. tensor of shape (d, k) where k is the number of neurons in the FOV
        c: torch.Tensor of shape (T, k) where T is number of frames in video
        to_torch: indicates whether the inputs are np.ndarrays that need to be converted to torch objects. Also implies that the result will be returned as a np.ndarray (the same format as the inputs)
    Output:
        b. torch.Tensor of shape (d, 1). Describes new static baseline
    """
    if to_torch:
        a_sp = scipy.sparse.csr_matrix(a)
        torch.sparse_coo_tensor(a_sp.nonzero(), a_sp.data, a_sp.shape).coalesce()
        c = torch.from_numpy(c).float()
        uv_mean = torch.from_numpy(uv_mean).float()
    mean_c = torch.mean(c, dim=0, keepdim=True).t()
    b = uv_mean - torch.sparse.mm(a, mean_c)

    if to_torch:
        return b.numpy()
    else:
        return b


def spatial_update_hals(
    u_sparse: torch.tensor,
    v: torch.tensor,
    a_sparse: torch.sparse_coo_tensor,
    c: torch.tensor,
    b: torch.tensor,
    q: Optional[torch.tensor] = None,
    blocks: Optional[Union[torch.tensor, list]] = None,
    mask_ab: Optional[torch.sparse_coo_tensor] = None,
):
    """
    Computes a spatial HALS updates:
    Params:

        Note: The first four parameters are the "PMD" representation of the data: it is given in a traditional SVD form: URsV, where UR is the left orthogonal basis, 's' represents the diagonal matrix, and V is the right orthogonal basis.
        u_sparse (torch.sparse_coo_tensor): Sparse matrix, with dimensions (d x R)
        v (torch.Tensor): Dimensions R x T. Dimensions R x T, where T is the number of frames, where all rows are orthonormal.
            Note:  V must contain the 1 x T vector of all 1's in its rowspan.
        a_sparse (torch.sparse_coo_tensor): dimensions d x k, where k represents the number of neural signals.
        c (torch.Tensor): Dimensions T x k
        b (torch.Tensor): Dimensions d x 1. Represents static background
        q (torch.tensor): This is the factorized ring model term; u@r@q@v gives you the full ring model movie
        blocks Optional[Union[torch.tensor, list]]: Describes which components can be updated in parallel. Typically a list of 1D tensors, each describing indices
        mask_ab (torch.sparse_coo_tensor): Dimensions (d x k). For each neuron, indicates the allowed support of neuron

    Returns:
        a_sparse: torch.sparse_coo_tensor. Dimensions d x k, containing updated spatial matrix

    TODO: Make 'a' input a sparse matrix
    """
    # Load all values onto device in torch
    device = v.device

    if mask_ab is None:
        mask_ab = a_sparse.bool()

    mask_ab = mask_ab.long().to_dense()
    nonzero_row_indices = torch.squeeze(torch.sum(mask_ab, dim=1).nonzero())
    mask_ab = torch.index_select(mask_ab, 0, nonzero_row_indices)

    Vc = torch.matmul(v, c)
    a_dense = torch.index_select(a_sparse, 0, nonzero_row_indices).to_dense()

    C_prime = torch.matmul(c.t(), c)
    C_prime_diag = torch.diag(C_prime)
    C_prime_diag[C_prime_diag == 0] = 1  # For division safety
    """
    We will now compute the following expression: 

    [UR(diag(s) - q)Vc - beVc]

    This is part of the 'residual video' that we regress onto the spatial components below
    """

    u_subset = torch.index_select(u_sparse, 0, nonzero_row_indices)

    if q is not None:
        eye_elt = torch.eye(u_sparse.shape[1], device=device, dtype=u_sparse.dtype)
        background_subtracted_projection = torch.sparse.mm(
            u_subset, torch.matmul((eye_elt - q), Vc)
        )
    else:
        eye_elt = torch.eye(u_sparse.shape[1], device=device, dtype=u_sparse.dtype)
        background_subtracted_projection = torch.sparse.mm(u_subset, Vc)
    baseline_projection = torch.matmul(
        torch.index_select(b, 0, nonzero_row_indices), torch.sum(c, dim=0, keepdim=True)
    )

    cumulator = background_subtracted_projection - baseline_projection

    threshold_func = torch.nn.ReLU(0)
    if blocks is None:
        blocks = torch.arange(c.shape[1], device=device).unsqueeze(1)
    for index_select_tensor in blocks:
        mask_apply = torch.index_select(mask_ab, 1, index_select_tensor)

        c_prime_i = C_prime.index_select(0, index_select_tensor).t()
        cumulator_i = cumulator.index_select(1, index_select_tensor)
        acc = torch.matmul(a_dense, c_prime_i)
        final_vec = (cumulator_i - acc) / C_prime_diag[None, index_select_tensor]
        curr_frame = torch.index_select(a_dense, 1, index_select_tensor)
        curr_frame += final_vec
        curr_frame *= mask_apply
        curr_frame = threshold_func(curr_frame)
        a_dense[:, index_select_tensor] = curr_frame

    pruned_indices = a_dense.nonzero()
    pruned_row, pruned_col = [pruned_indices[:, i] for i in range(2)]
    final_values = a_dense[pruned_row, pruned_col]
    real_row = nonzero_row_indices[pruned_row]

    a_sparse = torch.sparse_coo_tensor(
        torch.stack([real_row, pruned_col]), final_values, a_sparse.shape
    ).coalesce()
    return a_sparse


def temporal_update_hals(
    u_sparse: torch.sparse_coo_tensor,
    v: torch.tensor,
    a_sparse: torch.sparse_coo_tensor,
    c: torch.tensor,
    b: torch.tensor,
    q: Optional[torch.tensor] = None,
    c_nonneg: bool = True,
    blocks: Optional[Union[torch.tensor, list]] = None,
):
    """
    Inputs:
         Note: The first four parameters are the "PMD" representation of the data: it is given in a traditional SVD form: URsV, where UR is the left orthogonal basis, 's' represents the diagonal matrix, and V is the right orthogonal basis.
        u_sparse: torch.sparse_coo_tensor. Sparse matrix, with dimensions (d x R)
        v: torch.Tensor. Dimensions R' x T. Dimensions R' x T, where T is the number of frames, where all rows are orthonormal.
            Note:  V must contain the 1 x T vector of all 1's in its rowspan.
        a: (d1*d2, k)-shaped torch.sparse_coo_tensor
        c: (T, k)-shaped torch.Tensor
        b: (d1*d2, 1)-shaped torch.Tensor
        q Optional[torch.tensor]: This is the factorized ring model term; u@r@q@v gives you the full ring model movie
        c_nonneg (bool): Indicates whether "c" should be nonnegative or fully unconstrained. For voltage data, it should be unconstrained; for calcium it should be constrained.
        blocks Optional[Union[torch.tensor, list]]: Describes which components can be updated in parallel. Typically a list of 1D tensors, each describing indices

    Returns:
        c: (T, k)-shaped np.ndarray. Updated temporal components
    """
    device = v.device

    ##Precompute quantities used throughout all iterations

    # Find the tensor, e, (a 1 x R' shaped tensor) such that eV gives a 1 x T tensor consisting of all 1's
    # e = torch.matmul(torch.ones([1, v.shape[1]], device=device), v.t())

    # Step 1: Get aTURs
    aTU = torch.sparse.mm(a_sparse.t(), u_sparse)
    # aTUR = torch.sparse.mm(aTU, r)
    if q is not None:
        fluctuating_background_subtracted_projection = torch.sparse.mm(
            aTU, torch.eye(u_sparse.shape[1], device=device, dtype=u_sparse.dtype) - q
        )
    else:
        fluctuating_background_subtracted_projection = torch.sparse.mm(
            aTU, torch.eye(u_sparse.shape[1], device=device, dtype=u_sparse.dtype)
        )
    fluctuating_background_subtracted_projection = (
        fluctuating_background_subtracted_projection @ v
    )

    # Step 2: Get aTbe
    aTb = torch.matmul(a_sparse.t(), b)
    # static_background_projection = torch.matmul(aTb, e)
    static_background_projection = aTb

    # Step 3:
    cumulator = (
        fluctuating_background_subtracted_projection - static_background_projection
    )

    # cumulator = torch.matmul(cumulator, v)

    ata = torch.sparse.mm(a_sparse.t(), a_sparse)
    ata = ata.to_dense()
    diagonals = torch.diag(ata)

    if c_nonneg:
        threshold_function = torch.nn.ReLU()
    else:
        threshold_function = lambda x: x

    if blocks is None:
        blocks = torch.arange(c.shape[1], device=device).unsqueeze(1)
    for index_to_select in blocks:
        a_ia = torch.index_select(ata, 0, index_to_select)
        a_iaC = torch.matmul(a_ia, c.t())

        curr_trace = torch.index_select(c, 1, index_to_select)
        curr_trace += (
            (torch.index_select(cumulator, 0, index_to_select) - a_iaC)
            / torch.unsqueeze(diagonals[index_to_select], -1)
        ).t()
        curr_trace = threshold_function(curr_trace)
        c[:, index_to_select] = curr_trace

    return c
