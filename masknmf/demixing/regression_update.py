import scipy.sparse
import torch
from typing import *
from tqdm import tqdm
import math
from masknmf.demixing.demixing_utils import construct_graph_from_sparse_tensor, color_and_get_tensors


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
    q: Optional[Tuple[torch.tensor, torch.tensor]] = None,
    blocks: Optional[Union[torch.tensor, list]] = None,
    mask_ab: Optional[torch.sparse_coo_tensor] = None,
    frame_batch_size: int = 500,
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
        frame_batch_size (int): Roughly the number of dense frames of data that are expanded out in GPU memory

    Returns:
        a_sparse: torch.sparse_coo_tensor. Dimensions d x k, containing updated spatial matrix

    """
    # Load all values onto device in torch
    device = v.device

    if mask_ab is None:
        mask_ab = a_sparse.bool().coalesce()

    nonzero_row_indices = torch.unique(mask_ab.indices()[0, :])

    a_subset = torch.index_select(a_sparse, 0, nonzero_row_indices).coalesce()
    ctc = torch.matmul(c.t(), c)
    ctc_diag = torch.diag(ctc)
    ctc_diag[ctc_diag == 0] = 1  # For division safety
    """
    We will now compute the following expression: 
    This is part of the 'residual video' that we regress onto the spatial components below
    """

    u_subset = torch.index_select(u_sparse, 0, nonzero_row_indices)
    baseline_projection = torch.matmul(
        torch.index_select(b, 0, nonzero_row_indices), torch.sum(c, dim=0, keepdim=True)
    )
    threshold_func = torch.nn.ReLU(0)
    if blocks is None:
        blocks = torch.arange(c.shape[1], device=device).unsqueeze(1)
    for index_select_tensor in blocks:
        num_iters = math.ceil(index_select_tensor.shape[0] / frame_batch_size)
        curr_rows = []
        curr_cols = []
        curr_vals = []
        for i in range(num_iters):
            start_pt = i * frame_batch_size
            end_pt = min(start_pt + frame_batch_size, index_select_tensor.shape[0])
            subset_tensor = index_select_tensor[start_pt:end_pt]

            current_projection = torch.sparse.mm(u_subset, (v @ c[:, subset_tensor]))
            if q is not None:
                current_projection -= torch.sparse.mm(u_subset, (q[0]@(q[1]@c[:, subset_tensor])))
            current_projection -= baseline_projection[:, subset_tensor]

            a_crop = torch.index_select(a_subset, 1, subset_tensor).coalesce()
            ctc_subset = ctc[:, subset_tensor]
            current_projection -= torch.sparse.mm(a_subset, ctc_subset)
            current_projection /= ctc_diag[None, subset_tensor]

            subset_rows, subset_cols = a_crop.indices()
            subset_vals = a_crop.values()
            curr_cols.append(subset_tensor[subset_cols])
            curr_rows.append(subset_rows)
            curr_vals.append(subset_vals)

        curr_net_indices = torch.stack([torch.cat(curr_rows), torch.cat(curr_cols)])
        curr_net_values = torch.cat(curr_vals)
        net_indices = torch.cat([a_subset.indices(), curr_net_indices], dim=1)
        net_values = torch.cat([a_subset.values(), curr_net_values], dim=0)

        a_subset = torch.sparse_coo_tensor(net_indices, net_values, a_subset.shape
    ).coalesce()

        new_values = threshold_func(a_subset.values())
        a_subset = torch.sparse_coo_tensor(a_subset.indices(), new_values, a_subset.shape).coalesce()

    pruned_row, pruned_col = a_subset.indices()
    final_values = a_subset.values()
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
    q: Optional[Tuple[torch.tensor, torch.tensor]] = None,
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

    # Step 1: Get aTURs
    aTU = torch.sparse.mm(a_sparse.t(), u_sparse)
    if q is not None:
        fluctuating_background_subtracted_projection = torch.sparse.mm(aTU, v)
        fluctuating_background_subtracted_projection -= torch.sparse.mm(aTU, q[0]) @ q[1]
    else:
        fluctuating_background_subtracted_projection = torch.sparse.mm(
            aTU, v
        )


    # Step 2: Get aTbe
    aTb = torch.matmul(a_sparse.t(), b)
    static_background_projection = aTb

    # Step 3:
    cumulator = (
        fluctuating_background_subtracted_projection - static_background_projection
    )

    ata = torch.sparse.mm(a_sparse.t(), a_sparse)
    diagonals = _fast_a_squared_norm(a_sparse)

    if c_nonneg:
        threshold_function = torch.nn.ReLU()
    else:
        threshold_function = lambda x: x

    if blocks is None:
        blocks = torch.arange(c.shape[1], device=device).unsqueeze(1)
    for index_to_select in blocks:
        a_ia = torch.index_select(ata, 0, index_to_select)
        a_iaC = torch.sparse.mm(a_ia, c.t())

        curr_trace = torch.index_select(c, 1, index_to_select)
        curr_trace += (
            (torch.index_select(cumulator, 0, index_to_select) - a_iaC)
            / torch.unsqueeze(diagonals[index_to_select], -1)
        ).t()
        curr_trace = threshold_function(curr_trace)
        c[:, index_to_select] = curr_trace

    return c

def _fast_a_squared_norm(a: torch.sparse_coo_tensor):
    """
    Returns the l2 norm of each column of a
    Assumes "a" is coalesced
    """
    idx = a.indices()
    val = a.values()

    n_cols = a.shape[1]

    # squared values
    val2 = val * val
    col_idx = idx[1]

    # accumulate squares into a length-N vector
    col_sums = torch.zeros(n_cols, dtype=val.dtype, device=val.device)
    col_sums.scatter_add_(0, col_idx, val2)

    # take sqrt for L2 norm
    # col_norms = torch.sqrt(col_sums)
    return col_sums


def _affine_fit_scaling_update(v: torch.tensor,
                               a: torch.tensor,
                               c: torch.tensor,
                               b: torch.tensor,
                               m: torch.tensor,
                               ctauv: torch.tensor,
                               ata: torch.tensor,
                               ata_diag: torch.tensor,
                               c_sq: torch.tensor,
                               device: str = "cpu",
                               scale_nonneg: Optional[bool] = True,
                               blocks: Optional[Union[torch.tensor, list]] = None):
    """
    u is no longer needed since all quantities involving u are precomputed
    Args:
        v: shape (rank, T)
        a: shape (d, num_neurons)
        c: shape (num_frames, num_neurons)
        b: shape (d, 1)
    """
    catb_one = torch.sum(c * (torch.sparse.mm(a.T, b) @ torch.ones(1, v.shape[1], device=device, dtype=v.dtype)).T,
                         dim=0)  # Shape (num_neurons,)

    relu_obj = torch.nn.ReLU() if scale_nonneg else lambda x:x
    for index_select_tensor in blocks:
        aitam = ata[index_select_tensor] @ (m * c.T)
        ciaitam_val = torch.sum(c[:, index_select_tensor] * aitam.T, dim=0)
        numerator = ctauv[index_select_tensor] - catb_one[index_select_tensor] - ciaitam_val
        denominator = c_sq[index_select_tensor] * ata_diag[index_select_tensor]

        m[index_select_tensor] = relu_obj(m[index_select_tensor] + torch.nan_to_num(numerator / denominator, nan=0.0).unsqueeze(1))
    return m


def _affine_fit_baseline_update(uv_mean: torch.tensor,
                                a: torch.sparse_coo_tensor,
                                c_mean: torch.tensor,
                                m: torch.tensor):
    ac_mean = torch.sparse.mm(a, (m * c_mean))
    return uv_mean - ac_mean


def alternating_least_squares_affine_fit(u: torch.sparse_coo_tensor,
                                         v: torch.tensor,
                                         a: torch.sparse_coo_tensor,
                                         c: torch.tensor,
                                         num_iters: int =25,
                                         scale_nonneg: bool=True):
    adjacency_mat = torch.sparse.mm(a.t(), a)
    graph = construct_graph_from_sparse_tensor(adjacency_mat)
    blocks = color_and_get_tensors(graph, v.device)

    ctauv = torch.sum(c * ((torch.sparse.mm(u.T, a).T).to_dense() @ v).T, dim=0)  # Shape (num_neurons,)
    ata = torch.sparse.mm(a.T, a).to_dense()
    ata_diag = torch.diag(ata)
    c_sq = torch.sum(c * c, dim=0)

    uv_mean = torch.sparse.mm(u, torch.mean(v, dim=1, keepdim=True))
    c_mean = torch.mean(c.T, dim=1, keepdim=True)


    m = torch.ones(c.shape[1], 1, device=u.device, dtype=v.dtype)
    b = torch.ones(a.shape[0], 1, device=v.device, dtype=v.dtype)
    for _ in tqdm(range(num_iters)):
        m = _affine_fit_scaling_update(v,
                                       a,
                                       c,
                                       b,
                                       m,
                                       ctauv,
                                       ata,
                                       ata_diag,
                                       c_sq,
                                       device=v.device,
                                       scale_nonneg=scale_nonneg,
                                       blocks=blocks)
        b = _affine_fit_baseline_update(uv_mean, a, c_mean, m)
    return c*m.T, b

