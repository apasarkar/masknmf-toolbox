import scipy.sparse
import torch
from masknmf.utils.display import progress
import math
from masknmf.utils import SparseCOOTensor, SparseCSRTensor
from masknmf.demixing.demixing_utils import (construct_graph_from_sparse_tensor,
                                             color_and_get_tensors,
                                             csr_index_select_rows,
                                             decompress_csr_rows,
                                             _eval_sparse_dense_matmul)


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


def spatial_update_hals(u_sparse: SparseCSRTensor,
                        v: torch.Tensor,
                        a_sparse: SparseCSRTensor,
                        c: torch.Tensor,
                        b: torch.Tensor,
                        q: tuple[torch.Tensor, torch.Tensor] | None = None,
                        blocks: torch.Tensor | list | None = None,
                        mask_ab=None,
                        frame_batch_size=500):
    device = v.device
    c_sq_norm = torch.linalg.norm(c, dim=0) ** 2                       # (K,)
    normalized_c_sum = torch.sum(c, dim=0) / c_sq_norm                 # (K,)
    if blocks is None:
        blocks = torch.arange(c.shape[1], device=device).unsqueeze(1)

    index_dtype = a_sparse.col_indices().dtype
    column_lut = torch.zeros(a_sparse.shape[1], device=device, dtype=index_dtype)
    membership_lut = torch.zeros(a_sparse.shape[1], device=device, dtype=torch.bool)
    a_rows = decompress_csr_rows(a_sparse.crow_indices(), a_sparse.values().numel())
    a_cols = a_sparse.col_indices()
    a_values = a_sparse.values()

    for index_select_tensor in blocks:
        for start_pt in range(0, index_select_tensor.shape[0], frame_batch_size):
            subset_tensor = index_select_tensor[start_pt:start_pt + frame_batch_size]
            column_lut[subset_tensor] = torch.arange(subset_tensor.shape[0], device=device, dtype=index_dtype)
            membership_lut[subset_tensor] = True
            col_mask = membership_lut[a_cols]
            col_indices = a_cols[col_mask]
            remapped_col_indices = column_lut[col_indices]
            row_indices = a_rows[col_mask]
            curr_values = a_values[col_mask]

            # Only this subset's columns of (V - F1 F2) c / ||c||^2 and C^T c / ||c||^2
            c_sub = c[:, subset_tensor]                                # (T, s)
            diff_sub = v @ c_sub                                       # (R, s)
            if q is not None:
                diff_sub -= q[0] @ (q[1] @ c_sub)
            diff_sub /= c_sq_norm[subset_tensor]
            ctc_sub = (c.T @ c_sub) / c_sq_norm[subset_tensor]         # (K, s)

            # Each support entry needs only its own signal's column: gather, don't form (rows x s) products
            term1 = _eval_sparse_dense_matmul(u_sparse, diff_sub, row_indices, remapped_col_indices)
            term2 = _eval_sparse_dense_matmul(a_sparse, ctc_sub, row_indices, remapped_col_indices)
            curr_values += term1 - term2 - b[row_indices, 0] * normalized_c_sum[col_indices]
            curr_values.relu_()
            a_sparse.values().masked_scatter_(col_mask, curr_values)

            membership_lut[subset_tensor] = False
    return a_sparse

def temporal_update_hals(
    u_sparse: SparseCSRTensor,
    v: torch.Tensor,
    a_sparse: SparseCOOTensor,
    c: torch.Tensor,
    b: torch.Tensor,
    q: tuple[torch.Tensor, torch.Tensor] | None = None,
    c_nonneg: bool = True,
    blocks: torch.Tensor | list | None = None,
):
    """
    Inputs:
        u_sparse: torch.sparse_csr_tensor. Sparse matrix, with dimensions (d x R)
        v: torch.Tensor. Dimensions R' x T. Dimensions R' x T, where T is the number of frames, where all rows are orthonormal.
            Note:  V must contain the 1 x T vector of all 1's in its rowspan.
        a: (d1*d2, k)-shaped torch.sparse_coo_tensor
        c: (T, k)-shaped torch.Tensor
        b: (d1*d2, 1)-shaped torch.Tensor
        q (torch.Tensor | None): This is the factorized ring model term; u@r@q@v gives you the full ring model movie
        c_nonneg (bool): Indicates whether "c" should be nonnegative or fully unconstrained. For voltage data, it should be unconstrained; for calcium it should be constrained.
        blocks torch.Tensor | list: Describes which components can be updated in parallel. Typically a list of 1D tensors, each describing indices

    Returns:
        c: (T, k)-shaped np.ndarray. Updated temporal components
    """
    device = v.device

    ##Precompute quantities used throughout all iterations

    # Step 1: Get aTURs
    a_sparse_t = a_sparse.t().to_sparse_csr()
    a_sparse = a_sparse.to_sparse_csr()
    aTU = torch.sparse.mm(a_sparse_t, u_sparse)
    if q is not None:
        fluctuating_background_subtracted_projection = torch.sparse.mm(aTU, v)
        fluctuating_background_subtracted_projection -= torch.sparse.mm(aTU, q[0]) @ q[1]
    else:
        fluctuating_background_subtracted_projection = torch.sparse.mm(
            aTU, v
        )


    # Step 2: Get aTbe
    aTb = torch.matmul(a_sparse_t, b)
    static_background_projection = aTb

    # Step 3:
    cumulator = (
        fluctuating_background_subtracted_projection - static_background_projection
    )

    ata = torch.sparse.mm(a_sparse_t, a_sparse)
    diagonals = _fast_a_squared_norm(a_sparse)

    if c_nonneg:
        threshold_function = torch.nn.ReLU()
    else:
        threshold_function = lambda x: x

    if blocks is None:
        blocks = torch.arange(c.shape[1], device=device).unsqueeze(1)
    for index_to_select in blocks:
        a_ia = csr_index_select_rows(ata,  index_to_select)
        a_iaC = torch.sparse.mm(a_ia, c.t())

        curr_trace = torch.index_select(c, 1, index_to_select)
        curr_trace += (
            (torch.index_select(cumulator, 0, index_to_select) - a_iaC)
            / torch.unsqueeze(diagonals[index_to_select], -1)
        ).t()
        curr_trace = threshold_function(curr_trace)
        c[:, index_to_select] = curr_trace

    return c

def _fast_a_squared_norm(a: SparseCSRTensor) -> torch.Tensor:
    """
    Returns the squared l2 norm of each column of a sparse CSR tensor a, shape (d, K) -> (K,).
    """
    vals = a.values()
    out = torch.zeros(a.shape[1], dtype=vals.dtype, device=vals.device)
    return out.scatter_add_(0, a.col_indices(), vals * vals)

def _affine_fit_scaling_update(v: torch.Tensor,
                               a: torch.Tensor,
                               c: torch.Tensor,
                               b: torch.Tensor,
                               m: torch.Tensor,
                               ctauv: torch.Tensor,
                               ata: SparseCSRTensor,
                               ata_diag: torch.Tensor,
                               c_sq: torch.Tensor,
                               device: str = "cpu",
                               scale_nonneg: bool = True,
                               blocks: torch.Tensor | list | None = None):
    """
    u is no longer needed since all quantities involving u are precomputed
    Args:
        v: shape (rank, T)
        a: shape (d, num_neurons)
        c: shape (num_frames, num_neurons)
        b: shape (d, 1)
    """
    catb_one = torch.sum(c * (torch.sparse.mm(a.t(), b) @ torch.ones(1, v.shape[1], device=device, dtype=v.dtype)).T,
                         dim=0)  # Shape (num_neurons,)

    relu_obj = torch.nn.ReLU() if scale_nonneg else lambda x:x
    for index_select_tensor in blocks:
        ata_subset = csr_index_select_rows(ata, index_select_tensor)
        aitam = torch.sparse.mm(ata_subset, (m * c.T))
        ciaitam_val = torch.sum(c[:, index_select_tensor] * aitam.T, dim=0)
        numerator = ctauv[index_select_tensor] - catb_one[index_select_tensor] - ciaitam_val
        denominator = c_sq[index_select_tensor] * ata_diag[index_select_tensor]

        m[index_select_tensor] = relu_obj(m[index_select_tensor] + torch.nan_to_num(numerator / denominator, nan=0.0).unsqueeze(1))
    return m


def _affine_fit_baseline_update(uv_mean: torch.Tensor,
                                a: SparseCOOTensor,
                                c_mean: torch.Tensor,
                                m: torch.Tensor):
    ac_mean = torch.sparse.mm(a, (m * c_mean))
    return uv_mean - ac_mean


def alternating_least_squares_affine_fit(u: SparseCSRTensor,
                                         v: torch.Tensor,
                                         a: SparseCOOTensor,
                                         c: torch.Tensor,
                                         num_iters: int =25,
                                         scale_nonneg: bool=True):
    a = a.to_sparse_csr()
    a_t = a.t().to_sparse_csr()
    adjacency_mat = torch.sparse.mm(a.t(), a)
    graph = construct_graph_from_sparse_tensor(adjacency_mat)
    blocks = color_and_get_tensors(graph, v.device)

    ctauv = torch.sum(c * (torch.sparse.mm(torch.sparse.mm(a_t, u), v).T), dim=0) # Shape (num_neurons,)
    ata_diag = _fast_a_squared_norm(a)
    c_sq = torch.sum(c * c, dim=0)

    uv_mean = torch.sparse.mm(u, torch.mean(v, dim=1, keepdim=True))
    c_mean = torch.mean(c.T, dim=1, keepdim=True)


    m = torch.ones(c.shape[1], 1, device=u.device, dtype=v.dtype)
    b = torch.ones(a.shape[0], 1, device=v.device, dtype=v.dtype)
    for _ in progress(range(num_iters), desc="affine fit"):
        m = _affine_fit_scaling_update(v,
                                       a,
                                       c,
                                       b,
                                       m,
                                       ctauv,
                                       adjacency_mat,
                                       ata_diag,
                                       c_sq,
                                       device=v.device,
                                       scale_nonneg=scale_nonneg,
                                       blocks=blocks)
        b = _affine_fit_baseline_update(uv_mean, a, c_mean, m)
    return c*m.T, b

