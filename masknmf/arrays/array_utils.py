import numpy as np
import torch
def _as_index_array(term, n: int) -> np.ndarray:
    """Lists, tuples, numpy arrays and torch tensors -> 1-D int64 indices into an axis of length n."""
    if isinstance(term, torch.Tensor):
        term = term.detach().cpu().numpy()
    idx = np.asarray(term)
    if idx.dtype == bool:
        if idx.shape != (n,):
            raise IndexError(f"boolean mask has shape {idx.shape}, expected ({n},)")
        idx = np.flatnonzero(idx)
    elif idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
        raise IndexError(f"spatial index arrays must be 1-D integer or boolean, got shape {idx.shape}, dtype {idx.dtype}")
    if idx.size and (idx.min() < -n or idx.max() >= n):
        raise IndexError(f"spatial indices out of bounds for size {n}")
    return idx.astype(np.int64) % n


def normalize_spatial_indexer(item, spatial_shape: tuple[int, int]) -> tuple[slice | np.ndarray, ...]:
    """
    Normalize the spatial part of `item` (entries 1 and 2) for rectangular cropping.

    Every returned term is a slice or a 1-D int64 array, so no spatial dim is dropped:
    ints become length-1 slices, missing dims become slice(None), ranges become slices,
    and lists / bool masks / tensors become int arrays.
    Only one spatial axis may be an index array: two arrays would mean pointwise
    (non-rectangular) selection in numpy, which is not supported.
    """
    spatial = list(item[1:3]) if isinstance(item, tuple) else []
    spatial += [slice(None)] * (len(spatial_shape) - len(spatial))

    out = []
    for term, n in zip(spatial, spatial_shape):
        if isinstance(term, (int, np.integer)) or (isinstance(term, np.ndarray) and term.ndim == 0):
            i = int(term)
            if not -n <= i < n:
                raise IndexError(f"spatial index {i} out of bounds for size {n}")
            i %= n
            term = slice(i, i + 1)
        elif isinstance(term, range):
            term = slice(term.start, term.stop, term.step)
        elif not isinstance(term, slice):
            term = _as_index_array(term, n)
        if len(range(*term.indices(n))) == 0 if isinstance(term, slice) else term.size == 0:
            raise IndexError("empty spatial selection")
        out.append(term)

    if sum(isinstance(t, np.ndarray) for t in out) > 1:
        raise IndexError(
            "Only rectangular spatial indexing is supported: at most one spatial axis may be "
            "indexed with a list/array. Use a slice on the other axis, or index in two steps."
        )
    return tuple(out)


def test_spatial_crop_effect(spatial, spatial_shape) -> bool:
    """Expects the output of normalize_spatial_indexer. True if the crop selects anything other than the full FOV."""
    for term, n in zip(spatial, spatial_shape):
        if isinstance(term, slice):
            if term.indices(n) != (0, n, 1):
                return True
        elif term.size != n or not np.array_equal(term, np.arange(n)):
            return True
    return False