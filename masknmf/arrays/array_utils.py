import numpy as np
def normalize_spatial_indexer(item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
                              spatial_shape: tuple[int, int]):
    """Return (row, col) indexers that never drop a spatial dim: ints become length-1 slices, missing dims become slice(None)."""
    spatial = list(item[1:]) if isinstance(item, tuple) else []
    spatial += [slice(None)] * (len(spatial_shape) - len(spatial))
    out = []
    for term, n in zip(spatial, spatial_shape):
        if isinstance(term, (int, np.integer)) or (isinstance(term, np.ndarray) and term.ndim == 0):
            i = int(term)
            if not -n <= i < n:
                raise IndexError(f"spatial index {i} out of bounds for size {n}")
            i %= n # handle negatives before building the slice
            term = slice(i, i + 1)
        elif isinstance(term, range):
            term = slice(term.start, term.stop, term.step)
        out.append(term)
    return tuple(out)

def test_spatial_crop_effect(my_tuple, spatial_dims) -> bool:
    """
    Returns true if the tuple used for spatial cropping actually has an effect on the underlying data. Otherwise
    cropping can be an expensive and avoidable operation.
    """
    for k in range(len(my_tuple)):
        if isinstance(my_tuple[k], np.ndarray):
            if my_tuple[k].shape[0] < spatial_dims[k]:
                return True

        if isinstance(my_tuple[k], np.integer):
            return True

        if isinstance(my_tuple[k], int):
            return True

        if isinstance(my_tuple[k], slice):
            if test_slice_effect(my_tuple[k], spatial_dims[k]):
                return True
        if isinstance(my_tuple[k], range):
            if test_range_effect(my_tuple[k], spatial_dims[k]):
                return True
    return False