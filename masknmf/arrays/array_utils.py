import numpy as np
def normalize_spatial_indexer(item: int | list | np.ndarray | tuple[int | np.ndarray | slice | range],
                              spatial_shape: tuple[int, int]):
    """Return (row, col) indexers that never drop a spatial dim: ints become length-1 slices, missing dims become slice(None).
    Assumes that the first 3 entries of item if they exist correspond to frames, fov_height, fov_width"""
    spatial = list(item[1:min(len(item), 3)]) if isinstance(item, tuple) else []
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

def test_slice_effect(my_slice: slice, spatial_dim: int) -> bool:
    """
    Returns True if slice will actually have an effect
    """

    if not (
        (isinstance(my_slice.start, int) and my_slice.start == 0)
        or my_slice.start is None
    ):
        return True
    elif not (
        (isinstance(my_slice.stop, int) and my_slice.stop >= spatial_dim)
        or my_slice.stop is None
    ):
        return True
    elif not (
        my_slice.step is None or (isinstance(my_slice.step, int) and my_slice.step == 1)
    ):
        return True
    return False


def test_range_effect(my_range: range, spatial_dim: int) -> bool:
    """
    Returns True if the range will actually have an effect.

    Parameters:
    my_range (range): The range object to test.
    spatial_dim (int): The size of the dimension that the range is applied to.

    Returns:
    bool: True if the range will affect the selection; False otherwise.
    """
    # Check if the range starts from the beginning
    if my_range.start != 0:
        return True
    # Check if the range stops at the end of the dimension
    elif my_range.stop != spatial_dim:
        return True
    # Check if the range step is not 1
    elif my_range.step != 1:
        return True
    return False


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