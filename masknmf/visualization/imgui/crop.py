import numpy as np


def crop_origin(centroid, crop_shape) -> tuple:
    """Top-left FOV coordinate of a crop centred on centroid, roicat's convention."""
    h, w = crop_shape
    cy, cx = centroid
    return int(cy) - int(np.ceil(h / 2)), int(cx) - int(np.ceil(w / 2))


def crop_slices(top: int, left: int, crop_shape, fov_shape):
    """(dst, src) slice pairs for a crop that may run off the FOV; None if fully outside."""
    h, w = crop_shape
    y0, y1 = max(top, 0), min(top + h, fov_shape[0])
    x0, x1 = max(left, 0), min(left + w, fov_shape[1])
    if y1 <= y0 or x1 <= x0:
        return None
    return (slice(y0 - top, y1 - top), slice(x0 - left, x1 - left)), (slice(y0, y1), slice(x0, x1))


def context_crop(fov: np.ndarray, centroid, crop_shape) -> np.ndarray:
    """Crop of fov around centroid, zero-padded where it runs off the edge."""
    top, left = crop_origin(centroid, crop_shape)
    out = np.zeros(tuple(crop_shape), dtype=np.float32)
    slices = crop_slices(top, left, crop_shape, fov.shape)
    if slices is not None:
        dst, src = slices
        out[dst] = fov[src]
    return out
