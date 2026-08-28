import numpy as np


def crop_origin(centroid, crop_shape) -> tuple:
    """Top-left FOV coordinate of a crop centred on centroid, roicat's convention."""
    h, w = crop_shape
    cy, cx = centroid
    return int(cy) - int(np.ceil(h / 2)), int(cx) - int(np.ceil(w / 2))


def context_crop(fov: np.ndarray, centroid, crop_shape) -> np.ndarray:
    """Crop of fov around centroid, zero-padded where it runs off the edge."""
    h, w = crop_shape
    top, left = crop_origin(centroid, crop_shape)
    out = np.zeros((h, w), dtype=np.float32)
    y0, y1 = max(top, 0), min(top + h, fov.shape[0])
    x0, x1 = max(left, 0), min(left + w, fov.shape[1])
    if y1 > y0 and x1 > x0:
        out[y0 - top : y1 - top, x0 - left : x1 - left] = fov[y0:y1, x0:x1]
    return out
