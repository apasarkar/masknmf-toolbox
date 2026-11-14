"""
The pixel mask functions on a synthetic field of view with a dark vessel.
"""

import numpy as np
import pytest

from masknmf.visualization import pixel_mask


@pytest.fixture
def images():
    rng = np.random.default_rng(0)
    rows, cols = np.mgrid[:64, :80]
    mean = np.full((64, 80), 200.0)
    vessel = np.abs(rows - 30) < 4
    mean[vessel] = 50.0
    frames = rng.poisson(np.broadcast_to(mean, (100, 64, 80))).astype(np.float32)
    return pixel_mask.summary_images(frames=frames), vessel


def test_otsu_range_masks_the_vessel(images):
    summary, vessel = images
    params = pixel_mask.MaskParams(sigma=0.0, range_mean=(pixel_mask.otsu_percentile(image=summary["mean"]), 100.0))
    keep = pixel_mask.compute_mask(images=summary, params=params)
    assert keep.dtype == bool and keep.shape == vessel.shape
    np.testing.assert_array_equal(keep, ~vessel)


def test_cleanup_drops_small_regions_and_grows(images):
    _, vessel = images
    keep = ~vessel
    keep[5, 5] = False
    keep[30, 40] = True
    cleaned = pixel_mask.clean_mask(keep=keep, min_region=10, grow=0)
    np.testing.assert_array_equal(cleaned, ~vessel)
    grown = pixel_mask.clean_mask(keep=~vessel, min_region=0, grow=2)
    assert (~grown).sum() > vessel.sum() and not grown[vessel].any()
    shrunk = pixel_mask.clean_mask(keep=~vessel, min_region=0, grow=-2)
    assert (~shrunk).sum() < vessel.sum()


def test_brush_overrides_the_threshold(images):
    _, vessel = images
    brush = np.zeros(vessel.shape, dtype=np.uint8)
    pixel_mask.paint_disk(target=brush, row=10, col=10, radius=3, value=1)
    pixel_mask.paint_disk(target=brush, row=30, col=40, radius=2, value=2)
    keep = pixel_mask.apply_brush(keep=~vessel, brush=brush)
    assert not keep[10, 10] and keep[30, 40] and not keep[30, 10] and keep[50, 50]


def test_compose_tints_only_masked_pixels(images):
    summary, vessel = images
    rgb = pixel_mask.compose_rgb(image=summary["mean"], keep=~vessel, opacity=1.0)
    assert rgb.dtype == np.uint8 and rgb.shape == (*vessel.shape, 3)
    assert (rgb[vessel] == pixel_mask.COLOR_MASKED.astype(np.uint8)).all()
    assert (rgb[~vessel][:, 0] == rgb[~vessel][:, 1]).all()
