"""
Removing small detached islands ("speckle") from demixed spatial footprints.

Removes, from each spatial footprint, detached islands of at most `max_speckle_size` nonzero pixels, unless the
island contains that footprint's brightest pixel. Larger detached pieces are kept, so discontinuous structures
(axon segments, dendrite fragments) survive.

DemixingState.demix runs this once on the final footprints (NMFConfig.max_speckle_size), then refits the background
and traces. The functions here can also be used on any results:

    from masknmf.demixing.despeckle import despeckle, despeckle_summary, plot_despeckle
    a, fov = res.spatial_demixed.cpu(), res.fov_shape
    a_clean, removed = despeckle(a, fov, max_speckle_size=3)

`a` is a sparse tensor of shape (height * width, num_signals), pixels flattened in C (row-major) order, the same
layout as DemixingResults.spatial_demixed.
"""
from __future__ import annotations

import math

import numpy as np
import torch

_OFFSETS = {
    4: ((-1, 0), (0, -1), (0, 1), (1, 0)),
    8: ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
}


def _component_labels(keys: torch.Tensor, height: int, width: int, connectivity: int) -> torch.Tensor:
    """
    keys: sorted unique int64 keys signal * (height * width) + pixel. Returns, for each key, the smallest index
    (into keys) of a member of its connected component. Components never cross signals.
    """
    n = keys.numel()
    label = torch.arange(n, device=keys.device)
    if n == 0:
        return label
    num_px = height * width
    sig, pix = keys // num_px, keys % num_px
    y, x = pix // width, pix % width
    off = torch.tensor(_OFFSETS[connectivity], device=keys.device)
    yn, xn = y[:, None] + off[None, :, 0], x[:, None] + off[None, :, 1]
    inb = (yn >= 0) & (yn < height) & (xn >= 0) & (xn < width)
    nk = sig[:, None] * num_px + yn * width + xn
    center = torch.arange(n, device=keys.device)[:, None].expand_as(nk)
    nk, center = nk[inb], center[inb]
    pos = torch.searchsorted(keys, nk).clamp(max=n - 1)
    hit = keys[pos] == nk
    nbr, ctr = pos[hit], center[hit]
    for _ in range(n):  # converges in (largest component's graph diameter) passes
        new = label.clone()
        new.scatter_reduce_(0, ctr, label[nbr], reduce="amin")
        if torch.equal(new, label):
            break
        label = new
    return label


def despeckle(
    a: torch.Tensor,
    fov_shape: tuple[int, int],
    max_speckle_size: int = 3,
    connectivity: int = 8,
    rel_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        a: sparse (height * width, num_signals) footprints, C-order pixels. COO or CSR.
        fov_shape: (height, width).
        max_speckle_size: islands with this many pixels or fewer are removed, unless they hold the footprint's
            brightest pixel (by |value|).
        connectivity: 4 or 8. Use 8 for thin structures; with 4, a diagonal 1-px process breaks into single pixels.
        rel_threshold: two rules are applied. (1) Small islands of the whole footprint (all nonzero pixels) are
            removed. (2) Small islands among the "bright" pixels, |value| >= rel_threshold * (footprint's peak),
            are removed, where faint pixels do not count as a connection. Rule 2 is needed because footprints
            usually carry a faint halo that links bright specks to the cell body, so under rule 1 alone those
            specks survive. 0.1 matches the cutoff the masknmf viewers draw footprints at (rois.MASK_CUTOFF).
            At 0 the two rules coincide.

    Returns:
        a_clean: sparse COO, same shape, speckle removed (explicit zeros also dropped).
        removed: sparse COO, same shape, the removed pixels with their values (a_clean + removed == a, minus zeros).
    """
    height, width = fov_shape
    num_px = height * width
    if a.shape[0] != num_px:
        raise ValueError(f"a has {a.shape[0]} rows but fov_shape {fov_shape} has {num_px} pixels")
    a = (a if a.layout == torch.sparse_coo else a.to_sparse_coo()).coalesce()
    row, col = a.indices()
    vals = a.values()
    nz = vals != 0
    row, col, vals = row[nz], col[nz], vals[nz]

    keys = col * num_px + row
    order = torch.argsort(keys)
    keys, vals = keys[order], vals[order]
    sig = keys // num_px

    mag = vals.abs()
    peak = torch.full((a.shape[1],), -1.0, dtype=mag.dtype, device=mag.device)
    peak.scatter_reduce_(0, sig, mag, reduce="amax", include_self=True)

    def _not_speckle(k, m):
        """True for members of islands (among keys k) larger than max_speckle_size or holding the peak."""
        label = _component_labels(k, height, width, connectivity)
        sizes = torch.bincount(label, minlength=k.numel())
        holds_peak = torch.zeros(k.numel(), dtype=torch.bool, device=k.device)
        holds_peak[label[m == peak[k // num_px]]] = True
        return (sizes[label] > max_speckle_size) | holds_peak[label]

    # Removing a speck can cut a faint bridge and leave a new small island, so repeat until nothing changes.
    # The peak's island is never removed, so peak stays valid across passes.
    keep = torch.ones(keys.numel(), dtype=torch.bool, device=keys.device)
    while True:
        idx = keep.nonzero().squeeze(1)
        k, m = keys[idx], mag[idx]
        # Rule 1: small islands of the whole footprint (all nonzero pixels)
        ok = _not_speckle(k, m)
        # Rule 2: small islands among the bright pixels, where the faint halo does not count as a connection
        bright = m >= rel_threshold * peak[k // num_px]
        ok[bright] &= _not_speckle(k[bright], m[bright])
        if bool(ok.all()):
            break
        keep[idx[~ok]] = False

    def _coo(k, v):
        return torch.sparse_coo_tensor(torch.stack([k % num_px, k // num_px]), v, a.shape).coalesce()

    return _coo(keys[keep], vals[keep]), _coo(keys[~keep], vals[~keep])


def despeckle_summary(
    a: torch.Tensor, fov_shape: tuple[int, int], max_speckle_size: int = 3, connectivity: int = 8,
    rel_threshold: float = 0.1,
) -> dict:
    """Per-signal counts of removed pixels and the fraction of footprint mass (sum |a|) they carried."""
    a_clean, removed = despeckle(a, fov_shape, max_speckle_size, connectivity, rel_threshold)
    k = a.shape[1]
    removed_px = torch.bincount(removed.indices()[1], minlength=k)
    removed_mass = torch.zeros(k, dtype=removed.values().dtype).index_add_(
        0, removed.indices()[1], removed.values().abs())
    a_c = (a if a.layout == torch.sparse_coo else a.to_sparse_coo()).coalesce()
    total_mass = torch.zeros(k, dtype=a_c.values().dtype).index_add_(0, a_c.indices()[1], a_c.values().abs())
    frac = (removed_mass / total_mass.clamp_min(1e-12)).numpy()
    return {
        "num_signals": k,
        "signals_affected": int((removed_px > 0).sum()),
        "pixels_removed": int(removed_px.sum()),
        "pixels_removed_per_signal": removed_px.numpy(),
        "mass_fraction_removed_per_signal": frac,
        "max_mass_fraction_removed": float(frac.max()) if k else 0.0,
    }


def plot_despeckle(
    a: torch.Tensor,
    fov_shape: tuple[int, int],
    max_speckle_size: int = 3,
    connectivity: int = 8,
    rel_threshold: float = 0.1,
    indices: list[int] | None = None,
    num_to_show: int = 12,
    pad: int = 3,
    ncols: int = 4,
):
    """
    For each chosen signal: the original footprint, the cleaned one, and the original with removed pixels
    circled in red. Each panel is cropped to the footprint's bounding box plus `pad`.

    indices: which signals to show. Default: the `num_to_show` signals that lose the most pixels.
    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    height, width = fov_shape
    a_clean, removed = despeckle(a, fov_shape, max_speckle_size, connectivity, rel_threshold)
    if indices is None:
        n_rm = torch.bincount(removed.indices()[1], minlength=a.shape[1])
        indices = [int(i) for i in torch.argsort(n_rm, descending=True)[:num_to_show] if n_rm[i] > 0]
        if not indices:
            print("No speckle found with these settings.")
            return None

    def dense(t, k):
        img = torch.zeros(height * width, dtype=torch.float32)
        c = t.coalesce()
        m = c.indices()[1] == k
        img[c.indices()[0][m]] = c.values()[m].float()
        return img.reshape(height, width).numpy()

    a_coo = (a if a.layout == torch.sparse_coo else a.to_sparse_coo()).coalesce()
    nrows = math.ceil(len(indices) / ncols)
    fig, axes = plt.subplots(nrows, 3 * ncols, figsize=(3 * ncols * 1.9, nrows * 2.1), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for n, k in enumerate(indices):
        orig, clean, rm = dense(a_coo, k), dense(a_clean, k), dense(removed, k)
        ys, xs = np.nonzero(orig)
        if ys.size == 0:
            continue
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, height)
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, width)
        vmax = np.abs(orig).max()
        r, c0 = divmod(n, ncols)
        panels = axes[r, 3 * c0: 3 * c0 + 3]
        for ax, img, title in zip(panels, (orig, clean, orig),
                                  (f"#{k} original", f"cleaned", f"removed: {int((rm != 0).sum())} px")):
            ax.imshow(img[y0:y1, x0:x1], cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(title, fontsize=8)
        ry, rx = np.nonzero(rm[y0:y1, x0:x1])
        panels[2].scatter(rx, ry, s=40, facecolors="none", edgecolors="red", linewidths=1.2)
    fig.suptitle(f"despeckle: islands <= {max_speckle_size} px among pixels >= {rel_threshold:g} x peak removed "
                 f"({connectivity}-connectivity)", fontsize=10)
    fig.tight_layout()
    return fig
