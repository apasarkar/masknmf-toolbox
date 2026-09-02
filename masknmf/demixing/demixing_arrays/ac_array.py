from __future__ import annotations
from typing import Optional
import numpy as np
import torch
from typing import *
import numpy as np
from masknmf.arrays.array_interfaces import ArrayLike, TensorFlyWeight
import torch
from masknmf.demixing.demixing_arrays.demixing_array_utils import check_spatial_crop_effect

class ACArray(ArrayLike):
    """
    Factorized video for the spatial and temporal extracted sources from the data
    Computations happen transparently on GPU, if device = 'cuda' is specified
    """

    def __init__(
        self,
        fov_shape: tuple[int, int],
        flyweight: TensorFlyWeight,
        rescale: bool = False
    ):


        self._flyweight = flyweight
        self.flyweight.validate_attributes(["a", "c"])
        num_frames = self.c.shape[0]
        self._shape = tuple(map(int, (num_frames, *fov_shape)))

        self._pixel_mat = torch.arange(self.shape[1] * self.shape[2], device=self.device, dtype=torch.long).reshape(
            self.shape[1], self.shape[2])
        self._mask = torch.ones(self.a.shape[1], device=self.device, dtype=self.c.dtype)
        self._centers = None
        self._bbox = None
        self._contours = None

        self._rescale = rescale
        self._default_normalizer = torch.ones(self.shape[1], self.shape[2], device=self.device).float()
        if hasattr(self.flyweight, "normalizer"):
            if self.flyweight.normalizer is not None:
                if self.flyweight.normalizer.shape[0] != self.shape[1] or self.flyweight.normalizer.shape[1] != self.shape[2]:
                    raise ValueError("Normalizer from flyweight had dimensions not equal to the fov dimensions")

    @classmethod
    def from_tensors(cls,
                     fov_shape: tuple[int, int],
                     a: torch.sparse_coo_tensor,
                     c: torch.Tensor,
                     normalizer: torch.Tensor | None = None,
                     rescale: bool = False
                     ):
        """
        Args:
            fov_shape (tuple): (fov_dim1, fov_dim2)
            a (torch.sparse_coo_tensor): Shape (pixels, components)
            c (torch.Tensor). Shape (frames, components)
            normalizer (Optional[torch.Tensor]): A (height, width)-shaped tensor. Multiply the array pixelwise by this
                value to obtain results in the "raw data" space.
            rescale (bool): Whether or not to rescale the data to the raw data space. This determines the output of getitem
        """
        flyweight = TensorFlyWeight(a=a, c=c, normalizer=normalizer)
        return cls(fov_shape,
                   flyweight,
                   rescale=rescale)

    @classmethod
    def from_flyweight(cls,
                       fov_shape: tuple[int, int],
                       flyweight: TensorFlyWeight,
                       rescale: bool = False):
        return cls(fov_shape,
                   flyweight,
                   rescale=rescale)


    @property
    def flyweight(self) -> TensorFlyWeight:
        return self._flyweight

    @property
    def device(self):
        return self.flyweight.device


    def to(self, new_device):
        if self._flyweight.device != new_device:
            self._flyweight.to(new_device)
        self._move_local_tensors(new_device)

    def _move_local_tensors(self, new_device: str):
        self._pixel_mat = self._pixel_mat.to(new_device)
        self._mask = self._mask.to(new_device)
        self._default_normalizer = self._default_normalizer.to(new_device)


    @property
    def mask(self) -> torch.Tensor:
        return self._mask

    @mask.setter
    def mask(self, new_mask: torch.Tensor):
        self._mask = new_mask.to(self.device).bool().to(self.c.dtype) #Ensures it's all 1s and 0s

    @property
    def normalizer(self) -> torch.Tensor:
        if not hasattr(self.flyweight, "normalizer") or self.flyweight.normalizer is None:
            return self._default_normalizer
        return self.flyweight.normalizer

    @property
    def rescale(self):
        return self._rescale

    @rescale.setter
    def rescale(self, new_value: bool):
        self._rescale = new_value

    @property
    def contours(self) -> list[np.ndarray]:
        """
        Returns Each column describes a binary contour mask
        Since we are not computing statistics or doing any other analyses on this contour mask, this is computing via a simple 1-step dilation
        procedure which allows us to compute everything using the sparse "a" matrix.
        Returns:
            -  torch.sparse_coo_tensor of shape (num_pixels, num_signals).
        """
        if self._contours is None:
            self._contours = compute_contours(self.a, self.shape[1], self.shape[2])
        return self._contours

    @property
    def centers(self) -> torch.Tensor:
        """
        Returns a (num_signals, 2) shaped tensor describing the height, width dimensions of each signals spatial center
            of mass. The center of mass might not be on an active pixel, but this is ok
        """
        if self._centers is None:
            height, width = self.shape[1:]
            num_signals = self.a.shape[1]
            row, col = self.a.indices()
            values = self.a.values().float()

            # First get rid of values that are nonzero
            values_keep = values != 0

            row = row[values_keep]
            col = col[values_keep]
            values = values[values_keep]


            # Need to unvectorize row
            dim0_coords = row // width
            dim1_coords = row % width

            dim0_com_numerator = torch.zeros(num_signals, device=self.device).float()
            dim0_com_denominator = torch.zeros(num_signals, device=self.device).float()

            dim1_com_numerator = torch.zeros(num_signals, device=self.device).float()
            dim1_com_denominator = torch.zeros(num_signals, device=self.device).float()

            dim0_com_numerator.scatter_reduce_(0, col, (dim0_coords * values).float(), reduce="sum")
            dim0_com_denominator.scatter_reduce_(0, col, values, reduce="sum")

            dim1_com_numerator.scatter_reduce_(0, col, (dim1_coords * values).float(), reduce="sum")
            dim1_com_denominator.scatter_reduce_(0, col, values, reduce="sum")

            dim0_com = torch.nan_to_num(dim0_com_numerator / dim0_com_denominator, nan=0.0)
            dim1_com = torch.nan_to_num(dim1_com_numerator / dim1_com_denominator, nan=0.0)

            self._centers = torch.stack([dim0_com, dim1_com], dim=1)

        return self._centers

    @property
    def bbox(self) -> torch.Tensor:
        """
        Returns a torch tensor of shape (num_signals, 4). Each row contains 4 elements a1, a2, b1, b2 which define a bounding box
            of a neuron image like img[a1:a2, b1:b2]

        """
        if self._bbox is None:
            height, width = self.shape[1:]
            num_signals = self.a.shape[1]
            row, col = self.a.indices()
            values = self.a.values()

            #First get rid of values that are nonzero
            row = row[values != 0]
            col = col[values != 0]

            #Need to unvectorize row
            dim0_values = (row // width).long()
            dim1_values = (row % width).long()

            min_dim0 = torch.zeros(num_signals, device=self.device).long()
            max_dim0 = torch.zeros(num_signals, device=self.device).long()
            min_dim1 = torch.zeros(num_signals, device=self.device).long()
            max_dim1 = torch.zeros(num_signals, device=self.device).long()

            min_dim0.scatter_reduce_(0, col, dim0_values,  reduce="amin", include_self=False)
            max_dim0.scatter_reduce_(0, col, dim0_values, reduce="amax", include_self=False)
            min_dim1.scatter_reduce_(0, col, dim1_values, reduce="amin", include_self=False)
            max_dim1.scatter_reduce_(0, col, dim1_values, reduce="amax", include_self=False)

            self._bbox = torch.stack([min_dim0, max_dim0, min_dim1, max_dim1], dim=1)
        return self._bbox




    @property
    def c(self) -> torch.Tensor:
        """
        return temporal time courses of all signals, shape (frames, components)
        """
        return self.flyweight.c

    @property
    def a(self) -> torch.sparse_coo_tensor:
        """
        return spatial profiles of all signals as sparse matrix, shape (pixels, components)
        """
        return self.flyweight.a

    def export_a(self, apply_rescale: bool = False) -> np.ndarray:
        """
        returns the spatial components, where each component is a 2D image. output shape (fov dim1, fov dim 2, n_frames)
        """
        output = self.a.cpu().to_dense().numpy()
        output = output.reshape((self.shape[-2], self.shape[-1], -1))
        if apply_rescale:
            output *= self.normalizer[..., None].cpu().numpy()
        return output

    def export_c(self) -> np.ndarray:
        """
        returns the temporal traces, where each trace is a n_frames-shaped time series. output shape (n_frames, n_components)
        """
        return self.c.cpu().numpy()

    @property
    def dtype(self) -> str:
        """
        data type, default np.float32
        """
        return np.float32

    @property
    def shape(self) -> Tuple[int, int, int]:
        """
        Array shape (n_frames, dims_x, dims_y)
        """
        return self._shape

    @property
    def ndim(self) -> int:
        """
        Number of dimensions
        """
        return len(self.shape)

    def getitem_tensor(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> torch.Tensor:
        # Step 1: index the frames (dimension 0)
        frame_indexer, item = self._parse_indices(item)

        # Step 3: Now slice the data with frame_indexer (careful: if the ndims has shrunk, add a dim)
        c_crop = self.c[frame_indexer, :]
        if c_crop.ndim < self.c.ndim:
            c_crop = c_crop.unsqueeze(0)

        c_crop = c_crop * self._mask[None, :]

        # Step 4: First do spatial subselection before multiplying by c
        if isinstance(item, tuple) and check_spatial_crop_effect(item[1:3], self.shape[1:3]):

            pixel_space_crop = self._pixel_mat[item[1:3]]
            normalizer_crop = self.normalizer[item[1:3]][None, ...]
            a_indices = pixel_space_crop.flatten()
            a_crop = torch.index_select(self.a, 0, a_indices)
            implied_fov = pixel_space_crop.shape
            product = torch.sparse.mm(a_crop, c_crop.T)
            product = product.reshape(implied_fov + (-1,))
            product = product.permute(-1, *range(product.ndim - 1))

        else:
            a_crop = self.a
            normalizer_crop  = self.normalizer[None, :, :]
            implied_fov = self.shape[-2], self.shape[-1]
            product = torch.sparse.mm(a_crop, c_crop.T)
            product = product.reshape((implied_fov[0], implied_fov[1], -1))
            product = product.permute(2, 0, 1)

        if self.rescale:
            product *= normalizer_crop

        return product

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, Tuple[Union[int, np.ndarray, slice, range]]],
    ) -> np.ndarray:
        product = self.getitem_tensor(item)
        product = product.cpu().numpy().astype(self.dtype)
        return product



"""
Fast, fully-vectorized contour extraction for factorized spatial footprints.

Design notes
------------
Everything operates on a set of ``(pixel, component)`` pairs encoded as a single
sorted int64 key::

    key = component * (height * width) + (y * width + x)

Because the key is component-major, a sorted key array is *already* grouped by
component and sorted by pixel within each component, which makes the final
"split into one array per component" step a ``bincount`` plus ``np.split`` with
no Python loop.

Set operations (union / difference / intersection / membership) become
``torch.unique`` and ``torch.searchsorted`` on that sorted array

Cost of every operator below is O(nnz * K) where K is the connectivity (4 or 8),
plus one sort.
"""


_OFFSETS = {
    4: ((-1, 0), (0, -1), (0, 1), (1, 0)),
    8: ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)),
}


class PixelSet:
    """An immutable set of (pixel, component) pairs supporting morphology."""

    __slots__ = ("keys", "height", "width", "num_components")

    def __init__(
        self,
        keys: torch.Tensor,
        height: int,
        width: int,
        num_components: int,
        already_sorted: bool = False,
    ):
        self.height = int(height)
        self.width = int(width)
        self.num_components = int(num_components)
        self.keys = keys if already_sorted else torch.unique(keys)

    # ------------------------------------------------------------------ #
    # construction / conversion
    # ------------------------------------------------------------------ #

    @classmethod
    def from_sparse(
        cls,
        a: torch.Tensor,
        height: int,
        width: int,
        rel_threshold: float = 0.0,
        abs_threshold: float = 0.0,
    ) -> tuple["PixelSet", torch.Tensor]:
        """
        Build a support set from a sparse (pixels, components) matrix.

        Args:
            a: sparse COO tensor, shape (height * width, num_components).
            rel_threshold: keep pixel only if |value| >= rel_threshold * peak
                of its own component. This is the single cheapest quality win:
                0.05-0.15 removes the faint halo that causes ragged boundaries.
            abs_threshold: keep pixel only if |value| > abs_threshold.

        Returns:
            (support, values) where ``values`` is aligned elementwise with
            ``support.keys`` (needed by connected-component filtering).
        """
        a = a.coalesce()
        row, col = a.indices()
        vals = a.values()
        mag = vals.abs()

        keep = mag > abs_threshold
        if rel_threshold > 0:
            peak = torch.zeros(a.shape[1], device=vals.device, dtype=mag.dtype)
            peak.scatter_reduce_(0, col, mag, reduce="amax", include_self=False)
            keep = keep & (mag >= rel_threshold * peak[col])

        row, col, vals = row[keep], col[keep], vals[keep]
        keys = col * (height * width) + row
        order = torch.argsort(keys)
        return (
            cls(keys[order], height, width, a.shape[1], already_sorted=True),
            vals[order],
        )

    def decode(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (y, x, component) for every member, in key order."""
        per_frame = self.height * self.width
        comp = torch.div(self.keys, per_frame, rounding_mode="floor")
        pix = self.keys - comp * per_frame
        y = torch.div(pix, self.width, rounding_mode="floor")
        return y, pix - y * self.width, comp

    def _like(self, keys: torch.Tensor, already_sorted: bool = False) -> "PixelSet":
        return PixelSet(
            keys, self.height, self.width, self.num_components, already_sorted
        )

    # ------------------------------------------------------------------ #
    # primitives
    # ------------------------------------------------------------------ #

    def _neighbor_table(self, connectivity: int = 8):
        """
        Returns (neighbor_keys, center_index, n_in_bounds).

        ``neighbor_keys[i]`` is an in-bounds neighbor of member
        ``center_index[i]``. ``n_in_bounds[j]`` counts how many of member j's
        K neighbors fell inside the FOV -- needed so that footprints touching
        the image border are not eroded away.
        """
        device = self.keys.device
        offsets = torch.tensor(_OFFSETS[connectivity], device=device, dtype=torch.long)
        y, x, comp = self.decode()

        yn = y[:, None] + offsets[None, :, 0]
        xn = x[:, None] + offsets[None, :, 1]
        # NOTE: >= 0, not > 0. The original code used > 0 and silently dropped
        # the entire first row and first column of the FOV.
        in_bounds = (yn >= 0) & (yn < self.height) & (xn >= 0) & (xn < self.width)

        nk = comp[:, None] * (self.height * self.width) + yn * self.width + xn
        center = torch.arange(self.keys.numel(), device=device)[:, None].expand_as(nk)
        return nk[in_bounds], center[in_bounds], in_bounds.sum(dim=1)

    def contains(self, query: torch.Tensor) -> torch.Tensor:
        """Boolean membership test for an arbitrary key tensor."""
        n = self.keys.numel()
        if n == 0:
            return torch.zeros_like(query, dtype=torch.bool)
        pos = torch.searchsorted(self.keys, query).clamp(max=n - 1)
        return self.keys[pos] == query

    # ------------------------------------------------------------------ #
    # morphology
    # ------------------------------------------------------------------ #

    def dilate(self, connectivity: int = 8) -> "PixelSet":
        nk, _, _ = self._neighbor_table(connectivity)
        return self._like(torch.cat([self.keys, nk]))

    def erode(self, connectivity: int = 8, border_is_inside: bool = True) -> "PixelSet":
        nk, center, n_in_bounds = self._neighbor_table(connectivity)
        counts = torch.zeros(self.keys.numel(), device=self.keys.device, dtype=torch.long)
        counts.scatter_add_(0, center, self.contains(nk).long())
        required = (
            n_in_bounds
            if border_is_inside
            else torch.full_like(n_in_bounds, len(_OFFSETS[connectivity]))
        )
        return self._like(self.keys[counts == required], already_sorted=True)

    def opened(self, connectivity: int = 8) -> "PixelSet":
        """Erode then dilate: removes isolated pixels and thin spurs."""
        return self.erode(connectivity).dilate(connectivity).intersection(self)

    def closed(self, connectivity: int = 8) -> "PixelSet":
        """Dilate then erode: fills single-pixel notches and pinholes."""
        return self.dilate(connectivity).erode(connectivity).union(self)

    def majority(self, connectivity: int = 8, min_count: Optional[int] = None) -> "PixelSet":
        """
        Majority (a.k.a. mode) filter: a pixel is in the output iff at least
        ``min_count`` of the K+1 pixels in its neighborhood (including itself)
        are in the input. This is the classic one-pass binary smoother and is
        usually what you actually want for "the boundary looks jagged".
        """
        k = len(_OFFSETS[connectivity])
        if min_count is None:
            min_count = (k + 1) // 2 + 1  # 5 of 9 for 8-connectivity
        candidates = self.dilate(connectivity)
        nk, center, _ = candidates._neighbor_table(connectivity)
        counts = torch.zeros(
            candidates.keys.numel(), device=self.keys.device, dtype=torch.long
        )
        counts.scatter_add_(0, center, self.contains(nk).long())
        counts = counts + self.contains(candidates.keys).long()
        return self._like(candidates.keys[counts >= min_count], already_sorted=True)

    def smooth(self, connectivity: int = 8, passes: int = 1) -> "PixelSet":
        """Close -> open -> majority. A good default cleanup chain."""
        out = self
        for _ in range(passes):
            out = out.closed(connectivity).opened(connectivity).majority(connectivity)
        return out

    # ------------------------------------------------------------------ #
    # set algebra
    # ------------------------------------------------------------------ #

    def difference(self, other: "PixelSet") -> "PixelSet":
        return self._like(self.keys[~other.contains(self.keys)], already_sorted=True)

    def intersection(self, other: "PixelSet") -> "PixelSet":
        return self._like(self.keys[other.contains(self.keys)], already_sorted=True)

    def union(self, other: "PixelSet") -> "PixelSet":
        return self._like(torch.cat([self.keys, other.keys]))

    # ------------------------------------------------------------------ #
    # boundaries
    # ------------------------------------------------------------------ #

    def inner_boundary(self, connectivity: int = 8) -> "PixelSet":
        """
        Members that touch the outside: ``A \\ erode(A)``.

        Prefer this over ``outer_boundary`` for display -- it never leaves the
        footprint, so contours cannot spill past the FOV or overlap a
        neighboring cell's interior.
        """
        return self.difference(self.erode(connectivity))

    def outer_boundary(self, connectivity: int = 8) -> "PixelSet":
        """``dilate(A) \\ A`` -- what the original implementation computed."""
        return self.dilate(connectivity).difference(self)

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #

    def to_contour_list(self) -> list[np.ndarray]:
        """
        One (n_points, 2) int array of [y, x] coordinates per component.

        Replaces the original O(nnz) Python loop with a bincount + np.split,
        which returns zero-copy views.
        """
        y, x, comp = self.decode()
        counts = torch.bincount(comp, minlength=self.num_components)
        coords = torch.stack([y, x], dim=1).cpu().numpy()
        splits = np.cumsum(counts.cpu().numpy())[:-1]
        return np.split(coords, splits)


# ---------------------------------------------------------------------- #
# connected components
# ---------------------------------------------------------------------- #


def largest_component_only(
    support: PixelSet,
    values: torch.Tensor,
    connectivity: int = 8,
    max_iter: int = 64,
    check_every: int = 4,
) -> tuple[PixelSet, torch.Tensor]:
    """
    Keep, for each component, only the connected blob carrying the most mass.

    Label propagation by iterated min-scatter over the neighbor graph. Each
    iteration is one vectorized pass; the number of iterations needed is the
    graph diameter of the footprint (tens of steps for a typical soma), so this
    is meaningfully more expensive than a single morphological pass -- but it is
    still fully on-device with no Python loop over components.

    Use this when opening is not enough, i.e. when a footprint has a *large*
    detached blob rather than a single stray pixel.
    """
    n = support.keys.numel()
    if n == 0:
        return support, values

    nk, center, _ = support._neighbor_table(connectivity)
    pos = torch.searchsorted(support.keys, nk).clamp(max=n - 1)
    hit = support.keys[pos] == nk
    nbr_idx, ctr_idx = pos[hit], center[hit]

    label = torch.arange(n, device=support.keys.device)
    for it in range(max_iter):
        new_label = label.clone()
        new_label.scatter_reduce_(0, ctr_idx, label[nbr_idx], reduce="amin")
        converged = (it % check_every == check_every - 1) and torch.equal(new_label, label)
        label = new_label
        if converged:
            break

    uniq, inv = torch.unique(label, return_inverse=True)
    mass = torch.zeros(uniq.numel(), device=values.device, dtype=values.dtype)
    mass.scatter_add_(0, inv, values.abs())

    comp = support.decode()[2]
    label_comp = torch.zeros(uniq.numel(), device=comp.device, dtype=torch.long)
    label_comp.scatter_reduce_(0, inv, comp, reduce="amax", include_self=False)

    best = torch.zeros(support.num_components, device=values.device, dtype=values.dtype)
    best.scatter_reduce_(0, label_comp, mass, reduce="amax", include_self=False)

    keep = (mass >= best[label_comp])[inv]
    return (
        PixelSet(
            support.keys[keep],
            support.height,
            support.width,
            support.num_components,
            already_sorted=True,
        ),
        values[keep],
    )


# ---------------------------------------------------------------------- #
# drop-in replacement for ACArray.contours
# ---------------------------------------------------------------------- #


def compute_contours(
    a: torch.Tensor,
    height: int,
    width: int,
    rel_threshold: float = 0.10,
    connectivity: int = 8,
    smooth_passes: int = 1,
    drop_detached: bool = True,
    inner: bool = True,
) -> list[np.ndarray]:
    """
    Args:
        a: sparse (pixels, components) footprint matrix.
        rel_threshold: per-component fraction of peak below which pixels are
            discarded. 0 reproduces the old "all nonzero" behavior.
        smooth_passes: rounds of close -> open -> majority. 0 disables.
        drop_detached: run connected-component filtering (see caveat above).
        inner: return the inner boundary rather than the outer ring.

    Returns:
        list of (n_points, 2) [y, x] arrays, one per component.
    """
    support, values = PixelSet.from_sparse(a, height, width, rel_threshold=rel_threshold)

    if drop_detached:
        support, values = largest_component_only(support, values, connectivity)
    if smooth_passes:
        support = support.smooth(connectivity, passes=smooth_passes)

    boundary = (
        support.inner_boundary(connectivity)
        if inner
        else support.outer_boundary(connectivity)
    )
    return boundary.to_contour_list()
