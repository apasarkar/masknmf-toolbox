"""
Opens one recording's masknmf results and its suite2p results in two viewer windows and keeps them in step:
the frame slider, the zoom and pan of the video panels, and the selected cell.

Each window is an ordinary ``SingleSessionDemixingVis``. Linking them is three hooks on top:

- the frame index: each viewer's ``ReferenceIndices`` reports a change and the other is set to the same time;
- the camera: each figure's pan-zoom controller also drives the other figure's cameras;
- the selection: a click on a cell in one viewer selects the cell in the other whose footprint overlaps it most
  (largest intersection over union); a cell nothing overlaps clears the other viewer's selection.

Run it on the demo movie:

    python examples/synced_viewers.py \\
        --masknmf ~/data/20261029T104702_two-photon-calcium/results.hdf5 \\
        --suite2p ~/data/demoMovie_demo/zplane01_tp00001-02000 \\
        --raw ~/data/demoMovie.tif --fs 10
"""
import argparse

import numpy as np
import scipy.sparse
import torch

import fastplotlib as fpl

import masknmf
from masknmf.cli import timings
from masknmf.io import OpenedResults


def footprint_matches(spatial_a: torch.Tensor, spatial_b: torch.Tensor) -> np.ndarray:
    """
    The cell of ``b`` each cell of ``a`` overlaps most, by intersection over union of the footprint supports.

    Parameters
    ----------
    spatial_a, spatial_b : torch.Tensor
        Sparse footprints, (pixels, cells), over the same field of view.

    Returns
    -------
    np.ndarray
        One index into ``b`` per cell of ``a``; -1 where no footprint of ``b`` touches it.
    """
    support_a = _support(spatial_a)
    support_b = _support(spatial_b)
    intersection = (support_a.T @ support_b).toarray()  ## (cells_a, cells_b)
    area_a = np.asarray(support_a.sum(axis=0)).ravel()
    area_b = np.asarray(support_b.sum(axis=0)).ravel()
    iou = intersection / np.maximum(area_a[:, None] + area_b[None, :] - intersection, 1)
    best = iou.argmax(axis=1)
    best[iou.max(axis=1) == 0] = -1
    return best


def _support(spatial: torch.Tensor) -> scipy.sparse.csc_matrix:
    """The footprints' supports as a 0/1 scipy matrix, (pixels, cells)."""
    coo = spatial.detach().coalesce().cpu()
    rows, cols = coo.indices().numpy()
    ones = np.ones(len(rows), np.float32)
    return scipy.sparse.csc_matrix((ones, (rows, cols)), shape=tuple(coo.shape))


class LinkedVis(masknmf.SingleSessionDemixingVis):
    """A demixing viewer that passes its cell selection on to the viewers it is linked with."""

    def __init__(self, *args, **kwargs):
        self._partners: dict[LinkedVis, np.ndarray] = {}  ## partner -> matched cell in it per own cell
        self._relaying = False
        super().__init__(*args, **kwargs)

    def link(self, other: "LinkedVis"):
        """Both ways: the frame index, the cameras and the selection of ``self`` and ``other`` move together."""
        own = self.demixing_results.spatial_demixed
        theirs = other.demixing_results.spatial_demixed
        self._partners[other] = footprint_matches(own, theirs)
        other._partners[self] = footprint_matches(theirs, own)
        _link_indices(self, other)
        _link_cameras(self, other)

    def _select_component(self, component):
        super()._select_component(component)
        if self._relaying or not isinstance(component, (int, np.integer)):
            return
        for partner, matches in self._partners.items():
            partner.select_relayed(int(matches[int(component)]))

    def deselect(self):
        super().deselect()
        if self._relaying:
            return
        for partner in self._partners:
            partner.relay(partner.deselect)

    def select_relayed(self, component: int):
        """Select ``component``, or clear the selection when it is -1, without passing it on again."""
        if component < 0:
            self.relay(self.deselect)
        else:
            self.relay(self._select_component, component)

    def relay(self, action, *args):
        self._relaying = True
        try:
            action(*args)
        finally:
            self._relaying = False


def _link_indices(a: LinkedVis, b: LinkedVis):
    a.reference_index.add_event_handler(lambda indices: _follow_time(b, indices))
    b.reference_index.add_event_handler(lambda indices: _follow_time(a, indices))


def _follow_time(viewer: LinkedVis, indices: dict):
    """Sets ``viewer`` to the time in ``indices`` unless it is already there, which stops the two from ping-ponging."""
    if viewer.reference_index["time"] != indices["time"]:
        viewer.reference_index.set({"time": indices["time"]})


def _link_cameras(a: LinkedVis, b: LinkedVis):
    """Every video panel's camera in one figure joins the other figure's pan-zoom controller."""
    figure_a, figure_b = a.fov_widget.figure, b.fov_widget.figure
    controller_a = next(iter(figure_a)).controller
    controller_b = next(iter(figure_b)).controller
    for subplot in figure_b:
        controller_a.add_camera(subplot.camera)
    for subplot in figure_a:
        controller_b.add_camera(subplot.camera)


def open_viewer(path: str, raw, frame_rate: float | None, device: str, title: str) -> LinkedVis:
    opened = OpenedResults.open(path, raw=raw, device=device)
    for note in opened.skipped:
        print(note)
    viewer = LinkedVis(
        demixing_results=opened.results,
        frame_timings=timings(opened.results.shape[0], frame_rate),
        device=device,
        results_path=path,
        raw=opened.raw,
        registered=opened.registered,
    )
    viewer.show()
    viewer.fov_widget.figure.canvas.set_title(title)
    return viewer


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="masknmf and suite2p results of one recording, side by side and in step")
    parser.add_argument("--masknmf", required=True, help="masknmf results.hdf5")
    parser.add_argument("--suite2p", required=True, help="suite2p plane folder (stat.npy, F.npy, ops.npy, data.bin)")
    parser.add_argument("--raw", required=True, help="the movie both were run on")
    parser.add_argument("--fs", type=float, default=None, help="acquisition rate in Hz; frames when not given")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args(argv)

    device = str(masknmf.utils.torch_select_device()) if args.device == "auto" else args.device
    raw = masknmf.io.load_movie(args.raw)
    viewer_masknmf = open_viewer(args.masknmf, raw, args.fs, device, "masknmf")
    viewer_suite2p = open_viewer(args.suite2p, raw, args.fs, device, "suite2p")
    viewer_masknmf.link(viewer_suite2p)
    fpl.loop.run()


if __name__ == "__main__":
    main()
