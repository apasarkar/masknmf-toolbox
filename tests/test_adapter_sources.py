"""suite2p plane folders, caiman files and miniALI .mat files open like results files, through their adapters."""

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

import masknmf
from masknmf import cli
from masknmf.demixing._adapters import CaimanResults, MiniALIResults, Suite2pResults, adapter_for, suite2p_frame_rate
from masknmf.demixing._base_results import BaseResults
from masknmf.io import OpenedResults, load_movie

FRAMES, HEIGHT, WIDTH = 40, 16, 20


def write_suite2p_plane(folder, with_bin: bool = True, fs: float = 7.5) -> None:
    """A suite2p plane folder with two ROIs, as suite2p <= 0.14 writes it: ops.npy flat, F and Fneu (rois, frames)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    stat = []
    for y0, x0 in ((4, 5), (10, 13)):
        yy, xx = np.mgrid[y0 - 2:y0 + 3, x0 - 2:x0 + 3]
        stat.append({"ypix": yy.ravel(), "xpix": xx.ravel(), "lam": np.ones(yy.size, np.float32) / yy.size,
                     "overlap": np.zeros(yy.size, bool), "npix": yy.size})
    np.save(folder / "stat.npy", np.array(stat, dtype=object), allow_pickle=True)
    np.save(folder / "F.npy", rng.random((2, FRAMES)).astype(np.float32) * 100)
    np.save(folder / "Fneu.npy", rng.random((2, FRAMES)).astype(np.float32) * 10)
    ops = {"Ly": HEIGHT, "Lx": WIDTH, "nframes": FRAMES, "fs": fs, "neucoeff": 0.7,
           "yoff": rng.integers(-2, 3, FRAMES).astype(np.float32), "xoff": rng.integers(-2, 3, FRAMES).astype(np.float32)}
    np.save(folder / "ops.npy", ops, allow_pickle=True)
    if with_bin:
        (rng.random((FRAMES, HEIGHT, WIDTH)) * 1000).astype(np.int16).tofile(folder / "data.bin")


def write_caiman_file(path, with_ring_model: bool = False) -> None:
    """A results file as caiman's CNMF.save writes it: estimates/A sparse in Fortran pixel order, C, dims."""
    rng = np.random.default_rng(1)
    pixels = HEIGHT * WIDTH
    A = sp.lil_matrix((pixels, 3), dtype=np.float32)
    for k, (y0, x0) in enumerate(((3, 4), (8, 10), (12, 16))):
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                A[(y0 + dy) + HEIGHT * (x0 + dx), k] = 1.0
    A = A.tocsc()
    with h5py.File(path, "w") as f:
        est = f.create_group("estimates")
        group = est.create_group("A")
        group["data"], group["indices"], group["indptr"], group["shape"] = A.data, A.indices, A.indptr, np.array(A.shape)
        est["C"] = rng.random((3, FRAMES)).astype(np.float32)
        est["dims"] = np.array([HEIGHT, WIDTH])
        est["idx_components"] = np.array([0, 2])
        if with_ring_model:
            W = sp.eye(pixels, dtype=np.float32, format="csc")
            group = est.create_group("W")
            group["data"], group["indices"], group["indptr"], group["shape"] = W.data, W.indices, W.indptr, np.array(W.shape)
            est["b0"] = np.zeros(pixels, np.float32)


def write_miniali_file(path, num_cells: int = 2) -> None:
    """Writes ALI_Result_new.mat as mini_ali_run saves it with -v7.3: arrays transposed, footprint a cell of {idx, values} cells."""
    rng = np.random.default_rng(2)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        f["reference_image"] = (rng.random((HEIGHT, WIDTH)) * 100).T
        f["dfdx"] = rng.standard_normal((HEIGHT, WIDTH)).T
        f["dfdy"] = rng.standard_normal((HEIGHT, WIDTH)).T
        f["shift"] = (rng.standard_normal((2, FRAMES)) * 0.1).T
        f["cell_traces"] = rng.standard_normal((num_cells, FRAMES)).T
        cells = []
        for k, (r0, c0) in enumerate(((4, 5), (10, 13))[:num_cells]):
            rows, cols = np.meshgrid(np.arange(r0 - 2, r0 + 3), np.arange(c0 - 2, c0 + 3), indexing="ij")
            idx = (cols * HEIGHT + rows).ravel(order="F") + 1  # 1-based, column-major
            d_idx = refs.create_dataset(f"idx{k}", data=idx.astype(np.float64)[:, None])
            d_val = refs.create_dataset(f"val{k}", data=rng.random(idx.size)[None, :])
            pair = refs.create_dataset(f"pair{k}", data=np.array([d_idx.ref, d_val.ref], dtype=h5py.ref_dtype)[:, None])
            cells.append(pair.ref)
        f.create_dataset("footprint", data=np.array(cells, dtype=h5py.ref_dtype)[None, :])


def write_miniali_recording(folder, frames: int = FRAMES + 4) -> None:
    """Writes a recording folder as the microscope does: frames1.bin, uint8 and headerless, and output_data.mat."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    (rng.random((frames, HEIGHT, WIDTH)) * 255).astype(np.uint8).tofile(folder / "frames1.bin")
    with h5py.File(folder / "output_data.mat", "w") as f:
        refs = f.create_group("#refs#")
        entries = []
        for name, kind in (("daq", "DAQ"), ("cam", "Camera")):
            entry = refs.create_group(name)
            entry["deviceType"] = np.array([ord(c) for c in kind], np.uint16)[:, None]
            if kind == "Camera":
                entry["ROI"] = np.array([1.0, WIDTH, 1.0, HEIGHT])[:, None]
                entry["frames_requested"] = np.array([[float(frames)]])
            entries.append(entry.ref)
        f.create_dataset("Device_Data", data=np.array(entries, dtype=h5py.ref_dtype)[:, None])


@pytest.fixture
def plane(tmp_path):
    folder = tmp_path / "suite2p" / "plane0"
    write_suite2p_plane(folder)
    return folder


@pytest.fixture
def caiman_path(tmp_path):
    path = tmp_path / "cnmf.hdf5"
    write_caiman_file(path)
    return path


@pytest.fixture
def miniali(tmp_path):
    folder = tmp_path / "miniALI_matlab"
    folder.mkdir()
    write_miniali_file(folder / "ALI_Result_new.mat")
    write_miniali_recording(tmp_path / "recording")
    return folder / "ALI_Result_new.mat"


def test_suite2p_planes_are_found_from_a_plane_its_stat_file_or_the_folder_of_planes(tmp_path):
    for name in ("plane0", "plane1", "plane10"):
        write_suite2p_plane(tmp_path / "suite2p" / name)
    (tmp_path / "suite2p" / "combined").mkdir()
    planes = [tmp_path / "suite2p" / name for name in ("plane0", "plane1", "plane10")]
    assert Suite2pResults.expand(tmp_path / "suite2p") == planes
    assert Suite2pResults.expand(planes[1]) == [planes[1]]
    assert Suite2pResults.expand(planes[1] / "stat.npy") == [planes[1]]
    assert Suite2pResults.expand(tmp_path) == []
    assert Suite2pResults.expand(tmp_path / "missing") == []
    assert adapter_for(planes[2]) is Suite2pResults and adapter_for(planes[2] / "stat.npy") is Suite2pResults
    assert adapter_for(tmp_path / "suite2p") is Suite2pResults and adapter_for(tmp_path) is None


def test_a_caiman_file_is_told_from_a_results_file_and_anything_else(tmp_path, caiman_path):
    assert CaimanResults.matches(caiman_path) and adapter_for(caiman_path) is CaimanResults
    with h5py.File(tmp_path / "other.hdf5", "w") as f:
        f["x"] = 1
    assert not CaimanResults.matches(tmp_path / "other.hdf5") and adapter_for(tmp_path / "other.hdf5") is None
    assert adapter_for(tmp_path / "missing.hdf5") is None


def test_a_suite2p_plane_opens_with_its_data_bin_as_the_residuals_movie_and_its_shifts(plane):
    opened = OpenedResults.open(plane)
    assert isinstance(opened.results, BaseResults) and not isinstance(opened.results, masknmf.DemixingResults)
    assert opened.results.shape == (FRAMES, HEIGHT, WIDTH)
    assert opened.results.spatial_demixed.shape[1] == 2
    assert opened.shifts.shape == (FRAMES, 2) and opened.shifts.dtype == np.float32
    assert opened.results.residual_array is not None and opened.results.residual_array.shape == (FRAMES, HEIGHT, WIDTH)
    assert opened.raw is None and opened.registered is None and opened.skipped == []
    assert opened.path == plane and opened.results.frame_rate == 7.5
    # stat.npy names the same plane
    assert OpenedResults.open(plane / "stat.npy").results.shape == (FRAMES, HEIGHT, WIDTH)


def test_a_suite2p_plane_without_data_bin_opens_without_a_residual(tmp_path):
    folder = tmp_path / "plane0"
    write_suite2p_plane(folder, with_bin=False)
    opened = OpenedResults.open(folder)
    assert opened.results.residual_array is None and opened.skipped == []
    with pytest.raises(ValueError, match="take no prefix"):
        OpenedResults.open(folder, prefix="global")


def test_a_caiman_file_opens_with_its_accepted_components_and_a_matching_raw_movie(tmp_path, caiman_path):
    opened = OpenedResults.open(caiman_path)
    assert opened.results.shape == (FRAMES, HEIGHT, WIDTH) and opened.results.spatial_demixed.shape[1] == 2
    assert opened.raw is None and opened.shifts is None and opened.skipped == []
    movie = np.zeros((FRAMES, HEIGHT, WIDTH), np.float32)
    # a file without a ring model keeps the movie as the raw panel and says why there is no residual
    opened = OpenedResults.open(caiman_path, raw=movie)
    assert opened.raw is movie and opened.results.residual_array is None
    assert len(opened.skipped) == 1 and "ring-model" in opened.skipped[0]
    opened = OpenedResults.open(caiman_path, raw=np.zeros((FRAMES + 1, HEIGHT, WIDTH), np.float32))
    assert opened.raw is None and "does not match" in opened.skipped[0]
    path_ring = tmp_path / "cnmfe.hdf5"
    write_caiman_file(path_ring, with_ring_model=True)
    opened = OpenedResults.open(path_ring, raw=movie)
    assert opened.results.residual_array is not None and opened.skipped == []


def test_view_lists_suite2p_planes_and_caiman_files_beside_results_files(tmp_path, caiman_path, capsys):
    for name in ("plane0", "plane1"):
        write_suite2p_plane(tmp_path / "suite2p" / name, with_bin=name == "plane0")
    cli.main(["view", str(tmp_path / "suite2p"), str(caiman_path), str(tmp_path / "suite2p" / "plane1" / "stat.npy"), "--list"])
    out = capsys.readouterr().out
    assert out.count("plane0") == 1 and out.count("plane1") == 1
    assert "suite2p: 2 ROIs, data.bin" in out and "suite2p: 2 ROIs, no data.bin" in out
    assert "caiman: 3 components, 2 accepted" in out


def test_view_refuses_to_classify_or_prefix_adapter_results(tmp_path, plane, caiman_path, capsys):
    with pytest.raises(SystemExit):
        cli.main(["view", str(plane), "--classify"])
    assert "holds suite2p results" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        cli.main(["view", str(caiman_path), "--prefix", "global"])
    assert "take no prefix" in capsys.readouterr().err


def test_the_frame_rate_comes_from_suite2p_ops_in_any_version():
    assert suite2p_frame_rate({"fs": 30}) == 30.0
    assert suite2p_frame_rate({"io": {"fs": 15.5}, "Ly": 4}) == 15.5
    assert suite2p_frame_rate({"Ly": 4}) is None


def test_the_launcher_opens_a_suite2p_plane(tmp_path, plane):
    from test_launcher_defaults import render_frames

    from masknmf import launcher

    window = launcher.Launcher()
    window.page = "open"
    window.paths_open["source"] = str(plane / "stat.npy")
    window.inspect()
    assert window.found["kind"] == "suite2p" and window.found["files"] == [str(plane)]
    assert window.found["detail"] == "plane0: suite2p: 2 ROIs, data.bin"
    assert window.available(key="demixing") and not window.available(key="classification")
    assert window.problems() == []
    args = cli.build_parser(spec=None).parse_args(window.build_argv())
    assert args.handler is cli.command_view and args.results == [str(plane / "stat.npy")] and args.raw is None

    # the suite2p folder of planes opens one viewer per plane
    write_suite2p_plane(tmp_path / "suite2p" / "plane1", with_bin=False)
    window.paths_open["source"] = str(tmp_path / "suite2p")
    window.inspect()
    assert window.found["kind"] == "suite2p" and window.found["detail"] == "2 to open" and len(window.found["files"]) == 2
    assert window.available(key="demixing") and window.build_argv() == ["view", str(tmp_path / "suite2p")]
    render_frames(window=window)


def test_a_miniali_file_is_found_from_the_file_or_its_folder(tmp_path, miniali):
    assert MiniALIResults.expand(miniali) == [miniali] and MiniALIResults.expand(miniali.parent) == [miniali]
    assert adapter_for(miniali) is MiniALIResults and adapter_for(miniali.parent) is MiniALIResults
    with h5py.File(tmp_path / "other.mat", "w") as f:
        f["x"] = 1
    assert adapter_for(tmp_path / "other.mat") is None and adapter_for(tmp_path / "recording") is None
    assert MiniALIResults.describe(miniali) == "miniali: 2 cells, 40 frames, no Background, recomputed"


def test_a_miniali_recording_folder_loads_as_a_movie(tmp_path, miniali):
    movie = load_movie(tmp_path / "recording")
    assert movie.shape == (FRAMES + 4, HEIGHT, WIDTH) and movie.dtype == np.uint8


def test_a_miniali_file_opens_with_its_recording_folder_and_drops_the_frames_past_the_shifts(tmp_path, miniali):
    opened = OpenedResults.open(miniali, raw=tmp_path / "recording")
    assert isinstance(opened.results, MiniALIResults) and opened.results.shape == (FRAMES, HEIGHT, WIDTH)
    assert opened.results.num_frames_raw == FRAMES + 4 and opened.results.num_frames_analyzed == FRAMES
    assert opened.raw is opened.results.raw_array and opened.raw_source == tmp_path / "recording"
    assert opened.shifts.shape == (FRAMES, 2) and isinstance(opened.registered, masknmf.GradientRegistrationArray)
    assert opened.results.residual_array.shape == (FRAMES, HEIGHT, WIDTH) and opened.skipped == []
    with pytest.raises(ValueError, match="need the raw movie"):
        OpenedResults.open(miniali)
    with pytest.raises(ValueError, match="the results 40"):
        OpenedResults.open(miniali, raw=np.zeros((FRAMES - 1, HEIGHT, WIDTH), np.uint8))


def test_view_lists_a_miniali_file_and_the_launcher_opens_it(tmp_path, miniali, capsys):
    cli.main(["view", str(miniali), "--list"])
    assert "miniali: 2 cells, 40 frames" in capsys.readouterr().out

    from masknmf import launcher

    window = launcher.Launcher()
    window.page = "open"
    window.paths_open["source"] = str(miniali)
    window.inspect()
    assert window.found["kind"] == "miniali" and window.found["files"] == [str(miniali)]
    assert window.available(key="demixing") and not window.available(key="classification")
    assert window.problems() == ["miniali results need the raw movie: its recording folder (the .bin movie and output_data.mat)"]
    window.paths_open["raw"] = str(tmp_path / "recording")
    assert window.problems() == []
    args = cli.build_parser(spec=None).parse_args(window.build_argv())
    assert args.handler is cli.command_view and args.results == [str(miniali)] and args.raw == str(tmp_path / "recording")
