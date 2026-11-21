"""A figure on the http canvas, served to the fake browser in ``asgi_client`` as uvicorn would serve it."""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("uvicorn")

sys.path.insert(0, str(Path(__file__).parent))
from asgi_client import Browser, Lifespan, frame_of_size, get, request, ws_reply  # noqa: E402

from masknmf.visualization.canvas_server import SESSION_KEY_HEADER, BrowserCanvas, CanvasServer, client_js, page_html  # noqa: E402

KEY = [(SESSION_KEY_HEADER.encode(), b"sesame")]


class FigureVis:
    """The least a CanvasServer serves: a figure, shown and closed."""

    def __init__(self, canvas):
        import fastplotlib as fpl

        self.figure = fpl.Figure(size=(900, 640), canvas=canvas)
        self.figure[0, 0].add_image(np.random.default_rng(0).random((64, 64), dtype=np.float32))

    def show(self):
        self.figure.show()

    def close(self):
        self.figure.close()


def test_the_page_escapes_its_title_and_the_client_follows_https():
    assert "<title>a &lt;b&gt; &amp; c</title>" in page_html("a <b> & c", "Opening ...")
    assert "location.protocol === 'https:' ? 'wss://'" in client_js()


async def stream(server):
    status, _ = await get(server.asgi, "/")
    assert status == 403
    status, body = await get(server.asgi, "/", KEY)
    assert status == 200 and "figure" in body.decode()
    assert await ws_reply(server.asgi, "/") == {"type": "websocket.close", "code": 1008}

    await Lifespan(server.asgi).start()
    browser = Browser(server.asgi, headers=KEY)
    await browser.connect()
    # a HiDPI browser gets frames at one pixel per css pixel
    await browser.event(type="resize", width=800, height=600, pwidth=1600, pheight=1200, ratio=2)
    await frame_of_size(browser, (800, 600))

    # once the view stops changing, one lossless frame follows, then nothing
    while True:
        msg, _ = await browser.next_frame(timeout=15)
        if msg["mimetype"] == "image/png":
            break
    with pytest.raises(TimeoutError):
        await browser.next_frame(timeout=1.0)
    await browser.disconnect()
    server.close()


def test_a_served_figure_streams_frames_only_to_the_key_and_goes_still():
    canvas = BrowserCanvas(size=(900, 640), max_fps=60.0)
    server = CanvasServer(FigureVis(canvas), title="figure", waiting="Opening ...", host="127.0.0.1", port=0, key="sesame")
    asyncio.run(stream(server))


def write_results(path, groups):
    import h5py

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for group in groups:
            f.create_group(group)


def test_the_picker_lists_one_row_per_results_file_by_folder_newest_run_first(tmp_path, monkeypatch):
    from masknmf.io import group_name_compression, group_name_demixing
    from masknmf.visualization import results_picker

    stages = [group_name_compression(), group_name_demixing()]
    old = tmp_path / "mouse" / "fov1" / "20260101T000000_two-photon-calcium"
    new = tmp_path / "mouse" / "fov1" / "20260202T000000_two-photon-calcium"
    write_results(old / "results.hdf5", stages)
    write_results(old / "results.20260101T010000.curated.hdf5", stages)
    write_results(new / "results.hdf5", stages[:1])
    write_results(tmp_path / "mouse" / "other.results.hdf5", [])
    for skipped in ("_archive/run", "store.zarr", "movie.n5", "node_modules/pkg", "remote"):
        write_results(tmp_path / skipped / "results.hdf5", stages)
    write_results(tmp_path / "unnamed_store" / "results.hdf5", stages)
    (tmp_path / "unnamed_store" / "zarr.json").write_text("{}")
    monkeypatch.setattr(results_picker, "fuse_mounts", lambda: {str(tmp_path / "remote")})

    files = results_picker.find_results(tmp_path)
    assert [(f.folder, f.run, f.name) for f in files] == [
        ("mouse/fov1", new.name, "results.hdf5"),
        ("mouse/fov1", old.name, "results.20260101T010000.curated.hdf5"),
    ]
    assert files[0].stages == ["compression"] and files[1].stages == ["compression", "demixing"]
    assert files[1].num_curated == 1 and files[0].pipeline == "two-photon-calcium"

    page = results_picker.picker_html(str(tmp_path), files, path_open=files[1].path, error="a <b>")
    assert page.count("class='file") == 2 and page.count("class='file open'") == 1
    assert "a &lt;b&gt;" in page and "curated (1)" in page
    # mouse holds only fov1, so the tree shows them as one folder
    assert "<summary>mouse/fov1<" in page and "id='recent'" not in page


def test_the_picker_tree_sorts_folders_by_number_and_lists_the_newest_files_on_top():
    from masknmf.visualization.results_picker import ResultsFile, folder_tree, picker_html

    files = [
        ResultsFile(f"/data/sweep/epochs_{n}/results.hdf5", f"sweep/epochs_{n}", f"epochs_{n}", ["compression"], 0, 1, float(n))
        for n in (100, 10, 30, 5, 50, 1000, 20, 40, 60)
    ]
    tree = folder_tree(files)
    assert list(tree["folders"]) == ["sweep"]
    page = picker_html("/data", files)
    order = [page.index(f"<summary>epochs_{n}<") for n in (5, 10, 20, 30, 40, 50, 60, 100, 1000)]
    assert order == sorted(order)
    recent = page[page.index("id='recent'") : page.index("id='tree'")]
    assert recent.count("class='file") == 8 and "epochs_1000/results.hdf5" in recent and "epochs_5/" not in recent


class StubViewer:
    """Stands in for the demixing viewer and keeps what it was opened with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def show(self):
        pass

    def close(self):
        pass


def test_the_picker_lists_and_opens_miniali_results_beside_masknmf_ones(tmp_path):
    import h5py
    from test_adapter_sources import write_miniali_file, write_miniali_split

    from masknmf.io import group_name_demixing
    from masknmf.visualization import results_picker

    write_results(tmp_path / "rec" / "20260101T000000_one-photon-culture" / "results.hdf5", [group_name_demixing()])
    (tmp_path / "Analysis" / "rec").mkdir(parents=True)
    write_miniali_file(tmp_path / "Analysis" / "rec" / "ALI_Result_new.mat")
    write_miniali_split(tmp_path / "Analysis" / "rec2")
    with h5py.File(tmp_path / "rec" / "output_data.mat", "w") as f:
        f["x"] = 1

    files = results_picker.find_results(tmp_path)
    assert [(f.folder, f.pipeline, f.name) for f in files] == [
        ("Analysis/rec", "miniali", "ALI_Result_new.mat"),
        ("Analysis/rec2", "miniali", "signals.h5"),
        ("rec", "one-photon-culture", "results.hdf5"),
    ]
    assert files[0].stages == ["miniali: 2 cells, 40 frames, no Background, recomputed"]

    server = results_picker.PickerServer(tmp_path, open_viewer=StubViewer, args=None, host="127.0.0.1", port=0)
    server.open(files[0].path)
    assert server.vis.viewer.kwargs["names_present"] == files[0].stages
    server.close()


async def picker_requests(server):
    status, headers, _ = await request(server.asgi, "GET", "/")
    assert status == 303 and headers[b"location"] == b"files"
    status, body = await get(server.asgi, "/files")
    assert status == 200 and "results.hdf5" in body.decode()
    status, body = await get(server.asgi, "/open?path=/etc/passwd")
    assert status == 200 and "is not a results file below" in body.decode()


def test_the_picker_sends_a_browser_to_the_list_and_refuses_a_path_outside_it(tmp_path):
    from masknmf.io import group_name_demixing
    from masknmf.visualization.results_picker import PickerServer

    write_results(tmp_path / "mouse" / "results.hdf5", [group_name_demixing()])
    server = PickerServer(tmp_path, open_viewer=None, args=None, host="127.0.0.1", port=0)

    asyncio.run(picker_requests(server))
    server.close()
