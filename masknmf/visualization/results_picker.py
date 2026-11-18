"""Serve a folder of results files to a browser: a page listing them, and the demixing viewer on the one picked.

``masknmf serve FOLDER`` runs this. ``files`` lists every results file below the folder as a tree of folders, the
runs in each newest first, under the most recently changed files; a filter box narrows it and a click or Enter
opens a file in the one viewer this process holds, at ``/``. The first pick builds the viewer, later picks load
into it in place.
"""

from __future__ import annotations

import argparse
import fnmatch
import html
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Callable
from urllib.parse import parse_qs, quote

import psutil
from rendercanvas.http import resources

from masknmf.demixing.curation import CURATED_SUFFIX, latest_results, results_stem
from masknmf.io import stage_groups
from masknmf.visualization.canvas_server import BrowserCanvas, CanvasServer, page_html

__all__ = ["PickerServer", "ResultsFile", "find_results", "picker_html"]

PATTERNS_RESULTS = ("*results*.hdf5", "*results*.h5")
# a run folder is named <timestamp>_<pipeline>
PATTERN_RUN = re.compile(r"^(\d{8}T\d{6})_(.+)$")
# folders of chunked arrays, code or caches: thousands of entries and never a results file
SUFFIXES_STORE = (".zarr", ".n5")
NAMES_SKIPPED = {"node_modules", "__pycache__", "site-packages"}
MARKERS_ZARR = {".zgroup", ".zarray", "zarr.json"}
NUM_RECENT = 8
# a folder holding this many files or fewer starts with its subfolders open
NUM_FILES_OPEN_ALL = 30

_PICKER = Template("""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>$title</title>
    <style>
        body { margin: 0; padding: 0 24px 24px; background: #1e1f22; color: #ddd; font: 14px system-ui, sans-serif; }
        a { color: #7ab4ff; text-decoration: none; }
        [hidden] { display: none !important; }
        header { position: sticky; top: 0; z-index: 1; display: flex; gap: 16px; align-items: baseline;
                 flex-wrap: wrap; padding: 14px 0 10px; background: #1e1f22; }
        header .root { font: 13px monospace; color: #999; }
        header .muted, summary .muted { color: #888; font-weight: 400; }
        #filter { flex: 1; min-width: 240px; padding: 6px 10px; background: #2b2d31; color: #ddd;
                  border: 1px solid #444; border-radius: 4px; font: inherit; }
        header button { padding: 2px 8px; background: none; color: #aaa; border: 1px solid #444; border-radius: 4px;
                        font: 12px system-ui, sans-serif; cursor: pointer; }
        .error { padding: 8px 12px; margin-bottom: 12px; background: #4a2326; border-radius: 4px; }
        h2 { margin: 8px 0 4px; font-size: 13px; font-weight: 600; color: #999; text-transform: uppercase; }
        .scroll { overflow-x: auto; }
        details > .children { margin-left: 18px; }
        summary { padding: 4px 0; font-weight: 600; cursor: pointer; }
        summary .muted { margin-left: 10px; font-size: 12px; }
        table { border-collapse: collapse; width: 100%; min-width: 860px; table-layout: fixed; margin-bottom: 4px; }
        td { padding: 4px 12px 4px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
             border-top: 1px solid #2f3136; }
        td.when { width: 130px; } td.pipeline { width: 190px; } td.stages { width: 260px; }
        td.curated { width: 90px; } td.num { width: 70px; }
        tr.file { cursor: pointer; }
        tr.file:hover, tr.file.active { background: #2b2d31; }
        tr.file.active td:first-child { box-shadow: inset 2px 0 #aaa; padding-left: 8px; }
        tr.open td:first-child { box-shadow: inset 3px 0 #7ab4ff; padding-left: 8px; }
        td.folder, td.stages, td.curated, td.num { color: #999; }
        td.num { text-align: right; font-variant-numeric: tabular-nums; }
        body.opening { cursor: progress; opacity: 0.6; }
    </style>
</head>
<body>
    <header>
        <strong>masknmf results</strong>
        <span class='root'>$root</span>
        <span class='muted'>$count</span>
        <input id='filter' placeholder='filter by folder, run, pipeline or stage; arrows and Enter open' autofocus>
        <button id='expand'>expand all</button>
        <button id='collapse'>collapse all</button>
        $viewer
    </header>
    $error
    $recent
    $tree
    <script>
        const filter = document.getElementById('filter')
        const recent = document.getElementById('recent')
        const folders = [...document.querySelectorAll('#tree details')]
        let active = null

        function visibleRows () {
            // NOTE: offsetParent stays set inside a closed details in Chrome, so ancestors are checked instead
            return [...document.querySelectorAll('tr.file')].filter((row) => row.closest('[hidden], details:not([open])') === null)
        }
        function setActive (row) {
            if (active !== null) { active.classList.remove('active') }
            active = row === undefined ? null : row
            if (active !== null) {
                active.classList.add('active')
                active.scrollIntoView({ block: 'nearest' })
            }
        }
        function openRow (row) {
            document.body.classList.add('opening')
            window.location = row.dataset.href
        }

        filter.addEventListener('input', () => {
            const words = filter.value.toLowerCase().split(/\\s+/).filter(Boolean)
            if (recent !== null) { recent.hidden = words.length > 0 }
            for (const row of document.querySelectorAll('#tree tr.file')) {
                row.hidden = !words.every((word) => row.dataset.text.includes(word))
            }
            for (const folder of folders) {
                const hit = folder.querySelector('tr.file:not([hidden])') !== null
                folder.hidden = !hit
                folder.open = words.length > 0 ? hit : folder.dataset.open === '1'
            }
            setActive(visibleRows()[0])
        })
        filter.addEventListener('keydown', (event) => {
            const rows = visibleRows()
            const index = rows.indexOf(active)
            if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
                event.preventDefault()
                const step = event.key === 'ArrowDown' ? 1 : -1
                setActive(rows[Math.min(rows.length - 1, Math.max(0, index + step))])
            } else if (event.key === 'Enter' && rows.length > 0) {
                openRow(index >= 0 ? active : rows[0])
            } else if (event.key === 'Escape') {
                filter.value = ''
                filter.dispatchEvent(new Event('input'))
            }
        })
        document.addEventListener('keydown', (event) => {
            if (event.key === '/' && document.activeElement !== filter) {
                event.preventDefault()
                filter.focus()
            }
        })
        document.getElementById('expand').addEventListener('click', () => folders.forEach((folder) => { folder.open = true }))
        document.getElementById('collapse').addEventListener('click', () => folders.forEach((folder) => { folder.open = false }))
        for (const row of document.querySelectorAll('tr.file')) {
            row.addEventListener('click', () => openRow(row))
        }
    </script>
</body>
</html>
""")


@dataclass
class ResultsFile:
    """One results file below the served folder, with what its row on the page shows.

    ``folder`` is the folder it is listed under, relative to the served one: the folder holding its run folder,
    or the folder holding the file when that is not a run folder.
    """

    path: str
    folder: str
    run: str
    stages: list[str]
    num_curated: int
    num_bytes: int
    time_modified: float

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def is_curated(self) -> bool:
        return self.path.endswith(CURATED_SUFFIX)

    @property
    def time_run(self) -> float:
        """When the run started, from its folder's name; the file's modification time for any other folder."""
        match = PATTERN_RUN.match(self.run)
        if match is None:
            return self.time_modified
        return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").timestamp()

    @property
    def pipeline(self) -> str:
        match = PATTERN_RUN.match(self.run)
        return "" if match is None else match.group(2)


def find_results(folder: str | os.PathLike) -> list[ResultsFile]:
    """
    Find the masknmf results files below a folder, one per results file: its newest curated file when it has one.

    Not entered: hidden folders, folders starting with ``_`` (archives), zarr and n5 stores (by name or by a
    zarr's own marker files), ``node_modules``, ``__pycache__``, ``site-packages`` and FUSE mounts such as sshfs,
    where a sleeping remote blocks the listing. Symlinked folders are not followed.

    Parameters
    ----------
    folder : str | os.PathLike
        The folder to search.

    Returns
    -------
    list[ResultsFile]
        Sorted by folder, then newest run first.
    """
    folder = Path(folder).expanduser().resolve()
    mounts_fuse = fuse_mounts()
    paths = []
    for dirpath, dirnames, filenames in os.walk(folder):
        if MARKERS_ZARR.intersection(filenames):
            dirnames[:] = []
            continue
        dirnames[:] = [
            name
            for name in dirnames
            if not name.startswith((".", "_"))
            and not name.endswith(SUFFIXES_STORE)
            and name not in NAMES_SKIPPED
            and os.path.join(dirpath, name) not in mounts_fuse
        ]
        paths += [
            os.path.join(dirpath, name)
            for name in filenames
            if any(fnmatch.fnmatch(name, pattern) for pattern in PATTERNS_RESULTS)
        ]
    num_curated = {}
    for path in paths:
        key = (os.path.dirname(path), results_stem(path))
        num_curated[key] = num_curated.get(key, 0) + path.endswith(CURATED_SUFFIX)

    files = []
    for path in latest_results(paths):
        stages = stage_groups(path)
        if len(stages) == 0:
            continue
        folder_run = Path(path).parent
        is_run = PATTERN_RUN.match(folder_run.name) is not None
        folder_listed = folder_run.parent if is_run and folder_run != folder else folder_run
        stat = os.stat(path)
        files.append(
            ResultsFile(
                path=path,
                folder=folder_listed.relative_to(folder).as_posix(),
                run=folder_run.name,
                stages=[stage_label(stage) for stage in stages],
                num_curated=num_curated[(os.path.dirname(path), results_stem(path))],
                num_bytes=stat.st_size,
                time_modified=stat.st_mtime,
            )
        )
    return sorted(files, key=lambda file: (file.folder, -file.time_run, file.name))


def fuse_mounts() -> set[str]:
    """Returns the mount points of FUSE filesystems (sshfs, rclone and the like)."""
    return {partition.mountpoint for partition in psutil.disk_partitions(all=True) if "fuse" in partition.fstype}


def stage_label(stage: str) -> str:
    """A stage group's name as the page shows it: ``RigidRegistrationArray`` is registration, ``global/DemixingResults`` global/demixing."""
    prefix, _, name = stage.rpartition("/")
    label = next(word for word in ("registration", "compression", "demixing") if word in name.lower())
    return f"{prefix}/{label}" if prefix else label


def format_bytes(num_bytes: int) -> str:
    """A file size in the largest unit that keeps it at or above 1."""
    size = float(num_bytes)
    for unit in ("B", "kB", "MB", "GB"):
        if size < 1000:
            return f"{size:.0f} {unit}"
        size /= 1000
    return f"{size:.1f} TB"


def folder_tree(files: list[ResultsFile]) -> dict:
    """
    Nest files by their folder, merging a folder whose only content is one folder into it (``a/b``).

    Parameters
    ----------
    files : list[ResultsFile]
        What ``find_results`` returned.

    Returns
    -------
    dict
        ``{"folders": {name: subtree}, "files": [...]}``; the served folder's own files are at the top.
    """
    root = {"folders": {}, "files": []}
    for file in files:
        node = root
        for part in Path(file.folder).parts:
            node = node["folders"].setdefault(part, {"folders": {}, "files": []})
        node["files"].append(file)
    return merge_chains(root)


def merge_chains(node: dict) -> dict:
    """Returns ``node`` with every child folder that holds only one folder merged with it."""
    folders = {}
    for name, child in node["folders"].items():
        child = merge_chains(child)
        # NOTE: one merge suffices; merge_chains(child) has already merged child's own children
        if len(child["folders"]) == 1 and len(child["files"]) == 0:
            name_sub, child = next(iter(child["folders"].items()))
            name = f"{name}/{name_sub}"
        folders[name] = child
    return {"folders": folders, "files": node["files"]}


def files_below(node: dict) -> list[ResultsFile]:
    """Returns the files of a tree node and of every folder below it."""
    return node["files"] + [file for child in node["folders"].values() for file in files_below(child)]


def natural_key(name: str) -> list:
    """Sorts ``epochs_10`` before ``epochs_100``: digit runs compare as numbers."""
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def format_time(timestamp: float) -> str:
    """A timestamp as the page shows it, to the minute."""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def rows_html(files: list[ResultsFile], path_open: str | None, with_folder: bool) -> str:
    """A table with one row per file; ``with_folder`` adds the folder each is listed under."""
    rows = []
    for file in files:
        curated = f"curated ({file.num_curated})" if file.is_curated else ""
        text = " ".join([file.folder, file.run, file.name, *file.stages, curated]).lower()
        cell_folder = f"<td class='folder'>{html.escape(file.folder)}</td>" if with_folder else ""
        rows.append(
            f"<tr class='file{' open' if file.path == path_open else ''}' data-text='{html.escape(text, quote=True)}' "
            f"data-href='open?path={quote(file.path, safe='')}' title='{html.escape(file.path, quote=True)}'>"
            f"{cell_folder}<td class='when'>{format_time(file.time_run)}</td>"
            f"<td class='pipeline'>{html.escape(file.pipeline)}</td><td class='name'>{html.escape(file.name)}</td>"
            f"<td class='stages'>{html.escape(', '.join(file.stages))}</td><td class='curated'>{curated}</td>"
            f"<td class='num'>{format_bytes(file.num_bytes)}</td></tr>"
        )
    return f"<table>{''.join(rows)}</table>"


def folder_html(name: str, node: dict, path_open: str | None, is_open: bool) -> str:
    """
    One folder of the tree: a collapsible header with its file count and newest change, its runs, its subfolders.

    Parameters
    ----------
    name : str
        The folder's name, or ``a/b`` for merged folders.
    node : dict
        Its subtree from ``folder_tree``.
    path_open : str, optional
        The file the viewer holds; the folders above it start open.
    is_open : bool
        Start open whatever ``path_open`` is. Subfolders start open when this folder holds at most
        ``NUM_FILES_OPEN_ALL`` files.

    Returns
    -------
    str
        The html.
    """
    below = files_below(node)
    is_open = is_open or any(file.path == path_open for file in below)
    files = sorted(node["files"], key=lambda file: (-file.time_run, file.name))
    table = rows_html(files, path_open, with_folder=False) if files else ""
    children = "".join(
        folder_html(name_child, child, path_open, is_open=len(below) <= NUM_FILES_OPEN_ALL)
        for name_child, child in sorted(node["folders"].items(), key=lambda item: natural_key(item[0]))
    )
    newest = max(file.time_modified for file in below)
    return (
        f"<details data-open='{int(is_open)}'{' open' if is_open else ''}><summary>{html.escape(name)}"
        f"<span class='muted'>{len(below)} · {format_time(newest)}</span></summary>"
        f"<div class='children'>{table}{children}</div></details>"
    )


def picker_html(folder: str, files: list[ResultsFile], path_open: str | None = None, error: str = "") -> str:
    """
    The page listing the results files: the most recently changed on top, then all of them as a folder tree.

    Parameters
    ----------
    folder : str
        The served folder, shown in the header.
    files : list[ResultsFile]
        What ``find_results`` returned.
    path_open : str, optional
        The file the viewer holds; its row is marked and the header links to the viewer.
    error : str
        Why the last pick did not open.

    Returns
    -------
    str
        The html.
    """
    tree = folder_tree(files)
    recent = ""
    if len(files) > NUM_RECENT:
        newest = sorted(files, key=lambda file: -file.time_modified)[:NUM_RECENT]
        recent = f"<section id='recent'><h2>recent</h2><div class='scroll'>{rows_html(newest, path_open, with_folder=True)}</div></section>"
    blocks = [rows_html(tree["files"], path_open, with_folder=False)] if tree["files"] else []
    blocks += [
        folder_html(name, child, path_open, is_open=True)
        for name, child in sorted(tree["folders"].items(), key=lambda item: natural_key(item[0]))
    ]
    return _PICKER.substitute(
        title=f"{Path(folder).name} - masknmf",
        root=html.escape(folder),
        count=f"{len(files)} files",
        viewer="" if path_open is None else f"<a href='./'>viewer: {html.escape(Path(path_open).name)}</a>",
        error=f"<div class='error'>{html.escape(error)}</div>" if error else "",
        recent=recent,
        tree=(
            f"<section id='tree'><h2>all</h2><div class='scroll'>{''.join(blocks)}</div></section>"
            if files
            else "<p>No masknmf results files below this folder.</p>"
        ),
    )


class PickedViewer:
    """What a CanvasServer shows and closes: nothing until a file is picked, then that file's demixing viewer."""

    def __init__(self):
        self.viewer = None

    @property
    def figure(self):
        return self.viewer.figure

    def show(self):
        if self.viewer is not None:
            self.viewer.show()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


class PickerServer(CanvasServer):
    """
    The results files below a folder at ``files``, and the demixing viewer on the picked one at ``/``.

    One viewer, so one file open at a time: a pick replaces what every connected browser sees.

    Parameters
    ----------
    folder : str | os.PathLike
        The folder whose results files are listed; only those can be opened.
    open_viewer : Callable
        Builds the viewer of the first pick: ``open_viewer(args=, filepath_results=, names_present=, canvas=)``.
    args : argparse.Namespace
        The parsed ``serve`` options, handed to ``open_viewer``.
    host, port : str, int
        Where ``serve`` listens.
    """

    def __init__(self, folder: str | os.PathLike, open_viewer: Callable, args: argparse.Namespace, *, host: str, port: int):
        self.folder = str(Path(folder).expanduser().resolve())
        self.open_viewer = open_viewer
        self.args = args
        self.path_open = None
        self._canvas = BrowserCanvas(size=(1600, 1000), max_fps=60.0)
        super().__init__(
            PickedViewer(), title=f"{Path(self.folder).name} - masknmf", waiting="Opening ...", host=host, port=port
        )

    @property
    def canvas(self):
        return self._canvas

    def open(self, path: str) -> None:
        """Show the results file at ``path`` in the viewer; it has to be one ``find_results`` lists."""
        if path not in [file.path for file in find_results(folder=self.folder)]:
            raise ValueError(f"{path} is not a results file below {self.folder}")
        if self.vis.viewer is None:
            self.vis.viewer = self.open_viewer(
                args=self.args, filepath_results=path, names_present=stage_groups(path), canvas=self.canvas
            )
            self.vis.show()
        else:
            self.vis.viewer.load_results(path)
        self.path_open = path
        name = Path(path).name
        resources["index.html"] = "text/html", page_html(f"{name} - masknmf", f"Opening {name} ...", home="files", home_label="files")

    async def asgi(self, scope, receive, send):
        """The picker's pages, then the canvas: ``files`` lists, ``open?path=`` picks, ``/`` is the viewer."""
        if scope["type"] != "http":
            await super().asgi(scope, receive, send)
            return
        name = scope["path"].rsplit("/", 1)[-1]
        if name == "open":
            path = parse_qs(scope.get("query_string", b"").decode()).get("path", [""])[0]
            try:
                self.open(path)
            # NOTE: SystemExit is how the cli's checks end a command; here it ends the pick, not the server
            except (OSError, ValueError, SystemExit) as e:
                reason = str(e) if not isinstance(e, SystemExit) else "it could not be opened; the server's terminal says why"
                await respond(send, picker_html(self.folder, find_results(self.folder), self.path_open, f"{path}: {reason}"))
                return
            await redirect(send, "./")
        elif name == "files":
            await respond(send, picker_html(self.folder, find_results(self.folder), self.path_open))
        elif name == "" and self.path_open is None:
            await redirect(send, "files")
        else:
            await super().asgi(scope, receive, send)


async def respond(send, body: str) -> None:
    """Send one html page."""
    await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/html; charset=utf-8")]})
    await send({"type": "http.response.body", "body": body.encode()})


async def redirect(send, location: str) -> None:
    """Send the browser to ``location``, relative to the page it asked for."""
    await send({"type": "http.response.start", "status": 303, "headers": [(b"location", location.encode())]})
    await send({"type": "http.response.body", "body": b""})
