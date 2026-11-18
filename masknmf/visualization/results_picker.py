"""Serve a folder of results files to a browser: a page listing them, and the demixing viewer on the one picked.

``masknmf serve FOLDER`` runs this. ``files`` lists every results file below the folder, one table per folder
directly under it, the runs of each subfolder together and newest first; a click on a row opens the file in the one viewer this process holds, at ``/``.
The first pick builds the viewer, later picks load into it in place.
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

from rendercanvas.http import resources

from masknmf.demixing.curation import CURATED_SUFFIX, latest_results, results_stem
from masknmf.io import stage_groups
from masknmf.visualization.canvas_server import BrowserCanvas, CanvasServer, page_html

__all__ = ["PickerServer", "ResultsFile", "find_results", "picker_html"]

PATTERNS_RESULTS = ("*results*.hdf5", "*results*.h5")
# a run folder is named <timestamp>_<pipeline>
PATTERN_RUN = re.compile(r"^(\d{8}T\d{6})_(.+)$")

_PICKER = Template("""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>$title</title>
    <style>
        body { margin: 0; padding: 16px 24px; background: #1e1f22; color: #ddd; font: 14px system-ui, sans-serif; }
        a { color: #7ab4ff; text-decoration: none; }
        header { display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; margin-bottom: 12px; }
        header .root { font: 13px monospace; color: #999; }
        #filter { flex: 1; min-width: 200px; padding: 6px 10px; background: #2b2d31; color: #ddd;
                  border: 1px solid #444; border-radius: 4px; font: inherit; }
        .error { padding: 8px 12px; margin-bottom: 12px; background: #4a2326; border-radius: 4px; }
        details { margin-bottom: 10px; }
        summary { padding: 6px 0; font-weight: 600; cursor: pointer; }
        summary span { font-weight: 400; color: #888; margin-left: 8px; }
        .scroll { overflow-x: auto; }
        table { border-collapse: collapse; width: 100%; min-width: 900px; table-layout: fixed; }
        td { padding: 5px 12px 5px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
             border-top: 1px solid #2f3136; }
        td.when { width: 130px; } td.pipeline { width: 190px; } td.name { width: 230px; }
        td.stages { width: 250px; } td.curated { width: 90px; } td.num { width: 70px; }
        tr.file { cursor: pointer; }
        tr.file:hover { background: #2b2d31; }
        tr.open td:first-child { box-shadow: inset 3px 0 #7ab4ff; padding-left: 8px; }
        td.stages, td.curated, td.num { color: #999; }
        td.num { text-align: right; font-variant-numeric: tabular-nums; }
        body.opening { cursor: progress; opacity: 0.6; }
    </style>
</head>
<body>
    <header>
        <strong>masknmf results</strong>
        <span class='root'>$root</span>
        <input id='filter' placeholder='filter by dataset, run, pipeline or stage' autofocus>
        $viewer
    </header>
    $error
    $groups
    <script>
        const filter = document.getElementById('filter')
        filter.addEventListener('input', () => {
            const words = filter.value.toLowerCase().split(/\\s+/).filter(Boolean)
            for (const group of document.querySelectorAll('details')) {
                let shown = 0
                for (const row of group.querySelectorAll('tr.file')) {
                    const hit = words.every((word) => row.dataset.text.includes(word))
                    row.style.display = hit ? '' : 'none'
                    shown += hit
                }
                group.style.display = shown ? '' : 'none'
            }
        })
        for (const row of document.querySelectorAll('tr.file')) {
            row.addEventListener('click', () => {
                document.body.classList.add('opening')
                window.location = row.dataset.href
            })
        }
    </script>
</body>
</html>
""")


@dataclass
class ResultsFile:
    """One results file below the served folder, with what its row on the page shows."""

    path: str
    group: str
    subfolder: str
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

    Hidden folders, folders starting with ``_`` (archives) and .zarr stores are not entered.

    Parameters
    ----------
    folder : str | os.PathLike
        The folder to search.

    Returns
    -------
    list[ResultsFile]
        Sorted by the folder directly under ``folder``, then subfolder, then newest run first.
    """
    folder = Path(folder).expanduser().resolve()
    paths = []
    for dirpath, dirnames, filenames in os.walk(folder):
        dirnames[:] = [name for name in dirnames if not name.startswith((".", "_")) and not name.endswith(".zarr")]
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
        parts = (folder_run.parent if is_run and folder_run != folder else folder_run).relative_to(folder).parts
        stat = os.stat(path)
        files.append(
            ResultsFile(
                path=path,
                group=parts[0] if parts else ".",
                subfolder="/".join(parts[1:]),
                run=folder_run.name,
                stages=[stage_label(stage) for stage in stages],
                num_curated=num_curated[(os.path.dirname(path), results_stem(path))],
                num_bytes=stat.st_size,
                time_modified=stat.st_mtime,
            )
        )
    return sorted(files, key=lambda file: (file.group, file.subfolder, -file.time_run, file.name))


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


def picker_html(folder: str, files: list[ResultsFile], path_open: str | None = None, error: str = "") -> str:
    """
    The page listing the results files: one table per top-level folder, a row per file, a filter box on top.

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
    groups = {}
    for file in files:
        groups.setdefault(file.group, []).append(file)
    blocks = []
    # the folder with the newest run comes first
    for group, members in sorted(groups.items(), key=lambda item: -max(file.time_run for file in item[1])):
        rows = []
        for file in members:
            when = datetime.fromtimestamp(file.time_run).strftime("%Y-%m-%d %H:%M")
            curated = f"curated ({file.num_curated})" if file.is_curated else ""
            text = " ".join([group, file.subfolder, file.run, file.name, *file.stages, curated]).lower()
            rows.append(
                f"<tr class='file{' open' if file.path == path_open else ''}' data-text='{html.escape(text, quote=True)}' "
                f"data-href='open?path={quote(file.path, safe='')}' title='{html.escape(file.path, quote=True)}'>"
                f"<td>{html.escape(file.subfolder)}</td><td class='when'>{when}</td>"
                f"<td class='pipeline'>{html.escape(file.pipeline)}</td><td class='name'>{html.escape(file.name)}</td>"
                f"<td class='stages'>{html.escape(', '.join(file.stages))}</td><td class='curated'>{curated}</td>"
                f"<td class='num'>{format_bytes(file.num_bytes)}</td></tr>"
            )
        blocks.append(
            f"<details open><summary>{html.escape(group)}<span>{len(members)}</span></summary>"
            f"<div class='scroll'><table>{''.join(rows)}</table></div></details>"
        )
    return _PICKER.substitute(
        title=f"{Path(folder).name} - masknmf",
        root=html.escape(folder),
        viewer="" if path_open is None else f"<a href='./'>viewer: {html.escape(Path(path_open).name)}</a>",
        error=f"<div class='error'>{html.escape(error)}</div>" if error else "",
        groups="".join(blocks) if blocks else "<p>No masknmf results files below this folder.</p>",
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
