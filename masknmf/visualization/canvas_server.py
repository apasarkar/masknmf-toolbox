"""Serve one fastplotlib figure to browsers over HTTP, one figure per process.

rendercanvas's ``http`` backend streams the canvas as JPEG frames over a
websocket and takes the browser's pointer and key events back. A
:class:`CanvasServer` puts uvicorn in front of it. ``masknmf serve`` serves
the demixing viewer this way; any object with ``figure``, ``show()`` and
``close()`` whose figure is on a :class:`BrowserCanvas` can be served.

A server process must call :func:`select_http_backend` before fastplotlib is
imported, so fastplotlib's loop is the one the canvas runs on.
"""

from __future__ import annotations

import hmac
import html
import importlib
import logging
import os
import sys
import time
from importlib import resources as package_resources
from string import Template

import numpy as np
from rendercanvas.http import HttpRenderCanvas, resources
from rendercanvas.http import asgi as canvas_app

__all__ = [
    "SESSION_KEY_ENV",
    "SESSION_KEY_HEADER",
    "BrowserCanvas",
    "CanvasServer",
    "client_js",
    "page_html",
    "require",
    "select_http_backend",
]

logger = logging.getLogger("masknmf.canvas_server")

# a server started behind a proxy accepts only requests carrying its key, so no
# other local user can reach it on its localhost port
SESSION_KEY_ENV = "CANVAS_SESSION_KEY"
SESSION_KEY_HEADER = "x-canvas-session-key"

_PAGE = Template("""<!DOCTYPE html>
<html>
<head>
    <meta charset='utf-8'>
    <title>$title</title>
    <script type='module' src='renderview.js'></script>
    <script type='module' src='renderview-client.js'></script>
    <link rel='stylesheet' href='renderview.css'>
    <style>
        html, body { margin: 0; height: 100%; overflow: auto; background: #1e1f22; color: #ddd;
                     font-family: system-ui, sans-serif; }
        #canvas p { width: 100%; height: 100%; margin: 0; display: flex; justify-content: center;
                    align-items: center; color: #888; }
        #canvas .renderview-resizer { width: 18px; height: 18px;
                    background: linear-gradient(135deg, transparent 55%, rgba(200, 200, 200, 0.7) 55%); }
        .badge { position: fixed; bottom: 6px; padding: 2px 8px; font: 12px monospace; color: #bbb;
                 background: rgba(0, 0, 0, 0.45); border-radius: 4px; text-decoration: none; }
        #status { right: 8px; }
        #fit { right: 8px; bottom: 30px; cursor: pointer; }
        #home { left: 8px; }
        .badge button { font: inherit; margin-left: 6px; }
    </style>
</head>
<body>
    <div id='canvas' class='is-resizable' style='display:block; width:100vw; height:100vh; --line-thickness:0'>
        <p>$waiting</p>
    </div>
    <div id='fit' class='badge' title='drag the bottom-right corner to resize; this fits the window again'
         onclick='fitWindow()'>fit window</div>
    $home
    <div id='status' class='badge'></div>
    <script>
        function fitWindow() {
            const c = document.getElementById('canvas');
            c.style.width = '100vw';
            c.style.height = '100vh';
        }
        // the app can set the canvas to a fixed size (its minimum layout);
        // a browser window resized afterwards takes the canvas with it again
        window.addEventListener('resize', fitWindow);
    </script>
</body>
</html>
""")


def page_html(title: str, waiting: str, home: str | None = None, home_label: str = "sessions") -> str:
    """The page holding the canvas: full window, a fit badge, the connection status.

    ``waiting`` shows until the first frame arrives; ``home`` adds a badge
    named ``home_label`` linking back there (a session list, when a proxy serves several).
    """
    link = (
        f"<a id='home' class='badge' href='{home}' >{html.escape(home_label)}</a>"
        if home
        else ""
    )
    return _PAGE.substitute(
        title=html.escape(title), waiting=html.escape(waiting), home=link
    )


def client_js() -> str:
    """Returns the browser client, renderview on a canvas with H.264 decoded by WebCodecs."""
    return (
        package_resources.files("masknmf.visualization")
        .joinpath("canvas_client.js")
        .read_text(encoding="utf-8")
    )


def require(module: str):
    """Imports ``module`` from the ``server`` extra; the ImportError names the extra."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise ImportError(f'serving to browsers needs {module}: uv pip install "masknmf[server]"') from e


def select_http_backend() -> None:
    """Make ``rendercanvas.auto``, and so fastplotlib's loop, the http backend.

    NDWidget fetches frames on fastplotlib's loop. Under the offscreen backend
    that loop never runs and scrubbing stalls; over ssh the auto choice is the
    terminal backend. Has no effect once ``rendercanvas.auto`` is imported.
    """
    if "rendercanvas.auto" in sys.modules:
        logger.warning(
            "rendercanvas.auto is already imported; fastplotlib's loop stays on its backend"
        )
        return
    os.environ.pop("RENDERCANVAS_FORCE_OFFSCREEN", None)
    os.environ.pop("WGPU_FORCE_OFFSCREEN", None)
    os.environ["RENDERCANVAS_BACKEND"] = "http"


class BrowserCanvas(HttpRenderCanvas):
    """An http canvas that sends only frames that changed, as H.264 where the browser decodes it.

    Departures from ``HttpRenderCanvas``:

    - It keeps the size it was asked for until a browser reports its own;
      otherwise the first draw after a browser connects is on a 1 x 1 canvas,
      where every subplot's viewport is negative.
    - It drops a frame identical to the last one sent: an imgui figure draws
      every frame whether anything moved or not, which streamed 60 fps of jpeg
      while idle and cancelled the lossless frame meant to follow the last change.
    - While the view changes, browsers that can decode H.264 (WebCodecs, so
      https or localhost) get it at ``bitrate`` Mbit/s, a tenth of the bytes of
      jpeg on a recording; the rest get jpeg of ``quality``. A still view is
      sent once as png either way.
    - Frames are rendered at no more than ``max_pixel_ratio`` physical pixels
      per css pixel, so a HiDPI browser gets a quarter of the pixels and upscales.
    """

    def __init__(
        self,
        *args,
        quality: int = 80,
        bitrate: float = 8.0,
        max_pixel_ratio: float = 1.0,
        **kwargs,
    ):
        self._last_array = None
        self._lossless_sent = False
        # the base class reads _quality when it encodes jpeg
        self._quality = int(quality)
        self.bitrate = float(bitrate)
        self.max_pixel_ratio = float(max_pixel_ratio)
        self._h264_clients = set()
        self._encoder = None
        self._codec = None
        self._seq = 0
        self._force_key = False
        super().__init__(*args, **kwargs)

    def _on_event(self, event, id):
        kind = event.get("type")
        if kind == "_codecs":
            if event.get("h264"):
                self._h264_clients.add(id)
            self._force_key = True
            return
        if kind == "_keyframe":
            self._force_key = True
            return
        if kind == "_clients_change":
            self._h264_clients &= set(event["ids"])
            self._force_key = True
        ratio = event.get("ratio", 1.0) if kind == "resize" else 1.0
        if ratio > self.max_pixel_ratio:
            scale = self.max_pixel_ratio / ratio
            event = {
                **event,
                "pwidth": max(1, round(event["pwidth"] * scale)),
                "pheight": max(1, round(event["pheight"] * scale)),
                "ratio": self.max_pixel_ratio,
            }
        super()._on_event(event, id)

    def _rc_set_logical_size(self, width, height):
        super()._rc_set_logical_size(width, height)
        if canvas_app.get_count() == 0:
            self._size_info.set_physical_size(
                max(1, round(width)), max(1, round(height)), 1.0
            )

    def _rc_close(self):
        # an NVENC context left open while wgpu's device goes hangs the process at exit
        self._encoder = None
        super()._rc_close()

    def _rc_request_draw(self):
        # the base also cancels the pending lossless frame here; a changed
        # frame reschedules it in _encode_frame, an identical one must not
        if not self._draw_requested:
            self._draw_requested = True
            self._schedule_maybe_draw()

    def _encode_frame(self, array, is_lossless_redraw=False):
        if is_lossless_redraw:
            self._lossless_sent = True
        elif self._last_array is not None and np.array_equal(array, self._last_array):
            if self._lossless_draw_info is None and not self._lossless_sent:
                self._schedule_lossless_draw(self._last_array)
            # a browser that connected since still needs the frame
            self._send_last_frame_to_ready_clients()
            return
        else:
            # the bitmap may be a buffer the next frame reuses
            array = self._last_array = array.copy()
            self._lossless_sent = False
        clients = self._frame_info_per_client.keys()
        if not is_lossless_redraw and clients and clients <= self._h264_clients:
            self._encode_h264(array)
        else:
            super()._encode_frame(array, is_lossless_redraw)

    def _encode_h264(self, array):
        """Sends ``array`` as one H.264 access unit and schedules the png that follows it."""
        av = require("av")
        height, width = array.shape[0] // 2 * 2, array.shape[1] // 2 * 2
        if self._encoder is None or (self._encoder.width, self._encoder.height) != (
            width,
            height,
        ):
            self._encoder = _open_h264(av, width, height, self.bitrate)
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(array[:height, :width]), format="rgba"
        ).reformat(format="yuv420p")
        if self._force_key:
            frame.pict_type = av.video.frame.PictureType.I
            self._force_key = False
        packets = self._encoder.encode(frame)
        data = b"".join(bytes(p) for p in packets)
        if not data:
            return
        key = any(p.is_keyframe for p in packets)
        if key:
            self._codec = _avc1_codec(data) or self._codec
        msg = {
            "type": "framebufferdata",
            "nbuffers": 1,
            "mimetype": "video/h264",
            "key": key,
            "codec": self._codec,
            "seq": self._seq,
            "width": width,
            "height": height,
            "timestamp": time.time(),
            "index": 0,
        }
        self._seq += 1
        self._last_frame = msg, [data]
        self._ref_index += 1
        self._schedule_lossless_draw(array)
        self._send_last_frame_to_ready_clients()


def _open_h264(av, width: int, height: int, bitrate: float):
    """Opens an encoder that returns each frame's packet at once: NVENC, else libx264."""
    for name, options in (
        (
            "h264_nvenc",
            {
                "preset": "p1",
                "tune": "ull",
                "zerolatency": "1",
                "delay": "0",
                "forced-idr": "1",
            },
        ),
        ("libx264", {"preset": "ultrafast", "tune": "zerolatency"}),
    ):
        encoder = av.CodecContext.create(name, "w")
        encoder.width, encoder.height, encoder.pix_fmt = width, height, "yuv420p"
        encoder.bit_rate = int(bitrate * 1e6)
        encoder.framerate = 30
        encoder.max_b_frames = 0
        # keyframes only on demand: a new browser asks for one
        encoder.gop_size = 1_000_000
        encoder.options = options
        try:
            encoder.open()
        except (av.FFmpegError, ValueError) as e:
            logger.info("no %s (%s)", name, e)
            continue
        logger.info("h264 frames from %s at %g Mbit/s", name, bitrate)
        return encoder
    raise RuntimeError("PyAV has neither h264_nvenc nor libx264")


def _avc1_codec(data: bytes) -> str | None:
    """Returns the WebCodecs codec string (``avc1.PPCCLL``) from an access unit's SPS."""
    i = data.find(b"\x00\x00\x01")
    while i >= 0 and i + 6 < len(data):
        if data[i + 3] & 0x1F == 7:
            return "avc1." + data[i + 4 : i + 7].hex()
        i = data.find(b"\x00\x00\x01", i + 3)
    return None


class CanvasServer:
    """A shown figure on a :class:`BrowserCanvas`, ready for uvicorn to serve.

    Frames are drawn once a browser connects. ``serve()`` runs uvicorn in
    this thread until interrupted or the canvas closes; ``serve_async()``
    does the same inside a running asyncio loop; ``asgi`` mounts into a
    larger ASGI app.

    Parameters
    ----------
    vis
        What is served: an object with ``figure``, ``show()`` and ``close()``
        whose figure is on a :class:`BrowserCanvas`.
    title : str
        The page title.
    waiting : str
        What the page says until the first frame.
    host, port : str, int
        Where ``serve`` listens.
    key : str, optional
        Accept only requests whose ``x-canvas-session-key`` header carries it.
    """

    def __init__(
        self,
        vis,
        *,
        title: str,
        waiting: str,
        host: str,
        port: int,
        key: str | None = None,
    ):
        self.vis = vis
        self.title = title
        self.host = str(host)
        self.port = int(port)
        self.key = key
        resources["index.html"] = "text/html", page_html(title, waiting)
        resources["renderview-client.js"] = "text/javascript", client_js()
        self._server = None
        self._closed = False
        self.vis.show()
        self.canvas.add_event_handler(self.stop, "close")

    @property
    def figure(self):
        return self.vis.figure

    @property
    def canvas(self):
        return self.vis.figure.canvas

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{host}:{self.port}/"

    def clients(self) -> int:
        """How many browsers are connected."""
        return int(canvas_app.get_count())

    async def asgi(self, scope, receive, send):
        """The ASGI app: rendercanvas's, behind the session key when there is one."""
        if self.key is not None and scope["type"] in ("http", "websocket"):
            got = dict(scope.get("headers") or []).get(SESSION_KEY_HEADER.encode(), b"")
            if not hmac.compare_digest(got, self.key.encode()):
                if scope["type"] == "websocket":
                    await send({"type": "websocket.close", "code": 1008})
                    return
                await send(
                    {
                        "type": "http.response.start",
                        "status": 403,
                        "headers": [(b"content-type", b"text/plain")],
                    }
                )
                await send({"type": "http.response.body", "body": b"Forbidden"})
                return
        await canvas_app(scope, receive, send)

    def _config(self, uvicorn, **kwargs):
        # a bound method reads as asgi2 to uvicorn's sniffing; frames are
        # jpeg, deflating them costs cpu for nothing
        return uvicorn.Config(
            self.asgi,
            host=self.host,
            port=self.port,
            log_level="warning",
            interface="asgi3",
            ws_per_message_deflate=False,
            **kwargs,
        )

    def stop(self, event=None) -> None:
        """Ask a running ``serve`` to return; safe from any thread."""
        if self._server is not None:
            self._server.should_exit = True

    def serve(self) -> None:
        """Run the server in this thread until it is interrupted or the canvas closes."""
        uvicorn = require("uvicorn")
        self._server = uvicorn.Server(self._config(uvicorn))
        logger.info("%s at %s", self.title, self.url)
        try:
            self._server.run()
        finally:
            self.close()

    async def serve_async(self) -> None:
        """Run the server inside the current asyncio loop until it is stopped."""
        uvicorn = require("uvicorn")
        self._server = uvicorn.Server(self._config(uvicorn))
        logger.info("%s at %s", self.title, self.url)
        try:
            await self._server.serve()
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.vis.close()
