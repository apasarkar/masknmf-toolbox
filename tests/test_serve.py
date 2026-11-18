"""A figure on the http canvas, served to the fake browser in ``asgi_client`` as uvicorn would serve it."""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("uvicorn")

sys.path.insert(0, str(Path(__file__).parent))
from asgi_client import Browser, Lifespan, frame_of_size, get, ws_reply  # noqa: E402

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

