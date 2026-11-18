"""A fake browser and server lifecycle for driving an ASGI app as uvicorn would.

What ``renderview-client.js`` does on rendercanvas's websocket (connect,
report a size, acknowledge frames, send events), and uvicorn's lifespan,
whose startup kicks the rendercanvas loop on the running asyncio loop.
"""

from __future__ import annotations

import asyncio
import json
import struct

import simplejpeg


async def request(
    app, method: str, path: str, body: bytes = b"", headers=()
) -> tuple[int, dict, bytes]:
    """One http request; ``path`` may carry a query. Returns status, headers, body."""
    path, _, query = path.partition("?")
    sent = []

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query.encode(),
        "headers": list(headers),
    }
    await app(scope, receive, send)
    start = sent[0]
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return start["status"], dict(start.get("headers") or []), body


async def get(app, path: str, headers=()) -> tuple[int, bytes]:
    status, _, body = await request(app, "GET", path, headers=headers)
    return status, body


async def ws_reply(app, path: str = "/", headers=()) -> dict:
    """The first message an app sends a websocket: an accept, or a close refusing it."""
    inbox: asyncio.Queue = asyncio.Queue()
    outbox: asyncio.Queue = asyncio.Queue()
    scope = {"type": "websocket", "path": path, "headers": list(headers)}
    task = asyncio.create_task(app(scope, inbox.get, outbox.put))
    first = await asyncio.wait_for(outbox.get(), 10)
    await inbox.put({"type": "websocket.disconnect"})
    await asyncio.wait_for(task, 10)
    return first


class Browser:
    """A fake browser tab on the websocket: what renderview-client.js does."""

    def __init__(self, app, path: str = "/", headers=()):
        self.app = app
        self.path = path
        self.headers = list(headers)
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.frames: list[tuple[dict, list[bytes]]] = []
        self.task = None

    async def connect(self):
        async def receive():
            return await self.inbox.get()

        async def send(message):
            await self.outbox.put(message)

        self.task = asyncio.create_task(
            self.app(
                {"type": "websocket", "path": self.path, "headers": self.headers},
                receive,
                send,
            )
        )
        first = await asyncio.wait_for(self.outbox.get(), 5)
        assert first["type"] == "websocket.accept"

    async def event(self, **event):
        event.setdefault("timestamp", 0)
        await self.inbox.put({"type": "websocket.receive", "text": json.dumps(event)})

    async def resize(self, width: int, height: int):
        await self.event(
            type="resize",
            width=width,
            height=height,
            pwidth=width,
            pheight=height,
            ratio=1,
        )

    async def next_frame(self, timeout: float = 30.0) -> tuple[dict, list[bytes]]:
        """Wait for the next frame, acknowledge it as the client does."""
        while True:
            message = await asyncio.wait_for(self.outbox.get(), timeout)
            if "text" not in message:
                continue
            msg = json.loads(message["text"])
            if msg.get("type") != "framebufferdata":
                continue
            buffers = []
            for _ in range(msg["nbuffers"]):
                message = await asyncio.wait_for(self.outbox.get(), timeout)
                buffers.append(message["bytes"])
            self.frames.append((msg, buffers))
            await self.event(
                type="_framefeedback",
                index=msg["index"],
                localtime=0,
                timestamp=msg["timestamp"],
            )
            return msg, buffers

    async def disconnect(self):
        await self.inbox.put({"type": "websocket.disconnect"})
        await asyncio.wait_for(self.task, 5)


def frame_size(buffer: bytes) -> tuple[int, int]:
    """(width, height) of a jpeg or png frame."""
    if buffer[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", buffer[16:24])
    height, width, _cs, _ss = simplejpeg.decode_jpeg_header(buffer)
    return width, height


async def frame_of_size(browser: Browser, size: tuple[int, int], tries: int = 12):
    """The next frame that is ``size`` (width, height) pixels."""
    for _ in range(tries):
        msg, buffers = await browser.next_frame()
        width, height = frame_size(buffers[0])
        if (width, height) == tuple(size):
            return msg, buffers
    raise AssertionError(f"no {size} frame; last was {(width, height)}")


class Lifespan:
    """uvicorn's lifespan: startup kicks the rendercanvas loop on this asyncio loop."""

    def __init__(self, app):
        self.app = app
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.task = None

    async def start(self):
        async def receive():
            return await self.inbox.get()

        async def send(message):
            pass

        await self.inbox.put({"type": "lifespan.startup"})
        self.task = asyncio.create_task(self.app({"type": "lifespan"}, receive, send))

    async def stop(self):
        await self.inbox.put({"type": "lifespan.shutdown"})
        await asyncio.wait_for(self.task, 5)
