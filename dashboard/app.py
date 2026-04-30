"""FastAPI entrypoint for the new master dashboard.

Run with::

    python -m uvicorn dashboard.app:app --app-dir code --host 127.0.0.1 --port 8765

…from the project root. ``--app-dir code`` puts ``code/`` on ``sys.path`` so
``dashboard`` is importable without turning ``code/`` itself into a package
(which would shadow Python's stdlib ``code`` module).

No authentication, no rewrite of any data — the dashboard only reads JSONL
artefacts produced by the hash-KWS cluster.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import loaders, paths

logger = logging.getLogger("hash_kws.dashboard")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

APP_DIR: Path = Path(__file__).resolve().parent
TEMPLATE_DIR: Path = APP_DIR / "templates"
STATIC_DIR: Path = APP_DIR / "static"

app = FastAPI(
    title="Hash-KWS Cluster Dashboard",
    description=(
        "Read-only live dashboard for the distributed ESP32 hash-KWS "
        "cluster (real + emulated + master)."
    ),
    docs_url="/docs",
    redoc_url=None,
)

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> JSONResponse:
    """Trivial liveness probe — does not touch the filesystem."""
    return JSONResponse({"ok": True, "ts": time.time()})


@app.get("/api/snapshot")
async def api_snapshot() -> JSONResponse:
    """Single snapshot of the cluster state for one-shot pulls."""
    snapshot = await asyncio.to_thread(loaders.build_snapshot)
    return JSONResponse(snapshot)


@app.get("/api/stream")
async def api_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events stream of snapshots, one per second.

    Clients subscribe via ``EventSource("/api/stream")``. When the client
    disconnects, the loop exits and the underlying request is closed.
    """
    interval_sec = 1.0

    async def event_generator() -> AsyncIterator[bytes]:
        logger.info("SSE client connected from %s", request.client)
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("SSE client disconnected")
                    break
                snapshot = await asyncio.to_thread(loaders.build_snapshot)
                payload = json.dumps(snapshot, default=_json_default)
                yield f"event: snapshot\ndata: {payload}\n\n".encode("utf-8")
                await asyncio.sleep(interval_sec)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover — defensive
            logger.exception("SSE generator crashed")
            raise

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        event_generator(), media_type="text/event-stream", headers=headers
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Single-page dashboard UI.

    The template is self-contained — all CSS/JS is inlined to keep the demo
    footprint small and avoid static-serving surprises.
    """
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "project_root": str(paths.PROJECT_ROOT),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """Fallback serialiser for objects json.dumps would otherwise reject."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


__all__ = ["app"]
