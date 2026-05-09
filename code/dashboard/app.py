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
import os
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request
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


_INGEST_TOKEN_ENV = "HASH_KWS_INGEST_TOKEN"

_STREAM_TARGETS: dict[str, Path] = {
    "node1": paths.TELEMETRY_DIR / "node1" / "events.jsonl",
    "node2": paths.TELEMETRY_DIR / "node2" / "events.jsonl",
    "node3": paths.TELEMETRY_DIR / "node3" / "events.jsonl",
    "node4": paths.TELEMETRY_DIR / "node4" / "events.jsonl",
    "fusion": paths.FUSION_DECISIONS,
}


def _check_token(provided: str | None) -> None:
    expected = os.environ.get(_INGEST_TOKEN_ENV, "").strip()
    if not expected:
        return  # ingest open if no token configured (LAN-friendly default)
    if not provided or provided.strip() != expected:
        raise HTTPException(status_code=401, detail="bad ingest token")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


@app.post("/api/ingest")
async def api_ingest(
    body: dict[str, Any],
    x_hash_kws_token: str | None = Header(default=None, alias="X-Hash-KWS-Token"),
) -> JSONResponse:
    """Accept one parsed event from a remote bridge and append it to JSONL.

    Body shape:
      {
        "stream": "node1" | "node2" | "node3" | "node4" | "fusion",
        "event":  {... parsed hash_evt ...}
      }
    """
    _check_token(x_hash_kws_token)
    stream = str(body.get("stream", "")).strip()
    event = body.get("event")
    if stream not in _STREAM_TARGETS:
        raise HTTPException(status_code=400, detail=f"unknown stream: {stream!r}")
    if not isinstance(event, dict) or not event:
        raise HTTPException(status_code=400, detail="event must be a non-empty object")
    target = _STREAM_TARGETS[stream]
    await asyncio.to_thread(_append_jsonl, target, event)
    return JSONResponse({"ok": True, "stream": stream, "wrote": str(target)})


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
