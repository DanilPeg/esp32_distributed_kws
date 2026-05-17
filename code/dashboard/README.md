# Hash-KWS Cluster Dashboard

New master/aggregator dashboard for the distributed ESP32 hash-KWS demo.

Replaces `code/web_portal/` (deprecated). Read-only: it never writes to any
JSONL or state file — it only tails artefacts that producers create.

## Data sources

| File | Producer | What it contains |
| --- | --- | --- |
| `notes/Journal/hash_kws_telemetry/state.json` | `hash_kws_serial_bridge.py` | Current bridge state (port, last seen, counters). |
| `notes/Journal/hash_kws_telemetry/events.jsonl` | `hash_kws_serial_bridge.py` | Merged stream of all normalised events. |
| `notes/Journal/hash_kws_telemetry/node{1,2,3}/events.jsonl` | bridge + `hash_kws_cluster_sim.py` | Per-node streams (real, emulated, master). |
| `notes/Journal/hash_kws_fusion/decisions.jsonl` | `hash_kws_dual_audio_fusion.py`, `hash_kws_cluster_sim.py` | Fused decisions (agree / single-node / waiting). |
| `notes/Journal/hash_kws_fusion/state.json` | fusion scripts | Fusion engine snapshot. |
| `notes/Journal/hash_kws_cluster/state.json` | `hash_kws_cluster_sim.py` | Cluster simulation status. |

Missing files are tolerated — the UI still renders cards for the canonical
nodes defined in `paths.NODE_ROLES`.

## Install

```powershell
pip install fastapi uvicorn jinja2
```

No other runtime dependencies. Python 3.10+.

## Run

Recommended — use the launcher at the project root. It resolves `code/`
relative to its own location so the working directory does not matter:

```powershell
python C:\Users\Danil\diploma_esp32_distributed_nn\run_dashboard.py
# or, from anywhere inside the repo
python run_dashboard.py
```

Optional flags: `--host`, `--port 8765`, `--reload`, `--log-level info`.

Alternatively, the uvicorn-direct form works too, but **only** when the
current working directory is the project root — `--app-dir code` is
interpreted relative to cwd:

```powershell
cd C:\Users\Danil\diploma_esp32_distributed_nn
python -m uvicorn dashboard.app:app --app-dir code --host 127.0.0.1 --port 8765
```

(We avoid making `code/` itself a package because that name would shadow
Python's stdlib `code` module, hence the `--app-dir` detour.)

Then open <http://127.0.0.1:8765/>.

## Endpoints

- `GET /` — single-page dashboard UI.
- `GET /api/snapshot` — one-shot cluster snapshot (JSON).
- `GET /api/stream` — Server-Sent Events stream, one `snapshot` event per second.
- `GET /health` — liveness probe.
- `GET /docs` — auto-generated OpenAPI page.

The UI opens an `EventSource` against `/api/stream`; if the connection drops
it falls back to polling `/api/snapshot` every 2 seconds until the stream
recovers.

## Layout

```
code/dashboard/
├── __init__.py
├── paths.py          # repo-root-anchored JSONL paths
├── loaders.py        # tail_jsonl + per-node summary + snapshot builder
├── app.py            # FastAPI app (/, /api/snapshot, /api/stream, /health)
├── templates/
│   └── index.html    # single-page UI (inlined CSS + JS)
└── README.md
```

## What the UI shows

- **Cluster overview** — one card per node (real / emulated / master) with
  online status (based on `host_time` recency), last kind seen, latest top1
  label + score, speech flag, `recent_max`, and ESP-NOW `tx_ok`/`tx_fail`.
- **Counters (tail window)** — totals across the event tail (not all time),
  plus a per-label tag cloud of the most frequent `top1` labels.
- **Live inference feed** — newest-first table of the last ~30 events from
  `hash_kws_telemetry/events.jsonl`.
- **Fusion decisions** — newest-first table of the last ~30 entries from
  `hash_kws_fusion/decisions.jsonl`.

## Notes

- SSE cadence is 1 s. Change `interval_sec` in `app.py::api_stream` to tune.
- Tail depth is set per panel in `loaders.build_snapshot`.
- No auth, no CORS, no network writes. Local development tool only.
