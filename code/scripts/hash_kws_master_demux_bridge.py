"""All-in-one bridge for the master ESP32.

The master sketches (`hash_kws_master.ino` and `hash_kws_master_web.ino`)
forward two kinds of `hash_evt` lines to Serial:

  - per-node telemetry, copied from each received ESP-NOW packet:
        hash_evt kind=infer node=2 t=1234 invoke_ms=230 top1=yes top1_score=200 margin=42 recent_max=87 seq=12

  - master-side state and the fusion decisions:
        hash_evt kind=boot node=master role=master_aggregator ...
        hash_evt kind=espnow phase=init status=ok node=master mac=...
        hash_evt kind=fusion node=master label=yes score=4736 margin=2048 voters=3 mode=0 packets=14 rejected=0

This bridge reads the master's USB Serial and routes events:

  * kind=infer/episode/emit with node=N (1..3)
        -> notes/Journal/hash_kws_telemetry/node{N}/events.jsonl
  * kind=fusion (master)
        -> notes/Journal/hash_kws_fusion/decisions.jsonl
           (translated to the dashboard's audio_fusion_agree schema)
  * kind=boot, kind=espnow (master)
        -> notes/Journal/hash_kws_fusion/state.json (master_state)

So the host setup collapses to two windows:

    python code\\scripts\\hash_kws_master_demux_bridge.py --port COMx
    python run_dashboard.py

…regardless of how many inference nodes are physically present. The master
relays everything that comes in over ESP-NOW.

Optional `--remote-url` shifts JSONL writes to a remote dashboard's
`/api/ingest` for the two-PC setup.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import serial  # type: ignore
except Exception:
    serial = None


ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_DIR = ROOT / "notes" / "Journal" / "hash_kws_telemetry"
FUSION_DIR    = ROOT / "notes" / "Journal" / "hash_kws_fusion"

EVT_PREFIX = "hash_evt "
KEY_VAL_RE = re.compile(r"(?P<key>\w+)=(?P<val>[^\s]+)")

NODE_KIND_TO_STREAM = {"infer", "episode", "emit"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_text(path: Path, line: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def atomic_write_json(path: Path, payload: dict) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def coerce(raw: str) -> Any:
    if raw == "":
        return raw
    if raw.lstrip("-").isdigit():
        try:
            return int(raw)
        except Exception:
            return raw
    return raw


def parse_event(line: str) -> dict | None:
    idx = line.find(EVT_PREFIX)
    if idx < 0:
        return None
    payload = line[idx + len(EVT_PREFIX):].strip()
    out: dict = {}
    for m in KEY_VAL_RE.finditer(payload):
        out[m.group("key")] = coerce(m.group("val"))
    if "kind" not in out:
        return None
    return out


def post_event_remote(
    url: str,
    stream: str,
    event: dict,
    *,
    token: str = "",
    timeout: float = 2.0,
    retries: int = 2,
) -> tuple[bool, str]:
    body = json.dumps({"stream": stream, "event": event}, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Hash-KWS-Token"] = token
    last_err = ""
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if 200 <= resp.status < 300:
                    return True, ""
                last_err = f"http {resp.status}"
        except urllib.error.URLError as exc:
            last_err = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            time.sleep(0.2 * (attempt + 1))
    return False, last_err


def fusion_to_decision(evt: dict, sample_idx: int) -> dict:
    label = str(evt.get("label", ""))
    score = int(evt.get("score", 0)) if isinstance(evt.get("score"), (int, str)) else 0
    margin = int(evt.get("margin", 0)) if isinstance(evt.get("margin"), (int, str)) else 0
    voters = int(evt.get("voters", 0)) if isinstance(evt.get("voters"), (int, str)) else 0
    mode = int(evt.get("mode", 0)) if isinstance(evt.get("mode"), (int, str)) else 0
    packets = int(evt.get("packets", 0)) if isinstance(evt.get("packets"), (int, str)) else 0
    rejected = int(evt.get("rejected", 0)) if isinstance(evt.get("rejected"), (int, str)) else 0
    return {
        "kind": "audio_fusion_agree",
        "host_time": utc_now_iso(),
        "label": label,
        "score_sum": score,
        "score_avg": float(score),
        "margin": margin,
        "nodes": [str(i) for i in range(1, voters + 1)],
        "votes": [],
        "window_sec": 1.2,
        "min_score": 0,
        "min_margin": 0,
        "aggregator_node": "master",
        "aggregator_label": "real_master_esp32",
        "simulated_master": False,
        "aggregator_mode": mode,
        "packets_seen": packets,
        "packets_rejected": rejected,
        "sample_idx": sample_idx,
    }


def normalize_node_event(evt: dict, raw_line: str) -> dict:
    out = dict(evt)
    out["host_time"] = utc_now_iso()
    out["raw_line"]  = raw_line.rstrip("\r\n")
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Demux master ESP32 Serial into per-node + fusion JSONL streams.")
    p.add_argument("--port", required=True, help="Serial port of the master, e.g. COM7")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--telemetry-dir", type=Path, default=TELEMETRY_DIR,
                   help="Where to write per-node events.jsonl (one subdir per node)")
    p.add_argument("--fusion-dir", type=Path, default=FUSION_DIR,
                   help="Where to write fusion decisions and master state")
    p.add_argument("--echo", action="store_true", help="Echo serial lines to stdout")
    p.add_argument("--remote-url", default="",
                   help="If set, POST events to this dashboard's /api/ingest "
                        "instead of writing local JSONL.")
    p.add_argument("--remote-token", default="",
                   help="Optional shared token for X-Hash-KWS-Token header")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if serial is None:
        print("pyserial is required. Install with: pip install pyserial", file=sys.stderr)
        return 2

    raw_path = args.fusion_dir / "master_demux_raw.log"
    state_path = args.fusion_dir / "state.json"

    state = {
        "updated_at": utc_now_iso(),
        "mode": "master_demux_bridge",
        "port": args.port,
        "fusion_count": 0,
        "events_per_node": {"1": 0, "2": 0, "3": 0},
        "rejected_lines": 0,
    }
    atomic_write_json(state_path, state)

    print(f"[demux] opening {args.port} @ {args.baud}")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except serial.SerialException as exc:
        print(f"Cannot open {args.port}: {exc}", file=sys.stderr)
        return 1

    sample_idx = 0
    try:
        while True:
            try:
                raw = ser.readline()
            except serial.SerialException as exc:
                print(f"Serial error: {exc}", file=sys.stderr)
                return 1
            if not raw:
                continue
            line = raw.decode("utf-8", "replace")
            append_text(raw_path, line)
            if args.echo:
                sys.stdout.write(line)
                sys.stdout.flush()

            evt = parse_event(line)
            if evt is None:
                continue
            kind = str(evt.get("kind", ""))
            node_field = evt.get("node", "")
            node_str = str(node_field).strip()

            if kind == "fusion":
                decision = fusion_to_decision(evt, sample_idx)
                if args.remote_url:
                    ok, err = post_event_remote(
                        args.remote_url, "fusion", decision, token=args.remote_token,
                    )
                    if not ok:
                        sys.stderr.write(f"[demux] remote fusion ingest failed: {err}\n")
                else:
                    append_jsonl(args.fusion_dir / "decisions.jsonl", decision)
                state["fusion_count"] = int(state.get("fusion_count", 0)) + 1
                state["last_fusion"] = decision
                sample_idx += 1
            elif kind in NODE_KIND_TO_STREAM and node_str.isdigit():
                node_id = int(node_str)
                event_payload = normalize_node_event(evt, line)
                if args.remote_url:
                    stream_name = f"node{node_id}"
                    ok, err = post_event_remote(
                        args.remote_url, stream_name, event_payload, token=args.remote_token,
                    )
                    if not ok:
                        sys.stderr.write(f"[demux] remote {stream_name} ingest failed: {err}\n")
                else:
                    target = args.telemetry_dir / f"node{node_id}" / "events.jsonl"
                    append_jsonl(target, event_payload)
                key = str(node_id)
                state["events_per_node"][key] = state["events_per_node"].get(key, 0) + 1
            elif kind in ("boot", "espnow") and node_str == "master":
                state[f"last_{kind}"] = evt
            else:
                # Unrecognised line shape — keep counter for diagnostics only.
                state["rejected_lines"] = int(state.get("rejected_lines", 0)) + 1

            state["updated_at"] = utc_now_iso()
            atomic_write_json(state_path, state)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
