"""Tiny serial bridge for the master ESP32 in the hash ensemble demo.

The master sketch (`code/firmware/hash_kws_master/hash_kws_master.ino`)
prints lines like::

    hash_evt kind=fusion node=master label=yes score=4736 margin=2048 voters=3 mode=0 packets=14 rejected=0

The dashboard already understands the older ``kind=audio_fusion_agree``
format used by ``hash_kws_cluster_sim.py``. This bridge translates the
master's lines into that format so no dashboard changes are needed.

Usage::

    python code\\scripts\\hash_ensemble_master_bridge.py --port COMx [--echo]

Run alongside ``run_dashboard.py`` and one ``hash_kws_serial_bridge.py``
per inference node.
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

try:
    import serial  # type: ignore
except Exception:
    serial = None


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


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FUSION_PATH = ROOT / "notes" / "Journal" / "hash_kws_fusion" / "decisions.jsonl"
DEFAULT_STATE_PATH = ROOT / "notes" / "Journal" / "hash_kws_fusion" / "state.json"
DEFAULT_RAW_PATH = ROOT / "notes" / "Journal" / "hash_kws_fusion" / "master_raw.log"

LABELS_DEFAULT = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "unknown", "silence",
]

EVT_PREFIX = "hash_evt "

KEY_VAL_RE = re.compile(r"(?P<key>\w+)=(?P<val>[^\s]+)")


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


def parse_event(line: str) -> dict | None:
    idx = line.find(EVT_PREFIX)
    if idx < 0:
        return None
    payload = line[idx + len(EVT_PREFIX):].strip()
    out: dict = {}
    for m in KEY_VAL_RE.finditer(payload):
        key = m.group("key")
        val = m.group("val")
        if val.lstrip("-").isdigit():
            try:
                out[key] = int(val)
                continue
            except Exception:
                pass
        out[key] = val
    if "kind" not in out:
        return None
    return out


def fusion_to_decision(evt: dict, sample_idx: int) -> dict:
    """Translate `kind=fusion` from the master into `kind=audio_fusion_agree`."""
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
        "votes": [],  # master does not echo individual votes; deliberate
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bridge ESP32 master serial output into the dashboard's fusion stream.")
    p.add_argument("--port", required=True, help="Serial port of the master, e.g. COM7")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--decisions-path", type=Path, default=DEFAULT_FUSION_PATH,
                   help="Where to append fusion decisions JSONL")
    p.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH,
                   help="Where to write the latest fusion state JSON")
    p.add_argument("--raw-path", type=Path, default=DEFAULT_RAW_PATH,
                   help="Where to append raw serial lines")
    p.add_argument("--echo", action="store_true", help="Echo serial lines to stdout")
    p.add_argument(
        "--remote-url",
        default="",
        help=(
            "If set, POST translated fusion decisions to this dashboard's "
            "/api/ingest instead of writing local decisions.jsonl. "
            "Example: http://192.168.1.50:8765/api/ingest"
        ),
    )
    p.add_argument(
        "--remote-token",
        default="",
        help="Optional shared token sent in X-Hash-KWS-Token header",
    )
    return p


def main() -> int:
    args = build_arg_parser().parse_args()
    if serial is None:
        print("pyserial is required. Install with: pip install pyserial", file=sys.stderr)
        return 2

    ensure_parent(args.decisions_path)
    state = {
        "updated_at": utc_now_iso(),
        "mode": "real_master_esp32",
        "port": args.port,
        "fusion_count": 0,
        "last_label": None,
    }
    atomic_write_json(args.state_path, state)

    print(f"[master_bridge] opening {args.port} @ {args.baud}")
    sample_idx = 0
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except serial.SerialException as exc:
        print(f"Cannot open {args.port}: {exc}", file=sys.stderr)
        return 1

    try:
        while True:
            try:
                raw = ser.readline()
            except serial.SerialException as exc:
                print(f"Serial error: {exc}", file=sys.stderr)
                return 1
            if not raw:
                continue
            try:
                line = raw.decode("utf-8", "replace")
            except Exception:
                continue
            append_text(args.raw_path, line)
            if args.echo:
                sys.stdout.write(line)
                sys.stdout.flush()
            evt = parse_event(line)
            if evt is None:
                continue
            kind = str(evt.get("kind", ""))
            if kind == "fusion":
                decision = fusion_to_decision(evt, sample_idx)
                if args.remote_url:
                    ok, err = post_event_remote(
                        args.remote_url,
                        "fusion",
                        decision,
                        token=args.remote_token,
                    )
                    if not ok:
                        sys.stderr.write(f"[master_bridge] remote ingest failed: {err}\n")
                else:
                    append_jsonl(args.decisions_path, decision)
                state["updated_at"] = utc_now_iso()
                state["fusion_count"] = state.get("fusion_count", 0) + 1
                state["last_label"] = decision["label"]
                state["last_score"] = decision["score_sum"]
                state["last_margin"] = decision["margin"]
                state["last_voters"] = len(decision["nodes"])
                state["last_aggregator_mode"] = decision["aggregator_mode"]
                atomic_write_json(args.state_path, state)
                sample_idx += 1
            elif kind == "boot":
                state["last_boot"] = evt
                state["updated_at"] = utc_now_iso()
                atomic_write_json(args.state_path, state)
            elif kind == "espnow":
                state["last_espnow"] = evt
                state["updated_at"] = utc_now_iso()
                atomic_write_json(args.state_path, state)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
