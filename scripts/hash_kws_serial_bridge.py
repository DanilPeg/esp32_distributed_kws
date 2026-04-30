from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import serial  # type: ignore
except Exception:
    serial = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TELEMETRY_DIR = ROOT / "notes" / "Journal" / "hash_kws_telemetry"
DEFAULT_STATE_PATH = DEFAULT_TELEMETRY_DIR / "state.json"
DEFAULT_EVENTS_PATH = DEFAULT_TELEMETRY_DIR / "events.jsonl"
DEFAULT_RAW_PATH = DEFAULT_TELEMETRY_DIR / "raw.log"

HASH_EVT_PREFIX = "hash_evt "
HEARD_RE = re.compile(r"Heard\s+(?P<label>\S+)\s+\((?P<score>\d+)\)\s+@(?P<t>\d+)ms")
HASH_DBG_RE = re.compile(
    r"hash_dbg\s+t=(?P<t>\d+)\s+slices=(?P<slices>\d+)\s+invoke=(?P<invoke_ms>\d+)ms\s+"
    r"top1=(?P<top1>[^\(\s]+)\((?P<top1_score>\d+)\)\s+"
    r"top2=(?P<top2>[^\(\s]+)\((?P<top2_score>\d+)\)\s+"
    r"top3=(?P<top3>[^\(\s]+)\((?P<top3_score>\d+)\)"
)
HASH_EP_START_RE = re.compile(
    r"hash_ep\s+start\s+t=(?P<t>\d+)\s+recent_max=(?P<recent_max>\d+)"
)
HASH_EP_END_RE = re.compile(
    r"hash_ep\s+end\s+t=(?P<t>\d+)\s+dur=(?P<dur>\d+)\s+invokes=(?P<invokes>\d+)\s+"
    r"best=(?P<best>[^\(\s]+)\((?P<best_score>\d+)\)\s+peak_t=(?P<peak_t>\d+)"
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def append_text(path: Path, line: str) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def coerce_value(raw: str) -> Any:
    if raw == "":
        return raw
    if raw.lstrip("-").isdigit():
        try:
            return int(raw)
        except Exception:
            return raw
    return raw


def parse_hash_evt(line: str) -> dict[str, Any] | None:
    marker_index = line.find(HASH_EVT_PREFIX)
    if marker_index < 0:
        return None
    payload = line[marker_index + len(HASH_EVT_PREFIX) :].strip()
    event: dict[str, Any] = {}
    for token in payload.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        event[key] = coerce_value(value)
    if "kind" not in event:
        return None
    return event


def parse_legacy_line(line: str) -> dict[str, Any] | None:
    heard = HEARD_RE.search(line)
    if heard:
        return {
            "kind": "emit",
            "label": heard.group("label"),
            "score": int(heard.group("score")),
            "t": int(heard.group("t")),
            "mode": "legacy_heard",
        }

    dbg = HASH_DBG_RE.search(line)
    if dbg:
        return {
            "kind": "infer",
            "t": int(dbg.group("t")),
            "slices": int(dbg.group("slices")),
            "invoke_ms": int(dbg.group("invoke_ms")),
            "top1": dbg.group("top1"),
            "top1_score": int(dbg.group("top1_score")),
            "top2": dbg.group("top2"),
            "top2_score": int(dbg.group("top2_score")),
            "top3": dbg.group("top3"),
            "top3_score": int(dbg.group("top3_score")),
            "mode": "legacy_hash_dbg",
        }

    ep_start = HASH_EP_START_RE.search(line)
    if ep_start:
        return {
            "kind": "episode",
            "phase": "start",
            "t": int(ep_start.group("t")),
            "recent_max": int(ep_start.group("recent_max")),
            "mode": "legacy_hash_ep",
        }

    ep_end = HASH_EP_END_RE.search(line)
    if ep_end:
        return {
            "kind": "episode",
            "phase": "end",
            "t": int(ep_end.group("t")),
            "dur": int(ep_end.group("dur")),
            "invokes": int(ep_end.group("invokes")),
            "best": ep_end.group("best"),
            "best_score": int(ep_end.group("best_score")),
            "peak_t": int(ep_end.group("peak_t")),
            "mode": "legacy_hash_ep",
        }

    return None


def normalize_event(event: dict[str, Any], raw_line: str) -> dict[str, Any]:
    payload = dict(event)
    payload["host_time"] = utc_now_iso()
    payload["raw_line"] = raw_line.rstrip("\r\n")
    return payload


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "bridge_started_at": utc_now_iso(),
            "raw_lines": 0,
            "parsed_events": 0,
        }
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "bridge_started_at": utc_now_iso(),
            "raw_lines": 0,
            "parsed_events": 0,
        }


def update_state(
    state: dict[str, Any],
    *,
    port: str,
    baud: int,
    node_id: str,
    node_label: str,
    raw_line: str,
    event: dict[str, Any] | None,
) -> dict[str, Any]:
    now = utc_now_iso()
    state["port"] = port
    state["baud"] = baud
    state["node_id"] = node_id
    state["node_label"] = node_label
    state["last_seen_at"] = now
    state["last_raw_line"] = raw_line.rstrip("\r\n")
    state["raw_lines"] = int(state.get("raw_lines", 0)) + 1
    if event is None:
        return state

    state["parsed_events"] = int(state.get("parsed_events", 0)) + 1
    state["last_event"] = event
    kind = str(event.get("kind", ""))
    if kind == "ready":
        state["ready"] = event
    elif kind == "activity":
        state["activity"] = event
    elif kind == "infer":
        state["last_infer"] = event
    elif kind == "emit":
        state["last_emit"] = event
    elif kind == "episode":
        state["last_episode"] = event
    return state


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Read ESP32 hash KWS serial logs and write structured telemetry."
    )
    parser.add_argument("--port", required=True, help="Serial port, for example COM7")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument(
        "--node-id",
        default="",
        help="Logical node id to attach when firmware lines do not include node=<id>",
    )
    parser.add_argument(
        "--node-label",
        default="",
        help="Human-readable node label for state/logging",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_STATE_PATH,
        help="Where to write the latest telemetry state JSON",
    )
    parser.add_argument(
        "--events-path",
        type=Path,
        default=DEFAULT_EVENTS_PATH,
        help="Where to append parsed events as JSONL",
    )
    parser.add_argument(
        "--raw-path",
        type=Path,
        default=DEFAULT_RAW_PATH,
        help="Where to append raw serial lines",
    )
    parser.add_argument(
        "--echo",
        action="store_true",
        help="Echo serial lines to stdout while bridging",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if serial is None:
        print(
            "pyserial is required. Install it with: pip install pyserial",
            file=sys.stderr,
        )
        return 2

    state = load_state(args.state_path)
    state["bridge_started_at"] = state.get("bridge_started_at", utc_now_iso())
    if args.node_id:
        state["node_id"] = args.node_id
    if args.node_label:
        state["node_label"] = args.node_label
    atomic_write_json(args.state_path, state)

    try:
        with serial.Serial(args.port, args.baud, timeout=0.25) as ser:
            print(
                f"hash_kws_serial_bridge connected to {args.port} @ {args.baud}",
                file=sys.stderr,
            )
            while True:
                raw_bytes = ser.readline()
                if not raw_bytes:
                    continue
                line = raw_bytes.decode("utf-8", errors="replace")
                if args.echo:
                    print(line, end="")
                append_text(args.raw_path, line)

                parsed = parse_hash_evt(line)
                if parsed is None:
                    parsed = parse_legacy_line(line)
                if parsed is not None:
                    if args.node_id and "node" not in parsed and "node_id" not in parsed:
                        parsed["node"] = args.node_id
                    if args.node_label:
                        parsed["node_label"] = args.node_label
                event = normalize_event(parsed, line) if parsed else None
                if event is not None:
                    append_jsonl(args.events_path, event)
                state = update_state(
                    state,
                    port=args.port,
                    baud=args.baud,
                    node_id=args.node_id,
                    node_label=args.node_label,
                    raw_line=line,
                    event=event,
                )
                atomic_write_json(args.state_path, state)
    except KeyboardInterrupt:
        print("hash_kws_serial_bridge stopped", file=sys.stderr)
        return 0
    except Exception as exc:
        state["bridge_error_at"] = utc_now_iso()
        state["bridge_error"] = str(exc)
        atomic_write_json(args.state_path, state)
        print(f"serial bridge failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
