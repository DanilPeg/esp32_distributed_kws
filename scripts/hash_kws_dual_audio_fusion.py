from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TELEMETRY_DIR = ROOT / "notes" / "Journal" / "hash_kws_telemetry"
DEFAULT_FUSION_DIR = ROOT / "notes" / "Journal" / "hash_kws_fusion"
IGNORED_LABELS = {"silence", "unknown", ""}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_iso(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return time.time()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # On Windows, Path.replace() fails with PermissionError when the target is
    # concurrently held open for reading (the live dashboard polls state files
    # every second). Retry a few times with tiny backoffs before giving up.
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            tmp_path.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            # ~10 ms, 20 ms, 40 ms, 80 ms, 160 ms, 320 ms: total <0.7 s worst case.
            import time as _time
            _time.sleep(0.01 * (2 ** attempt))
    # Final fallback: write directly (non-atomic) so the producer never crashes
    # a long-running watch loop because of a transient reader lock.
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        try:
            tmp_path.unlink()
        except OSError:
            pass
    except OSError as exc:
        # Re-raise the original PermissionError so the operator sees the real
        # cause, not the fallback's symptom.
        raise last_error if last_error is not None else exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.lstrip("\ufeff").strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def event_node(event: dict[str, Any], fallback: str) -> str:
    value = event.get("node", event.get("node_id", fallback))
    return str(value)


def event_candidate(event: dict[str, Any], fallback_node: str) -> dict[str, Any] | None:
    kind = str(event.get("kind", ""))
    label = ""
    score = 0
    margin = 0
    mode = kind
    if kind == "emit":
        label = str(event.get("label", ""))
        score = int(event.get("score", 0) or 0)
        mode = str(event.get("mode", "emit"))
    elif kind == "infer":
        label = str(event.get("top1", ""))
        score = int(event.get("top1_score", 0) or 0)
        top2 = int(event.get("top2_score", 0) or 0)
        margin = score - top2
        mode = "infer"
    elif kind == "episode" and str(event.get("phase", "")) == "end":
        label = str(event.get("best", ""))
        score = int(event.get("best_score", 0) or 0)
        mode = "episode_end"
    else:
        return None

    if label in IGNORED_LABELS:
        return None

    host_time = str(event.get("host_time", utc_now_iso()))
    return {
        "node": event_node(event, fallback_node),
        "kind": kind,
        "mode": mode,
        "label": label,
        "score": score,
        "margin": margin,
        "host_time": host_time,
        "host_ts": parse_iso(host_time),
        "device_t": event.get("t"),
        "raw_event": event,
    }


@dataclass(frozen=True)
class FusionConfig:
    window_sec: float = 0.75
    min_score: int = 145
    min_margin: int = 12
    require_agreement: bool = True
    allow_single_node_fallback: bool = True


def fuse_candidates(
    candidates: Iterable[dict[str, Any]],
    config: FusionConfig,
) -> list[dict[str, Any]]:
    ordered = sorted(candidates, key=lambda item: float(item["host_ts"]))
    decisions: list[dict[str, Any]] = []
    used: set[int] = set()
    for index, first in enumerate(ordered):
        if index in used:
            continue
        if int(first["score"]) < config.min_score:
            continue
        window = [
            (other_index, other)
            for other_index, other in enumerate(ordered)
            if other_index not in used
            and abs(float(other["host_ts"]) - float(first["host_ts"])) <= config.window_sec
        ]
        by_node: dict[str, dict[str, Any]] = {}
        for other_index, other in window:
            node = str(other["node"])
            if node not in by_node or int(other["score"]) > int(by_node[node]["score"]):
                by_node[node] = other
        votes: dict[str, list[dict[str, Any]]] = {}
        for other in by_node.values():
            if int(other["score"]) < config.min_score:
                continue
            if str(other["kind"]) == "infer" and int(other.get("margin", 0)) < config.min_margin:
                continue
            votes.setdefault(str(other["label"]), []).append(other)

        best_label = ""
        best_votes: list[dict[str, Any]] = []
        best_score_sum = -1
        for label, label_votes in votes.items():
            score_sum = sum(int(item["score"]) for item in label_votes)
            if len(label_votes) > len(best_votes) or (
                len(label_votes) == len(best_votes) and score_sum > best_score_sum
            ):
                best_label = label
                best_votes = label_votes
                best_score_sum = score_sum

        if not best_votes:
            continue

        agreed = len(best_votes) >= 2
        if config.require_agreement and not agreed and not config.allow_single_node_fallback:
            continue

        decision_kind = "audio_fusion_agree" if agreed else "audio_single_node_fallback"
        if config.require_agreement and not agreed:
            decision_kind = "audio_waiting_for_peer"

        decision = {
            "kind": decision_kind,
            "host_time": utc_now_iso(),
            "label": best_label,
            "score_sum": best_score_sum,
            "score_avg": round(best_score_sum / max(len(best_votes), 1), 3),
            "nodes": [str(item["node"]) for item in best_votes],
            "votes": best_votes,
            "window_sec": config.window_sec,
            "min_score": config.min_score,
            "min_margin": config.min_margin,
        }
        decisions.append(decision)
        vote_ids = {id(item) for item in best_votes}
        for other_index, other in window:
            if id(other) in vote_ids:
                used.add(other_index)
    return decisions


def collect_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index, path in enumerate(paths, start=1):
        fallback_node = str(index)
        for event in read_jsonl(path):
            candidate = event_candidate(event, fallback_node)
            if candidate is not None:
                candidates.append(candidate)
    return candidates


def run_once(args: argparse.Namespace) -> int:
    config = FusionConfig(
        window_sec=args.window_ms / 1000.0,
        min_score=args.min_score,
        min_margin=args.min_margin,
        require_agreement=args.require_agreement,
        allow_single_node_fallback=args.allow_single_node_fallback,
    )
    candidates = collect_candidates(args.events)
    decisions = fuse_candidates(candidates, config)
    state = {
        "updated_at": utc_now_iso(),
        "input_events": [str(path) for path in args.events],
        "candidate_count": len(candidates),
        "decision_count": len(decisions),
        "last_decision": decisions[-1] if decisions else None,
        "config": config.__dict__,
    }
    atomic_write_json(args.state_path, state)
    for decision in decisions:
        append_jsonl(args.decisions_path, decision)
    if args.print_decisions:
        for decision in decisions:
            print(json.dumps(decision, ensure_ascii=False))
    return 0 if decisions else 1


def run_demo() -> int:
    base = datetime.now(timezone.utc)
    demo_events = [
        {
            "kind": "emit",
            "node": "1",
            "label": "yes",
            "score": 183,
            "host_time": base.isoformat(),
        },
        {
            "kind": "emit",
            "node": "2",
            "label": "yes",
            "score": 174,
            "host_time": base.isoformat(),
        },
        {
            "kind": "emit",
            "node": "1",
            "label": "no",
            "score": 170,
            "host_time": base.isoformat(),
        },
        {
            "kind": "emit",
            "node": "2",
            "label": "go",
            "score": 171,
            "host_time": base.isoformat(),
        },
    ]
    candidates = [
        event_candidate(event, str(index))
        for index, event in enumerate(demo_events, start=1)
    ]
    decisions = fuse_candidates(
        [candidate for candidate in candidates if candidate is not None],
        FusionConfig(require_agreement=True, allow_single_node_fallback=False),
    )
    print(json.dumps(decisions, ensure_ascii=False, indent=2))
    return 0 if decisions and decisions[0]["label"] == "yes" else 1


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fuse two audio KWS telemetry streams into distributed decisions."
    )
    parser.add_argument(
        "--events",
        type=Path,
        nargs="+",
        default=[
            DEFAULT_TELEMETRY_DIR / "node1" / "events.jsonl",
            DEFAULT_TELEMETRY_DIR / "node2" / "events.jsonl",
        ],
        help="Input node events JSONL files produced by hash_kws_serial_bridge.py.",
    )
    parser.add_argument(
        "--decisions-path",
        type=Path,
        default=DEFAULT_FUSION_DIR / "decisions.jsonl",
        help="Where to append fused decisions.",
    )
    parser.add_argument(
        "--state-path",
        type=Path,
        default=DEFAULT_FUSION_DIR / "state.json",
        help="Where to write latest fusion state.",
    )
    parser.add_argument("--window-ms", type=int, default=750)
    parser.add_argument("--min-score", type=int, default=145)
    parser.add_argument("--min-margin", type=int, default=12)
    parser.add_argument("--require-agreement", action="store_true", default=True)
    parser.add_argument(
        "--allow-single-node-fallback",
        action="store_true",
        help="Emit a fallback decision when only one node has a strong candidate.",
    )
    parser.add_argument("--print-decisions", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Run an in-memory smoke demo.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.demo:
        return run_demo()
    return run_once(args)


if __name__ == "__main__":
    raise SystemExit(main())
