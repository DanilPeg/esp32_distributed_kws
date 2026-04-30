from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hash_kws_dual_audio_fusion import (
    FusionConfig,
    append_jsonl,
    atomic_write_json,
    event_candidate,
    fuse_candidates,
    read_jsonl,
    utc_now_iso,
)


ROOT = Path(__file__).resolve().parents[2]
TELEMETRY_DIR = ROOT / "notes" / "Journal" / "hash_kws_telemetry"
FUSION_DIR = ROOT / "notes" / "Journal" / "hash_kws_fusion"
CLUSTER_DIR = ROOT / "notes" / "Journal" / "hash_kws_cluster"


COMMAND_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
IGNORED_LABELS = {"silence", "unknown", ""}


def stable_key(payload: dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def bounded_score(value: int) -> int:
    return max(0, min(255, int(value)))


def infer_label_and_score(event: dict[str, Any]) -> tuple[str, int, int]:
    kind = str(event.get("kind", ""))
    if kind == "emit":
        return str(event.get("label", "")), int(event.get("score", 0) or 0), 32
    if kind == "infer":
        top1 = str(event.get("top1", ""))
        top1_score = int(event.get("top1_score", 0) or 0)
        top2_score = int(event.get("top2_score", 0) or 0)
        return top1, top1_score, max(0, top1_score - top2_score)
    if kind == "episode" and str(event.get("phase", "")) == "end":
        return str(event.get("best", "")), int(event.get("best_score", 0) or 0), 24
    return "", 0, 0


def make_virtual_peer_event(
    source_event: dict[str, Any],
    *,
    rng: random.Random,
    agree_probability: float,
    score_jitter: int,
    node_id: str,
) -> dict[str, Any] | None:
    label, score, margin = infer_label_and_score(source_event)
    if label in IGNORED_LABELS or score <= 0:
        return None
    if rng.random() > agree_probability:
        alternatives = [item for item in COMMAND_LABELS if item != label]
        label = rng.choice(alternatives)
        score = bounded_score(score - rng.randint(10, 34))
        margin = max(8, margin - rng.randint(8, 20))
    else:
        score = bounded_score(score + rng.randint(-score_jitter, score_jitter))
        margin = max(0, margin + rng.randint(-8, 10))

    host_time = str(source_event.get("host_time") or utc_now_iso())
    return {
        "kind": "infer",
        "node": node_id,
        "node_label": "virtual_audio_peer",
        "simulated": True,
        "source_node": source_event.get("node", source_event.get("node_id", "1")),
        "source_key": stable_key(source_event),
        "host_time": host_time,
        "t": source_event.get("t"),
        "top1": label,
        "top1_score": score,
        "top2": "unknown",
        "top2_score": bounded_score(score - margin),
        "top3": "silence",
        "top3_score": max(0, bounded_score(score - margin) - 7),
        "mode": "virtual_peer_from_real",
        "raw_line": (
            f"sim_hash_evt kind=infer node={node_id} top1={label} "
            f"top1_score={score} source={source_event.get('node', '1')}"
        ),
    }


def make_scenario_event(label: str, *, node_id: str, score: int) -> dict[str, Any]:
    return {
        "kind": "infer",
        "node": node_id,
        "node_label": "virtual_audio_peer",
        "simulated": True,
        "host_time": utc_now_iso(),
        "top1": label,
        "top1_score": bounded_score(score),
        "top2": "unknown",
        "top2_score": bounded_score(score - 32),
        "top3": "silence",
        "top3_score": bounded_score(score - 44),
        "mode": "virtual_peer_scenario",
        "raw_line": f"sim_hash_evt kind=infer node={node_id} top1={label} top1_score={score}",
    }


def write_virtual_state(path: Path, *, node_id: str, event: dict[str, Any] | None, count: int) -> None:
    state = {
        "updated_at": utc_now_iso(),
        "node_id": node_id,
        "node_label": "virtual_audio_peer",
        "simulated": True,
        "generated_events": count,
        "last_event": event,
    }
    atomic_write_json(path, state)


def decision_key(decision: dict[str, Any]) -> str:
    votes = decision.get("votes", [])
    source_keys = [str(vote.get("raw_event", {}).get("source_key", "")) for vote in votes]
    payload = {
        "label": decision.get("label"),
        "nodes": decision.get("nodes"),
        "source_keys": source_keys,
        "score_sum": decision.get("score_sum"),
    }
    return stable_key(payload)


def run_once(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    rng = random.Random(args.seed)
    sim_state = load_json(args.sim_state_path)
    processed = set(sim_state.get("processed_source_keys", []))
    emitted_decisions = set(sim_state.get("emitted_decision_keys", []))

    real_events = read_jsonl(args.real_events)
    virtual_events = read_jsonl(args.virtual_events)
    generated: list[dict[str, Any]] = []

    for event in real_events[-args.real_scan_limit :]:
        source_key = stable_key(event)
        if source_key in processed:
            continue
        candidate = event_candidate(event, "1")
        if candidate is None:
            continue
        if int(candidate.get("score", 0)) < args.min_score:
            continue
        processed.add(source_key)
        virtual = make_virtual_peer_event(
            event,
            rng=rng,
            agree_probability=args.agree_probability,
            score_jitter=args.score_jitter,
            node_id=args.virtual_node_id,
        )
        if virtual is not None:
            append_jsonl(args.virtual_events, virtual)
            generated.append(virtual)

    if args.scenario and not real_events:
        for label in args.scenario.split(","):
            label = label.strip()
            if not label:
                continue
            virtual = make_scenario_event(
                label,
                node_id=args.virtual_node_id,
                score=args.scenario_score,
            )
            append_jsonl(args.virtual_events, virtual)
            generated.append(virtual)

    all_candidates = []
    for event in read_jsonl(args.real_events):
        candidate = event_candidate(event, "1")
        if candidate is not None:
            all_candidates.append(candidate)
    for event in read_jsonl(args.virtual_events):
        candidate = event_candidate(event, args.virtual_node_id)
        if candidate is not None:
            all_candidates.append(candidate)

    config = FusionConfig(
        window_sec=args.window_ms / 1000.0,
        min_score=args.min_score,
        min_margin=args.min_margin,
        require_agreement=True,
        allow_single_node_fallback=False,
    )
    decisions = fuse_candidates(all_candidates, config)
    new_decisions = []
    for decision in decisions:
        key = decision_key(decision)
        if key in emitted_decisions:
            continue
        emitted_decisions.add(key)
        decision["aggregator_node"] = args.master_node_id
        decision["aggregator_label"] = "virtual_master"
        decision["simulated_master"] = True
        append_jsonl(args.decisions_path, decision)
        append_jsonl(
            args.master_events,
            {
                "kind": "fusion",
                "node": args.master_node_id,
                "node_label": "virtual_master",
                "simulated": True,
                "host_time": decision.get("host_time", utc_now_iso()),
                "label": decision.get("label"),
                "score": decision.get("score_avg"),
                "nodes": decision.get("nodes"),
                "raw_line": (
                    f"sim_hash_evt kind=fusion node={args.master_node_id} "
                    f"label={decision.get('label')} nodes={','.join(decision.get('nodes', []))}"
                ),
            },
        )
        new_decisions.append(decision)

    write_virtual_state(
        args.virtual_state,
        node_id=args.virtual_node_id,
        event=generated[-1] if generated else (virtual_events[-1] if virtual_events else None),
        count=len(read_jsonl(args.virtual_events)),
    )

    cluster_state = {
        "updated_at": utc_now_iso(),
        "mode": "hybrid_simulation",
        "real_node": "1",
        "virtual_peer_node": args.virtual_node_id,
        "virtual_master_node": args.master_node_id,
        "real_events_path": str(args.real_events),
        "virtual_events_path": str(args.virtual_events),
        "master_events_path": str(args.master_events),
        "decisions_path": str(args.decisions_path),
        "generated_peer_events": len(generated),
        "new_decisions": len(new_decisions),
        "last_generated_peer_event": generated[-1] if generated else None,
        "last_decision": new_decisions[-1] if new_decisions else None,
        "config": {
            "agree_probability": args.agree_probability,
            "window_ms": args.window_ms,
            "min_score": args.min_score,
            "min_margin": args.min_margin,
        },
    }
    atomic_write_json(args.cluster_state, cluster_state)

    sim_state = {
        "updated_at": utc_now_iso(),
        "processed_source_keys": sorted(processed),
        "emitted_decision_keys": sorted(emitted_decisions),
    }
    atomic_write_json(args.sim_state_path, sim_state)
    return len(new_decisions), cluster_state


def watch(args: argparse.Namespace) -> int:
    print("hash_kws_cluster_sim started", flush=True)
    while True:
        new_count, state = run_once(args)
        if args.print_state and (new_count or args.verbose):
            print(json.dumps(state, ensure_ascii=False), flush=True)
        time.sleep(args.poll_sec)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate missing audio KWS nodes and a master aggregator around one real ESP32 node."
    )
    parser.add_argument(
        "--real-events",
        type=Path,
        default=TELEMETRY_DIR / "node1" / "events.jsonl",
    )
    parser.add_argument(
        "--virtual-events",
        type=Path,
        default=TELEMETRY_DIR / "node2" / "events.jsonl",
    )
    parser.add_argument(
        "--virtual-state",
        type=Path,
        default=TELEMETRY_DIR / "node2" / "state.json",
    )
    parser.add_argument(
        "--master-events",
        type=Path,
        default=TELEMETRY_DIR / "node3" / "events.jsonl",
    )
    parser.add_argument(
        "--decisions-path",
        type=Path,
        default=FUSION_DIR / "decisions.jsonl",
    )
    parser.add_argument(
        "--cluster-state",
        type=Path,
        default=CLUSTER_DIR / "state.json",
    )
    parser.add_argument(
        "--sim-state-path",
        type=Path,
        default=CLUSTER_DIR / "sim_state.json",
    )
    parser.add_argument("--virtual-node-id", default="2")
    parser.add_argument("--master-node-id", default="3")
    parser.add_argument("--agree-probability", type=float, default=0.86)
    parser.add_argument("--score-jitter", type=int, default=14)
    parser.add_argument("--window-ms", type=int, default=1200)
    parser.add_argument("--min-score", type=int, default=145)
    parser.add_argument("--min-margin", type=int, default=8)
    parser.add_argument("--real-scan-limit", type=int, default=80)
    parser.add_argument("--seed", type=int, default=240424)
    parser.add_argument("--scenario", default="")
    parser.add_argument("--scenario-score", type=int, default=176)
    parser.add_argument("--poll-sec", type=float, default=0.5)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--print-state", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.watch:
        return watch(args)
    new_count, state = run_once(args)
    if args.print_state:
        print(json.dumps(state, ensure_ascii=False, indent=2))
    return 0 if new_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
