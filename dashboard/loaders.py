"""Read-only loaders for the dashboard.

The dashboard must cope with three realities:

1. JSONL files may not exist yet (fresh checkout, nothing emitted).
2. Files are appended by long-running producers while we read.
3. Lines may be partially written or corrupted — we tolerate and skip.

All helpers are synchronous. The FastAPI layer wraps them in
``asyncio.to_thread`` so the event loop is never blocked.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import enrich, paths


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------


def load_json_file(path: Path) -> dict[str, Any] | None:
    """Read a JSON file, returning ``None`` if the file is missing or invalid.

    State files are tiny (hundreds of bytes); no streaming needed.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        # Producers rewrite state.json atomically, but an in-flight swap is
        # possible — treat it as "no data this tick".
        return None


def tail_jsonl(path: Path, limit: int = 25) -> list[dict[str, Any]]:
    """Return up to ``limit`` most recent JSON objects from a JSONL file.

    Uses a reverse block-wise read so we do not load the whole file for live
    tailing. Lines that fail to decode are skipped silently.
    """
    if limit <= 0:
        return []
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return []
    except OSError:
        return []
    if size == 0:
        return []

    chunk_size = 64 * 1024
    collected: list[bytes] = []
    leftover = b""
    try:
        with path.open("rb") as fh:
            pos = size
            while pos > 0 and len(collected) <= limit:
                read_size = min(chunk_size, pos)
                pos -= read_size
                fh.seek(pos)
                chunk = fh.read(read_size) + leftover
                lines = chunk.split(b"\n")
                # The earliest fragment in a chunk might be an incomplete line;
                # stash it so it is prepended next iteration.
                leftover = lines[0] if pos > 0 else b""
                usable = lines[1:] if pos > 0 else lines
                # Walk from newest to oldest within this chunk.
                for raw in reversed(usable):
                    if not raw.strip():
                        continue
                    collected.append(raw)
                    if len(collected) >= limit:
                        break
    except OSError:
        return []

    events: list[dict[str, Any]] = []
    # ``collected`` is newest-first — decode then flip so the UI gets
    # chronological order.
    for raw in collected:
        try:
            events.append(json.loads(raw.decode("utf-8", errors="replace")))
        except json.JSONDecodeError:
            continue
    events.reverse()
    return events[-limit:]


# ---------------------------------------------------------------------------
# Per-node aggregation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class NodeSummary:
    """Compact view of a single node used by the snapshot endpoint."""

    key: str
    role: str
    label: str
    last_event_host_time: str | None
    last_event_age_sec: float | None
    last_kind: str | None
    speech: bool | None
    recent_max: int | None
    latest_top1: str | None
    latest_top1_score: int | None
    tx_ok: int | None
    tx_fail: int | None
    events_tail: list[dict[str, Any]]
    online: bool  # derived from last_event_age_sec


def _derive_node_summary(
    node_key: str,
    events: list[dict[str, Any]],
    *,
    online_threshold_sec: float = 8.0,
) -> NodeSummary:
    meta = paths.NODE_ROLES.get(node_key, {"role": "unknown", "label": node_key})
    last_event_host_time: str | None = None
    last_event_age_sec: float | None = None
    last_kind: str | None = None
    speech: bool | None = None
    recent_max: int | None = None
    latest_top1: str | None = None
    latest_top1_score: int | None = None
    tx_ok: int | None = None
    tx_fail: int | None = None

    # Events come back in chronological order; iterate from newest to oldest to
    # fill fields with the most recent available data.
    for event in reversed(events):
        if last_event_host_time is None:
            last_event_host_time = event.get("host_time")
            last_kind = event.get("kind")
        if speech is None and "speech" in event:
            try:
                speech = bool(int(event.get("speech")))
            except (TypeError, ValueError):
                speech = bool(event.get("speech"))
        if recent_max is None and "recent_max" in event:
            try:
                recent_max = int(event["recent_max"])
            except (TypeError, ValueError):
                pass
        if latest_top1 is None and event.get("kind") == "infer":
            latest_top1 = event.get("top1")
            try:
                latest_top1_score = int(event.get("top1_score", 0))
            except (TypeError, ValueError):
                latest_top1_score = None
        if tx_ok is None and "tx_ok" in event:
            try:
                tx_ok = int(event["tx_ok"])
            except (TypeError, ValueError):
                pass
        if tx_fail is None and "tx_fail" in event:
            try:
                tx_fail = int(event["tx_fail"])
            except (TypeError, ValueError):
                pass
        if all(
            value is not None
            for value in (
                last_event_host_time,
                speech,
                recent_max,
                latest_top1,
                tx_ok,
                tx_fail,
            )
        ):
            break

    if last_event_host_time is not None:
        last_event_age_sec = _host_time_age_sec(last_event_host_time)

    online = (
        last_event_age_sec is not None
        and last_event_age_sec <= online_threshold_sec
    )

    return NodeSummary(
        key=node_key,
        role=meta["role"],
        label=meta["label"],
        last_event_host_time=last_event_host_time,
        last_event_age_sec=last_event_age_sec,
        last_kind=last_kind,
        speech=speech,
        recent_max=recent_max,
        latest_top1=latest_top1,
        latest_top1_score=latest_top1_score,
        tx_ok=tx_ok,
        tx_fail=tx_fail,
        events_tail=events[-10:],
        online=online,
    )


def _host_time_age_sec(host_time: str) -> float | None:
    """Best-effort conversion of an ISO8601 host_time into age-in-seconds."""
    from datetime import datetime, timezone

    try:
        ts = host_time
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        parsed = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    return max(0.0, (now - parsed).total_seconds())


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------


def _per_node_events(limit_per_node: int) -> dict[str, list[dict[str, Any]]]:
    if not paths.TELEMETRY_DIR.exists():
        return {}
    nodes: dict[str, list[dict[str, Any]]] = {}
    for node_dir in sorted(paths.TELEMETRY_DIR.glob(paths.NODE_GLOB_PREFIX)):
        if not node_dir.is_dir():
            continue
        events = tail_jsonl(node_dir / "events.jsonl", limit=limit_per_node)
        # Include a per-node card only when data is actually flowing. The
        # real node (node1) is seeded unconditionally in build_snapshot, but
        # node2/node3 (emulated/master) must not clutter the single-node
        # view just because their directories linger on disk.
        if events:
            nodes[node_dir.name] = events
    return nodes


def _counters(
    node_events: dict[str, list[dict[str, Any]]],
    fusion_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    per_label: dict[str, int] = {}
    infer_total = 0
    emit_total = 0
    espnow_tx_total = 0
    for events in node_events.values():
        for event in events:
            kind = event.get("kind")
            if kind == "infer":
                infer_total += 1
                label = event.get("top1")
                if label:
                    per_label[label] = per_label.get(label, 0) + 1
            elif kind == "emit":
                emit_total += 1
            elif kind == "espnow":
                espnow_tx_total += 1

    fusion_total = len(fusion_decisions)
    agreement_count = sum(
        1 for d in fusion_decisions if d.get("kind") == "audio_fusion_agree"
    )
    agreement_rate = (
        agreement_count / fusion_total if fusion_total else None
    )

    return {
        "infer_total_tail": infer_total,
        "emit_total_tail": emit_total,
        "espnow_tx_total_tail": espnow_tx_total,
        "fusion_total_tail": fusion_total,
        "fusion_agreement_tail": agreement_count,
        "fusion_agreement_rate_tail": agreement_rate,
        "per_label_tail": per_label,
    }


def build_snapshot(
    *,
    events_limit: int = 30,
    per_node_limit: int = 40,
    fusion_limit: int = 30,
) -> dict[str, Any]:
    """Assemble the full snapshot consumed by the dashboard UI.

    The returned dict is JSON-serialisable and stable across runs — the UI
    relies on the field names listed here, so treat them as a contract.
    """
    telemetry_state = load_json_file(paths.TELEMETRY_STATE)
    fusion_state = load_json_file(paths.FUSION_STATE)
    cluster_state = load_json_file(paths.CLUSTER_STATE)

    recent_events = tail_jsonl(paths.TELEMETRY_EVENTS, limit=events_limit)
    node_events = _per_node_events(limit_per_node=per_node_limit)
    fusion_decisions = tail_jsonl(paths.FUSION_DECISIONS, limit=fusion_limit)

    # Make sure canonical nodes show up even when no events have landed yet —
    # UI should render three cards from the very first page load.
    # Always render the real node card. Emulated (node2) and master (node3)
    # only appear when the cluster simulator is actually writing their JSONL;
    # this keeps the single-node (--no-sim / --single-node) view clean.
    node_events.setdefault("node1", [])

    # Producers sometimes write only to per-node files (e.g. when the bridge
    # is launched with --events-path pointing into notes/Journal/hash_kws_
    # telemetry/node1/events.jsonl for the cluster-sim setup). In that case
    # the merged events.jsonl stays empty but per-node files are populated.
    # Surface a merged view so the "Live inference feed" panel stays useful
    # without forcing the operator to run the bridge twice.
    if not recent_events:
        merged: list[dict[str, Any]] = []
        for events in node_events.values():
            merged.extend(events)
        merged.sort(key=lambda e: e.get("host_time") or "")
        if merged:
            recent_events = merged[-events_limit:]

    node_summaries = [
        _derive_node_summary(key, events)
        for key, events in sorted(node_events.items())
    ]

    counters = _counters(node_events, fusion_decisions)

    has_any_data = (
        bool(telemetry_state)
        or bool(recent_events)
        or any(bool(v) for v in node_events.values())
        or bool(cluster_state)
        or bool(fusion_decisions)
    )

    snapshot = {
        "generated_at": time.time(),
        "available": has_any_data,
        "telemetry_state": telemetry_state,
        "fusion_state": fusion_state,
        "cluster_state": cluster_state,
        "recent_events": recent_events,
        "fusion_decisions": fusion_decisions,
        "nodes": [_node_summary_as_dict(n) for n in node_summaries],
        "counters": counters,
    }
    return enrich.enrich_snapshot(snapshot, node_events)


def _node_summary_as_dict(n: NodeSummary) -> dict[str, Any]:
    return {
        "key": n.key,
        "role": n.role,
        "label": n.label,
        "last_event_host_time": n.last_event_host_time,
        "last_event_age_sec": n.last_event_age_sec,
        "last_kind": n.last_kind,
        "speech": n.speech,
        "recent_max": n.recent_max,
        "latest_top1": n.latest_top1,
        "latest_top1_score": n.latest_top1_score,
        "tx_ok": n.tx_ok,
        "tx_fail": n.tx_fail,
        "events_tail": n.events_tail,
        "online": n.online,
    }
