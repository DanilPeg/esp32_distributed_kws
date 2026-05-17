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
            if kind in {"infer", "episode"}:
                infer_total += 1
                label = event.get("top1") or event.get("best")
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


def _build_camera_summary(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Squash recent camera-link events into a single dashboard-card payload.

    Walks the tail of `hash_kws_camera/events.jsonl` (newest last). Extracts:
      - last_trigger:   most recent kind=cam_trigger (label, trigger_id, age)
      - last_reply:     most recent kind=cam_reply   (label, score, latency, age)
      - counters:       triggers_total, replies_total, stale_total (over tail)
      - last_pair:      whether last_reply belongs to last_trigger
    Empty events → all-None payload (template hides empty card if needed).
    """
    summary: dict[str, Any] = {
        "available": bool(events),
        "trigger_word": None,
        "last_trigger": None,
        "last_reply": None,
        "last_status": None,
        "triggers_total_tail": 0,
        "replies_total_tail": 0,
        "stale_total_tail": 0,
        "status_total_tail": 0,
    }
    if not events:
        return summary

    last_trigger: dict[str, Any] | None = None
    last_reply: dict[str, Any] | None = None
    last_status: dict[str, Any] | None = None
    for event in events:
        kind = event.get("kind")
        if kind == "cam_trigger":
            summary["triggers_total_tail"] += 1
            last_trigger = event
            tw = event.get("trigger_label")
            if tw is not None:
                summary["trigger_word"] = tw
        elif kind == "cam_reply":
            summary["replies_total_tail"] += 1
            try:
                if int(event.get("stale", 0)) == 1:
                    summary["stale_total_tail"] += 1
            except (TypeError, ValueError):
                pass
            last_reply = event
        elif kind == "cam_status":
            summary["status_total_tail"] += 1
            last_status = event

    if last_trigger is not None:
        summary["last_trigger"] = {
            "trigger_id":   last_trigger.get("trigger_id"),
            "trigger_word": last_trigger.get("trigger_label"),
            "t_ms":         last_trigger.get("t_ms"),
            "host_time":    last_trigger.get("host_time"),
            "age_sec":      _host_time_age_sec(last_trigger.get("host_time") or ""),
        }
    if last_reply is not None:
        latency = last_reply.get("latency_ms")
        score   = last_reply.get("score")
        try:
            latency_int = int(latency) if latency is not None else None
        except (TypeError, ValueError):
            latency_int = None
        try:
            score_float = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_float = None
        summary["last_reply"] = {
            "trigger_id": last_reply.get("trigger_id"),
            "label":      last_reply.get("label"),
            "score":      score_float,
            "status":     last_reply.get("status"),
            "latency_ms": latency_int,
            "fb_ms":      last_reply.get("fb_ms"),
            "invoke_ms":  last_reply.get("invoke_ms"),
            "stale":      last_reply.get("stale"),
            "host_time":  last_reply.get("host_time"),
            "age_sec":    _host_time_age_sec(last_reply.get("host_time") or ""),
        }
    if last_trigger is not None and last_reply is not None:
        summary["last_pair_matches"] = (
            last_trigger.get("trigger_id") == last_reply.get("trigger_id")
        )
    if last_status is not None:
        # Numeric fields come in as int from bridge; keep them as-is and just
        # surface what the UI needs.
        summary["last_status"] = {
            "channel":           last_status.get("channel"),
            "uptime_ms":         last_status.get("uptime_ms"),
            "packets_seen":      last_status.get("packets_seen"),
            "triggers_received": last_status.get("triggers_received"),
            "triggers_rejected": last_status.get("triggers_rejected"),
            "inferences_done":   last_status.get("inferences_done"),
            "replies_sent":      last_status.get("replies_sent"),
            "replies_failed":    last_status.get("replies_failed"),
            "free_heap_kb":      last_status.get("free_heap_kb"),
            "psram_free_kb":     last_status.get("psram_free_kb"),
            "host_time":         last_status.get("host_time"),
            "age_sec":           _host_time_age_sec(last_status.get("host_time") or ""),
        }
    return summary


def build_snapshot(
    *,
    events_limit: int = 30,
    per_node_limit: int = 40,
    fusion_limit: int = 30,
    camera_limit: int = 30,
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
    camera_events = tail_jsonl(paths.CAMERA_EVENTS, limit=camera_limit)
    camera_summary = _build_camera_summary(camera_events)

    # Always render the three canonical inference node cards (node1/2/3)
    # from the very first page load, even before any events have arrived.
    # Empty cards show "no inference yet" and stay offline until the
    # demux bridge starts pumping data.
    for canonical_key in paths.NODE_DISPLAY_ORDER:
        node_events.setdefault(canonical_key, [])

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

    # Render in canonical order (node1, node2, node3) first, then any
    # non-canonical nodes (future expansion).
    ordered_keys = list(paths.NODE_DISPLAY_ORDER) + sorted(
        k for k in node_events.keys() if k not in paths.NODE_DISPLAY_ORDER
    )
    node_summaries = [
        _derive_node_summary(key, node_events.get(key) or [])
        for key in ordered_keys
        if key in node_events
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
        "camera": camera_summary,
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


# ---------------------------------------------------------------------------
# Wire-format snapshot — same JSON contract as the on-board master's
# WebSocket messages. Lets the FastAPI dashboard reuse the master's
# UI shell verbatim with only the transport changed (SSE polling instead
# of WebSocket).
# ---------------------------------------------------------------------------

_LABEL_NAMES_KWS12 = (
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "unknown", "silence",
)


def _label_name_to_idx(name: str | None) -> int | None:
    if name is None:
        return None
    try:
        return _LABEL_NAMES_KWS12.index(name)
    except ValueError:
        return None


def _node_lat_stats(events: list[dict[str, Any]]) -> dict[str, int]:
    """Build an on-board-style {min,med,p95,max,count} from recent infer events."""
    invokes: list[int] = []
    for e in events:
        if e.get("kind") != "infer":
            continue
        v = e.get("invoke_ms")
        try:
            ms = int(v) if v is not None else None
        except (TypeError, ValueError):
            ms = None
        if ms is not None and ms > 0:
            invokes.append(ms)
    if not invokes:
        return {"min": 0, "med": 0, "p95": 0, "max": 0, "count": 0}
    invokes.sort()
    n = len(invokes)
    p95_idx = min(n - 1, (n * 95) // 100)
    return {
        "min": invokes[0],
        "med": invokes[n // 2],
        "p95": invokes[p95_idx],
        "max": invokes[-1],
        "count": n,
    }


def _wire_node(node_key: str, events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reshape per-node JSONL tail into the on-board master's WS `node` payload."""
    node_id = 0
    if node_key.startswith("node"):
        try:
            node_id = int(node_key[len("node"):])
        except ValueError:
            node_id = 0
    last_label: int | None = None
    last_score = 0
    last_margin = 0
    last_invoke = 0
    last_recent_max = 0
    last_kind = 0
    last_seq = 0
    for e in reversed(events):
        if last_label is None:
            label_idx = _label_name_to_idx(e.get("top1"))
            if label_idx is not None:
                last_label = label_idx
                try:
                    last_score = int(e.get("top1_score", 0))
                except (TypeError, ValueError):
                    last_score = 0
                try:
                    last_margin = int(e.get("margin", 0))
                except (TypeError, ValueError):
                    last_margin = 0
                try:
                    last_invoke = int(e.get("invoke_ms", 0))
                except (TypeError, ValueError):
                    last_invoke = 0
                try:
                    last_recent_max = int(e.get("recent_max", 0))
                except (TypeError, ValueError):
                    last_recent_max = 0
                try:
                    last_seq = int(e.get("seq", 0))
                except (TypeError, ValueError):
                    last_seq = 0
                kind_str = e.get("kind", "infer")
                last_kind = {"infer": 0, "episode": 1, "emit": 2}.get(kind_str, 0)
                break
    # Age of the freshest event in the tail (any kind), so the UI can compute
    # `lastMs = Date.now() - last_age_ms` and let tiles correctly transition
    # online → stale → offline as data goes silent. Without this the snapshot
    # replay every 1 s would forever pin `lastMs` to "now".
    last_age_ms = 0
    if events:
        age_sec = _host_time_age_sec(events[-1].get("host_time") or "")
        if age_sec is not None:
            last_age_ms = int(age_sec * 1000)
    return {
        "node":         node_id,
        "label":        last_label if last_label is not None else 11,  # 11=silence
        "score":        last_score,
        "margin":       last_margin,
        "packets":      len(events),
        "invoke_ms":    last_invoke,
        "kind":         last_kind,
        "recent_max":   last_recent_max,
        "seq":          last_seq,
        "lat":          _node_lat_stats(events),
        "last_age_ms":  last_age_ms,
    }


def _wire_fusion_entry(d: dict[str, Any]) -> dict[str, Any]:
    """Reshape one fusion decision JSONL entry into on-board master's WS `fusion`."""
    label_idx = _label_name_to_idx(d.get("label"))
    if label_idx is None:
        label_idx = 11  # 11=silence
    try:
        score = int(d.get("score_sum") or d.get("score_avg") or 0)
    except (TypeError, ValueError):
        score = 0
    try:
        margin = int(d.get("margin", 0))
    except (TypeError, ValueError):
        margin = 0
    nodes_list = d.get("nodes") or d.get("votes") or []
    if not isinstance(nodes_list, list):
        nodes_list = []
    voters = len(nodes_list)
    mode = int(d.get("aggregator_mode", 0) or 0)
    mode_name = ("mean_logits", "temperature_scaled", "learned_weights")
    mode_str = mode_name[mode] if 0 <= mode < 3 else "mean_logits"
    # Age in ms — UI sets `at = Date.now() - age_ms` so the hero correctly
    # fades after HERO_FADE_MS instead of pinning "last" to the current tick.
    age_ms = 0
    age_sec = _host_time_age_sec(d.get("host_time") or "")
    if age_sec is not None:
        age_ms = int(age_sec * 1000)
    return {
        "label":   label_idx,
        "score":   score,
        "margin":  margin,
        "voters":  voters,
        "time_ms": 0,
        "mode":    mode,
        "mode_name": mode_str,
        "age_ms":  age_ms,
    }


def _wire_counters(
    node_events: dict[str, list[dict[str, Any]]],
    fusion_decisions: list[dict[str, Any]],
    camera_summary: dict[str, Any],
    process_start_sec: float,
) -> dict[str, Any]:
    """On-board-style counters: same field names the master's web JS expects.

    Cumulative figures (packets, rejected, fusion total, aggregator mode) come
    from the latest fusion decision — the bridge stamps each entry with the
    master's `packets_seen` / `packets_rejected` / `aggregator_mode` and a
    monotonic `sample_idx`. Falling back to tail length gives sensible 0 / N
    values when no fusion has fired yet.
    """
    audio_online = 0
    now = time.time()
    for events in node_events.values():
        if events:
            last = events[-1]
            age = _host_time_age_sec(last.get("host_time") or "")
            if age is not None and age <= 8.0:
                audio_online += 1
    video_online = 0
    last_reply = camera_summary.get("last_reply") if camera_summary else None
    if last_reply:
        ageR = last_reply.get("age_sec")
        if isinstance(ageR, (int, float)) and ageR <= 8.0:
            video_online = 1

    mode_name = ("mean_logits", "temperature_scaled", "learned_weights")
    last_decision = fusion_decisions[-1] if fusion_decisions else {}

    def _int_or(field: str, default: int = 0) -> int:
        try:
            return int(last_decision.get(field, default))
        except (TypeError, ValueError):
            return default

    packets_cum  = _int_or("packets_seen")
    rejected_cum = _int_or("packets_rejected")
    agg_mode     = _int_or("aggregator_mode")
    # Cumulative fusion count: bridge tags every decision with monotonic
    # sample_idx, so (last+1) is exact. Fallback to tail size when absent.
    sample_idx = _int_or("sample_idx", -1)
    fusion_cum = (sample_idx + 1) if sample_idx >= 0 else len(fusion_decisions)

    return {
        "fusion":         fusion_cum,
        "packets":        packets_cum,
        "rejected":       rejected_cum,
        "agg_mode":       agg_mode,
        "agg_mode_name":  mode_name[agg_mode] if 0 <= agg_mode < 3 else "mean_logits",
        "uptime_s":       int(max(0, now - process_start_sec)),
        "audio_total":    3,
        "audio_online":   audio_online,
        "video_online":   video_online,
    }


# Process start (so wire snapshot can report a sensible uptime_s).
_PROCESS_START_SEC = time.time()


def build_wire_snapshot(
    *,
    per_node_limit: int = 60,
    fusion_limit: int = 30,
    camera_limit: int = 30,
) -> dict[str, Any]:
    """Return a snapshot shaped exactly like the on-board master's WS `snapshot`
    message. The FastAPI dashboard template uses this so the UI code can be
    a near-verbatim copy of `code/firmware/hash_kws_master_web/web_page.h`."""
    node_events = _per_node_events(limit_per_node=per_node_limit)
    fusion_decisions = tail_jsonl(paths.FUSION_DECISIONS, limit=fusion_limit)
    camera_events = tail_jsonl(paths.CAMERA_EVENTS, limit=camera_limit)
    camera_summary = _build_camera_summary(camera_events)

    nodes_wire: list[dict[str, Any]] = []
    for key in paths.NODE_DISPLAY_ORDER:
        events = node_events.get(key) or []
        if not events:
            continue
        nodes_wire.append(_wire_node(key, events))

    # Newest first, like the master's ring rendering.
    fusion_wire = [_wire_fusion_entry(d) for d in reversed(fusion_decisions)]

    counters_wire = _wire_counters(
        node_events, fusion_decisions, camera_summary, _PROCESS_START_SEC
    )

    return {
        "type":      "snapshot",
        "nodes":     nodes_wire,
        "fusion":    fusion_wire,
        "counters":  counters_wire,
        "camera":    camera_summary,
    }
