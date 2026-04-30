"""Derived views attached to the live snapshot.

Kept in a separate module so loaders.py stays small and robust. All helpers
are pure functions over the already-collected per-node events and fusion
decisions — no IO.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable


# Labels we never show as recognised commands.
NOISE_LABELS: frozenset[str] = frozenset(
    {"silence", "_silence_", "unknown", "_unknown_"}
)

# Score floor for promoting a plain kind=infer event to the command history.
INFER_COMMAND_MIN_SCORE: int = 140

# Time window in milliseconds within which consecutive same-(node,label)
# detections are merged into one command card with a count multiplier.
# Set close to typical episode length so that talking "yes" once produces a
# single card with count=N, not N separate cards.
COMMAND_DEDUP_WINDOW_MS: int = 2000


def _iter_all_events(
    node_events: dict[str, list[dict[str, Any]]],
) -> Iterable[dict[str, Any]]:
    for events in node_events.values():
        for event in events:
            yield event


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _stringify_node(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _parse_ts_ms(host_time: Any) -> int | None:
    """Parse an ISO8601 host_time into a millisecond epoch. Returns None on
    failure so callers can fall back to looser bucketing."""
    if not isinstance(host_time, str):
        return None
    s = host_time.strip()
    if not s:
        return None
    # datetime.fromisoformat handles "Z" only on 3.11+; normalise.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    try:
        return int(dt.timestamp() * 1000)
    except (OverflowError, ValueError):
        return None


def extract_commands(
    node_events: dict[str, list[dict[str, Any]]],
    *,
    limit: int = 12,
    window_ms: int = COMMAND_DEDUP_WINDOW_MS,
) -> list[dict[str, Any]]:
    """Return the most recent recognised commands, newest first.

    Multiple consecutive same-(node, label) detections within `window_ms`
    are merged into a single entry with a `count` multiplier and the max
    score across the window. Prefers `kind=emit` source when present in
    the merged window. Filters out noise labels and low-confidence infers.
    """
    all_events: list[dict[str, Any]] = list(_iter_all_events(node_events))
    all_events.sort(key=lambda e: e.get("host_time") or "")

    groups: list[dict[str, Any]] = []
    latest_by_key: dict[tuple[str, str], dict[str, Any]] = {}

    for event in all_events:
        kind = event.get("kind")
        if kind == "emit":
            label = event.get("label")
            if not label or label in NOISE_LABELS:
                continue
            score = _safe_int(event.get("score"))
            source = "emit"
            mode = event.get("mode") or "emit"
        elif kind == "infer":
            label = event.get("top1")
            if not label or label in NOISE_LABELS:
                continue
            score = _safe_int(event.get("top1_score"))
            if score < INFER_COMMAND_MIN_SCORE:
                continue
            source = "infer"
            mode = "infer"
        else:
            continue

        node = _stringify_node(event.get("node"))
        host_time = event.get("host_time")
        ts_ms = _parse_ts_ms(host_time)
        key = (node, label)
        last = latest_by_key.get(key)
        if (
            last is not None
            and ts_ms is not None
            and last["last_ts"] is not None
            and (ts_ms - last["last_ts"]) <= window_ms
        ):
            # Merge into existing group.
            last["count"] += 1
            last["last_ts"] = ts_ms
            last["host_time"] = host_time  # newest timestamp surfaces.
            if score > last["score"]:
                last["score"] = score
            if source == "emit" and last["source"] != "emit":
                last["source"] = "emit"
                last["mode"] = mode
        else:
            entry = {
                "label": label,
                "score": score,
                "node": node,
                "host_time": host_time,
                "source": source,
                "mode": mode,
                "count": 1,
                "first_ts": ts_ms,
                "last_ts": ts_ms,
            }
            groups.append(entry)
            latest_by_key[key] = entry

    # Newest first; drop the internal ts fields before returning.
    groups.sort(key=lambda g: g["last_ts"] if g["last_ts"] is not None else -1,
                reverse=True)
    out: list[dict[str, Any]] = []
    for g in groups[:limit]:
        out.append({
            "label": g["label"],
            "score": g["score"],
            "node": g["node"],
            "host_time": g["host_time"],
            "source": g["source"],
            "mode": g["mode"],
            "count": g["count"],
        })
    return out


def compute_latency_stats(
    node_events: dict[str, list[dict[str, Any]]],
    *,
    series_cap: int = 40,
) -> dict[str, Any] | None:
    """Aggregate invoke_ms across recent kind=infer events."""
    infer_events: list[dict[str, Any]] = [
        e for e in _iter_all_events(node_events) if e.get("kind") == "infer"
    ]
    infer_events.sort(key=lambda e: e.get("host_time") or "")

    series: list[int] = []
    for event in infer_events[-series_cap:]:
        value = event.get("invoke_ms")
        try:
            series.append(int(float(value)))
        except (TypeError, ValueError):
            continue

    if not series:
        return None

    sorted_values = sorted(series)
    count = len(sorted_values)

    def pct(p: float) -> float:
        idx = max(0, min(count - 1, int(round(p * (count - 1)))))
        return float(sorted_values[idx])

    return {
        "count": count,
        "min_ms": float(sorted_values[0]),
        "p50_ms": pct(0.5),
        "p95_ms": pct(0.95),
        "max_ms": float(sorted_values[-1]),
        "avg_ms": sum(sorted_values) / count,
        "series_ms": series,
    }


def activity_pulse(
    node_events: dict[str, list[dict[str, Any]]],
    *,
    node_key: str = "node1",
    limit: int = 60,
) -> list[int]:
    """Return the last `limit` recent_max samples from the real node."""
    events = node_events.get(node_key) or []
    series: list[int] = []
    for event in events[-limit:]:
        if event.get("kind") not in {"activity", "infer"}:
            continue
        value = event.get("recent_max")
        try:
            series.append(int(float(value)))
        except (TypeError, ValueError):
            continue
    return series


def enrich_snapshot(
    snapshot: dict[str, Any],
    node_events: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Mutate and return the snapshot with derived fields added."""
    snapshot["commands"] = extract_commands(node_events)
    snapshot["invoke_latency"] = compute_latency_stats(node_events)
    snapshot["activity_pulse"] = activity_pulse(node_events)
    return snapshot
