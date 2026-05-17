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

# Score floor for promoting an inference event (kind=infer or kind=episode)
# into the recognised-commands history.
INFER_COMMAND_MIN_SCORE: int = 140

# Master relays packets to Serial as kind=episode/infer/emit. We treat the
# first two as the same data source for latency and command extraction.
LATENCY_KINDS: frozenset[str] = frozenset({"infer", "episode"})

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
    fusion_decisions: list[dict[str, Any]] | None = None,
    limit: int = 12,
    window_ms: int = COMMAND_DEDUP_WINDOW_MS,
) -> list[dict[str, Any]]:
    """Return the most recent recognised commands, newest first.

    Source preference:
      1. ``fusion_decisions`` (master's audio_fusion_agree) when provided —
         this is the *real* system output: per-node guesses are diagnostic,
         only fusion is what the demo claims to recognise.
      2. Falls back to per-node infer/episode/emit events when no fusion is
         available (e.g. single-board dev mode without master).

    Filters out noise labels (unknown / silence). Multiple consecutive
    same-label detections within ``window_ms`` are merged into one entry
    with a ``count`` multiplier and max score across the window.
    """

    if fusion_decisions:
        return _commands_from_fusion(fusion_decisions, limit=limit, window_ms=window_ms)
    return _commands_from_node_events(node_events, limit=limit, window_ms=window_ms)


def _commands_from_fusion(
    fusion_decisions: list[dict[str, Any]],
    *,
    limit: int,
    window_ms: int,
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    latest_by_label: dict[str, dict[str, Any]] = {}

    decisions = sorted(fusion_decisions, key=lambda d: d.get("host_time") or "")
    for decision in decisions:
        if decision.get("kind") != "audio_fusion_agree":
            continue
        label = decision.get("label")
        if not label or label in NOISE_LABELS:
            continue
        score = _safe_int(
            decision.get("score_avg")
            if decision.get("score_avg") is not None
            else decision.get("score_sum"),
        )
        margin = _safe_int(decision.get("margin"))
        host_time = decision.get("host_time")
        ts_ms = _parse_ts_ms(host_time)
        last = latest_by_label.get(label)
        if (
            last is not None
            and ts_ms is not None
            and last["last_ts"] is not None
            and (ts_ms - last["last_ts"]) <= window_ms
        ):
            last["count"] += 1
            last["last_ts"] = ts_ms
            last["host_time"] = host_time
            if score > last["score"]:
                last["score"] = score
        else:
            entry = {
                "label": label,
                "score": score,
                "margin": margin,
                "node": "master",
                "host_time": host_time,
                "source": "fusion",
                "mode": "fusion",
                "count": 1,
                "voters": len(decision.get("nodes") or []),
                "first_ts": ts_ms,
                "last_ts": ts_ms,
            }
            groups.append(entry)
            latest_by_label[label] = entry

    groups.sort(key=lambda g: g["last_ts"] if g["last_ts"] is not None else -1,
                reverse=True)
    out: list[dict[str, Any]] = []
    for g in groups[:limit]:
        out.append({
            "label": g["label"],
            "score": g["score"],
            "margin": g.get("margin", 0),
            "node": g["node"],
            "host_time": g["host_time"],
            "source": g["source"],
            "mode": g["mode"],
            "count": g["count"],
            "voters": g.get("voters", 0),
        })
    return out


def _commands_from_node_events(
    node_events: dict[str, list[dict[str, Any]]],
    *,
    limit: int,
    window_ms: int,
) -> list[dict[str, Any]]:
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
        elif kind in LATENCY_KINDS:
            label = event.get("top1") or event.get("best")
            if not label or label in NOISE_LABELS:
                continue
            score_raw = event.get("top1_score")
            if score_raw is None:
                score_raw = event.get("best_score")
            score = _safe_int(score_raw)
            if score < INFER_COMMAND_MIN_SCORE:
                continue
            source = kind
            mode = kind
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
    """Aggregate invoke_ms across recent kind=infer / kind=episode events."""
    infer_events: list[dict[str, Any]] = [
        e for e in _iter_all_events(node_events) if e.get("kind") in LATENCY_KINDS
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


def compute_per_node_latency(
    node_events: dict[str, list[dict[str, Any]]],
    *,
    series_cap: int = 30,
) -> dict[str, dict[str, Any]]:
    """Return per-node invoke_ms stats keyed by node_key.

    Each value mirrors compute_latency_stats fields so the UI can render a
    sparkline + min/p50/p95/max line per node card.
    """
    out: dict[str, dict[str, Any]] = {}
    for node_key, events in node_events.items():
        infer_events = [e for e in events if e.get("kind") in LATENCY_KINDS]
        infer_events.sort(key=lambda e: e.get("host_time") or "")
        series: list[int] = []
        for event in infer_events[-series_cap:]:
            value = event.get("invoke_ms")
            try:
                series.append(int(float(value)))
            except (TypeError, ValueError):
                continue
        if not series:
            continue
        sorted_values = sorted(series)
        count = len(sorted_values)

        def pct(p: float, sv: list[int] = sorted_values, n: int = count) -> float:
            idx = max(0, min(n - 1, int(round(p * (n - 1)))))
            return float(sv[idx])

        out[node_key] = {
            "count": count,
            "min_ms": float(sorted_values[0]),
            "p50_ms": pct(0.5),
            "p95_ms": pct(0.95),
            "max_ms": float(sorted_values[-1]),
            "avg_ms": sum(sorted_values) / count,
            "series_ms": series,
        }
    return out


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
        if event.get("kind") not in {"activity", "infer", "episode"}:
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
    fusion_decisions = snapshot.get("fusion_decisions") or []
    snapshot["commands"] = extract_commands(
        node_events, fusion_decisions=fusion_decisions,
    )
    snapshot["invoke_latency"] = compute_latency_stats(node_events)
    snapshot["per_node_latency"] = compute_per_node_latency(node_events)
    snapshot["activity_pulse"] = activity_pulse(node_events)
    return snapshot
