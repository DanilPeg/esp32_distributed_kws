from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

try:
    from .paths import CONFIG_PATH
    from .state import load_state, save_state
    from .queue import load_queue, mark_done
    from .evolution import schedule_evolution_task
    from .health import run_health_checks
    from .executor import process_queue
    from .events import append_event
    from .notify import enqueue_message, format_cycle_report
    from .utils import MSK_TZ, is_quiet_hours_msk, load_json, utc_now_iso
except ImportError:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.paths import CONFIG_PATH
    from agent.state import load_state, save_state
    from agent.queue import load_queue, mark_done
    from agent.evolution import schedule_evolution_task
    from agent.health import run_health_checks
    from agent.executor import process_queue
    from agent.events import append_event
    from agent.notify import enqueue_message, format_cycle_report
    from agent.utils import MSK_TZ, is_quiet_hours_msk, load_json, utc_now_iso


def _minutes_since(ts: str) -> float | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0
    except Exception:
        return None


def _parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _task_age_minutes(task: dict) -> float | None:
    return _minutes_since(task.get("started_at") or task.get("created_at") or "")


def _should_auto_close(task: dict, config: dict) -> bool:
    mode = str(config.get("auto_close_mode", "off")).strip().lower()
    if mode in ("", "off", "false", "0", "none"):
        return False
    codex_status = ""
    codex_meta = task.get("codex_start") or {}
    if isinstance(codex_meta, dict):
        codex_status = str(codex_meta.get("status", "")).lower()
    if mode == "prepared":
        return True
    if mode == "codex":
        return codex_status == "started"
    if mode == "codex_or_idle":
        if codex_status == "started":
            return True
        idle_min = int(config.get("auto_close_idle_minutes", 60))
        age = _task_age_minutes(task)
        return age is not None and age >= idle_min
    if mode == "idle":
        idle_min = int(config.get("auto_close_idle_minutes", 60))
        age = _task_age_minutes(task)
        return age is not None and age >= idle_min
    return False


def _auto_close_running(config: dict, queue: dict) -> list:
    closed = []
    for task in list(queue.get("running", [])):
        if _should_auto_close(task, config):
            finished = mark_done(task.get("id", ""), status="auto_closed")
            if finished:
                closed.append(finished)
                append_event("task_auto_closed", {"task_id": finished.get("id")})
    return closed


def load_config() -> Dict[str, Any]:
    return load_json(CONFIG_PATH, {})


def cycle_once() -> Dict[str, Any]:
    config = load_config()
    state = load_state()

    # Health check cadence
    health_interval = int(config.get("health_check_interval_minutes", 60))
    minutes = _minutes_since(state.get("last_health_check_at", ""))
    if minutes is None or minutes >= health_interval:
        run_health_checks()
        state["last_health_check_at"] = utc_now_iso()

    # Evolution scheduling
    scheduled = None
    if bool(config.get("evolution_enabled", False)) or bool(state.get("evolution_enabled", False)):
        evo_interval = int(config.get("evolution_interval_minutes", 240))
        evo_minutes = _minutes_since(state.get("last_evolution_at", ""))
        if evo_minutes is None or evo_minutes >= evo_interval:
            if is_quiet_hours_msk(config):
                now_msk = datetime.now(MSK_TZ).strftime("%H:%M")
                append_event(
                    "evolution_suppressed_quiet_hours",
                    {"msk_time": now_msk, "window": config.get("quiet_hours_msk", {})},
                )
            else:
                scheduled = schedule_evolution_task(config, state)

    # Optional queue processing
    auto_process = bool(config.get("auto_process_queue", False) or config.get("autonomous_apply", False))
    processed = []
    if auto_process:
        max_tasks = int(config.get("max_tasks_per_cycle", 1))
        processed = process_queue(max_tasks=max_tasks, config=config)

    # Refresh queue & auto-close running tasks if configured
    queue = load_queue()
    closed_tasks = _auto_close_running(config, queue)
    if closed_tasks:
        queue = load_queue()

    state["pending_tasks"] = len(queue.get("pending", []))

    append_event("cycle_complete", {"pending": state["pending_tasks"]})
    if config.get("notify_telegram", False):
        last_report_at = state.get("last_report_at", "")
        last_dt = _parse_iso(last_report_at)
        completed_recent = []
        if last_dt:
            for t in queue.get("completed", []):
                finished = _parse_iso(t.get("finished_at", ""))
                if finished and finished > last_dt:
                    completed_recent.append(t)
        else:
            completed_recent = queue.get("completed", [])
        report = format_cycle_report(
            scheduled=scheduled,
            processed=processed,
            pending_count=len(queue.get("pending", [])),
            running_count=len(queue.get("running", [])),
            state=state,
            running_tasks=queue.get("running", []),
            completed_tasks=completed_recent,
            force=bool(config.get("telegram_report_on_cycle", False)),
        )
        if report:
            enqueue_message(report, config)
        state["last_report_at"] = utc_now_iso()

    save_state(state)
    return state


def watch(interval_sec: int) -> None:
    while True:
        cycle_once()
        import time
        time.sleep(interval_sec)


def main() -> int:
    parser = argparse.ArgumentParser(description="Agent runtime loop")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("once", help="Run one evolution/health cycle")
    w = sub.add_parser("watch", help="Run loop repeatedly")
    w.add_argument("--interval", type=int, default=300)
    args = parser.parse_args()

    if args.cmd == "once":
        cycle_once()
    else:
        watch(int(args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
