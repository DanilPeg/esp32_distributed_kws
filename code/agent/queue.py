from __future__ import annotations

import uuid
from typing import Any, Dict

from .paths import QUEUE_PATH
from .utils import load_json, save_json, utc_now_iso
from .events import append_event


def default_queue() -> Dict[str, Any]:
    return {"pending": [], "running": [], "completed": []}


def load_queue() -> Dict[str, Any]:
    data = load_json(QUEUE_PATH, default_queue())
    if not isinstance(data, dict):
        return default_queue()
    data.setdefault("pending", [])
    data.setdefault("running", [])
    data.setdefault("completed", [])
    return data


def save_queue(queue: Dict[str, Any]) -> None:
    save_json(QUEUE_PATH, queue)


def enqueue_task(task: Dict[str, Any]) -> Dict[str, Any]:
    queue = load_queue()
    task = dict(task)
    task.setdefault("id", uuid.uuid4().hex[:8])
    task.setdefault("created_at", utc_now_iso())
    task.setdefault("status", "pending")
    queue.setdefault("pending", []).append(task)
    save_queue(queue)
    append_event("task_enqueued", {"task_id": task["id"], "type": task.get("type")})
    return task


def pop_next_task() -> Dict[str, Any] | None:
    queue = load_queue()
    pending = queue.get("pending", [])
    if not pending:
        return None
    task = pending.pop(0)
    task["status"] = "running"
    task["started_at"] = utc_now_iso()
    queue.setdefault("running", []).append(task)
    save_queue(queue)
    append_event("task_started", {"task_id": task["id"], "type": task.get("type")})
    return task


def mark_done(task_id: str, status: str = "done") -> Dict[str, Any] | None:
    queue = load_queue()
    running = queue.get("running", [])
    remaining = []
    finished_task = None
    for t in running:
        if t.get("id") == task_id:
            t["status"] = status
            t["finished_at"] = utc_now_iso()
            finished_task = t
            append_event(
                "task_done",
                {
                    "task_id": task_id,
                    "status": status,
                    "finished_at": t["finished_at"],
                    "title": t.get("title") or "",
                },
            )
        else:
            remaining.append(t)
    queue["running"] = remaining
    if finished_task:
        queue.setdefault("completed", []).append(finished_task)
    save_queue(queue)
    return finished_task
