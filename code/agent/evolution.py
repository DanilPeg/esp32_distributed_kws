from __future__ import annotations

from typing import Any, Dict

from .events import append_event
from .queue import enqueue_task
from .utils import utc_now_iso


def next_block_id(config: Dict[str, Any], state: Dict[str, Any]) -> int:
    blocks = config.get("review_blocks", [])
    if not blocks:
        return -1
    last_id = int(state.get("last_block_id", -1))
    return (last_id + 1) % len(blocks)


def build_review_task(block: Dict[str, Any]) -> Dict[str, Any]:
    title = f"Daily review: {block.get('name')}"
    desc = (
        f"Review block: {block.get('name')}\n"
        f"Paths: {', '.join(block.get('paths', []))}\n"
        f"Focus: {block.get('focus', '')}\n"
        "Deliverable: short notes in journal + any concrete fixes to propose."
    )
    return {
        "type": "review",
        "title": title,
        "description": desc,
        "needs_agent": True,
    }


def schedule_evolution_task(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any] | None:
    blocks = config.get("review_blocks", [])
    if not blocks:
        return None
    block_id = next_block_id(config, state)
    block = blocks[block_id]
    task = build_review_task(block)
    queued = enqueue_task(task)
    append_event("evolution_scheduled", {"block_id": block_id, "task_id": queued["id"]})
    state["last_block_id"] = block_id
    state["last_evolution_at"] = utc_now_iso()
    state["pending_tasks"] = int(state.get("pending_tasks", 0)) + 1
    return queued
