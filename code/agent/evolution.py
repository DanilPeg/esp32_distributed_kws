from __future__ import annotations

from typing import Any, Dict, List

from .events import append_event
from .queue import enqueue_task
from .utils import utc_now_iso


def next_block_id(config: Dict[str, Any], state: Dict[str, Any]) -> int:
    blocks = config.get("review_blocks", [])
    if not blocks:
        return -1
    last_id = int(state.get("last_block_id", -1))
    return (last_id + 1) % len(blocks)


def _task_modes(config: Dict[str, Any]) -> List[str]:
    raw = config.get("evolution_task_modes", [])
    if not isinstance(raw, list):
        return ["review"]
    modes = [str(m).strip().lower() for m in raw if str(m).strip()]
    return modes or ["review"]


def next_task_mode(config: Dict[str, Any], state: Dict[str, Any]) -> str:
    modes = _task_modes(config)
    last_mode = str(state.get("last_task_mode", "")).strip().lower()
    if last_mode in modes:
        idx = modes.index(last_mode)
        next_mode = modes[(idx + 1) % len(modes)]
    else:
        next_mode = modes[0]
    state["last_task_mode"] = next_mode
    return next_mode


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


def build_edit_task(block: Dict[str, Any]) -> Dict[str, Any]:
    paths = block.get("paths", [])
    path_str = ", ".join(paths)
    focus = block.get("focus", "")
    notebook_hint = ""
    if any(p.startswith("code/training") or p.startswith("code/analysis") for p in paths):
        notebook_hint = "Include notebook updates if relevant.\n"
    desc = (
        f"Edit block: {block.get('name')}\n"
        f"Paths: {path_str}\n"
        f"Focus: {focus}\n"
        f"{notebook_hint}"
        "Goal: apply small, safe improvements or fixes (code or docs). "
        "Keep changes minimal, list them in the report."
    )
    return {
        "type": "edit",
        "title": f"Edit: {block.get('name')}",
        "description": desc,
        "needs_agent": True,
    }


def schedule_evolution_task(config: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any] | None:
    blocks = config.get("review_blocks", [])
    if not blocks:
        return None
    block_id = next_block_id(config, state)
    block = blocks[block_id]
    mode = next_task_mode(config, state)
    if mode == "edit":
        task = build_edit_task(block)
    else:
        task = build_review_task(block)
    queued = enqueue_task(task)
    append_event("evolution_scheduled", {"block_id": block_id, "task_id": queued["id"], "mode": mode})
    state["last_block_id"] = block_id
    state["last_evolution_at"] = utc_now_iso()
    state["pending_tasks"] = int(state.get("pending_tasks", 0)) + 1
    return queued
