from __future__ import annotations

from typing import Any, Dict

from .paths import STATE_PATH
from .utils import load_json, save_json


def default_state() -> Dict[str, Any]:
    return {
        "version": "0.1",
        "evolution_enabled": False,
        "autonomous_apply": False,
        "last_evolution_at": "",
        "last_health_check_at": "",
        "last_report_at": "",
        "last_block_id": -1,
        "pending_tasks": 0,
        "completed_tasks": 0,
        "notes": "",
    }


def load_state() -> Dict[str, Any]:
    return load_json(STATE_PATH, default_state())


def save_state(state: Dict[str, Any]) -> None:
    save_json(STATE_PATH, state)
