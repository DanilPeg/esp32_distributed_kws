from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


REPO_ROOT = repo_root()
AGENT_DIR = REPO_ROOT / "notes" / "Journal" / "agent"
CONFIG_PATH = REPO_ROOT / "code" / "agent" / "agent_config.json"
STATE_PATH = AGENT_DIR / "state.json"
QUEUE_PATH = AGENT_DIR / "queue.json"
EVENTS_PATH = AGENT_DIR / "events.jsonl"
HEALTH_PATH = AGENT_DIR / "health.json"
TASKS_DIR = AGENT_DIR / "tasks"
