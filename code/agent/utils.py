from __future__ import annotations

import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Dict


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


MSK_TZ = timezone(timedelta(hours=3))


def _parse_hhmm(value: str) -> time | None:
    if not value:
        return None
    try:
        raw = value.strip()
        if not raw:
            return None
        parts = raw.split(":")
        if len(parts) != 2:
            return None
        hour = int(parts[0])
        minute = int(parts[1])
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return None
        return time(hour=hour, minute=minute)
    except Exception:
        return None


def is_quiet_hours_msk(config: Dict[str, Any]) -> bool:
    window = config.get("quiet_hours_msk") or {}
    if not isinstance(window, dict):
        return False
    start_raw = str(window.get("start", "")).strip()
    end_raw = str(window.get("end", "")).strip()
    start = _parse_hhmm(start_raw)
    end = _parse_hhmm(end_raw) if end_raw else None
    if not start:
        return False
    now = datetime.now(MSK_TZ).time()
    if not end:
        return now >= start
    if start <= end:
        return start <= now < end
    return now >= start or now < end


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def save_json(path: Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def ensure_file(path: Path, default_text: str = "") -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(default_text, encoding="utf-8")
