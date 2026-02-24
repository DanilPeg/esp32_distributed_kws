from __future__ import annotations

import json
from typing import Any, Dict

from .paths import EVENTS_PATH
from .utils import utc_now_iso


def append_event(event_type: str, data: Dict[str, Any] | None = None) -> None:
    payload = {
        "ts": utc_now_iso(),
        "type": event_type,
    }
    if data:
        payload.update(data)
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
