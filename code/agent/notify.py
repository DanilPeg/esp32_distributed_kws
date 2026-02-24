from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

from .events import append_event
from .paths import REPO_ROOT
from .utils import utc_now_iso


TELEGRAM_DIR = REPO_ROOT / "notes" / "Journal" / "telegram"
OUTBOX_PATH = TELEGRAM_DIR / "outbox.jsonl"
LAST_TRIGGER_PATH = TELEGRAM_DIR / "last_trigger_chat.json"
TELEGRAM_ENV = REPO_ROOT / "code" / "scripts" / "telegram.env"
BRIDGE_SCRIPT = REPO_ROOT / "code" / "scripts" / "telegram_bridge.py"


def _load_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        data[key.strip()] = val.strip().strip('"').strip("'")
    return data


def _load_last_trigger_chat_id() -> str:
    if not LAST_TRIGGER_PATH.exists():
        return ""
    try:
        raw = LAST_TRIGGER_PATH.read_text(encoding="utf-8").lstrip("\ufeff")
        data = json.loads(raw)
        return str(data.get("chat_id", "")).strip()
    except Exception:
        return ""


def _load_default_chat_id() -> str:
    env = _load_env_file(TELEGRAM_ENV)
    return str(env.get("TELEGRAM_DEFAULT_CHAT_ID", "")).strip()


def resolve_chat_id(config: Dict[str, Any]) -> str:
    override = str(config.get("telegram_report_chat_id", "")).strip()
    if override:
        return override
    last_trigger = _load_last_trigger_chat_id()
    if last_trigger:
        return last_trigger
    return _load_default_chat_id()


def enqueue_message(text: str, config: Dict[str, Any], send_now: bool = True) -> bool:
    chat_id = resolve_chat_id(config)
    if not chat_id:
        append_event("telegram_report_skipped", {"reason": "missing_chat_id"})
        return False
    OUTBOX_PATH.parent.mkdir(parents=True, exist_ok=True)
    msg = {"chat_id": chat_id, "text": text}
    with OUTBOX_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    append_event("telegram_report_enqueued", {"chat_id": chat_id, "chars": len(text)})
    if send_now and BRIDGE_SCRIPT.exists():
        subprocess.run(["python", str(BRIDGE_SCRIPT), "push"], check=False)
    return True


MSK_TZ = timezone(timedelta(hours=3))


def _fmt_time(ts: str) -> str:
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts)
    except Exception:
        return ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(MSK_TZ).strftime("%Y-%m-%d %H:%M:%S MSK")


def format_cycle_report(
    scheduled: Dict[str, Any] | None,
    processed: List[Dict[str, Any]],
    pending_count: int,
    running_count: int,
    state: Dict[str, Any],
    running_tasks: List[Dict[str, Any]] | None = None,
    completed_tasks: List[Dict[str, Any]] | None = None,
    force: bool = False,
) -> str:
    if not scheduled and not processed and not force:
        return ""
    now_utc = datetime.now(timezone.utc)
    lines = [
        "Отчет эволюции",
        f"UTC: {now_utc.isoformat()}",
        f"MSK: {now_utc.astimezone(MSK_TZ).strftime('%Y-%m-%d %H:%M:%S MSK')}",
    ]
    if not scheduled and not processed:
        lines.append("Изменения: нет новых задач в этом цикле.")
    if scheduled:
        title = scheduled.get("title") or scheduled.get("type") or "задача"
        lines.append(f"Запланировано: {title} (id={scheduled.get('id')})")
    if processed:
        lines.append(f"Подготовлено: {len(processed)}")
        for t in processed:
            title = t.get("title") or t.get("type") or "задача"
            started = _fmt_time(t.get("started_at", ""))
            actions = t.get("actions") or []
            action_str = ", ".join(actions) if actions else "нет"
            lines.append(
                f"- id={t.get('id')}; задача={title}; начало={started}; действия={action_str}"
            )
    if completed_tasks:
        lines.append(f"Завершено: {len(completed_tasks)}")
        for t in completed_tasks:
            title = t.get("title") or t.get("type") or "задача"
            finished = _fmt_time(t.get("finished_at", ""))
            lines.append(f"- id={t.get('id')}; задача={title}; конец={finished}")
    if running_tasks:
        lines.append(f"В работе: {len(running_tasks)}")
        for t in running_tasks[:5]:
            title = t.get("title") or t.get("type") or "задача"
            started = _fmt_time(t.get("started_at", ""))
            lines.append(f"- id={t.get('id')}; задача={title}; начало={started}")
        if len(running_tasks) > 5:
            lines.append(f"- еще в работе: {len(running_tasks) - 5}")
    lines.append(f"Очередь: pending={pending_count} running={running_count}")
    lines.append(f"Последняя эволюция: {_fmt_time(state.get('last_evolution_at') or '')}")
    return "\n".join(lines)
