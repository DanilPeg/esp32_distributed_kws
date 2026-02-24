from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .paths import REPO_ROOT, HEALTH_PATH
from .utils import save_json, utc_now_iso
from .events import append_event


def _file_age_minutes(path: Path) -> float | None:
    if not path.exists():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 60.0


def _check_file_recent(path: Path, max_age_min: int, label: str) -> Dict[str, str]:
    age = _file_age_minutes(path)
    if age is None:
        return {"check": label, "status": "fail", "detail": "missing"}
    if age <= max_age_min:
        return {"check": label, "status": "ok", "detail": f"age_min={int(age)}"}
    return {"check": label, "status": "warn", "detail": f"age_min={int(age)}"}


def run_health_checks() -> Dict[str, object]:
    checks: List[Dict[str, str]] = []

    checks.append(_check_file_recent(REPO_ROOT / "notes" / "Journal" / "web_portal_uvicorn.out.log", 120, "portal_stdout_recent"))
    checks.append(_check_file_recent(REPO_ROOT / "notes" / "Journal" / "web_portal_uvicorn.err.log", 120, "portal_stderr_recent"))
    checks.append(_check_file_recent(REPO_ROOT / "notes" / "Journal" / "portal_watchdog.log", 180, "portal_watchdog_recent"))

    # daily summary freshness
    today = datetime.now().strftime("%Y-%m-%d")
    summary_path = REPO_ROOT / "notes" / "Journal" / "summaries" / f"{today}.md"
    checks.append(_check_file_recent(summary_path, 1440, "daily_summary_present"))

    status = "ok"
    if any(c["status"] == "fail" for c in checks):
        status = "fail"
    elif any(c["status"] == "warn" for c in checks):
        status = "warn"

    payload = {
        "last_check": utc_now_iso(),
        "status": status,
        "checks": checks,
    }
    save_json(HEALTH_PATH, payload)
    append_event("health_check", {"status": status})
    return payload
