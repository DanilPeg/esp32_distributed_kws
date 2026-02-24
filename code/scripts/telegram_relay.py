#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def inbox_path() -> Path:
    return repo_root() / "notes" / "Journal" / "telegram" / "inbox.jsonl"


def relay_dir() -> Path:
    return repo_root() / "notes" / "Journal" / "telegram"


def state_path() -> Path:
    return relay_dir() / "relay_state.json"


def load_state() -> dict:
    if not state_path().exists():
        return {"last_update_id": 0}
    return json.loads(state_path().read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    relay_dir().mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def read_inbox_lines() -> list[dict]:
    if not inbox_path().exists():
        return []
    lines = inbox_path().read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def format_prompt(msgs: list[dict]) -> str:
    latest = msgs[-1] if msgs else {}
    return (
        f"Telegram: new message(s)={len(msgs)} "
        f"latest_update_id={latest.get('update_id')}. "
        "Check notes/Journal/telegram/inbox.jsonl and reply."
    )


def write_pending(text: str) -> Path:
    path = relay_dir() / "pending.md"
    path.write_text(text, encoding="utf-8")
    return path


def copy_to_clipboard(text: str) -> None:
    tmp = relay_dir() / "pending_clipboard.txt"
    tmp.write_text(text, encoding="utf-8")
    cmd = f"Get-Content -Raw -Path '{tmp}' | Set-Clipboard"
    subprocess.run(["powershell", "-NoProfile", "-Command", cmd], check=False)


def relay_once(args: argparse.Namespace) -> int:
    state = load_state()
    last_id = int(state.get("last_update_id", 0))
    msgs = [m for m in read_inbox_lines() if int(m.get("update_id", 0)) > last_id]
    if not msgs:
        return 0
    msgs.sort(key=lambda m: int(m.get("update_id", 0)))
    newest_id = int(msgs[-1].get("update_id", 0))
    text = format_prompt(msgs)
    write_pending(text)
    if args.clipboard:
        copy_to_clipboard(text)
    state["last_update_id"] = newest_id
    save_state(state)
    return len(msgs)


def watch(args: argparse.Namespace) -> None:
    interval = max(5, int(args.interval))
    print(f"[relay] watching inbox every {interval}s. Ctrl+C to stop.")
    while True:
        count = relay_once(args)
        if count and args.print:
            print(f"[relay] new messages: {count}")
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Relay Telegram inbox to a pending CLI prompt.")
    parser.add_argument("--clipboard", action="store_true", help="Copy pending prompt to clipboard")
    parser.add_argument("--print", action="store_true", help="Print when new messages arrive")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("once", help="Relay new messages once")
    w = sub.add_parser("watch", help="Watch inbox and relay new messages")
    w.add_argument("--interval", type=int, default=5, help="Polling interval seconds (default 5)")

    args = parser.parse_args()
    if args.cmd == "once":
        relay_once(args)
    elif args.cmd == "watch":
        watch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
