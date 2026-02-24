#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from datetime import datetime


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def inbox_path() -> Path:
    return repo_root() / "notes" / "Journal" / "telegram" / "inbox.jsonl"


def state_path() -> Path:
    return repo_root() / "notes" / "Journal" / "telegram" / "command_state.json"


def outbox_path() -> Path:
    return repo_root() / "notes" / "Journal" / "telegram" / "outbox.jsonl"


def bridge_script() -> Path:
    return repo_root() / "code" / "scripts" / "telegram_bridge.py"


def env_file_path() -> Path:
    return Path(__file__).resolve().parent / "telegram_commands.env"


def load_env_file(path: Path) -> dict:
    if not path.exists():
        return {}
    data = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        data[key.strip()] = val.strip().strip('"').strip("'")
    return data


def get_config() -> dict:
    cfg = {}
    cfg.update(load_env_file(env_file_path()))
    cfg.update({k: v for k, v in os.environ.items() if v is not None})
    return cfg


def parse_id_list(raw: str) -> set[int]:
    if not raw:
        return set()
    parts = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parts.append(token)
    out = set()
    for p in parts:
        try:
            out.add(int(p))
        except ValueError:
            continue
    return out


def parse_str_list(raw: str) -> set[str]:
    if not raw:
        return set()
    parts = []
    for token in raw.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        parts.append(token.lower())
    return set(parts)


def load_state() -> dict:
    if not state_path().exists():
        return {"last_update_id": 0}
    return json.loads(state_path().read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    state_path().parent.mkdir(parents=True, exist_ok=True)
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


def is_allowed(msg: dict, cfg: dict) -> bool:
    allowed_users = parse_id_list(cfg.get("TELEGRAM_ALLOWED_USER_IDS", ""))
    allowed_chats = parse_id_list(cfg.get("TELEGRAM_ALLOWED_CHAT_IDS", ""))
    user_id = int(msg.get("from", {}).get("id", 0))
    chat_id = int(msg.get("chat_id", 0))
    if allowed_users and user_id not in allowed_users:
        return False
    if allowed_chats and chat_id not in allowed_chats:
        return False
    return True


def normalize_command(text: str) -> tuple[str, str]:
    raw = text.strip().split()[0]
    if raw.startswith("/"):
        raw = raw[1:]
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    args = text.strip().split(maxsplit=1)
    arg_text = args[1] if len(args) > 1 else ""
    return raw.lower(), arg_text


def enqueue_reply(chat_id: int, text: str) -> None:
    outbox_path().parent.mkdir(parents=True, exist_ok=True)
    with outbox_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps({"chat_id": chat_id, "text": text}, ensure_ascii=False) + "\n")


def push_outbox() -> None:
    cmd = [
        "python",
        str(bridge_script()),
        "push",
    ]
    subprocess.run(cmd, check=False)


def run_ps(script: Path) -> tuple[int, str]:
    if not script.exists():
        return 1, f"script not found: {script}"
    cmd = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(script),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def run_git(args: list[str]) -> tuple[int, str]:
    cmd = ["git"] + args
    proc = subprocess.run(cmd, cwd=repo_root(), capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def cmd_status(cfg: dict) -> str:
    heartbeat = repo_root() / "notes" / "Journal" / "telegram" / "simple_relay_heartbeat.txt"
    state = repo_root() / "notes" / "Journal" / "telegram" / "state.json"
    hb = heartbeat.read_text(encoding="utf-8").strip() if heartbeat.exists() else "no heartbeat"
    last_update = "n/a"
    if state.exists():
        try:
            last_update = str(json.loads(state.read_text(encoding="utf-8")).get("last_update_id"))
        except Exception:
            last_update = "n/a"
    return f"status: heartbeat={hb}; last_update_id={last_update}"


def cmd_help() -> str:
    return (
        "commands:\n"
        "/status - relay heartbeat and last update id\n"
        "/start_codex - start a Codex PowerShell session\n"
        "/evolve [on|off|status|run] - control evolution loop\n"
        "/agent [status|queue|run] - agent runtime status\n"
        "/task <title> | <description> - enqueue a custom agent task\n"
        "/merge <branch> [--force] - merge a task branch into current branch\n"
        "/whoami - show chat_id and user_id\n"
        "/help - this message"
    )


def cmd_whoami(msg: dict) -> str:
    user = msg.get("from", {})
    return (
        f"chat_id: {msg.get('chat_id')}\n"
        f"user_id: {user.get('id')}\n"
        f"username: {user.get('username') or user.get('first_name')}"
    )


def cmd_start_codex(cfg: dict) -> str:
    script = cfg.get("START_CODEX_SCRIPT", "").strip()
    if not script:
        script = str(repo_root() / "code" / "scripts" / "start_codex_session.ps1")
    code, out = run_ps(Path(script))
    if code == 0:
        return out or "ok: codex session start requested"
    return f"error: {out or 'start_codex failed'}"


def agent_paths() -> tuple[Path, Path]:
    base = repo_root() / "notes" / "Journal" / "agent"
    return base / "state.json", base / "queue.json"


def load_agent_state() -> dict:
    state_path, _ = agent_paths()
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_agent_state(state: dict) -> None:
    state_path, _ = agent_paths()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def load_agent_queue() -> dict:
    _, queue_path = agent_paths()
    if not queue_path.exists():
        return {"pending": [], "running": []}
    try:
        return json.loads(queue_path.read_text(encoding="utf-8"))
    except Exception:
        return {"pending": [], "running": []}


def agent_config_path() -> Path:
    return repo_root() / "code" / "agent" / "agent_config.json"


def load_agent_config() -> dict:
    path = agent_config_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_agent_config(cfg: dict) -> None:
    path = agent_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_evolve(arg_text: str) -> str:
    arg = (arg_text or "").strip().lower()
    cfg = load_agent_config()
    state = load_agent_state()

    if arg in ("on", "start", "enable"):
        cfg["evolution_enabled"] = True
        state["evolution_enabled"] = True
        save_agent_config(cfg)
        save_agent_state(state)
        return "evolution: ON"
    if arg in ("off", "stop", "disable"):
        cfg["evolution_enabled"] = False
        state["evolution_enabled"] = False
        save_agent_config(cfg)
        save_agent_state(state)
        return "evolution: OFF"
    if arg in ("run", "once"):
        cmd = ["python", str(repo_root() / "code" / "agent" / "runner.py"), "once"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return "evolution: ran one cycle"
        return f"evolution: run failed: {proc.stderr.strip() or proc.stdout.strip()}"

    # default: status
    pending = load_agent_queue().get("pending", [])
    enabled = bool(state.get("evolution_enabled") or cfg.get("evolution_enabled"))
    last_evo = state.get("last_evolution_at") or "-"
    last_health = state.get("last_health_check_at") or "-"
    return (
        "evolution status:\n"
        f"enabled: {int(enabled)}\n"
        f"pending: {len(pending)}\n"
        f"last_evolution_at: {last_evo}\n"
        f"last_health_check_at: {last_health}"
    )


def cmd_agent(arg_text: str) -> str:
    arg = (arg_text or "").strip().lower()
    state = load_agent_state()
    queue = load_agent_queue()
    pending = queue.get("pending", [])
    running = queue.get("running", [])
    if arg in ("queue", "q"):
        if not pending:
            return "agent queue: empty"
        lines = ["agent queue (pending):"]
        for t in pending[:10]:
            title = t.get("title") or t.get("description") or t.get("type") or "task"
            lines.append(f"- {t.get('id')}: {title[:80]}")
        return "\n".join(lines)
    if arg in ("run", "once"):
        cmd = ["python", str(repo_root() / "code" / "agent" / "runner.py"), "once"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0:
            return "agent: ran one cycle"
        return f"agent: run failed: {proc.stderr.strip() or proc.stdout.strip()}"
    # default: status
    return (
        "agent status:\n"
        f"pending: {len(pending)}\n"
        f"running: {len(running)}\n"
        f"last_evolution_at: {state.get('last_evolution_at') or '-'}\n"
        f"last_health_check_at: {state.get('last_health_check_at') or '-'}"
    )


def _ensure_agent_imports():
    try:
        import sys
        sys.path.insert(0, str(repo_root() / "code"))
        from agent.queue import enqueue_task  # type: ignore
        return enqueue_task, None
    except Exception as exc:
        return None, str(exc)


def cmd_task(arg_text: str) -> str:
    raw = (arg_text or "").strip()
    if not raw:
        return "usage: /task <title> | <description>"
    if "|" in raw:
        title, desc = raw.split("|", 1)
    elif "::" in raw:
        title, desc = raw.split("::", 1)
    else:
        title, desc = raw, ""
    title = title.strip()
    desc = desc.strip()
    if not title:
        return "usage: /task <title> | <description>"
    if not desc:
        desc = "No description provided."
    enqueue_task, err = _ensure_agent_imports()
    if enqueue_task is None:
        return f"error: could not enqueue task ({err})"
    task = enqueue_task({
        "type": "custom",
        "title": title,
        "description": desc,
        "needs_agent": True,
    })
    return (
        "task queued:\n"
        f"id: {task.get('id')}\n"
        f"title: {title}\n"
        "next: will be processed by the agent loop (or run /agent run)"
    )


def cmd_merge(arg_text: str) -> str:
    raw = (arg_text or "").strip()
    if not raw:
        return "usage: /merge <branch> [--force]"
    tokens = raw.split()
    branch = tokens[0].strip()
    force = "--force" in tokens[1:]
    if not branch:
        return "usage: /merge <branch> [--force]"
    # verify git exists
    code, out = run_git(["rev-parse", "--git-dir"])
    if code != 0:
        return f"error: git not initialized ({out})"
    # verify branch exists
    code, out = run_git(["rev-parse", "--verify", branch])
    if code != 0:
        return f"error: branch not found: {branch}"
    # check dirty status
    code, status = run_git(["status", "--porcelain"])
    if code != 0:
        return f"error: git status failed: {status}"
    if status.strip() and not force:
        return "error: working tree not clean. Use /merge <branch> --force if you want to merge anyway."
    code, out = run_git(["merge", branch])
    if code != 0:
        return f"merge failed:\n{out}"
    return f"merge ok: {branch}"


def process_once(cfg: dict) -> int:
    state = load_state()
    last_id = int(state.get("last_update_id", 0))
    msgs = [m for m in read_inbox_lines() if int(m.get("update_id", 0)) > last_id]
    if not msgs:
        return 0
    msgs.sort(key=lambda m: int(m.get("update_id", 0)))
    silent_cmds = parse_str_list(cfg.get("TELEGRAM_SILENT_COMMANDS", ""))
    handled = 0
    for msg in msgs:
        update_id = int(msg.get("update_id", 0))
        text = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            state["last_update_id"] = update_id
            continue
        if not is_allowed(msg, cfg):
            state["last_update_id"] = update_id
            continue
        cmd, arg_text = normalize_command(text)
        if cmd in silent_cmds:
            state["last_update_id"] = update_id
            continue
        chat_id = int(msg.get("chat_id", 0))
        if cmd in ("help", "start"):
            reply = cmd_help()
        elif cmd == "status":
            reply = cmd_status(cfg)
        elif cmd == "whoami":
            reply = cmd_whoami(msg)
        elif cmd == "start_codex":
            reply = cmd_start_codex(cfg)
        elif cmd == "evolve":
            reply = cmd_evolve(arg_text)
        elif cmd == "agent":
            reply = cmd_agent(arg_text)
        elif cmd == "task":
            reply = cmd_task(arg_text)
        elif cmd == "merge":
            reply = cmd_merge(arg_text)
        else:
            reply = "unknown command. use /help"
        enqueue_reply(chat_id, reply)
        push_outbox()
        handled += 1
        state["last_update_id"] = update_id
    save_state(state)
    return handled


def watch(cfg: dict, interval: int) -> None:
    while True:
        process_once(cfg)
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Process Telegram commands from inbox.jsonl.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("once", help="Process new commands once")
    w = sub.add_parser("watch", help="Watch inbox and process commands")
    w.add_argument("--interval", type=int, default=5)
    args = parser.parse_args()
    cfg = get_config()
    if args.cmd == "once":
        process_once(cfg)
    else:
        watch(cfg, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
