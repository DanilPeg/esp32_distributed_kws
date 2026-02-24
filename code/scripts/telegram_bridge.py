#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
import uuid
import mimetypes
import io
import urllib.parse
import urllib.request
import urllib.error
import socket
import ssl
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def journal_dir() -> Path:
    return repo_root() / "notes" / "Journal" / "telegram"


def last_trigger_chat_path() -> Path:
    return journal_dir() / "last_trigger_chat.json"


def get_last_trigger_chat_id() -> str:
    path = last_trigger_chat_path()
    if not path.exists():
        return ""
    try:
        raw = path.read_text(encoding="utf-8").lstrip("\ufeff")
        data = json.loads(raw)
        return str(data.get("chat_id", "")).strip()
    except Exception:
        return ""


def env_file_path() -> Path:
    return Path(__file__).resolve().parent / "telegram.env"


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


def require_token(cfg: dict) -> str:
    token = cfg.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN. Set it in env or in code/scripts/telegram.env")
    return token


def get_default_chat_id(cfg: dict) -> str:
    return str(cfg.get("TELEGRAM_DEFAULT_CHAT_ID", "")).strip()


def get_default_session_id(cfg: dict) -> str:
    return str(cfg.get("TELEGRAM_DEFAULT_SESSION_ID", "")).strip()


def state_path() -> Path:
    return journal_dir() / "state.json"


def load_state() -> dict:
    path = state_path()
    if not path.exists():
        return {"last_update_id": 0, "last_sent_line": 0}
    raw = path.read_text(encoding="utf-8").lstrip("\ufeff")
    return json.loads(raw)


def save_state(state: dict) -> None:
    journal_dir().mkdir(parents=True, exist_ok=True)
    state_path().write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def get_http_settings(cfg: dict) -> tuple[int, int, float]:
    def _get_int(key: str, default: int) -> int:
        try:
            return int(cfg.get(key, default))
        except Exception:
            return default

    def _get_float(key: str, default: float) -> float:
        try:
            return float(cfg.get(key, default))
        except Exception:
            return default

    timeout = _get_int("TELEGRAM_HTTP_TIMEOUT_SEC", 15)
    retries = _get_int("TELEGRAM_HTTP_RETRIES", 2)
    backoff = _get_float("TELEGRAM_HTTP_BACKOFF_SEC", 1.5)
    return timeout, retries, backoff


def request_json(url: str, params: dict | None = None, data: dict | None = None, cfg: dict | None = None) -> dict:
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    if data is not None:
        payload = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=payload)
    else:
        req = urllib.request.Request(url)
    if cfg is None:
        cfg = get_config()
    timeout, retries, backoff = get_http_settings(cfg)
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, socket.timeout, ssl.SSLError):
            if attempt >= retries:
                raise
            attempt += 1
            time.sleep(backoff * attempt)


def request_json_raw(url: str, data: bytes, headers: dict, cfg: dict | None = None) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers)
    if cfg is None:
        cfg = get_config()
    timeout, retries, backoff = get_http_settings(cfg)
    attempt = 0
    while True:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except (urllib.error.URLError, socket.timeout, ssl.SSLError):
            if attempt >= retries:
                raise
            attempt += 1
            time.sleep(backoff * attempt)


def iso_time(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def extract_session_id(text: str) -> str:
    if not text:
        return ""
    for marker in ("session_id:", "session:", "#session:"):
        if marker in text:
            tail = text.split(marker, 1)[1].strip()
            return tail.split()[0].strip()
    return ""


def pull_updates(args: argparse.Namespace) -> None:
    cfg = get_config()
    token = require_token(cfg)
    default_session_id = get_default_session_id(cfg)
    state = load_state()
    offset = state.get("last_update_id", 0) + 1

    url = f"https://api.telegram.org/bot{token}/getUpdates"
    resp = request_json(url, params={"offset": offset, "timeout": 0}, cfg=cfg)
    if not resp.get("ok"):
        raise RuntimeError(f"Telegram API error: {resp}")

    results = resp.get("result", [])
    if not results:
        return

    journal_dir().mkdir(parents=True, exist_ok=True)
    inbox_path = journal_dir() / "inbox.jsonl"

    max_update_id = state.get("last_update_id", 0)
    with inbox_path.open("a", encoding="utf-8") as f:
        for upd in results:
            update_id = upd.get("update_id", 0)
            msg = upd.get("message") or upd.get("edited_message") or {}
            chat = msg.get("chat", {})
            sender = msg.get("from", {})
            text = msg.get("text") or msg.get("caption") or ""
            session_id = extract_session_id(text) or default_session_id

            entry = {
                "update_id": update_id,
                "msg_id": msg.get("message_id"),
                "chat_id": chat.get("id"),
                "from": {
                    "id": sender.get("id"),
                    "username": sender.get("username"),
                    "first_name": sender.get("first_name"),
                },
                "text": text,
                "timestamp": iso_time(msg.get("date", int(time.time()))),
                "session_id": session_id,
                "type": "text" if text else "non_text",
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if update_id > max_update_id:
                max_update_id = update_id

    state["last_update_id"] = max_update_id
    save_state(state)


def send_message(token: str, chat_id: str, text: str, reply_to_msg_id: str | None = None, cfg: dict | None = None) -> None:
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text}
    if reply_to_msg_id:
        data["reply_to_message_id"] = reply_to_msg_id
    resp = request_json(url, data=data, cfg=cfg)
    if not resp.get("ok"):
        raise RuntimeError(f"Telegram API error: {resp}")


def send_document(token: str, chat_id: str, file_path: Path, caption: str | None = None, cfg: dict | None = None) -> None:
    url = f"https://api.telegram.org/bot{token}/sendDocument"
    boundary = f"----tg{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    buf = io.BytesIO()
    def write(s: str) -> None:
        buf.write(s.encode("utf-8"))

    write(f"--{boundary}\r\n")
    write('Content-Disposition: form-data; name="chat_id"\r\n\r\n')
    write(f"{chat_id}\r\n")
    if caption:
        write(f"--{boundary}\r\n")
        write('Content-Disposition: form-data; name="caption"\r\n\r\n')
        write(caption + "\r\n")

    write(f"--{boundary}\r\n")
    write(f'Content-Disposition: form-data; name="document"; filename="{file_path.name}"\r\n')
    write(f"Content-Type: {mime}\r\n\r\n")
    with file_path.open("rb") as f:
        buf.write(f.read())
    write("\r\n")
    write(f"--{boundary}--\r\n")

    data = buf.getvalue()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    resp = request_json_raw(url, data=data, headers=headers, cfg=cfg)
    if not resp.get("ok"):
        raise RuntimeError(f"Telegram API error: {resp}")


def push_outbox(args: argparse.Namespace) -> None:
    cfg = get_config()
    token = require_token(cfg)
    default_chat_id = get_default_chat_id(cfg)

    outbox_path = journal_dir() / "outbox.jsonl"
    if not outbox_path.exists():
        return

    state = load_state()
    last_sent_line = int(state.get("last_sent_line", 0))
    lines = outbox_path.read_text(encoding="utf-8").splitlines()

    for idx, line in enumerate(lines, start=1):
        if idx <= last_sent_line:
            continue
        if not line.strip():
            last_sent_line = idx
            continue
        msg = json.loads(line.lstrip("\ufeff"))
        chat_id = str(msg.get("chat_id") or default_chat_id).strip()
        text = msg.get("text") or msg.get("summary") or ""
        reply_to_msg_id = msg.get("reply_to_msg_id")
        if not chat_id:
            raise RuntimeError("Missing chat_id for outbox message. Set TELEGRAM_DEFAULT_CHAT_ID or add chat_id.")
        if msg.get("type") == "document" or msg.get("file_path"):
            file_path = msg.get("file_path")
            if not file_path:
                raise RuntimeError("Missing file_path for document message.")
            caption = msg.get("caption") or text or ""
            send_document(token, chat_id, Path(file_path), caption, cfg=cfg)
        else:
            if not text:
                raise RuntimeError("Missing text for outbox message.")
            send_message(token, chat_id, text, reply_to_msg_id, cfg=cfg)
        last_sent_line = idx
        state["last_sent_line"] = last_sent_line
    save_state(state)


def enqueue_outbox(args: argparse.Namespace) -> None:
    cfg = get_config()
    default_chat_id = get_default_chat_id(cfg)
    text_path = Path(args.text_file)
    if not text_path.exists():
        raise RuntimeError(f"text file not found: {text_path}")
    text = text_path.read_text(encoding="utf-8").lstrip("\ufeff")
    chat_id = str(args.chat_id or get_last_trigger_chat_id() or default_chat_id).strip()
    if not chat_id:
        raise RuntimeError("Missing chat_id for enqueue. Set TELEGRAM_DEFAULT_CHAT_ID or pass --chat-id.")
    if not text.strip():
        raise RuntimeError("Text file is empty.")
    outbox_path = journal_dir() / "outbox.jsonl"
    outbox_path.parent.mkdir(parents=True, exist_ok=True)
    msg = {"chat_id": chat_id, "text": text}
    with outbox_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def enqueue_file(args: argparse.Namespace) -> None:
    cfg = get_config()
    default_chat_id = get_default_chat_id(cfg)
    raw_path = Path(args.file_path)
    if not raw_path.is_absolute():
        raw_path = repo_root() / raw_path
    file_path = raw_path.resolve()
    root = repo_root().resolve()
    if root not in file_path.parents and file_path != root:
        raise RuntimeError("file_path must be inside the repository.")
    if not file_path.exists():
        raise RuntimeError(f"file not found: {file_path}")
    chat_id = str(args.chat_id or get_last_trigger_chat_id() or default_chat_id).strip()
    if not chat_id:
        raise RuntimeError("Missing chat_id for enqueue-file. Set TELEGRAM_DEFAULT_CHAT_ID or pass --chat-id.")
    caption = args.caption or ""
    outbox_path = journal_dir() / "outbox.jsonl"
    outbox_path.parent.mkdir(parents=True, exist_ok=True)
    msg = {"chat_id": chat_id, "type": "document", "file_path": str(file_path), "caption": caption}
    with outbox_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")


def sync(args: argparse.Namespace) -> None:
    pull_updates(args)
    push_outbox(args)


def watch(args: argparse.Namespace) -> None:
    interval = int(args.interval)
    if interval < 1:
        interval = 1
    print(f"[watch] sync every {interval}s. Press Ctrl+C to stop.")
    while True:
        sync(args)
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Telegram bridge: pull to inbox, push outbox.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("pull", help="Fetch new Telegram messages to notes/Journal/telegram/inbox.jsonl")
    sub.add_parser("push", help="Send queued messages from notes/Journal/telegram/outbox.jsonl")
    sub.add_parser("sync", help="Pull then push")
    enqueue_parser = sub.add_parser("enqueue", help="Append a UTF-8 text file as a Telegram outbox message")
    enqueue_parser.add_argument("--text-file", required=True, help="Path to UTF-8 text file to send")
    enqueue_parser.add_argument("--chat-id", default="", help="Override chat_id (default uses TELEGRAM_DEFAULT_CHAT_ID)")
    enqueue_file_parser = sub.add_parser("enqueue-file", help="Append a file to Telegram outbox (document)")
    enqueue_file_parser.add_argument("--file-path", required=True, help="Path to file (must be inside repo)")
    enqueue_file_parser.add_argument("--caption", default="", help="Optional caption")
    enqueue_file_parser.add_argument("--chat-id", default="", help="Override chat_id (default uses TELEGRAM_DEFAULT_CHAT_ID)")
    watch_parser = sub.add_parser("watch", help="Continuously sync on interval (seconds)")
    watch_parser.add_argument("--interval", type=int, default=300, help="Interval in seconds (default: 300)")

    args = parser.parse_args()

    try:
        if args.cmd == "pull":
            pull_updates(args)
        elif args.cmd == "push":
            push_outbox(args)
        elif args.cmd == "sync":
            sync(args)
        elif args.cmd == "watch":
            watch(args)
        elif args.cmd == "enqueue":
            enqueue_outbox(args)
        elif args.cmd == "enqueue-file":
            enqueue_file(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
