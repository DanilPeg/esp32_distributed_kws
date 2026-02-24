import argparse
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "notes" / "Journal"
NGROK_EXE = ROOT / "code" / "scripts" / "tools" / "ngrok" / "ngrok.exe"


def log(msg: str, log_path: Path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")


def http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status in (200, 303, 307, 308)
    except Exception:
        return False


def internet_ok(timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection(("1.1.1.1", 53), timeout=timeout):
            return True
    except Exception:
        return False


def ngrok_tunnel_ok(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return bool(data.get("tunnels"))
    except Exception:
        return False


def start_uvicorn(port: int, log_path: Path):
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app:app",
        "--app-dir",
        "code/web_portal",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
    ]
    out = (LOG_DIR / "web_portal_uvicorn.out.log").open("a", encoding="utf-8")
    err = (LOG_DIR / "web_portal_uvicorn.err.log").open("a", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(cmd, cwd=str(ROOT), stdout=out, stderr=err, creationflags=creationflags)
    log(f"uvicorn started on port {port}", log_path)


def start_ngrok(port: int, log_path: Path):
    if not NGROK_EXE.exists():
        log("ngrok.exe not found; skip start", log_path)
        return
    cmd = [str(NGROK_EXE), "http", str(port)]
    out = (LOG_DIR / "ngrok.out.log").open("a", encoding="utf-8")
    err = (LOG_DIR / "ngrok.err.log").open("a", encoding="utf-8")
    creationflags = 0
    if os.name == "nt":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(cmd, cwd=str(NGROK_EXE.parent), stdout=out, stderr=err, creationflags=creationflags)
    log("ngrok started", log_path)


def restart_ngrok(log_path: Path):
    try:
        subprocess.run(["taskkill", "/IM", "ngrok.exe", "/F"], capture_output=True, text=True)
    except Exception:
        pass
    time.sleep(0.5)
    log("ngrok restarted", log_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--interval-sec", type=int, default=20)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    log_path = LOG_DIR / "portal_watchdog.log"
    last_internet = None

    def tick():
        nonlocal last_internet
        local_ok = http_ok(f"http://127.0.0.1:{args.port}/")
        if not local_ok:
            start_uvicorn(args.port, log_path)

        internet = internet_ok()
        if internet:
            if not ngrok_tunnel_ok():
                restart_ngrok(log_path)
                start_ngrok(args.port, log_path)
        elif last_internet is True:
            log("internet down", log_path)
        last_internet = internet

    if args.once:
        tick()
        return

    log("portal_watchdog started", log_path)
    while True:
        tick()
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
