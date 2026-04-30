"""One-command launcher for the distributed hash-KWS demo.

Starts the three host-side processes required to visualise the real ESP32 +
virtual peer + master aggregator cluster:

1. ``hash_kws_serial_bridge.py`` — talks to the real board over serial and
   writes normalised events into ``notes/Journal/hash_kws_telemetry/node1/``.
2. ``hash_kws_cluster_sim.py`` — tails node1 events, manufactures virtual
   node2 and node3 events plus fusion decisions.
3. ``run_dashboard.py`` — the FastAPI master dashboard on localhost.

Usage (PowerShell or cmd)::

    python code\\scripts\\run_distributed_demo.py --port COM5

Press ``Ctrl+C`` once to shut everything down cleanly.
"""

from __future__ import annotations

import argparse
import atexit
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Launch bridge + cluster sim + dashboard in one go."
    )
    ap.add_argument(
        "--port",
        required=True,
        help="Serial port of the real ESP32 (e.g. COM5 on Windows, /dev/ttyUSB0 on Linux)",
    )
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--node-id", default="1", help="Logical id of the real board (default: 1)")
    ap.add_argument("--node-label", default="real_esp32")
    ap.add_argument("--dashboard-host", default="127.0.0.1")
    ap.add_argument("--dashboard-port", type=int, default=8765)
    ap.add_argument(
        "--no-dashboard",
        action="store_true",
        help="do not launch the dashboard (useful if you already run it separately)",
    )
    ap.add_argument(
        "--no-sim",
        action="store_true",
        help="do not launch the cluster simulator (single-node debug mode)",
    )
    ap.add_argument(
        "--no-bridge",
        action="store_true",
        help="do not launch the serial bridge (run it manually or replay from files)",
    )
    ap.add_argument(
        "--single-node",
        action="store_true",
        help="convenience alias: disable the cluster simulator so only the real ESP32 shows in the dashboard",
    )
    args = ap.parse_args()
    if args.single_node:
        args.no_sim = True
    return args


def _preflight_checks(args: argparse.Namespace) -> int:
    """Fail fast on missing deps instead of letting children crash silently.

    Returns a non-zero exit code to propagate, or 0 if we can proceed.
    """
    missing: list[tuple[str, str]] = []
    if not args.no_bridge:
        try:
            import serial  # noqa: F401 — pyserial probe
        except ImportError:
            missing.append(("pyserial", "pip install pyserial"))
    if not args.no_dashboard:
        try:
            import fastapi  # noqa: F401
            import uvicorn  # noqa: F401
            import jinja2  # noqa: F401
        except ImportError:
            missing.append(
                ("fastapi / uvicorn / jinja2", "pip install fastapi uvicorn jinja2")
            )
    if missing:
        sys.stderr.write("[demo] missing Python dependencies:\n")
        for pkg, hint in missing:
            sys.stderr.write(f"  - {pkg}  →  {hint}\n")
        sys.stderr.write(
            "\n[demo] install the listed packages in the Python you use to run this\n"
            "       launcher (same interpreter that spawns the children), then retry.\n"
        )
        return 2
    return 0


def main() -> int:
    args = parse_args()

    rc = _preflight_checks(args)
    if rc:
        return rc

    telemetry_dir = REPO_ROOT / "notes" / "Journal" / "hash_kws_telemetry"
    node1_dir = telemetry_dir / "node1"
    node1_dir.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[str, subprocess.Popen]] = []

    def launch(name: str, cmd: list[str]) -> subprocess.Popen:
        print(f"[demo] starting {name}:\n       " + " ".join(_shell_quote(c) for c in cmd))
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
        processes.append((name, proc))
        return proc

    if not args.no_bridge:
        bridge_cmd = [
            sys.executable,
            str(REPO_ROOT / "code" / "scripts" / "hash_kws_serial_bridge.py"),
            "--port", args.port,
            "--baud", str(args.baud),
            "--node-id", args.node_id,
            "--node-label", args.node_label,
            "--events-path", str(node1_dir / "events.jsonl"),
            "--state-path", str(node1_dir / "state.json"),
            "--raw-path", str(node1_dir / "raw.log"),
            "--echo",
        ]
        launch("bridge", bridge_cmd)
        # small delay so the bridge can claim the port before the sim starts
        # polling the events file
        time.sleep(0.6)

    if not args.no_sim:
        sim_cmd = [
            sys.executable,
            str(REPO_ROOT / "code" / "scripts" / "hash_kws_cluster_sim.py"),
            "--real-events", str(node1_dir / "events.jsonl"),
            "--watch",
            "--print-state",
        ]
        launch("cluster_sim", sim_cmd)

    if not args.no_dashboard:
        dashboard_cmd = [
            sys.executable,
            str(REPO_ROOT / "run_dashboard.py"),
            "--host", args.dashboard_host,
            "--port", str(args.dashboard_port),
        ]
        launch("dashboard", dashboard_cmd)

    def terminate_all() -> None:
        for name, proc in processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass
        deadline = time.time() + 5.0
        for name, proc in processes:
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                print(f"[demo] {name} did not exit in time, killing")
                proc.kill()

    atexit.register(terminate_all)

    def on_signal(sig: int, frame: object) -> None:
        print(f"\n[demo] signal {sig} received, shutting down cluster")
        terminate_all()
        sys.exit(0)

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    print()
    print("[demo] --------------------------------------------------")
    print(f"[demo] processes launched: {[n for n, _ in processes]}")
    if not args.no_dashboard:
        print(f"[demo] dashboard \u2192 http://{args.dashboard_host}:{args.dashboard_port}/")
    print("[demo] press Ctrl+C once to stop everything.")
    print("[demo] --------------------------------------------------")
    print()

    try:
        while True:
            time.sleep(1)
            for name, proc in processes:
                rc = proc.poll()
                if rc is not None:
                    print(f"[demo] child '{name}' exited with code {rc}. tearing down.")
                    terminate_all()
                    return rc if rc is not None else 1
    except KeyboardInterrupt:
        terminate_all()
        return 0


def _shell_quote(s: str) -> str:
    """Best-effort quoting for logging, not for shell execution."""
    if any(ch in s for ch in (" ", "\t", '"')):
        return '"' + s.replace('"', '\\"') + '"'
    return s


if __name__ == "__main__":
    sys.exit(main())
