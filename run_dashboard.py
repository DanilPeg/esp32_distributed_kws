"""Launcher for the hash-KWS cluster dashboard.

Resolves ``code/`` relative to this script's own location so the working
directory doesn't matter — you can run it from anywhere on your system:

    python C:\\Users\\Danil\\diploma_esp32_distributed_nn\\run_dashboard.py

or, equivalently:

    cd C:\\Users\\Danil\\diploma_esp32_distributed_nn
    python run_dashboard.py

Optional flags::

    --host 127.0.0.1      # bind address (default: 127.0.0.1)
    --port 8765           # TCP port   (default: 8765)
    --reload              # auto-reload on code changes (dev)
    --log-level info      # uvicorn log level

Requires ``fastapi``, ``uvicorn`` and ``jinja2`` in the active Python env.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the hash-KWS cluster dashboard.")
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="TCP port (default: 8765)")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    parser.add_argument("--log-level", default="info", help="uvicorn log level")
    args = parser.parse_args()

    here = Path(__file__).resolve().parent
    code_dir = here / "code"
    if not code_dir.is_dir():
        sys.stderr.write(
            f"[run_dashboard] expected {code_dir} to exist — are you running this\n"
            f"script from inside a checkout of the diploma repo?\n"
        )
        sys.exit(2)

    # Put ``code/`` on sys.path so ``dashboard`` is importable as a top-level
    # package, without making ``code/`` itself a package (which would shadow
    # the Python stdlib ``code`` module).
    code_str = str(code_dir)
    if code_str not in sys.path:
        sys.path.insert(0, code_str)

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "[run_dashboard] uvicorn is not installed. Install the server deps\n"
            "first:\n\n    pip install fastapi uvicorn jinja2\n\n"
        )
        sys.exit(1)

    # Import the app through the same sys.path we just configured. If this
    # raises, surface a friendly message with the actual cause.
    try:
        from dashboard.app import app  # noqa: F401 — side-effect: ensures importability
    except ImportError as exc:
        sys.stderr.write(
            f"[run_dashboard] failed to import dashboard.app: {exc}\n"
            f"Check that the following files exist:\n"
            f"  {code_dir / 'dashboard' / '__init__.py'}\n"
            f"  {code_dir / 'dashboard' / 'app.py'}\n"
        )
        sys.exit(1)

    print(f"[run_dashboard] serving on http://{args.host}:{args.port}")
    print(f"[run_dashboard] project root: {here}")
    print(f"[run_dashboard] open the UI at http://{args.host}:{args.port}/")
    uvicorn.run(
        "dashboard.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
