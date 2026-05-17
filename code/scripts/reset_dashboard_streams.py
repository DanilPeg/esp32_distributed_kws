"""Wipe all dashboard JSONL/state files so the next run starts clean.

By default truncates (size = 0) every stream the dashboard reads from:

    notes/Journal/hash_kws_telemetry/events.jsonl
    notes/Journal/hash_kws_telemetry/state.json
    notes/Journal/hash_kws_telemetry/node*/events.jsonl
    notes/Journal/hash_kws_telemetry/node*/state.json
    notes/Journal/hash_kws_telemetry/node*/raw.log
    notes/Journal/hash_kws_fusion/decisions.jsonl
    notes/Journal/hash_kws_fusion/state.json
    notes/Journal/hash_kws_fusion/master_demux_raw.log
    notes/Journal/hash_kws_cluster/state.json
    notes/Journal/hash_kws_cluster/sim_state.json

Plus stray atomic-write tmp files (*.tmp).

Usage:
    python code\\scripts\\reset_dashboard_streams.py            # truncate
    python code\\scripts\\reset_dashboard_streams.py --remove   # delete instead of truncate
    python code\\scripts\\reset_dashboard_streams.py --dry-run  # show what would happen
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
JOURNAL = REPO_ROOT / "notes" / "Journal"

TARGET_DIRS = [
    JOURNAL / "hash_kws_telemetry",
    JOURNAL / "hash_kws_fusion",
    JOURNAL / "hash_kws_cluster",
]
TARGET_GLOBS = [
    "**/*.jsonl",
    "**/*.log",
    "**/state.json",
    "**/sim_state.json",
    "**/*.tmp",
]


def collect_targets() -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for d in TARGET_DIRS:
        if not d.exists():
            continue
        for pattern in TARGET_GLOBS:
            for p in d.glob(pattern):
                if p.is_file() and p not in seen:
                    seen.add(p)
                    out.append(p)
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset hash-KWS dashboard streams.")
    parser.add_argument("--remove", action="store_true",
                        help="Delete files instead of truncating to 0 bytes.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print actions without changing anything.")
    args = parser.parse_args()

    targets = collect_targets()
    if not targets:
        print("Nothing to clean.")
        return 0

    verb = "DELETE" if args.remove else "TRUNCATE"
    if args.dry_run:
        verb = f"(dry) {verb}"

    for path in targets:
        rel = path.relative_to(REPO_ROOT)
        size = path.stat().st_size
        print(f"{verb}  {rel}  ({size} bytes)")
        if args.dry_run:
            continue
        try:
            if args.remove:
                path.unlink()
            else:
                with path.open("w", encoding="utf-8") as fh:
                    fh.truncate(0)
        except OSError as exc:
            print(f"  ! failed: {exc}", file=sys.stderr)

    print(f"\nProcessed {len(targets)} files.")
    print("Now restart your demux bridge and `run_dashboard.py`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
