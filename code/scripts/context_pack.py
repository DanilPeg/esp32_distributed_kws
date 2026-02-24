#!/usr/bin/env python3
"""Build a compact context pack for the agent."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def journal_dir() -> Path:
    return repo_root() / "notes" / "Journal"


def research_dir() -> Path:
    return repo_root() / "notes" / "Research"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def find_latest_checkpoint() -> Path | None:
    files = []
    for p in journal_dir().glob("*.*"):
        if p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in (".yaml", ".yml", ".md", ".txt"):
            continue
        if "telegram" in p.parts:
            continue
        try:
            txt = read_text(p)
            if "checkpoint_id:" in txt:
                files.append(p)
        except Exception:
            continue
    if not files:
        return None
    return max(files, key=lambda x: x.stat().st_mtime)


def latest_journal_entries(n: int) -> list[Path]:
    files = []
    for p in journal_dir().glob("*.*"):
        if p.is_dir():
            continue
        if p.name.startswith("."):
            continue
        if p.suffix.lower() not in (".yaml", ".yml", ".md", ".txt"):
            continue
        if "telegram" in p.parts:
            continue
        files.append(p)
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[:n]


def run_rg(query: str, max_snippets: int, context_lines: int) -> str:
    if not query:
        return ""
    rg = "rg"
    try:
        proc = subprocess.run(
            [
                rg,
                "-n",
                "-i",
                "--context",
                str(context_lines),
                "--max-count",
                str(max_snippets),
                query,
                str(research_dir()),
                str(journal_dir()),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return "rg not found"
    out = (proc.stdout or "").strip()
    return out


def build_context_pack(
    query: str,
    journal_count: int,
    max_snippets: int,
    context_lines: int,
) -> str:
    now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    bible_path = repo_root() / "BIBLE.md"
    checkpoint_path = find_latest_checkpoint()
    journals = latest_journal_entries(journal_count)
    retrieval = run_rg(query, max_snippets, context_lines)

    parts = []
    parts.append("# Context Pack")
    parts.append(f"Generated: {now}")
    parts.append(f"Query: {query or '(none)'}")
    parts.append("")

    parts.append("## BIBLE.md")
    parts.append(read_text(bible_path))
    parts.append("")

    parts.append("## Latest Checkpoint")
    if checkpoint_path:
        parts.append(f"File: {checkpoint_path}")
        parts.append(read_text(checkpoint_path))
    else:
        parts.append("(none found)")
    parts.append("")

    parts.append("## Latest Journal Entries")
    if journals:
        for p in journals:
            parts.append(f"File: {p}")
            parts.append(read_text(p))
            parts.append("")
    else:
        parts.append("(none found)")
        parts.append("")

    parts.append("## Retrieval Snippets (rg)")
    if retrieval:
        parts.append(retrieval)
    else:
        parts.append("(no snippets)")
    parts.append("")

    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact context pack.")
    parser.add_argument("--query", default="", help="Query for retrieval")
    parser.add_argument("--journal-count", type=int, default=2)
    parser.add_argument("--max-snippets", type=int, default=7)
    parser.add_argument("--context-lines", type=int, default=2)
    parser.add_argument(
        "--out",
        default=str(journal_dir() / "context_pack_latest.md"),
        help="Output file",
    )
    args = parser.parse_args()

    pack = build_context_pack(
        query=args.query,
        journal_count=args.journal_count,
        max_snippets=args.max_snippets,
        context_lines=args.context_lines,
    )
    Path(args.out).write_text(pack, encoding="utf-8")
    print(f"Context pack written to: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
