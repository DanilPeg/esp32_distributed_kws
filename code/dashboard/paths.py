"""Canonical filesystem paths the dashboard reads from.

Everything resolves relative to the project root so the dashboard behaves the
same whether you launch ``uvicorn`` from the repo root or from the
``code/dashboard`` directory. No writes happen through this module — the
dashboard is strictly read-only.
"""

from __future__ import annotations

from pathlib import Path

# code/dashboard/paths.py -> parents[2] == repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

JOURNAL_DIR: Path = PROJECT_ROOT / "notes" / "Journal"

TELEMETRY_DIR: Path = JOURNAL_DIR / "hash_kws_telemetry"
TELEMETRY_EVENTS: Path = TELEMETRY_DIR / "events.jsonl"
TELEMETRY_STATE: Path = TELEMETRY_DIR / "state.json"
TELEMETRY_RAW_LOG: Path = TELEMETRY_DIR / "raw.log"

FUSION_DIR: Path = JOURNAL_DIR / "hash_kws_fusion"
FUSION_DECISIONS: Path = FUSION_DIR / "decisions.jsonl"
FUSION_STATE: Path = FUSION_DIR / "state.json"

CLUSTER_DIR: Path = JOURNAL_DIR / "hash_kws_cluster"
CLUSTER_STATE: Path = CLUSTER_DIR / "state.json"

# Per-node event files live in TELEMETRY_DIR / <nodeN> / events.jsonl — we glob
# at read time so new nodes appear without code changes.
NODE_GLOB_PREFIX: str = "node*"

# Canonical node metadata for the hash-ensemble distributed demo.
# Three inference nodes broadcast over ESP-NOW; the dashboard sees their
# data through the master's USB Serial (demuxed by hash_kws_master_demux_bridge.py).
# The master itself owns the fusion stream — it appears in the Fusion
# decisions table, not in this per-node grid.
NODE_ROLES: dict[str, dict[str, str]] = {
    "node1": {"role": "inference", "label": "Node 1 — ens_a"},
    "node2": {"role": "inference", "label": "Node 2 — ens_b"},
    "node3": {"role": "inference", "label": "Node 3 — ens_c"},
}

# Order of node cards on the page — keeps node1 leftmost regardless of which
# JSONL file appeared first.
NODE_DISPLAY_ORDER: tuple[str, ...] = ("node1", "node2", "node3")
