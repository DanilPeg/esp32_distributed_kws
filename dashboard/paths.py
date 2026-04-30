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

# Canonical node metadata. These labels match the cluster sim's conventions:
#   node1 — real ESP32 with microphone
#   node2 — virtual / emulated audio peer (host-side)
#   node3 — virtual master / aggregator
# Unknown nodes inherit the "unknown" role so the UI still renders them.
NODE_ROLES: dict[str, dict[str, str]] = {
    "node1": {"role": "real", "label": "Real ESP32 (mic)"},
    "node2": {"role": "emulated", "label": "Emulated audio peer"},
    "node3": {"role": "master", "label": "Master / aggregator"},
}
