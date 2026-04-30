from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .config import ExperimentConfig
from .models import collect_layer_inventory, summarize_model


def _cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def export_model_bundle(
    model: nn.Module,
    experiment: ExperimentConfig,
    stage_name: str = "student",
) -> dict[str, Any]:
    artifact_dir = Path(experiment.export.artifacts_dir) / experiment.tag
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_stem = f"{experiment.export.model_stem}_{stage_name}"
    bundle_path = artifact_dir / f"{model_stem}.pt"
    metadata_path = artifact_dir / f"{model_stem}_metadata.json"

    model_summary = summarize_model(model, experiment)
    bundle = {
        "experiment": experiment.to_dict(),
        "stage_name": stage_name,
        "model_summary": model_summary,
        "layer_inventory": collect_layer_inventory(model),
        "state_dict": _cpu_state_dict(model),
    }
    torch.save(bundle, bundle_path)

    metadata = {
        "experiment": experiment.to_dict(),
        "stage_name": stage_name,
        "bundle_path": str(bundle_path),
        "model_summary": model_summary,
        "layer_inventory": collect_layer_inventory(model),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "bundle": {
            "path": str(bundle_path),
            "metadata_path": str(metadata_path),
        },
        "model_summary": model_summary,
    }
