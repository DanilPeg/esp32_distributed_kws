from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .config import ExperimentConfig


def get_run_dir(project_root: Path, experiment: ExperimentConfig) -> Path:
    run_dir = project_root / "code" / "training" / "hash_runs" / experiment.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _state_path(run_dir: Path) -> Path:
    return run_dir / "run_state.json"


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "run_summary.md"


def _read_state(run_dir: Path) -> dict[str, Any]:
    path = _state_path(run_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    _state_path(run_dir).write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def initialize_run_state(
    project_root: Path,
    experiment: ExperimentConfig,
    recipe_name: str,
    dataset_summary: dict[str, Any] | None = None,
) -> Path:
    run_dir = get_run_dir(project_root, experiment)
    state = _merge_dict(
        _read_state(run_dir),
        {
            "experiment": experiment.to_dict(),
            "recipe_name": recipe_name,
            "dataset_summary": dataset_summary or {},
            "stages": {},
            "artifacts": {},
            "notes": [],
        },
    )
    _write_state(run_dir, state)
    write_run_summary(run_dir)
    return run_dir


def save_json_artifact(run_dir: Path, name: str, payload: Any) -> Path:
    path = run_dir / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def save_text_artifact(run_dir: Path, name: str, text: str) -> Path:
    path = run_dir / name
    path.write_text(text, encoding="utf-8")
    return path


def save_history(run_dir: Path, stage_name: str, history: list[dict[str, float]]) -> Path:
    return save_json_artifact(run_dir, f"{stage_name}_history.json", history)


def save_metrics(run_dir: Path, stage_name: str, metrics: dict[str, Any]) -> Path:
    return save_json_artifact(run_dir, f"{stage_name}_metrics.json", metrics)


def save_model_summary(run_dir: Path, stage_name: str, summary_text: str) -> Path:
    return save_text_artifact(run_dir, f"{stage_name}_model_summary.txt", summary_text)


def save_history_plots(run_dir: Path, stage_name: str, history: list[dict[str, float]]) -> list[str]:
    if not history:
        return []
    keys = history[0].keys()
    plot_paths: list[str] = []
    groups = {
        "loss": [key for key in keys if "loss" in key],
        "metrics": [key for key in keys if "accuracy" in key and key != "epoch"],
        "lr": [key for key in keys if key == "lr"],
    }
    for suffix, selected_keys in groups.items():
        if not selected_keys:
            continue
        epochs = [entry["epoch"] for entry in history]
        plt.figure(figsize=(10, 4))
        for key in selected_keys:
            plt.plot(epochs, [entry[key] for entry in history], label=key)
        plt.title(f"{stage_name} {suffix}")
        plt.xlabel("Epoch")
        plt.grid(True, alpha=0.3)
        plt.legend()
        path = run_dir / f"{stage_name}_{suffix}.png"
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        plot_paths.append(str(path))
    return plot_paths


def update_stage_state(
    run_dir: Path,
    stage_name: str,
    metrics: dict[str, Any] | None = None,
    history_path: str | None = None,
    plot_paths: list[str] | None = None,
    summary_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    state = _read_state(run_dir)
    stage_block = state.setdefault("stages", {}).get(stage_name, {})
    update = {
        "metrics": metrics or stage_block.get("metrics", {}),
        "history_path": history_path or stage_block.get("history_path", ""),
        "plot_paths": plot_paths or stage_block.get("plot_paths", []),
        "summary_path": summary_path or stage_block.get("summary_path", ""),
    }
    if extra:
        update["extra"] = _merge_dict(stage_block.get("extra", {}), extra)
    state.setdefault("stages", {})[stage_name] = _merge_dict(stage_block, update)
    _write_state(run_dir, state)
    write_run_summary(run_dir)


def add_note(run_dir: Path, title: str, body: str) -> None:
    state = _read_state(run_dir)
    state.setdefault("notes", []).append({"title": title, "body": body})
    _write_state(run_dir, state)
    write_run_summary(run_dir)


def record_export_artifacts(run_dir: Path, metadata: dict[str, Any]) -> None:
    state = _read_state(run_dir)
    state["artifacts"] = _merge_dict(state.get("artifacts", {}), metadata)
    _write_state(run_dir, state)
    write_run_summary(run_dir)


def write_run_summary(run_dir: Path) -> Path:
    state = _read_state(run_dir)
    lines = [
        f"# Run Summary: {state.get('experiment', {}).get('tag', run_dir.name)}",
        "",
        f"Recipe: `{state.get('recipe_name', '')}`",
        "",
        "## Experiment",
        "",
        "```json",
        json.dumps(state.get("experiment", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Dataset",
        "",
        "```json",
        json.dumps(state.get("dataset_summary", {}), ensure_ascii=False, indent=2),
        "```",
        "",
        "## Stages",
        "",
    ]

    for stage_name, payload in state.get("stages", {}).items():
        lines.extend(
            [
                f"### {stage_name}",
                "",
                "```json",
                json.dumps(payload.get("metrics", {}), ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
        if payload.get("history_path"):
            lines.append(f"History: `{payload['history_path']}`")
        if payload.get("summary_path"):
            lines.append(f"Model summary: `{payload['summary_path']}`")
        for plot_path in payload.get("plot_paths", []):
            lines.append(f"Plot: `{plot_path}`")
        if payload.get("extra"):
            lines.extend(
                [
                    "Extra:",
                    "```json",
                    json.dumps(payload["extra"], ensure_ascii=False, indent=2),
                    "```",
                ]
            )
        lines.append("")

    if state.get("artifacts"):
        lines.extend(
            [
                "## Artifacts",
                "",
                "```json",
                json.dumps(state["artifacts"], ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )

    if state.get("notes"):
        lines.extend(["## Notes", ""])
        for note in state["notes"]:
            lines.append(f"### {note.get('title', 'note')}")
            lines.append("")
            lines.append(note.get("body", ""))
            lines.append("")

    summary_path = _summary_path(run_dir)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path
