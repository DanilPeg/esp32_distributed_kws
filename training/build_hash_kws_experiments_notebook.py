from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


RUNTIME_FILES = [
    "code/training/README-hash-kws.md",
    "code/training/requirements-kws-hash.txt",
    "code/training/requirements-kws-hash-exact-frontend.txt",
    "code/training/hash_kws_lab/__init__.py",
    "code/training/hash_kws_lab/config.py",
    "code/training/hash_kws_lab/recipes.py",
    "code/training/hash_kws_lab/data.py",
    "code/training/hash_kws_lab/models.py",
    "code/training/hash_kws_lab/trainer.py",
    "code/training/hash_kws_lab/reporting.py",
    "code/training/hash_kws_lab/export.py",
    "code/scripts/export_hash_kws_firmware.py",
    "code/firmware/hash_kws_runtime/README.md",
    "code/firmware/hash_kws_runtime/hash_model_types.h",
    "code/firmware/hash_kws_runtime/hash_model_settings.h",
    "code/firmware/hash_kws_runtime/hash_model_settings.cpp",
    "code/firmware/hash_kws_runtime/hash_model_data.h",
    "code/firmware/hash_kws_runtime/hash_model_data.cpp",
    "code/firmware/hash_kws_runtime/hash_recognize_commands.h",
    "code/firmware/hash_kws_runtime/hash_recognize_commands.cpp",
    "code/firmware/hash_kws_runtime/hash_kws_runner.h",
    "code/firmware/hash_kws_runtime/hash_kws_runner.cpp",
    "code/firmware/hash_kws_runtime/hash_micro_speech.cpp",
]


def md_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).lstrip("\n").splitlines(keepends=True),
    }


def build_runtime_payloads(repo_root: Path) -> dict[str, str]:
    payloads: dict[str, str] = {}
    for relative_path in RUNTIME_FILES:
        source_path = repo_root / relative_path
        payloads[relative_path] = source_path.read_text(encoding="utf-8")
    return payloads


def build_notebook(
    runtime_payloads: dict[str, str],
    *,
    notebook_title: str = "Hash KWS Experiment Lab",
    notebook_description: str = (
        "This notebook is the orchestration layer for the parallel hash-based KWS branch."
    ),
    notebook_goals: list[str] | None = None,
    pip_comment: str = '# !pip -q install "torch>=2.2" "torchaudio>=2.2" "matplotlib>=3.8,<4"',
    selected_recipe: str = "hash_deeper_fair_ce",
    next_iteration_ideas: list[str] | None = None,
) -> dict:
    payload_literal = json.dumps(runtime_payloads, ensure_ascii=False)
    notebook_goals = notebook_goals or [
        "reproduce the current analytic-hash DSCNN results cleanly;",
        "compare fair class mixes without losing the original user baseline;",
        "try likely upgrades such as augmentation, signed hashing, pointwise-only hashing, and KD;",
        "export compact bundles for a future custom ESP32 runtime.",
    ]
    next_iteration_ideas = next_iteration_ideas or [
        "Start from `hash_deeper_fair_ce` and only then compare against `hash_deeper_fair_augmented`.",
        "If the stronger fair recipe still beats the ordinary student branch, run `hash_deeper_fair_signed`.",
        "Use `dense_teacher_fair` only as a support branch to test whether KD helps a hash student that is already strong on CE.",
        "Keep `hash_pointwise_only_fair` in the loop because it is the most promising path for a later ESP32 custom runtime.",
    ]
    goals_md = "\n".join(f"- {goal}" for goal in notebook_goals)
    next_ideas_md = "\n".join(f"- {idea}" for idea in next_iteration_ideas)
    cells = [
        md_cell(
            f"""
            # {notebook_title}

            {notebook_description}

            Design goals:
            {goals_md}
            """
        ),
        code_cell(
            f"""
            # Uncomment if the notebook runtime is missing PyTorch audio packages:
            {pip_comment}
            """
        ),
        code_cell(
            """
            import importlib
            import json
            import sys
            from pathlib import Path

            FORCE_SYNC_RUNTIME_FILES = True
            BASE_RUNTIME_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
            PROJECT_ROOT = BASE_RUNTIME_DIR / "diploma_esp32_distributed_nn"
            TRAINING_ROOT = PROJECT_ROOT / "code" / "training"
            SCRIPTS_ROOT = PROJECT_ROOT / "code" / "scripts"
            FILE_PAYLOADS = json.loads(%PAYLOAD_LITERAL%)

            def ensure_runtime_files(root: Path, payloads: dict[str, str], overwrite: bool = False):
                created = []
                skipped = []
                for relative_path, content in payloads.items():
                    target_path = root / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    if target_path.exists() and not overwrite:
                        skipped.append(relative_path)
                        continue
                    target_path.write_text(content, encoding="utf-8")
                    created.append(relative_path)
                return created, skipped

            created_files, skipped_files = ensure_runtime_files(
                PROJECT_ROOT,
                FILE_PAYLOADS,
                overwrite=FORCE_SYNC_RUNTIME_FILES,
            )

            if str(TRAINING_ROOT) not in sys.path:
                sys.path.insert(0, str(TRAINING_ROOT))
            if str(SCRIPTS_ROOT) not in sys.path:
                sys.path.insert(0, str(SCRIPTS_ROOT))

            importlib.invalidate_caches()
            for module_name in list(sys.modules):
                if module_name == "hash_kws_lab" or module_name.startswith("hash_kws_lab."):
                    del sys.modules[module_name]

            print("Runtime project root:", PROJECT_ROOT)
            print("Runtime training root:", TRAINING_ROOT)
            print("Created files:", len(created_files))
            print("Skipped existing files:", len(skipped_files))
            print("Hash KWS package exists:", (TRAINING_ROOT / "hash_kws_lab").is_dir())
            """
            .replace("%PAYLOAD_LITERAL%", repr(payload_literal))
        ),
        code_cell(
            """
            import json
            from pathlib import Path

            import matplotlib.pyplot as plt
            import torch

            from hash_kws_lab.config import make_experiment
            from hash_kws_lab.data import ensure_torchaudio_available, prepare_dataloaders
            from hash_kws_lab.export import export_model_bundle
            from hash_kws_lab.models import build_student_model, build_teacher_model, summarize_model
            from hash_kws_lab.recipes import build_recipe_book
            from hash_kws_lab.reporting import (
                add_note,
                initialize_run_state,
                record_export_artifacts,
                save_history,
                save_history_plots,
                save_json_artifact,
                save_metrics,
                save_model_summary,
                save_text_artifact,
                update_stage_state,
                write_run_summary,
            )
            from hash_kws_lab.trainer import evaluate, train_student, train_teacher
            import export_hash_kws_firmware as firmware_exporter

            ensure_torchaudio_available()
            torch.manual_seed(13)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(13)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Device:", device)
            """
        ),
        code_cell(
            """
            base_experiment = make_experiment(tag="hash_kws12_iterlab_v1", vocabulary_preset="kws12")
            recipe_book = build_recipe_book(base_experiment)
            print("Available recipes:", list(recipe_book))

            SELECTED_RECIPE = "%SELECTED_RECIPE%"
            experiment = recipe_book[SELECTED_RECIPE]
            RUN_NOTES = ""
            print(json.dumps(experiment.to_dict(), indent=2))
            """
            .replace("%SELECTED_RECIPE%", selected_recipe)
        ),
        code_cell(
            """
            AUTO_MOUNT_DRIVE_FOR_TEACHER = True
            REUSE_TEACHER_CHECKPOINT = True
            TEACHER_CHECKPOINT_PATH = ""
            AUTO_FIND_LATEST_TEACHER_CHECKPOINT = True
            TEACHER_SEARCH_ROOT = "/content/drive/MyDrive/diploma_hash_kws_teacher_saves"

            def maybe_mount_drive_for_teacher() -> None:
                if not AUTO_MOUNT_DRIVE_FOR_TEACHER:
                    return
                drive_root = Path("/content/drive")
                if drive_root.exists() and (drive_root / "MyDrive").exists():
                    return
                try:
                    from google.colab import drive
                    drive.mount("/content/drive", force_remount=False)
                except Exception as exc:
                    print("Google Drive mount skipped:", exc)

            def resolve_teacher_checkpoint() -> Path | None:
                candidate = TEACHER_CHECKPOINT_PATH.strip()
                if candidate:
                    path = Path(candidate)
                    return path if path.exists() else None
                if not AUTO_FIND_LATEST_TEACHER_CHECKPOINT or not experiment.uses_teacher:
                    return None
                search_root = Path(TEACHER_SEARCH_ROOT)
                if not search_root.exists():
                    return None
                teacher_tag = experiment.teacher_reuse_tag or experiment.tag
                matches = sorted(
                    search_root.glob(f"{teacher_tag}_*/teacher_best.pt"),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
                return matches[0] if matches else None

            maybe_mount_drive_for_teacher()
            resolved_teacher_checkpoint = resolve_teacher_checkpoint()
            print("Teacher reuse enabled:", REUSE_TEACHER_CHECKPOINT)
            print("Resolved teacher checkpoint:", resolved_teacher_checkpoint)
            """
        ),
        code_cell(
            """
            import os

            SPEECHCOMMANDS_DATA_ROOT = ""
            if SPEECHCOMMANDS_DATA_ROOT:
                os.environ["SPEECHCOMMANDS_DATA_ROOT"] = SPEECHCOMMANDS_DATA_ROOT
            print("SPEECHCOMMANDS_DATA_ROOT =", os.environ.get("SPEECHCOMMANDS_DATA_ROOT", ""))
            """
        ),
        code_cell(
            """
            bundle = prepare_dataloaders(PROJECT_ROOT, experiment, device=device)
            loaders = bundle["loaders"]

            run_dir = initialize_run_state(
                PROJECT_ROOT,
                experiment,
                recipe_name=SELECTED_RECIPE,
                dataset_summary=bundle["summary"],
            )
            save_json_artifact(run_dir, "experiment.json", experiment.to_dict())
            save_json_artifact(run_dir, "dataset_summary.json", bundle["summary"])
            save_text_artifact(run_dir, "selected_recipe.txt", SELECTED_RECIPE + "\\n")
            if RUN_NOTES.strip():
                save_text_artifact(run_dir, "run_notes.txt", RUN_NOTES.strip() + "\\n")
                add_note(run_dir, "Operator notes", RUN_NOTES.strip())

            print("Data root:", bundle["data_root"])
            print("Run dir:", run_dir)
            print(json.dumps(bundle["summary"], indent=2))
            print("Cache summary:", json.dumps(bundle.get("cache_summary", {}), indent=2))
            print("Feature preview batch shape:", bundle["feature_preview_shape"])
            """
        ),
        code_cell(
            """
            preview_features, preview_targets = next(iter(loaders["train"]))
            print("Preview features:", preview_features.shape)
            print("Preview targets:", preview_targets.shape)

            plt.figure(figsize=(10, 4))
            plt.imshow(preview_features[0].numpy(), aspect="auto", origin="lower")
            plt.title("Example hash-KWS training feature map")
            plt.xlabel("Frames")
            plt.ylabel("Mel bins")
            plt.colorbar()
            plt.show()
            plt.close()
            """
        ),
        code_cell(
            """
            teacher = build_teacher_model(experiment)
            student = build_student_model(experiment)

            student_summary = summarize_model(student, experiment)
            student_summary_path = save_model_summary(
                run_dir,
                "student",
                str(student) + "\\n\\n" + json.dumps(student_summary, indent=2),
            )
            save_json_artifact(run_dir, "student_model_inventory.json", student_summary)
            print("Student summary:")
            print(json.dumps(student_summary, indent=2))

            teacher_summary = None
            teacher_summary_path = None
            if teacher is not None:
                teacher_summary = summarize_model(teacher, experiment)
                teacher_summary_path = save_model_summary(
                    run_dir,
                    "teacher",
                    str(teacher) + "\\n\\n" + json.dumps(teacher_summary, indent=2),
                )
                save_json_artifact(run_dir, "teacher_model_inventory.json", teacher_summary)
                print("Teacher summary:")
                print(json.dumps(teacher_summary, indent=2))
            else:
                print("Teacher is disabled for this recipe.")
            """
        ),
        code_cell(
            """
            teacher_checkpoint_path = run_dir / "teacher_best.pt"
            teacher_test_metrics = {}

            if teacher is not None:
                if resolved_teacher_checkpoint is not None and REUSE_TEACHER_CHECKPOINT:
                    checkpoint = torch.load(resolved_teacher_checkpoint, map_location="cpu")
                    teacher.load_state_dict(checkpoint["state_dict"], strict=True)
                    teacher = teacher.to(device)
                    add_note(
                        run_dir,
                        "Teacher reuse",
                        f"Loaded teacher checkpoint from `{resolved_teacher_checkpoint}`.",
                    )
                else:
                    teacher_train_result = train_teacher(teacher, loaders=loaders, experiment=experiment, device=device)
                    teacher.load_state_dict(teacher_train_result["best_state"], strict=True)
                    torch.save(
                        {
                            "experiment": experiment.to_dict(),
                            "state_dict": {key: value.detach().cpu().clone() for key, value in teacher.state_dict().items()},
                            "result": teacher_train_result,
                        },
                        teacher_checkpoint_path,
                    )
                    teacher_history_path = save_history(run_dir, "teacher", teacher_train_result["history"])
                    teacher_plot_paths = save_history_plots(run_dir, "teacher", teacher_train_result["history"])
                    teacher_test_metrics = evaluate(
                        teacher,
                        loaders["test"],
                        device=device,
                        top_k=experiment.train.top_k,
                        use_amp=experiment.train.use_amp,
                        desc="hash_teacher | test",
                    )
                    save_metrics(run_dir, "teacher", teacher_test_metrics)
                    update_stage_state(
                        run_dir,
                        "teacher",
                        metrics=teacher_test_metrics,
                        history_path=str(teacher_history_path),
                        plot_paths=teacher_plot_paths,
                        summary_path=str(teacher_summary_path) if teacher_summary_path else "",
                        extra={
                            "checkpoint_path": str(teacher_checkpoint_path),
                            "best_val_accuracy": teacher_train_result["best_val_accuracy"],
                            "best_epoch": teacher_train_result["best_epoch"],
                            "elapsed_sec": teacher_train_result["elapsed_sec"],
                        },
                    )
                if not teacher_test_metrics:
                    teacher_test_metrics = evaluate(
                        teacher,
                        loaders["test"],
                        device=device,
                        top_k=experiment.train.top_k,
                        use_amp=experiment.train.use_amp,
                        desc="hash_teacher | reused-test",
                    )
                    save_metrics(run_dir, "teacher", teacher_test_metrics)
                    update_stage_state(
                        run_dir,
                        "teacher",
                        metrics=teacher_test_metrics,
                        history_path="",
                        plot_paths=[],
                        summary_path=str(teacher_summary_path) if teacher_summary_path else "",
                        extra={
                            "checkpoint_path": str(resolved_teacher_checkpoint),
                            "reused_from_checkpoint": True,
                        },
                    )
                print("Teacher test metrics:", teacher_test_metrics)
            """
        ),
        code_cell(
            """
            student_checkpoint_path = run_dir / "student_best.pt"
            student_train_result = train_student(
                student,
                loaders=loaders,
                experiment=experiment,
                device=device,
                teacher=teacher,
            )
            student.load_state_dict(student_train_result["best_state"], strict=True)
            torch.save(
                {
                    "experiment": experiment.to_dict(),
                    "state_dict": {key: value.detach().cpu().clone() for key, value in student.state_dict().items()},
                    "result": student_train_result,
                },
                student_checkpoint_path,
            )

            student_history_path = save_history(run_dir, "student", student_train_result["history"])
            student_plot_paths = save_history_plots(run_dir, "student", student_train_result["history"])
            save_metrics(run_dir, "student", student_train_result["test_metrics"])
            update_stage_state(
                run_dir,
                "student",
                metrics=student_train_result["test_metrics"],
                history_path=str(student_history_path),
                plot_paths=student_plot_paths,
                summary_path=str(student_summary_path),
                extra={
                    "checkpoint_path": str(student_checkpoint_path),
                    "stage_summaries": student_train_result["stage_summaries"],
                },
            )
            print("Student test metrics:", student_train_result["test_metrics"])
            """
        ),
        code_cell(
            """
            export_metadata = export_model_bundle(student, experiment=experiment, stage_name="student")
            record_export_artifacts(run_dir, export_metadata)
            save_json_artifact(run_dir, "export_metadata_snapshot.json", export_metadata)
            print(json.dumps(export_metadata, indent=2))
            """
        ),
        code_cell(
            """
            AUTO_EXPORT_FIRMWARE_RUNTIME = True
            FIRMWARE_CALIBRATION_SPLIT = "validation"
            FIRMWARE_CALIBRATION_BATCHES = 8
            FIRMWARE_EXPORT_DIR = PROJECT_ROOT / "code" / "firmware" / "hash_kws_runtime"

            firmware_export_result = {}
            bundle_path = Path(export_metadata["bundle"]["path"])
            if AUTO_EXPORT_FIRMWARE_RUNTIME:
                firmware_export_result = firmware_exporter.export_bundle_to_firmware(
                    bundle_path=bundle_path,
                    output_dir=FIRMWARE_EXPORT_DIR,
                    project_root=PROJECT_ROOT,
                    device=device,
                    calibration_split=FIRMWARE_CALIBRATION_SPLIT,
                    calibration_batches=FIRMWARE_CALIBRATION_BATCHES,
                )
                record_export_artifacts(run_dir, {"firmware_export": firmware_export_result})
                save_json_artifact(run_dir, "firmware_export_snapshot.json", firmware_export_result)
                print(json.dumps(firmware_export_result, indent=2))
            else:
                print("Firmware export disabled.")
            """
        ),
        code_cell(
            """
            DRIVE_TEACHER_EXPORT_DIR = ""
            DRIVE_RUN_EXPORT_DIR = ""

            if DRIVE_TEACHER_EXPORT_DIR and (run_dir / "teacher_best.pt").exists():
                target_dir = Path(DRIVE_TEACHER_EXPORT_DIR) / run_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)
                teacher_export_path = target_dir / "teacher_best.pt"
                teacher_export_path.write_bytes((run_dir / "teacher_best.pt").read_bytes())
                print("Copied teacher checkpoint:", teacher_export_path)
            """
        ),
        code_cell(
            """
            import shutil

            summary_path = write_run_summary(run_dir)
            archive_base = run_dir.parent / f"{run_dir.name}_bundle"
            archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=run_dir)
            print("Run summary:", summary_path)
            print("Run archive:", archive_path)

            if DRIVE_RUN_EXPORT_DIR:
                drive_export_dir = Path(DRIVE_RUN_EXPORT_DIR)
                drive_export_dir.mkdir(parents=True, exist_ok=True)
                exported_archive = drive_export_dir / Path(archive_path).name
                shutil.copy2(archive_path, exported_archive)
                print("Copied archive:", exported_archive)
            """
        ),
        md_cell(
            f"""
            ## Next Iteration Ideas

            {next_ideas_md}
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    notebook_path = root / "hash_kws_experiments_esp32.ipynb"
    notebook_path.write_text(
        json.dumps(
            build_notebook(
                build_runtime_payloads(root.parents[1]),
                selected_recipe="hash_deeper_fair_ce",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(notebook_path)


if __name__ == "__main__":
    main()
