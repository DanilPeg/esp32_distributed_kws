from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


RUNTIME_FILES = [
    "code/training/README.md",
    "code/training/requirements-kws.txt",
    "code/training/kws_advanced/__init__.py",
    "code/training/kws_advanced/config.py",
    "code/training/kws_advanced/experiments.py",
    "code/training/kws_advanced/features.py",
    "code/training/kws_advanced/data.py",
    "code/training/kws_advanced/models.py",
    "code/training/kws_advanced/distillation.py",
    "code/training/kws_advanced/export.py",
    "code/training/kws_advanced/reporting.py",
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


def build_notebook(runtime_payloads: dict[str, str]) -> dict:
    payload_literal = json.dumps(runtime_payloads, ensure_ascii=False)
    cells = [
        md_cell(
            """
            # ESP32 KWS Experiment Lab

            This notebook is the orchestration layer for the rebuilt KWS training stack.

            Design goals:
            - experiment with multiple theories instead of one fixed recipe;
            - stay close to the `micro_speech` deployment path;
            - train a stronger student than the stock baseline;
            - export a full-int8 `TFLite` artifact and firmware-ready C array.
            """
        ),
        code_cell(
            """
            # Uncomment if the notebook environment is missing TensorFlow / tfmot,
            # or if TensorFlow fails to import because of NumPy 2.x:
            # !pip -q install "numpy<2" "tensorflow==2.15.*" "tensorflow-model-optimization>=0.7.0,<0.9" matplotlib
            """
        ),
        code_cell(
            """
            import json
            import importlib
            import sys
            from pathlib import Path

            FORCE_SYNC_RUNTIME_FILES = True
            BASE_RUNTIME_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
            PROJECT_ROOT = BASE_RUNTIME_DIR / "diploma_esp32_distributed_nn"
            TRAINING_ROOT = PROJECT_ROOT / "code" / "training"
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

            importlib.invalidate_caches()
            for module_name in list(sys.modules):
                if module_name == "kws_advanced" or module_name.startswith("kws_advanced."):
                    del sys.modules[module_name]

            print("Runtime project root:", PROJECT_ROOT)
            print("Runtime training root:", TRAINING_ROOT)
            print("Created files:", len(created_files))
            print("Skipped existing files:", len(skipped_files))
            print("Training package exists:", (TRAINING_ROOT / "kws_advanced").is_dir())
            print("Runtime source sync mode: force overwrite")
            """.replace("%PAYLOAD_LITERAL%", repr(payload_literal))
        ),
        code_cell(
            """
            import json

            import matplotlib.pyplot as plt
            import tensorflow as tf

            from kws_advanced.config import make_experiment
            from kws_advanced.data import prepare_datasets
            from kws_advanced.distillation import (
                build_callbacks_for_monitor,
                compile_classifier,
                maybe_quantize_aware_clone,
                train_student_model,
            )
            from kws_advanced.experiments import build_recipe_book
            from kws_advanced.export import export_experiment_artifacts
            from kws_advanced.models import (
                build_student_model,
                build_teacher_model,
                estimate_maccs,
                required_tflite_ops,
            )
            from kws_advanced.reporting import (
                add_note,
                initialize_run_state,
                record_export_artifacts,
                save_current_figure,
                save_history,
                save_history_plots,
                save_json_artifact,
                save_metrics,
                save_model_summary,
                save_text_artifact,
                update_stage_state,
                write_run_summary,
            )

            tf.keras.utils.set_random_seed(42)
            print("Project root:", PROJECT_ROOT)
            print("Training root:", TRAINING_ROOT)
            """
        ),
        code_cell(
            """
            base_experiment = make_experiment(tag="kws12_iterlab_v1", vocabulary_preset="kws12")
            recipe_book = build_recipe_book(base_experiment)
            print("Available recipes:", list(recipe_book))

            SELECTED_RECIPE = "distilled_student_v3_bcresnet_teacher_seq"
            experiment = recipe_book[SELECTED_RECIPE]
            RUN_NOTES = ""
            print(json.dumps(experiment.to_dict(), indent=2))
            """
        ),
        code_cell(
            """
            AUTO_MOUNT_DRIVE_FOR_TEACHER = True
            REUSE_TEACHER_CHECKPOINT = True
            TEACHER_CHECKPOINT_PATH = ""
            AUTO_FIND_LATEST_TEACHER_CHECKPOINT = True
            TEACHER_SEARCH_ROOT = "/content/drive/MyDrive/diploma_esp32_teacher_saves"

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
                if not AUTO_FIND_LATEST_TEACHER_CHECKPOINT:
                    return None
                search_root = Path(TEACHER_SEARCH_ROOT)
                if not search_root.exists():
                    return None
                teacher_tag = experiment.teacher_reuse_tag or experiment.tag
                matches = sorted(
                    search_root.glob(f"{teacher_tag}_*/teacher_best.keras"),
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

            # Optional dataset overrides for SSL-restricted or offline runtimes.
            # Set only one of these if automatic download fails.
            SPEECH_COMMANDS_DATASET_ROOT = ""
            SPEECH_COMMANDS_ARCHIVE_PATH = ""

            if SPEECH_COMMANDS_DATASET_ROOT:
                os.environ["SPEECH_COMMANDS_DATASET_ROOT"] = SPEECH_COMMANDS_DATASET_ROOT
            if SPEECH_COMMANDS_ARCHIVE_PATH:
                os.environ["SPEECH_COMMANDS_ARCHIVE_PATH"] = SPEECH_COMMANDS_ARCHIVE_PATH

            print("SPEECH_COMMANDS_DATASET_ROOT =", os.environ.get("SPEECH_COMMANDS_DATASET_ROOT", ""))
            print("SPEECH_COMMANDS_ARCHIVE_PATH =", os.environ.get("SPEECH_COMMANDS_ARCHIVE_PATH", ""))
            """
        ),
        code_cell(
            """
            import subprocess
            import tarfile

            DATA_DIR = PROJECT_ROOT / "data"
            ARCHIVE_PATH = DATA_DIR / "speech_commands_v0.02.tar.gz"
            EXTRACTED_DIR = DATA_DIR / "speech_commands_v0.02"
            SOURCE_URL = "https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

            def dataset_ready(root: Path) -> bool:
                required = [
                    root / "validation_list.txt",
                    root / "testing_list.txt",
                    root / "yes",
                    root / "no",
                ]
                return all(path.exists() for path in required)

            DATA_DIR.mkdir(parents=True, exist_ok=True)

            if dataset_ready(EXTRACTED_DIR):
                print("Dataset already prepared:", EXTRACTED_DIR)
            else:
                if not ARCHIVE_PATH.exists():
                    commands = [
                        ["bash", "-lc", f"wget --no-check-certificate -O '{ARCHIVE_PATH.as_posix()}' '{SOURCE_URL}'"],
                        ["bash", "-lc", f"curl -L -k '{SOURCE_URL}' -o '{ARCHIVE_PATH.as_posix()}'"],
                    ]
                    last_error = None
                    for command in commands:
                        try:
                            print("Trying:", " ".join(command[:2]), "...")
                            subprocess.run(command, check=True)
                            last_error = None
                            break
                        except Exception as exc:
                            last_error = exc
                    if last_error is not None:
                        raise RuntimeError(f"Failed to download dataset archive: {last_error}")

                EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
                with tarfile.open(ARCHIVE_PATH, "r:gz") as archive:
                    archive.extractall(path=EXTRACTED_DIR)

                if not dataset_ready(EXTRACTED_DIR):
                    raise RuntimeError(
                        "Extraction finished, but the dataset structure is incomplete: "
                        f"{EXTRACTED_DIR}"
                    )

                print("Dataset prepared:", EXTRACTED_DIR)
            """
        ),
        code_cell(
            """
            bundle = prepare_datasets(PROJECT_ROOT, experiment)
            teacher_train_ds = bundle["datasets"]["teacher_train"]
            teacher_val_ds = bundle["datasets"]["teacher_validation"]
            teacher_test_ds = bundle["datasets"]["teacher_test"]
            student_train_ds = bundle["datasets"]["student_train"]
            student_val_ds = bundle["datasets"]["student_validation"]
            student_test_ds = bundle["datasets"]["student_test"]
            distillation_train_ds = bundle["datasets"]["distillation_train"]
            distillation_val_ds = bundle["datasets"]["distillation_validation"]
            steps_per_epoch = bundle["steps_per_epoch"]

            run_dir = initialize_run_state(
                PROJECT_ROOT,
                experiment,
                recipe_name=SELECTED_RECIPE,
                dataset_summary=bundle["summary"],
            )
            teacher_dir = run_dir / "teacher"
            student_dir = run_dir / "student"
            qat_dir = run_dir / "qat"
            save_json_artifact(run_dir, "experiment.json", experiment.to_dict())
            save_json_artifact(run_dir, "dataset_summary.json", bundle["summary"])
            save_json_artifact(run_dir, "frontend_runtime.json", bundle["frontend"])
            save_text_artifact(run_dir, "selected_recipe.txt", SELECTED_RECIPE + "\\n")
            if RUN_NOTES.strip():
                save_text_artifact(run_dir, "run_notes.txt", RUN_NOTES.strip() + "\\n")
                add_note(run_dir, "Operator notes", RUN_NOTES.strip())

            print("Dataset root:", bundle["dataset_root"])
            print("Steps per epoch:", steps_per_epoch)
            print("Run dir:", run_dir)
            print(json.dumps(bundle["summary"], indent=2))
            print(json.dumps(bundle["frontend"], indent=2))
            """
        ),
        code_cell(
            """
            batch_features, batch_labels = next(iter(student_train_ds.take(1)))
            print("Student feature batch shape:", batch_features.shape)
            print("Label batch shape:", batch_labels.shape)

            plt.figure(figsize=(10, 4))
            plt.imshow(batch_features[0, :, :, 0].numpy().T, aspect="auto", origin="lower")
            plt.title("Example student training feature map")
            plt.xlabel("Frames")
            plt.ylabel("Channels")
            plt.colorbar()
            preview_path = save_current_figure(run_dir, "student_feature_map_preview.png")
            print("Saved preview:", preview_path)
            plt.show()
            plt.close()

            teacher_preview_features, _ = next(iter(teacher_train_ds.take(1)))
            if tuple(teacher_preview_features.shape[1:]) != tuple(batch_features.shape[1:]):
                plt.figure(figsize=(10, 4))
                plt.imshow(teacher_preview_features[0, :, :, 0].numpy().T, aspect="auto", origin="lower")
                plt.title("Example teacher training feature map")
                plt.xlabel("Frames")
                plt.ylabel("Channels")
                plt.colorbar()
                teacher_preview_path = save_current_figure(run_dir, "teacher_feature_map_preview.png")
                print("Saved teacher preview:", teacher_preview_path)
                plt.show()
                plt.close()
            """
        ),
        code_cell(
            """
            teacher = build_teacher_model(experiment)
            student = build_student_model(experiment)

            teacher_params = teacher.count_params()
            teacher_maccs = estimate_maccs(teacher)
            student_params = student.count_params()
            student_maccs = estimate_maccs(student)

            teacher_summary_path = save_model_summary(run_dir, teacher, "teacher")
            student_summary_path = save_model_summary(run_dir, student, "student")
            model_inventory = {
                "teacher": {
                    "params": teacher_params,
                    "maccs_rough": teacher_maccs,
                    "summary_path": str(teacher_summary_path),
                },
                "student": {
                    "params": student_params,
                    "maccs_rough": student_maccs,
                    "summary_path": str(student_summary_path),
                },
            }
            save_json_artifact(run_dir, "model_inventory.json", model_inventory)

            print("Teacher params:", teacher_params)
            print("Teacher MACs (rough):", teacher_maccs)
            print("Student params:", student_params)
            print("Student MACs (rough):", student_maccs)

            teacher.summary()
            student.summary()
            """
        ),
        code_cell(
            """
            teacher_reused = False
            teacher_checkpoint_path = resolved_teacher_checkpoint if REUSE_TEACHER_CHECKPOINT else None

            if teacher_checkpoint_path is not None and Path(teacher_checkpoint_path).exists():
                teacher = tf.keras.models.load_model(teacher_checkpoint_path)
                teacher_reused = True
                teacher_history = None
                teacher_summary_path = save_model_summary(run_dir, teacher, "teacher")
                add_note(
                    run_dir,
                    "Teacher reuse",
                    f"Loaded teacher checkpoint from `{teacher_checkpoint_path}`.",
                )
                print("Loaded teacher from:", teacher_checkpoint_path)
            else:
                teacher = compile_classifier(
                    teacher,
                    experiment=experiment,
                    learning_rate=experiment.train.teacher_lr,
                    epochs=experiment.train.teacher_epochs,
                    steps_per_epoch=steps_per_epoch,
                    label_smoothing=experiment.model.teacher_label_smoothing,
                    warmup_epochs=experiment.train.teacher_warmup_epochs,
                )
                teacher_history = teacher.fit(
                    teacher_train_ds,
                    validation_data=teacher_val_ds,
                    epochs=experiment.train.teacher_epochs,
                    callbacks=build_callbacks_for_monitor(
                        teacher_dir,
                        patience=experiment.train.teacher_early_stopping_patience,
                        monitor="val_accuracy",
                        mode="max",
                        include_checkpoint=True,
                        min_delta=experiment.train.teacher_early_stopping_min_delta,
                    ),
                )

            teacher_test_metrics = teacher.evaluate(teacher_test_ds, return_dict=True)
            teacher_history_path = save_history(run_dir, teacher_history, "teacher")
            save_metrics(run_dir, teacher_test_metrics, "teacher")
            teacher_plot_paths = save_history_plots(run_dir, teacher_history, "teacher")
            update_stage_state(
                run_dir,
                "teacher",
                metrics=teacher_test_metrics,
                history_path=str(teacher_history_path) if teacher_history_path else "",
                plot_paths=teacher_plot_paths,
                summary_path=str(teacher_summary_path),
                extra={
                    "params": teacher_params,
                    "maccs_rough": teacher_maccs,
                    "epoch_log_path": "" if teacher_reused else str(teacher_dir / "epoch_log.csv"),
                    "checkpoint_path": (
                        str(teacher_checkpoint_path)
                        if teacher_reused and teacher_checkpoint_path is not None
                        else str(teacher_dir / "best.keras")
                    ),
                    "reused_from_checkpoint": teacher_reused,
                },
            )
            print("Teacher test metrics:", teacher_test_metrics)
            """
        ),
        code_cell(
            """
            student_training = train_student_model(
                student=student,
                teacher=teacher,
                train_ds=distillation_train_ds,
                val_ds=distillation_val_ds,
                experiment=experiment,
                steps_per_epoch=steps_per_epoch,
                output_dir=student_dir,
            )
            student = student_training["student"]
            student_history = student_training["history"]
            student_training_extra = student_training["extra"]

            student = compile_classifier(
                student,
                experiment=experiment,
                learning_rate=experiment.train.student_lr,
                epochs=1,
                steps_per_epoch=max(1, steps_per_epoch),
            )
            student_test_metrics = student.evaluate(student_test_ds, return_dict=True)
            student_history_path = save_history(run_dir, student_history, "student")
            save_metrics(run_dir, student_test_metrics, "student")
            student_plot_paths = save_history_plots(run_dir, student_history, "student")
            update_stage_state(
                run_dir,
                "student",
                metrics=student_test_metrics,
                history_path=str(student_history_path) if student_history_path else "",
                plot_paths=student_plot_paths,
                summary_path=str(student_summary_path),
                extra={
                    "params": student_params,
                    "maccs_rough": student_maccs,
                    **student_training_extra,
                },
            )
            print("Student test metrics:", student_test_metrics)
            """
        ),
        code_cell(
            """
            final_model = student

            if experiment.train.use_qat:
                qat_candidate = maybe_quantize_aware_clone(student)
                if qat_candidate is None:
                    print("QAT skipped: tensorflow_model_optimization is not available.")
                    add_note(
                        run_dir,
                        "QAT skipped",
                        "tensorflow_model_optimization was not available in the notebook runtime.",
                    )
                else:
                    qat_candidate.set_weights(student.get_weights())
                    qat_summary_path = save_model_summary(run_dir, qat_candidate, "qat")
                    qat_candidate = compile_classifier(
                        qat_candidate,
                        experiment=experiment,
                        learning_rate=experiment.train.qat_lr,
                        epochs=experiment.train.qat_epochs,
                        steps_per_epoch=steps_per_epoch,
                    )
                    qat_history = qat_candidate.fit(
                        student_train_ds,
                        validation_data=student_val_ds,
                        epochs=experiment.train.qat_epochs,
                        callbacks=build_callbacks_for_monitor(
                            qat_dir,
                            patience=max(2, experiment.train.early_stopping_patience // 2),
                            monitor="val_accuracy",
                            mode="max",
                            include_checkpoint=True,
                        ),
                    )
                    qat_test_metrics = qat_candidate.evaluate(student_test_ds, return_dict=True)
                    qat_history_path = save_history(run_dir, qat_history, "qat")
                    save_metrics(run_dir, qat_test_metrics, "qat")
                    qat_plot_paths = save_history_plots(run_dir, qat_history, "qat")
                    update_stage_state(
                        run_dir,
                        "qat",
                        metrics=qat_test_metrics,
                        history_path=str(qat_history_path) if qat_history_path else "",
                        plot_paths=qat_plot_paths,
                        summary_path=str(qat_summary_path),
                        extra={
                            "epoch_log_path": str(qat_dir / "epoch_log.csv"),
                            "checkpoint_path": str(qat_dir / "best.keras"),
                        },
                    )
                    print("QAT student test metrics:", qat_test_metrics)
                    final_model = qat_candidate
            """
        ),
        code_cell(
            """
            metadata = export_experiment_artifacts(
                model=final_model,
                representative_data=student_train_ds,
                experiment=experiment,
                required_ops=required_tflite_ops(experiment.model.student_name),
            )
            record_export_artifacts(run_dir, metadata)
            save_json_artifact(run_dir, "export_metadata_snapshot.json", metadata)
            print(json.dumps(metadata, indent=2))
            """
        ),
        code_cell(
            """
            def plot_history(history, title):
                if history is None:
                    return
                hist = history.history
                keys = [key for key in hist if "accuracy" in key or key == "loss"]
                if not keys:
                    return
                plt.figure(figsize=(12, 4))
                for key in keys:
                    plt.plot(hist[key], label=key)
                plt.title(title)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show()
                plt.close()

            plot_history(teacher_history, "Teacher history")
            plot_history(student_history, "Student history")
            if "qat_history" in globals():
                plot_history(qat_history, "QAT history")
            """
        ),
        code_cell(
            """
            summary_path = write_run_summary(run_dir)
            print("Run summary:", summary_path)
            print("Teacher epoch log:", teacher_dir / "epoch_log.csv")
            print("Student epoch log:", student_dir / "epoch_log.csv")
            if experiment.train.use_qat:
                print("QAT epoch log:", qat_dir / "epoch_log.csv")
            """
        ),
        code_cell(
            """
            import shutil

            archive_base = run_dir.parent / f"{run_dir.name}_bundle"
            archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=run_dir)
            DRIVE_EXPORT_DIR = ""

            print("Run archive:", archive_path)
            if DRIVE_EXPORT_DIR:
                drive_export_dir = Path(DRIVE_EXPORT_DIR)
                drive_export_dir.mkdir(parents=True, exist_ok=True)
                exported_archive = drive_export_dir / Path(archive_path).name
                shutil.copy2(archive_path, exported_archive)
                print("Copied archive:", exported_archive)
            """
        ),
        md_cell(
            """
            ## Next Iteration Ideas

            - Compare `distilled_student_v2` against `distilled_student_v3`, `distilled_student_v3_teacher_plus`, and `distilled_student_v3_bcresnet_teacher`.
            - Use `baseline` and `fast_student` only as sanity references, not as the main optimization path.
            - Run `kws20_probe` after a stable `kws12` export exists.
            - If exact microfrontend import fails, keep the fallback pipeline for rapid experimentation, but verify transfer to firmware before trusting notebook gains.
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
    notebook_path = root / "kws_experiments_esp32.ipynb"
    notebook_path.write_text(
        json.dumps(build_notebook(build_runtime_payloads(root.parents[1])), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(notebook_path)


if __name__ == "__main__":
    main()
