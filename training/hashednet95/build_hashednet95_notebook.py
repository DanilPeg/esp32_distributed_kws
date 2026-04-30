from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


RUNTIME_FILES = [
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
    "code/training/hashednet95/__init__.py",
    "code/training/hashednet95/hashednet95_recipes.py",
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


def build_notebook(runtime_payloads: dict[str, str]) -> dict:
    payload_literal = json.dumps(runtime_payloads, ensure_ascii=False)
    cells = [
        md_cell(
            """
            # HashedNet95 KWS Lab

            Fast experiment notebook for the HashedNets-inspired ESP32 KWS branch.

            Main levers:
            - exact firmware microfrontend;
            - direct Speech Commands preparation under the runtime project data folder;
            - persistent Google Drive cache for exact features and run artifacts;
            - signed hashing;
            - per-layer codebook budgets;
            - virtual-width inflation up to the current ESP32 runtime limit;
            - optional HashNetDK-style teacher distillation.
            """
        ),
        code_cell(
            """
            # The next cell writes the minimal runtime files into /content first.
            # If Colab is missing packages, run after bootstrap:
            # !pip -q install -r /content/diploma_esp32_distributed_nn/code/training/requirements-kws-hash-exact-frontend.txt
            """
        ),
        code_cell(
            """
            import importlib
            import json
            import os
            import sys
            from pathlib import Path

            FORCE_SYNC_RUNTIME_FILES = True
            USE_GOOGLE_DRIVE_CACHE = True
            CACHE_SPEECHCOMMANDS_ON_DRIVE = False
            DRIVE_CACHE_ROOT = Path("/content/drive/MyDrive/diploma_kws_cache/hashednet95")
            SELECTED_RECIPE = "hn95_kd128_cached_schedule"
            TEACHER_CHECKPOINT_PATH = ""  # optional: /content/.../teacher_best.pt
            FORCE_TEACHER_RETRAIN = False
            SMOKE_MODE = False  # keeps architecture intact, only limits data for syntax/debug runs

            if USE_GOOGLE_DRIVE_CACHE and Path("/content").exists():
                try:
                    from google.colab import drive
                    drive.mount("/content/drive", force_remount=False)
                except Exception as exc:
                    print("Drive mount skipped:", exc)
            DRIVE_CACHE_ACTIVE = USE_GOOGLE_DRIVE_CACHE and (Path("/content/drive/MyDrive").exists())

            BASE_RUNTIME_DIR = Path("/content") if Path("/content").exists() else Path.cwd()
            PROJECT_ROOT = BASE_RUNTIME_DIR / "diploma_esp32_distributed_nn"

            TRAINING_ROOT = PROJECT_ROOT / "code" / "training"
            SCRIPTS_ROOT = PROJECT_ROOT / "code" / "scripts"
            HASHEDNET95_ROOT = TRAINING_ROOT / "hashednet95"
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

            for path in (TRAINING_ROOT, SCRIPTS_ROOT, HASHEDNET95_ROOT):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))

            importlib.invalidate_caches()
            for module_name in list(sys.modules):
                if (
                    module_name == "hash_kws_lab"
                    or module_name.startswith("hash_kws_lab.")
                    or module_name == "hashednet95"
                    or module_name.startswith("hashednet95.")
                ):
                    del sys.modules[module_name]

            local_speechcommands_root = PROJECT_ROOT / "data"
            drive_speechcommands_root = DRIVE_CACHE_ROOT / "speechcommands_v2"
            speechcommands_root = (
                drive_speechcommands_root
                if DRIVE_CACHE_ACTIVE and CACHE_SPEECHCOMMANDS_ON_DRIVE
                else local_speechcommands_root
            )
            feature_cache_root = DRIVE_CACHE_ROOT / "hash_feature_cache"
            teacher_logits_cache_root = DRIVE_CACHE_ROOT / "teacher_logits_cache"

            speechcommands_root.mkdir(parents=True, exist_ok=True)
            os.environ["SPEECHCOMMANDS_DATA_ROOT"] = str(speechcommands_root)

            if DRIVE_CACHE_ACTIVE:
                DRIVE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
                feature_cache_root.mkdir(parents=True, exist_ok=True)
                teacher_logits_cache_root.mkdir(parents=True, exist_ok=True)
                os.environ["HASH_KWS_FEATURE_CACHE_ROOT"] = str(feature_cache_root)
                os.environ["HASH_KWS_TEACHER_LOGITS_CACHE_ROOT"] = str(teacher_logits_cache_root)
            else:
                feature_cache_root = PROJECT_ROOT / "data" / "hash_feature_cache"
                teacher_logits_cache_root = PROJECT_ROOT / "data" / "teacher_logits_cache"
                feature_cache_root.mkdir(parents=True, exist_ok=True)
                teacher_logits_cache_root.mkdir(parents=True, exist_ok=True)
                os.environ["HASH_KWS_FEATURE_CACHE_ROOT"] = str(feature_cache_root)
                os.environ["HASH_KWS_TEACHER_LOGITS_CACHE_ROOT"] = str(teacher_logits_cache_root)

            os.chdir(PROJECT_ROOT)
            print("PROJECT_ROOT:", PROJECT_ROOT)
            print("SPEECHCOMMANDS_DATA_ROOT:", speechcommands_root)
            print("HASH_KWS_FEATURE_CACHE_ROOT:", feature_cache_root)
            print("HASH_KWS_TEACHER_LOGITS_CACHE_ROOT:", teacher_logits_cache_root)
            print("Runtime files written:", len(created_files))
            print("Runtime files skipped:", len(skipped_files))
            print("DRIVE_CACHE_ROOT:", DRIVE_CACHE_ROOT if DRIVE_CACHE_ACTIVE else "disabled")
            """
            .replace("%PAYLOAD_LITERAL%", repr(payload_literal))
        ),
        code_cell(
            """
            import json
            import shutil
            import time
            from dataclasses import replace
            from pathlib import Path

            import torch
            from torchaudio.datasets import SPEECHCOMMANDS

            from hash_kws_lab.config import make_experiment
            from hash_kws_lab.data import ensure_torchaudio_available, prepare_dataloaders
            from hash_kws_lab.export import export_model_bundle
            from hash_kws_lab.models import build_student_model, build_teacher_model, summarize_model
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
            from hash_kws_lab.trainer import evaluate, load_model_checkpoint, train_student, train_teacher
            from hashednet95_recipes import (
                build_hashednet95_recipe_book,
                describe_hashednet95_recipe,
                with_drive_cache_paths,
            )
            import export_hash_kws_firmware as firmware_exporter

            ensure_torchaudio_available()
            torch.manual_seed(13)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(13)
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("Device:", device)
            """
        ),
        code_cell(
            """
            SPEECHCOMMANDS_REQUIRED_FILES = (
                "validation_list.txt",
                "testing_list.txt",
            )
            SPEECHCOMMANDS_ARCHIVE = "speech_commands_v0.02.tar.gz"
            SPEECHCOMMANDS_REQUIRED_DIRS = (
                "_background_noise_",
                "yes",
                "no",
                "up",
                "down",
                "left",
                "right",
                "on",
                "off",
                "stop",
                "go",
            )

            def speechcommands_extracted_dir(root: Path) -> Path:
                return root / "SpeechCommands" / "speech_commands_v0.02"

            def speechcommands_is_complete(root: Path) -> bool:
                extracted = speechcommands_extracted_dir(root)
                return (
                    extracted.is_dir()
                    and all((extracted / name).is_file() for name in SPEECHCOMMANDS_REQUIRED_FILES)
                    and all((extracted / name).is_dir() for name in SPEECHCOMMANDS_REQUIRED_DIRS)
                )

            def reset_incomplete_speechcommands(root: Path) -> None:
                if speechcommands_is_complete(root):
                    return
                extracted_parent = root / "SpeechCommands"
                if extracted_parent.exists():
                    print("Removing incomplete Speech Commands extraction:", extracted_parent)
                    shutil.rmtree(extracted_parent)

            def remove_incomplete_speechcommands_archive(root: Path) -> None:
                archive_path = root / SPEECHCOMMANDS_ARCHIVE
                if archive_path.exists():
                    print("Removing incomplete Speech Commands archive:", archive_path)
                    archive_path.unlink()

            def ensure_speechcommands_downloaded(root: Path) -> dict[str, int]:
                root.mkdir(parents=True, exist_ok=True)
                reset_incomplete_speechcommands(root)
                if not speechcommands_is_complete(root):
                    print("Downloading Speech Commands v0.02 into:", root)
                    try:
                        SPEECHCOMMANDS(root=str(root), download=True, subset="validation")
                    except Exception:
                        reset_incomplete_speechcommands(root)
                        remove_incomplete_speechcommands_archive(root)
                        SPEECHCOMMANDS(root=str(root), download=True, subset="validation")

                if not speechcommands_is_complete(root):
                    raise RuntimeError(
                        "Speech Commands download/extraction did not produce validation_list.txt, "
                        "testing_list.txt, and required label directories under "
                        f"{speechcommands_extracted_dir(root)}"
                    )

                split_sizes = {}
                for subset in ("training", "validation", "testing"):
                    dataset = SPEECHCOMMANDS(root=str(root), download=False, subset=subset)
                    split_sizes[subset] = len(dataset)
                return split_sizes

            speechcommands_counts = ensure_speechcommands_downloaded(Path(os.environ["SPEECHCOMMANDS_DATA_ROOT"]))
            print("Speech Commands split sizes:", speechcommands_counts)
            """
        ),
        code_cell(
            """
            base = make_experiment(tag="hash_kws12_iterlab_v1", vocabulary_preset="kws12")
            recipes = build_hashednet95_recipe_book(base)
            print("Recipes:")
            for name in recipes:
                print(" -", name)

            experiment = recipes[SELECTED_RECIPE]
            if DRIVE_CACHE_ACTIVE:
                experiment = with_drive_cache_paths(experiment, DRIVE_CACHE_ROOT)

            if SMOKE_MODE:
                experiment = replace(
                    experiment,
                    dataset=replace(
                        experiment.dataset,
                        train_limit=2048,
                        val_limit=512,
                        test_limit=512,
                    ),
                    train=replace(
                        experiment.train,
                        teacher_epochs=min(experiment.train.teacher_epochs, 1),
                        student_pretrain_epochs=min(experiment.train.student_pretrain_epochs, 1),
                        student_epochs=1,
                        student_polish_epochs=0,
                    ),
                )

            torch.manual_seed(experiment.train.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(experiment.train.seed)

            description = describe_hashednet95_recipe(experiment)
            print(json.dumps({k: v for k, v in description.items() if k != "student_summary"}, indent=2))
            print("Student summary:")
            print(json.dumps(description["student_summary"], indent=2))
            """
        ),
        code_cell(
            """
            t0 = time.perf_counter()
            bundle = prepare_dataloaders(PROJECT_ROOT, experiment, device=device)
            loaders = bundle["loaders"]

            run_dir = initialize_run_state(
                PROJECT_ROOT,
                experiment,
                recipe_name=SELECTED_RECIPE,
                dataset_summary=bundle["summary"],
            )
            save_json_artifact(run_dir, "experiment.json", experiment.to_dict())
            save_json_artifact(run_dir, "hashednet95_description.json", description)
            save_json_artifact(run_dir, "dataset_summary.json", bundle["summary"])
            save_text_artifact(run_dir, "selected_recipe.txt", SELECTED_RECIPE + "\\n")
            add_note(
                run_dir,
                "HashedNet95 setup",
                "Drive cache is used for Speech Commands and exact feature cache."
                if USE_GOOGLE_DRIVE_CACHE else "Drive cache disabled.",
            )

            print("Data root:", bundle["data_root"])
            print("Run dir:", run_dir)
            print("Dataset summary:")
            print(json.dumps(bundle["summary"], indent=2))
            print("Cache summary:")
            print(json.dumps(bundle.get("cache_summary", {}), indent=2))
            print("Feature preview shape:", bundle["feature_preview_shape"])
            print("Data/cache prepare seconds:", round(time.perf_counter() - t0, 1))
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
            print("Student params:", {
                "trainable": student_summary["trainable_parameters"],
                "compact": student_summary["hash_compact_parameters"],
                "virtual": student_summary["virtual_dense_parameters"],
                "maccs": student_summary["maccs_rough"],
            })

            teacher_summary_path = None
            teacher_reuse_info = {}
            if teacher is not None:
                if TEACHER_CHECKPOINT_PATH:
                    teacher_reuse_info = load_model_checkpoint(teacher, TEACHER_CHECKPOINT_PATH, device=device)
                    add_note(
                        run_dir,
                        "Teacher checkpoint reuse",
                        json.dumps(teacher_reuse_info, indent=2),
                    )
                teacher_summary = summarize_model(teacher, experiment)
                teacher_summary_path = save_model_summary(
                    run_dir,
                    "teacher",
                    str(teacher) + "\\n\\n" + json.dumps(teacher_summary, indent=2),
                )
                save_json_artifact(run_dir, "teacher_model_inventory.json", teacher_summary)
                print("Teacher enabled.")
            else:
                print("Teacher disabled.")
            """
        ),
        code_cell(
            """
            teacher_test_metrics = {}
            if teacher is not None:
                if teacher_reuse_info and not FORCE_TEACHER_RETRAIN:
                    teacher_result = {
                        "history": [],
                        "best_state": {k: v.detach().clone() for k, v in teacher.state_dict().items()},
                        "best_val_accuracy": None,
                        "best_epoch": None,
                        "elapsed_sec": 0.0,
                    }
                    teacher_history_path = save_history(run_dir, "teacher", [])
                    teacher_plot_paths = []
                else:
                    teacher_result = train_teacher(teacher, loaders=loaders, experiment=experiment, device=device)
                    teacher.load_state_dict(teacher_result["best_state"], strict=True)
                    teacher_history_path = save_history(run_dir, "teacher", teacher_result["history"])
                    teacher_plot_paths = save_history_plots(run_dir, "teacher", teacher_result["history"])
                torch.save(
                    {
                        "experiment": experiment.to_dict(),
                        "state_dict": {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()},
                        "result": teacher_result,
                    },
                    run_dir / "teacher_best.pt",
                )
                teacher_test_metrics = evaluate(
                    teacher,
                    loaders["test"],
                    device=device,
                    top_k=experiment.train.top_k,
                    use_amp=experiment.train.use_amp,
                    desc="teacher | test",
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
                        "best_val_accuracy": teacher_result["best_val_accuracy"],
                        "best_epoch": teacher_result["best_epoch"],
                        "elapsed_sec": teacher_result["elapsed_sec"],
                        "checkpoint_reuse": teacher_reuse_info,
                        "force_teacher_retrain": FORCE_TEACHER_RETRAIN,
                    },
                )
                print("Teacher test:", teacher_test_metrics)
            """
        ),
        code_cell(
            """
            student_result = train_student(
                student,
                loaders=loaders,
                experiment=experiment,
                device=device,
                teacher=teacher,
            )
            student.load_state_dict(student_result["best_state"], strict=True)
            torch.save(
                {
                    "experiment": experiment.to_dict(),
                    "state_dict": {k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
                    "result": student_result,
                },
                run_dir / "student_best.pt",
            )

            student_history_path = save_history(run_dir, "student", student_result["history"])
            student_plot_paths = save_history_plots(run_dir, "student", student_result["history"])
            save_metrics(run_dir, "student", student_result["test_metrics"])
            update_stage_state(
                run_dir,
                "student",
                metrics=student_result["test_metrics"],
                history_path=str(student_history_path),
                plot_paths=student_plot_paths,
                summary_path=str(student_summary_path),
                extra={"stage_summaries": student_result["stage_summaries"]},
            )
            print("Student test:", student_result["test_metrics"])
            """
        ),
        code_cell(
            """
            export_metadata = export_model_bundle(student, experiment=experiment, stage_name="student")
            record_export_artifacts(run_dir, export_metadata)
            save_json_artifact(run_dir, "export_metadata_snapshot.json", export_metadata)

            firmware_export = firmware_exporter.export_bundle_to_firmware(
                bundle_path=Path(export_metadata["bundle"]["path"]),
                output_dir=PROJECT_ROOT / "code" / "firmware" / "hash_kws_runtime",
                project_root=PROJECT_ROOT,
                device=device,
                calibration_split="validation",
                calibration_batches=8,
            )
            record_export_artifacts(run_dir, {"firmware_export": firmware_export})
            save_json_artifact(run_dir, "firmware_export_snapshot.json", firmware_export)
            print(json.dumps(firmware_export, indent=2))
            """
        ),
        code_cell(
            """
            summary_path = write_run_summary(run_dir)
            archive_path = shutil.make_archive(str(run_dir.parent / f"{run_dir.name}_bundle"), "zip", root_dir=run_dir)
            print("Summary:", summary_path)
            print("Archive:", archive_path)

            if DRIVE_CACHE_ACTIVE:
                drive_run_dir = DRIVE_CACHE_ROOT / "runs"
                drive_run_dir.mkdir(parents=True, exist_ok=True)
                target = drive_run_dir / Path(archive_path).name
                shutil.copy2(archive_path, target)
                print("Copied run archive to Drive:", target)
            """
        ),
        md_cell(
            """
            ## Recommended Sweep Order

            1. `hn95_kd128_cached_schedule` - primary next run: current 128-channel student plus cached logits and cosine KD schedule.
            2. `hn95_kd128_hard_polish_cached` - same deploy shape, lower CE smoothing and longer hard-label polish.
            3. `hn95_kd128_pw2048_budget_cached_schedule` - same MACs, larger pointwise codebooks to test collision pressure.
            4. `hn95_kd112_latency_balanced_cached_schedule` - lower-latency guardrail if 128-channel firmware timing is too high.
            5. `hn95_kd128_rich_teacher160_cached_schedule` - training-only richer teacher, same 128-channel deploy student.
            6. `hn95_kd128_cached_schedule_s29` and `hn95_kd128_cached_schedule_s47` - training-seed repeat validation for the best candidate.
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
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    root = Path(__file__).resolve().parent
    notebook = build_notebook(build_runtime_payloads(root.parents[2]))
    output_path = root / "hashednet95_kws_colab.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
