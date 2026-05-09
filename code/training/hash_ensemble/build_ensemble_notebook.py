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
    "code/training/hash_ensemble/__init__.py",
    "code/training/hash_ensemble/ensemble_recipes.py",
    "code/training/hash_ensemble/aggregation.py",
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
            # Hash KWS Ensemble — 3 students + smart aggregation

            Trains three near-homogeneous HashedNet KWS students that differ only in
            pointwise codebook sizes and seed; they share an identical architecture
            and a single distilled teacher. After training, the notebook stacks
            test logits and runs every aggregator (mean_logits, mean_probs,
            conf_weighted, trimmed, majority_vote, temperature_scaled, learned_weights),
            plus diagnostics: oracle top-1, pairwise disagreement, per-class
            disagreement rate, single/ensemble dispersion.

            All artifacts (3 student bundles, 3 firmware exports, ensemble_results.json,
            aggregator_params.h) are zipped at the end and copied to Drive.

            See `notes/Journal/2026-05-09_hash_ensemble_plan.md` for the locked plan.
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
            DRIVE_CACHE_ROOT = Path("/content/drive/MyDrive/diploma_kws_cache/hash_ensemble")
            ENSEMBLE_VARIANT_NAMES = ["ens_a", "ens_b", "ens_c"]
            TEACHER_VARIANT_NAME = "ens_b"  # which recipe owns the teacher checkpoint
            SMOKE_MODE = False  # truncates dataset for quick syntax check

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
            HASH_ENSEMBLE_ROOT = TRAINING_ROOT / "hash_ensemble"
            FILE_PAYLOADS = json.loads(%PAYLOAD_LITERAL%)

            def ensure_runtime_files(root: Path, payloads: dict, overwrite: bool = False):
                created, skipped = [], []
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
                PROJECT_ROOT, FILE_PAYLOADS, overwrite=FORCE_SYNC_RUNTIME_FILES,
            )

            for path in (TRAINING_ROOT, SCRIPTS_ROOT, HASHEDNET95_ROOT, HASH_ENSEMBLE_ROOT):
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))

            importlib.invalidate_caches()
            for module_name in list(sys.modules):
                if (
                    module_name == "hash_kws_lab"
                    or module_name.startswith("hash_kws_lab.")
                    or module_name == "hashednet95"
                    or module_name.startswith("hashednet95.")
                    or module_name == "hash_ensemble"
                    or module_name.startswith("hash_ensemble.")
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
            print("DRIVE_CACHE_ACTIVE:", DRIVE_CACHE_ACTIVE)
            print("Variants to train:", ENSEMBLE_VARIANT_NAMES)
            print("Teacher variant:", TEACHER_VARIANT_NAME)
            print("Runtime files written/skipped:", len(created_files), "/", len(skipped_files))
            """
            .replace("%PAYLOAD_LITERAL%", repr(payload_literal))
        ),
        code_cell(
            """
            import json
            import shutil
            import time
            from copy import deepcopy
            from dataclasses import replace
            from pathlib import Path

            import numpy as np
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
            from hashednet95.hashednet95_recipes import with_drive_cache_paths
            from hash_ensemble.ensemble_recipes import build_ensemble_recipe_book, describe_variant
            from hash_ensemble import aggregation as agg
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
            SPEECHCOMMANDS_REQUIRED_FILES = ("validation_list.txt", "testing_list.txt")
            SPEECHCOMMANDS_ARCHIVE = "speech_commands_v0.02.tar.gz"
            SPEECHCOMMANDS_REQUIRED_DIRS = (
                "_background_noise_", "yes", "no", "up", "down", "left", "right",
                "on", "off", "stop", "go",
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

            def ensure_speechcommands_downloaded(root: Path) -> dict:
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
                    raise RuntimeError("Speech Commands extraction incomplete")
                return {
                    subset: len(SPEECHCOMMANDS(root=str(root), download=False, subset=subset))
                    for subset in ("training", "validation", "testing")
                }

            speechcommands_counts = ensure_speechcommands_downloaded(Path(os.environ["SPEECHCOMMANDS_DATA_ROOT"]))
            print("Speech Commands split sizes:", speechcommands_counts)
            """
        ),
        code_cell(
            """
            base = make_experiment(tag="hash_kws12_iterlab_v1", vocabulary_preset="kws12")
            recipes = build_ensemble_recipe_book(base)
            assert TEACHER_VARIANT_NAME in recipes, f"Unknown teacher variant: {TEACHER_VARIANT_NAME}"
            assert all(name in recipes for name in ENSEMBLE_VARIANT_NAMES)

            if DRIVE_CACHE_ACTIVE:
                recipes = {name: with_drive_cache_paths(recipe, DRIVE_CACHE_ROOT) for name, recipe in recipes.items()}

            if SMOKE_MODE:
                shrunk = {}
                for name, recipe in recipes.items():
                    shrunk[name] = replace(
                        recipe,
                        dataset=replace(recipe.dataset, train_limit=1024, val_limit=256, test_limit=256),
                        train=replace(
                            recipe.train,
                            teacher_epochs=1,
                            student_pretrain_epochs=1,
                            student_epochs=1,
                            student_polish_epochs=0,
                        ),
                    )
                recipes = shrunk

            ensemble_root = PROJECT_ROOT / "code" / "training" / "hash_artifacts" / "hash_ensemble"
            ensemble_root.mkdir(parents=True, exist_ok=True)

            print("Variants:")
            for name, recipe in recipes.items():
                d = describe_variant(recipe)
                print(f"  {name}: tag={d['tag']}  pw={d['pointwise_codebook_sizes']}  seed={d['seed']}")
            """
        ),
        md_cell(
            """
            ## Phase 1 — Train teacher once (anchored to the chosen variant)
            """
        ),
        code_cell(
            """
            teacher_recipe = recipes[TEACHER_VARIANT_NAME]
            teacher_run_dir = initialize_run_state(
                PROJECT_ROOT, teacher_recipe, recipe_name=f"hash_ensemble.{TEACHER_VARIANT_NAME}.teacher"
            )

            t0 = time.perf_counter()
            teacher_bundle = prepare_dataloaders(PROJECT_ROOT, teacher_recipe, device=device)
            teacher_loaders = teacher_bundle["loaders"]
            save_json_artifact(teacher_run_dir, "dataset_summary.json", teacher_bundle["summary"])
            print("Data prepare seconds:", round(time.perf_counter() - t0, 1))
            print("Dataset summary:", json.dumps(teacher_bundle["summary"], indent=2))

            teacher_model = build_teacher_model(teacher_recipe)
            teacher_summary = summarize_model(teacher_model, teacher_recipe)
            save_json_artifact(teacher_run_dir, "teacher_model_inventory.json", teacher_summary)
            print("Teacher params:", teacher_summary["trainable_parameters"])

            teacher_result = train_teacher(teacher_model, loaders=teacher_loaders, experiment=teacher_recipe, device=device)
            teacher_model.load_state_dict(teacher_result["best_state"], strict=True)
            teacher_test_metrics = evaluate(
                teacher_model,
                teacher_loaders["test"],
                device=device,
                top_k=teacher_recipe.train.top_k,
                use_amp=teacher_recipe.train.use_amp,
                desc="ensemble | teacher | test",
            )
            save_metrics(teacher_run_dir, "teacher", teacher_test_metrics)

            teacher_checkpoint_path = teacher_run_dir / "teacher_best.pt"
            torch.save(
                {
                    "experiment": teacher_recipe.to_dict(),
                    "state_dict": {k: v.detach().cpu().clone() for k, v in teacher_model.state_dict().items()},
                    "result": teacher_result,
                },
                teacher_checkpoint_path,
            )
            print("Teacher test metrics:", teacher_test_metrics)
            print("Teacher checkpoint:", teacher_checkpoint_path)
            """
        ),
        md_cell(
            """
            ## Phase 2 — Train each student variant (teacher reused, logits cached)
            """
        ),
        code_cell(
            """
            from hash_kws_lab.config import experiment_from_dict

            student_states: dict[str, dict] = {}

            for variant_name in ENSEMBLE_VARIANT_NAMES:
                recipe = recipes[variant_name]
                print(f"\\n=== Training student '{variant_name}' (tag={recipe.tag}) ===")
                run_dir = initialize_run_state(PROJECT_ROOT, recipe, recipe_name=f"hash_ensemble.{variant_name}")

                # Re-use the teacher's prepared loaders if dataset/feature configs match;
                # otherwise rebuild. For the ensemble track they always match.
                if variant_name == TEACHER_VARIANT_NAME:
                    bundle = teacher_bundle
                else:
                    bundle = prepare_dataloaders(PROJECT_ROOT, recipe, device=device)
                save_json_artifact(run_dir, "dataset_summary.json", bundle["summary"])

                student = build_student_model(recipe)
                summary = summarize_model(student, recipe)
                save_json_artifact(run_dir, "student_model_inventory.json", summary)
                print("Student params:", {
                    "compact": summary["hash_compact_parameters"],
                    "virtual": summary["virtual_dense_parameters"],
                    "maccs": summary["maccs_rough"],
                })

                # Re-load teacher state (the same teacher for all students; checkpoint is fixed)
                teacher_for_student = build_teacher_model(recipe)
                load_model_checkpoint(teacher_for_student, teacher_checkpoint_path, device=device)

                student_result = train_student(
                    student,
                    loaders=bundle["loaders"],
                    experiment=recipe,
                    device=device,
                    teacher=teacher_for_student,
                )
                student.load_state_dict(student_result["best_state"], strict=True)

                bundle_path = run_dir / "student_best.pt"
                torch.save(
                    {
                        "experiment": recipe.to_dict(),
                        "state_dict": {k: v.detach().cpu().clone() for k, v in student.state_dict().items()},
                        "result": {key: value for key, value in student_result.items() if key != "best_state"},
                        "test_metrics": student_result["test_metrics"],
                    },
                    bundle_path,
                )
                save_metrics(run_dir, "student", student_result["test_metrics"])
                save_history(run_dir, "student", student_result["history"])

                # Standard library bundle + per-variant firmware export
                export_metadata = export_model_bundle(student, experiment=recipe, stage_name="student")
                variant_firmware_dir = (
                    PROJECT_ROOT / "code" / "firmware" / f"hash_kws_runtime_{variant_name}"
                )
                variant_firmware_dir.mkdir(parents=True, exist_ok=True)
                firmware_export = firmware_exporter.export_bundle_to_firmware(
                    bundle_path=Path(export_metadata["bundle"]["path"]),
                    output_dir=variant_firmware_dir,
                    project_root=PROJECT_ROOT,
                    device=device,
                    calibration_split="validation",
                    calibration_batches=8,
                )
                record_export_artifacts(run_dir, {
                    "library_bundle": export_metadata,
                    "firmware_export": firmware_export,
                    "firmware_output_dir": str(variant_firmware_dir),
                })

                student_states[variant_name] = {
                    "recipe": recipe,
                    "run_dir": run_dir,
                    "bundle_path": bundle_path,
                    "summary": summary,
                    "test_metrics": student_result["test_metrics"],
                    "firmware_dir": variant_firmware_dir,
                    "stage_summaries": student_result["stage_summaries"],
                }
                print(f"--- '{variant_name}' done. test={student_result['test_metrics']}")

            print("\\nAll students trained.")
            """
        ),
        md_cell(
            """
            ## Phase 3 — Stack logits, evaluate ensemble, fit calibration / learned weights
            """
        ),
        code_cell(
            """
            @torch.no_grad()
            def collect_logits_and_labels(model, loader, device):
                model.eval()
                all_logits = []
                all_labels = []
                for batch in loader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        features, targets = batch[0], batch[1]
                    else:
                        raise TypeError("Unexpected batch shape")
                    features = features.to(device, non_blocking=True)
                    logits = model(features).detach().cpu().numpy()
                    all_logits.append(logits.astype(np.float64))
                    all_labels.append(targets.detach().cpu().numpy().astype(np.int64))
                return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)

            anchor_recipe = recipes[TEACHER_VARIANT_NAME]
            label_names = anchor_recipe.all_labels

            # Re-prepare (cached) loaders with shuffle off so all 3 models see the same order
            per_variant_logits_test: dict[str, np.ndarray] = {}
            per_variant_logits_val: dict[str, np.ndarray] = {}
            test_labels_ref = None
            val_labels_ref = None

            for variant_name in ENSEMBLE_VARIANT_NAMES:
                recipe = student_states[variant_name]["recipe"]
                bundle_path = student_states[variant_name]["bundle_path"]
                bundle = teacher_bundle if variant_name == TEACHER_VARIANT_NAME else prepare_dataloaders(
                    PROJECT_ROOT, recipe, device=device
                )
                # Build a fresh student and load its best state
                student = build_student_model(recipe).to(device)
                payload = torch.load(bundle_path, map_location=device)
                student.load_state_dict(payload["state_dict"], strict=True)

                test_logits, test_labels = collect_logits_and_labels(student, bundle["loaders"]["test"], device)
                val_logits, val_labels = collect_logits_and_labels(student, bundle["loaders"]["validation"], device)
                per_variant_logits_test[variant_name] = test_logits
                per_variant_logits_val[variant_name] = val_logits

                if test_labels_ref is None:
                    test_labels_ref = test_labels
                    val_labels_ref = val_labels
                else:
                    if not np.array_equal(test_labels_ref, test_labels):
                        raise RuntimeError("Test labels diverged between variants — loaders not deterministic?")
                    if not np.array_equal(val_labels_ref, val_labels):
                        raise RuntimeError("Val labels diverged between variants — loaders not deterministic?")

            test_logits_stack = np.stack([per_variant_logits_test[name] for name in ENSEMBLE_VARIANT_NAMES], axis=0)
            val_logits_stack = np.stack([per_variant_logits_val[name] for name in ENSEMBLE_VARIANT_NAMES], axis=0)
            print("Test logits stack:", test_logits_stack.shape, "labels:", test_labels_ref.shape)
            print("Val  logits stack:", val_logits_stack.shape,  "labels:", val_labels_ref.shape)

            ensemble_eval = agg.evaluate_aggregators(
                test_logits=test_logits_stack,
                test_labels=test_labels_ref,
                val_logits=val_logits_stack,
                val_labels=val_labels_ref,
                label_names=label_names,
            )

            per_model = {}
            for name in ENSEMBLE_VARIANT_NAMES:
                preds = per_variant_logits_test[name].argmax(axis=-1)
                top1 = float((preds == test_labels_ref).mean())
                topk_idx = np.argsort(per_variant_logits_test[name], axis=-1)[..., -3:]
                top3 = float((topk_idx == test_labels_ref.reshape(-1, 1)).any(axis=-1).mean())
                per_model[name] = {
                    "top1": top1,
                    "top3": top3,
                    "compact_params": int(student_states[name]["summary"]["hash_compact_parameters"]),
                    "virtual_params": int(student_states[name]["summary"]["virtual_dense_parameters"]),
                    "maccs": int(student_states[name]["summary"]["maccs_rough"]),
                }

            single_top1 = np.array([per_model[n]["top1"] for n in ENSEMBLE_VARIANT_NAMES])
            ensemble_eval["dispersion"] = {
                "single_mean": float(single_top1.mean()),
                "single_std": float(single_top1.std(ddof=0)),
                "single_top1_per_variant": per_model,
            }

            results = {
                "teacher_variant": TEACHER_VARIANT_NAME,
                "teacher_test_metrics": teacher_test_metrics,
                "per_model": per_model,
                "aggregators_test": ensemble_eval["aggregators"],
                "oracle_top1": ensemble_eval["oracle_top1"],
                "pairwise_disagreement": ensemble_eval["pairwise_disagreement"],
                "per_class_disagreement_rate": ensemble_eval["per_class_disagreement_rate"],
                "dispersion": ensemble_eval["dispersion"],
                "labels": label_names,
            }
            results_path = ensemble_root / "ensemble_results.json"
            results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print("\\nWrote", results_path)
            print(json.dumps({
                "per_model_top1": {n: per_model[n]["top1"] for n in ENSEMBLE_VARIANT_NAMES},
                "aggregators_top1": {k: v.get("top1") for k, v in ensemble_eval["aggregators"].items()},
                "oracle_top1": ensemble_eval["oracle_top1"],
            }, indent=2))
            """
        ),
        md_cell(
            """
            ## Phase 4 — Plots and aggregator_params.h
            """
        ),
        code_cell(
            """
            import matplotlib.pyplot as plt

            agg_top1 = {k: v["top1"] for k, v in ensemble_eval["aggregators"].items() if "top1" in v}
            best_single = max(per_model.values(), key=lambda x: x["top1"])["top1"]

            fig, ax = plt.subplots(figsize=(10, 4))
            keys = list(agg_top1.keys())
            vals = [agg_top1[k] for k in keys]
            ax.axhline(best_single, color="grey", linestyle="--", label=f"best single = {best_single:.4f}")
            ax.axhline(ensemble_eval["oracle_top1"], color="green", linestyle=":", label=f"oracle = {ensemble_eval['oracle_top1']:.4f}")
            bars = ax.bar(keys, vals, color="steelblue")
            for bar, value in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.001, f"{value:.4f}",
                        ha="center", va="bottom", fontsize=9)
            ax.set_ylabel("test top-1 accuracy")
            ax.set_title("Hash KWS ensemble — aggregator comparison")
            ax.set_ylim(min(vals) * 0.985, max(vals + [best_single, ensemble_eval["oracle_top1"]]) * 1.005)
            ax.legend()
            plt.xticks(rotation=20)
            plot_path = ensemble_root / "aggregator_comparison.png"
            plt.savefig(plot_path, bbox_inches="tight")
            plt.close()
            print("Wrote", plot_path)

            disagree = np.array(ensemble_eval["pairwise_disagreement"])
            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(disagree, cmap="magma", vmin=0)
            for i in range(disagree.shape[0]):
                for j in range(disagree.shape[1]):
                    ax.text(j, i, f"{disagree[i, j]:.3f}", ha="center", va="center",
                            color="white" if disagree[i, j] > disagree.max() * 0.5 else "black",
                            fontsize=10)
            ax.set_xticks(range(len(ENSEMBLE_VARIANT_NAMES)))
            ax.set_yticks(range(len(ENSEMBLE_VARIANT_NAMES)))
            ax.set_xticklabels(ENSEMBLE_VARIANT_NAMES)
            ax.set_yticklabels(ENSEMBLE_VARIANT_NAMES)
            ax.set_title("Pairwise disagreement rate")
            plt.colorbar(im, ax=ax)
            plot_path2 = ensemble_root / "pairwise_disagreement.png"
            plt.savefig(plot_path2, bbox_inches="tight")
            plt.close()
            print("Wrote", plot_path2)
            """
        ),
        code_cell(
            """
            # Generate aggregator_params.h with calibrated temperatures + learned weights
            ts_block = ensemble_eval["aggregators"].get("temperature_scaled", {})
            lw_block = ensemble_eval["aggregators"].get("learned_weights", {})
            temps = ts_block.get("T", [1.0, 1.0, 1.0])
            weights = lw_block.get("w", [1.0 / 3.0] * 3)

            header_path = ensemble_root / "aggregator_params.h"
            lines = [
                "// Auto-generated by code/training/hash_ensemble notebook.",
                "// One temperature per model + softmax-normalized 3-weight ensemble head.",
                "// Both fitted on validation split only — test metrics are not leaked.",
                "// Models order: " + ", ".join(ENSEMBLE_VARIANT_NAMES),
                "",
                "#ifndef HASH_KWS_AGGREGATOR_PARAMS_H_",
                "#define HASH_KWS_AGGREGATOR_PARAMS_H_",
                "",
                "#include <stddef.h>",
                "",
                "static const size_t kHashEnsembleNumModels = 3;",
                "static const size_t kHashEnsembleNumClasses = " + str(len(label_names)) + ";",
                "",
                "static const float kHashEnsembleTemperatures[kHashEnsembleNumModels] = {",
                ", ".join(f"{float(t):.8f}f" for t in temps),
                "};",
                "",
                "static const float kHashEnsembleLearnedWeights[kHashEnsembleNumModels] = {",
                ", ".join(f"{float(w):.8f}f" for w in weights),
                "};",
                "",
                "#endif  // HASH_KWS_AGGREGATOR_PARAMS_H_",
                "",
            ]
            header_path.write_text("\\n".join(lines), encoding="utf-8")
            print("Wrote", header_path)
            print("Temperatures:", temps)
            print("Learned weights:", weights)
            """
        ),
        md_cell(
            """
            ## Phase 5 — Bundle everything for download
            """
        ),
        code_cell(
            """
            stage_dir = ensemble_root / "stage"
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True)

            (stage_dir / "ensemble_results.json").write_bytes((ensemble_root / "ensemble_results.json").read_bytes())
            (stage_dir / "aggregator_params.h").write_bytes((ensemble_root / "aggregator_params.h").read_bytes())
            (stage_dir / "aggregator_comparison.png").write_bytes((ensemble_root / "aggregator_comparison.png").read_bytes())
            (stage_dir / "pairwise_disagreement.png").write_bytes((ensemble_root / "pairwise_disagreement.png").read_bytes())

            for variant_name in ENSEMBLE_VARIANT_NAMES:
                vdir = stage_dir / variant_name
                vdir.mkdir()
                shutil.copy2(student_states[variant_name]["bundle_path"], vdir / "student_best.pt")
                shutil.copytree(
                    student_states[variant_name]["firmware_dir"],
                    vdir / "firmware_export",
                    dirs_exist_ok=True,
                )

            archive_path = shutil.make_archive(str(ensemble_root / "hash_ensemble_bundle"), "zip", root_dir=stage_dir)
            print("Bundle archive:", archive_path)
            if DRIVE_CACHE_ACTIVE:
                drive_target = DRIVE_CACHE_ROOT / "runs" / Path(archive_path).name
                drive_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(archive_path, drive_target)
                print("Copied to Drive:", drive_target)
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
    output_path = root / "hash_ensemble_train_colab.ipynb"
    output_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
