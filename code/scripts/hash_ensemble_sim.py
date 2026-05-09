"""Host-side simulator for the 3-model hash KWS ensemble.

Subcommands:

  eval    — load three trained student bundles, run them on the test split,
            and produce per-model + ensemble metrics. Useful as an out-of-
            notebook verification of `ensemble_results.json`.

  demo    — replay random test-set samples through the three models and the
            aggregator, emitting JSONL events under the same paths the live
            dashboard already reads (notes/Journal/hash_kws_telemetry/node{1,2,3},
            notes/Journal/hash_kws_fusion/decisions.jsonl). Lets us drive the
            dashboard without three real ESP32 boards.

  smoke   — runs the demo path without any model checkpoints; useful for CI
            and for verifying the JSONL contract end-to-end.

Bundles must be the per-variant `student_best.pt` files written by the Colab
notebook (`hash_ensemble_train_colab.ipynb`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code" / "training"))
sys.path.insert(0, str(REPO_ROOT / "code" / "training" / "hashednet95"))
sys.path.insert(0, str(REPO_ROOT / "code" / "training" / "hash_ensemble"))

import aggregation as agg  # noqa: E402

TELEMETRY_DIR = REPO_ROOT / "notes" / "Journal" / "hash_kws_telemetry"
FUSION_DIR = REPO_ROOT / "notes" / "Journal" / "hash_kws_fusion"
CLUSTER_DIR = REPO_ROOT / "notes" / "Journal" / "hash_kws_cluster"

DEFAULT_LABELS = [
    "yes", "no", "up", "down", "left", "right",
    "on", "off", "stop", "go", "unknown", "silence",
]


# ---------------------------------------------------------------------------
# Optional torch import (eval / demo with real models needs it; smoke does not)
# ---------------------------------------------------------------------------


def _try_import_torch(*, with_dataloaders: bool = True):
    """Lazy-import torch + hash_kws_lab pieces.

    ``with_dataloaders=False`` skips the prepare_dataloaders import (which
    transitively pulls in TensorFlow for the exact-microfrontend feature
    extraction). Use it for paths that don't need real audio data, e.g. the
    `verify` smoke that just runs a dummy forward pass through each model.
    """

    try:
        import torch  # noqa: F401

        from hash_kws_lab.config import experiment_from_dict
        from hash_kws_lab.models import build_student_model

        out: dict[str, Any] = {
            "torch": __import__("torch"),
            "experiment_from_dict": experiment_from_dict,
            "build_student_model": build_student_model,
        }
        if with_dataloaders:
            from hash_kws_lab.data import prepare_dataloaders

            out["prepare_dataloaders"] = prepare_dataloaders
        return out
    except ImportError as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# JSONL writers — match the contract used by hash_kws_cluster_sim.py
# ---------------------------------------------------------------------------


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_node_event(node_id: int, payload: dict[str, Any]) -> None:
    append_jsonl(TELEMETRY_DIR / f"node{node_id}" / "events.jsonl", payload)


def write_fusion_decision(payload: dict[str, Any]) -> None:
    append_jsonl(FUSION_DIR / "decisions.jsonl", payload)


# ---------------------------------------------------------------------------
# Loading bundles + per-variant logits collection
# ---------------------------------------------------------------------------


def _load_bundle(path: Path, device, modules: dict):
    payload = modules["torch"].load(path, map_location=device)
    if "experiment" not in payload or "state_dict" not in payload:
        raise ValueError(f"{path}: not a hash_ensemble student bundle")
    experiment = modules["experiment_from_dict"](payload["experiment"])
    student = modules["build_student_model"](experiment).to(device)
    student.load_state_dict(payload["state_dict"], strict=True)
    student.eval()
    return experiment, student, payload


@_load_bundle.__class__.__call__.__class__ if False else (lambda f: f)
def _no_op_decorator(f):
    return f


def _collect_logits(student, loader, device, torch) -> tuple[np.ndarray, np.ndarray]:
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            features, targets = batch[0], batch[1]
            features = features.to(device, non_blocking=True)
            logits = student(features).detach().cpu().numpy().astype(np.float64)
            all_logits.append(logits)
            all_labels.append(targets.detach().cpu().numpy().astype(np.int64))
    return np.concatenate(all_logits, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# eval subcommand
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    modules = _try_import_torch()
    if "error" in modules:
        print(f"torch / hash_kws_lab unavailable: {modules['error']}")
        return 2
    torch = modules["torch"]
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    bundles = {name: Path(path) for name, path in zip(args.variant_names, args.bundles)}
    print(f"Loading {len(bundles)} student bundles on {device}...")

    experiments: dict[str, Any] = {}
    students: dict[str, Any] = {}
    for name, path in bundles.items():
        if not path.exists():
            raise FileNotFoundError(path)
        exp, student, _ = _load_bundle(path, device, modules)
        experiments[name] = exp
        students[name] = student

    anchor_name = args.variant_names[0]
    print(f"Preparing dataloaders from {anchor_name}'s experiment...")
    prep = modules["prepare_dataloaders"](REPO_ROOT, experiments[anchor_name], device=device)
    loaders = prep["loaders"]

    val_logits: dict[str, np.ndarray] = {}
    test_logits: dict[str, np.ndarray] = {}
    val_labels: np.ndarray | None = None
    test_labels: np.ndarray | None = None
    for name, student in students.items():
        v_logits, v_labels = _collect_logits(student, loaders["validation"], device, torch)
        t_logits, t_labels = _collect_logits(student, loaders["test"], device, torch)
        if val_labels is None:
            val_labels = v_labels
            test_labels = t_labels
        else:
            assert np.array_equal(val_labels, v_labels), "Validation order diverged"
            assert np.array_equal(test_labels, t_labels), "Test order diverged"
        val_logits[name] = v_logits
        test_logits[name] = t_logits

    val_stack = np.stack([val_logits[n] for n in args.variant_names], axis=0)
    test_stack = np.stack([test_logits[n] for n in args.variant_names], axis=0)

    label_names = experiments[anchor_name].all_labels
    summary = agg.evaluate_aggregators(
        test_logits=test_stack,
        test_labels=test_labels,
        val_logits=val_stack,
        val_labels=val_labels,
        label_names=label_names,
    )

    per_model = {}
    for name in args.variant_names:
        preds = test_logits[name].argmax(axis=-1)
        top1 = float((preds == test_labels).mean())
        topk_idx = np.argsort(test_logits[name], axis=-1)[..., -3:]
        top3 = float((topk_idx == test_labels.reshape(-1, 1)).any(axis=-1).mean())
        per_model[name] = {"top1": top1, "top3": top3}

    print("\nPer-model:")
    for name, m in per_model.items():
        print(f"  {name}: top1={m['top1']:.4f} top3={m['top3']:.4f}")
    print("\nAggregators:")
    for k, v in summary["aggregators"].items():
        if "top1" in v:
            print(f"  {k}: top1={v['top1']:.4f}")
    print(f"\nOracle top1: {summary['oracle_top1']:.4f}")

    if args.output:
        out_path = Path(args.output)
        atomic_write_json(out_path, {
            "per_model": per_model,
            "aggregators_test": summary["aggregators"],
            "oracle_top1": summary["oracle_top1"],
            "pairwise_disagreement": summary["pairwise_disagreement"],
            "per_class_disagreement_rate": summary["per_class_disagreement_rate"],
            "labels": label_names,
        })
        print(f"\nWrote {out_path}")
    return 0


# ---------------------------------------------------------------------------
# demo subcommand — replay test samples through the live dashboard contract
# ---------------------------------------------------------------------------


def _aggregator_logits_to_label_score(logits_avg: np.ndarray, label_names: list[str]) -> tuple[str, int, int]:
    top_idx = int(logits_avg.argmax())
    sorted_idx = np.argsort(logits_avg)
    top1 = float(logits_avg[sorted_idx[-1]])
    top2 = float(logits_avg[sorted_idx[-2]])
    score = max(0, min(255, int(round(top1))))
    margin = max(0, min(255, int(round(top1 - top2))))
    return label_names[top_idx], score, margin


def _emit_node_infer(
    node_id: int,
    label: str,
    score: int,
    margin: int,
    logits_int8: list[int],
    label_names: list[str],
    device_t: int,
    sample_idx: int,
) -> dict[str, Any]:
    other_scores = [int(v) for v in logits_int8]
    sorted_pairs = sorted(zip(label_names, other_scores), key=lambda x: x[1], reverse=True)
    top1_name, top1_val = sorted_pairs[0]
    top2_name, top2_val = sorted_pairs[1] if len(sorted_pairs) > 1 else (top1_name, top1_val)
    top3_name, top3_val = sorted_pairs[2] if len(sorted_pairs) > 2 else (top2_name, top2_val)
    record = {
        "kind": "infer",
        "node": str(node_id),
        "node_label": f"hash_ensemble_node{node_id}",
        "simulated": True,
        "source_node": node_id,
        "host_time": utc_now_iso(),
        "t": device_t,
        "sample_idx": sample_idx,
        "top1": top1_name,
        "top1_score": int(top1_val),
        "top2": top2_name,
        "top2_score": int(top2_val),
        "top3": top3_name,
        "top3_score": int(top3_val),
        "logits_int8": logits_int8,
        "mode": "hash_ensemble_sim",
    }
    write_node_event(node_id, record)
    return record


def _make_decision_record(
    label: str,
    score: int,
    margin: int,
    voters: list[dict[str, Any]],
    aggregator_mode: str,
    learned_weights: list[float] | None,
    temperatures: list[float] | None,
    sample_idx: int,
    true_label: str | None,
) -> dict[str, Any]:
    return {
        "kind": "audio_fusion_agree",
        "host_time": utc_now_iso(),
        "label": label,
        "score_sum": int(round(score)),
        "score_avg": float(score),
        "margin": int(margin),
        "nodes": [str(v["node_id"]) for v in voters],
        "votes": voters,
        "window_sec": 1.2,
        "min_score": 0,
        "min_margin": 0,
        "aggregator_node": "0",
        "aggregator_label": "host_master_sim",
        "simulated_master": True,
        "aggregator_mode": aggregator_mode,
        "aggregator_weights": learned_weights,
        "aggregator_temperatures": temperatures,
        "sample_idx": sample_idx,
        "true_label": true_label,
        "match_truth": (true_label == label) if true_label is not None else None,
    }


def _quantize_logits_to_int8(logits: np.ndarray) -> np.ndarray:
    """Map a float logit vector to int8 the same way the firmware packet does."""

    bound = max(1.0, float(np.abs(logits).max()))
    scale = 127.0 / bound
    quantized = np.round(logits * scale).astype(np.int64)
    quantized = np.clip(quantized, -127, 127)
    return quantized.astype(np.int8)


def _ensure_clean_streams(paths: list[Path]) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def cmd_demo(args: argparse.Namespace) -> int:
    modules = _try_import_torch()
    if "error" in modules:
        print(f"torch unavailable: {modules['error']}")
        return 2
    torch = modules["torch"]
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    bundles = {name: Path(path) for name, path in zip(args.variant_names, args.bundles)}
    experiments: dict[str, Any] = {}
    students: dict[str, Any] = {}
    for name, path in bundles.items():
        exp, student, _ = _load_bundle(path, device, modules)
        experiments[name] = exp
        students[name] = student

    anchor_name = args.variant_names[0]
    prep = modules["prepare_dataloaders"](REPO_ROOT, experiments[anchor_name], device=device)
    test_loader = prep["loaders"]["test"]
    label_names = experiments[anchor_name].all_labels

    if args.reset_streams:
        _ensure_clean_streams([
            *(TELEMETRY_DIR / f"node{i}" / "events.jsonl" for i in (1, 2, 3)),
            FUSION_DIR / "decisions.jsonl",
        ])

    cluster_state = {
        "updated_at": utc_now_iso(),
        "mode": "hash_ensemble_host_sim",
        "aggregator_mode": args.aggregator,
        "variants": list(args.variant_names),
        "samples_processed": 0,
        "agree_count": 0,
        "ensemble_top1_acc": 0.0,
    }
    atomic_write_json(CLUSTER_DIR / "state.json", cluster_state)

    weights = None
    temps = None
    if args.params_json and Path(args.params_json).exists():
        params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
        agg_block = params.get("aggregators_test", {})
        if "learned_weights" in agg_block and "w" in agg_block["learned_weights"]:
            weights = np.array(agg_block["learned_weights"]["w"], dtype=np.float64)
        if "temperature_scaled" in agg_block and "T" in agg_block["temperature_scaled"]:
            temps = np.array(agg_block["temperature_scaled"]["T"], dtype=np.float64)

    seen = 0
    correct = 0
    rng = np.random.default_rng(args.seed)
    sample_indices: list[int] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            features, targets = batch[0], batch[1]
            features_dev = features.to(device, non_blocking=True)
            per_model_float: list[np.ndarray] = []
            per_model_int8: list[np.ndarray] = []
            for name in args.variant_names:
                logits = students[name](features_dev).detach().cpu().numpy().astype(np.float64)
                per_model_float.append(logits)
                per_model_int8.append(np.array([_quantize_logits_to_int8(row) for row in logits]))

            stack = np.stack(per_model_float, axis=0)  # [N, B, C]

            for b in range(stack.shape[1]):
                if seen >= args.samples:
                    break
                sample_indices.append(seen)

                # Per-model events (each "node" reports an infer)
                for node_idx, name in enumerate(args.variant_names):
                    int8_row = per_model_int8[node_idx][b]
                    top_idx = int(stack[node_idx, b].argmax())
                    sorted_idx = np.argsort(stack[node_idx, b])
                    score = max(0, min(255, int(round(float(stack[node_idx, b][sorted_idx[-1]])))))
                    margin = max(0, min(255, int(round(float(stack[node_idx, b][sorted_idx[-1]] - stack[node_idx, b][sorted_idx[-2]])))))
                    _emit_node_infer(
                        node_id=node_idx + 1,
                        label=label_names[top_idx],
                        score=score,
                        margin=margin,
                        logits_int8=[int(v) for v in int8_row.tolist()],
                        label_names=label_names,
                        device_t=int(time.time() * 1000) & 0xFFFFFFFF,
                        sample_idx=seen,
                    )

                # Aggregator
                if args.aggregator == "mean_logits":
                    aggregated = agg.mean_logits(stack[:, b : b + 1])[0]
                elif args.aggregator == "temperature_scaled":
                    if temps is None:
                        temps_used = np.ones(stack.shape[0], dtype=np.float64)
                    else:
                        temps_used = temps
                    aggregated = agg.temperature_scaled_mean(stack[:, b : b + 1], temps_used)[0]
                elif args.aggregator == "learned_weights":
                    if weights is None:
                        weights_used = np.full(stack.shape[0], 1.0 / stack.shape[0], dtype=np.float64)
                    else:
                        weights_used = weights
                    aggregated = agg.learned_weights_mean(stack[:, b : b + 1], weights_used)[0]
                else:
                    raise ValueError(f"Unknown aggregator: {args.aggregator}")

                ens_label, ens_score, ens_margin = _aggregator_logits_to_label_score(aggregated, label_names)
                truth_idx = int(targets[b].item())
                truth_label = label_names[truth_idx]
                if ens_label == truth_label:
                    correct += 1

                voter_records = []
                for node_idx in range(stack.shape[0]):
                    top_label_idx = int(stack[node_idx, b].argmax())
                    voter_records.append({
                        "node_id": node_idx + 1,
                        "label": label_names[top_label_idx],
                        "score": int(round(float(stack[node_idx, b][top_label_idx]))),
                    })

                decision = _make_decision_record(
                    label=ens_label,
                    score=ens_score,
                    margin=ens_margin,
                    voters=voter_records,
                    aggregator_mode=args.aggregator,
                    learned_weights=weights.tolist() if weights is not None else None,
                    temperatures=temps.tolist() if temps is not None else None,
                    sample_idx=seen,
                    true_label=truth_label,
                )
                write_fusion_decision(decision)
                seen += 1

                if args.verbose:
                    per_model_labels = ", ".join(
                        f"{n}:{label_names[int(stack[i, b].argmax())]}"
                        for i, n in enumerate(args.variant_names)
                    )
                    flag = "OK " if ens_label == truth_label else "ERR"
                    print(f"[{seen:04d}] {flag}  truth={truth_label:<8s}  ens={ens_label:<8s}  ({per_model_labels})")

            if seen >= args.samples:
                break

    cluster_state["updated_at"] = utc_now_iso()
    cluster_state["samples_processed"] = seen
    cluster_state["agree_count"] = correct
    cluster_state["ensemble_top1_acc"] = (correct / seen) if seen else 0.0
    atomic_write_json(CLUSTER_DIR / "state.json", cluster_state)
    print(f"\nProcessed {seen} samples; ensemble top1 = {correct}/{seen} = {cluster_state['ensemble_top1_acc']:.4f}")
    return 0


# ---------------------------------------------------------------------------
# smoke subcommand — JSONL path verification with no models
# ---------------------------------------------------------------------------


def cmd_smoke(args: argparse.Namespace) -> int:
    rng = np.random.default_rng(0)
    label_names = list(DEFAULT_LABELS)
    n_classes = len(label_names)

    if args.reset_streams:
        _ensure_clean_streams([
            *(TELEMETRY_DIR / f"node{i}" / "events.jsonl" for i in (1, 2, 3)),
            FUSION_DIR / "decisions.jsonl",
        ])

    cluster_state = {
        "updated_at": utc_now_iso(),
        "mode": "hash_ensemble_host_sim_SMOKE",
        "samples_processed": 0,
    }
    atomic_write_json(CLUSTER_DIR / "state.json", cluster_state)

    correct = 0
    for i in range(args.samples):
        true_idx = int(rng.integers(0, len(label_names) - 2))  # not silence
        # 3 noisy classifiers
        per_model = []
        for k in range(3):
            row = rng.normal(0.0, 1.0, size=n_classes).astype(np.float64)
            row[true_idx] += 4.0 + 0.3 * k
            per_model.append(row)
        stack = np.stack(per_model, axis=0)  # [3, C]
        for k in range(3):
            int8_row = _quantize_logits_to_int8(stack[k])
            top_idx = int(stack[k].argmax())
            sorted_idx = np.argsort(stack[k])
            score = max(0, min(255, int(round(float(stack[k][sorted_idx[-1]])))))
            margin = max(0, min(255, int(round(float(stack[k][sorted_idx[-1]] - stack[k][sorted_idx[-2]])))))
            _emit_node_infer(
                node_id=k + 1,
                label=label_names[top_idx],
                score=score,
                margin=margin,
                logits_int8=[int(v) for v in int8_row.tolist()],
                label_names=label_names,
                device_t=int(time.time() * 1000) & 0xFFFFFFFF,
                sample_idx=i,
            )
        avg = agg.mean_logits(stack[:, np.newaxis, :])[0]
        ens_label, ens_score, ens_margin = _aggregator_logits_to_label_score(avg, label_names)
        truth_label = label_names[true_idx]
        if ens_label == truth_label:
            correct += 1
        voter_records = [
            {"node_id": k + 1, "label": label_names[int(stack[k].argmax())], "score": int(round(float(stack[k].max())))}
            for k in range(3)
        ]
        write_fusion_decision(_make_decision_record(
            label=ens_label,
            score=ens_score,
            margin=ens_margin,
            voters=voter_records,
            aggregator_mode="mean_logits",
            learned_weights=None,
            temperatures=None,
            sample_idx=i,
            true_label=truth_label,
        ))

    cluster_state["updated_at"] = utc_now_iso()
    cluster_state["samples_processed"] = args.samples
    cluster_state["ensemble_top1_acc"] = correct / max(args.samples, 1)
    atomic_write_json(CLUSTER_DIR / "state.json", cluster_state)
    print(f"smoke: {correct}/{args.samples} ensemble correct -> {cluster_state['ensemble_top1_acc']:.3f}")
    print(f"streams under {TELEMETRY_DIR} / {FUSION_DIR} / {CLUSTER_DIR}")
    return 0


# ---------------------------------------------------------------------------
# verify subcommand — load every .pt and run one dummy forward pass
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    modules = _try_import_torch(with_dataloaders=False)
    if "error" in modules:
        print(f"torch / hash_kws_lab unavailable: {modules['error']}")
        return 2
    torch = modules["torch"]
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Verifying {len(args.bundles)} bundles on {device}...")

    failures = 0
    for name, path_str in zip(args.variant_names, args.bundles):
        path = Path(path_str)
        if not path.exists():
            print(f"[{name}] MISSING: {path}")
            failures += 1
            continue
        try:
            payload = torch.load(path, map_location=device)
            experiment = modules["experiment_from_dict"](payload["experiment"])
            student = modules["build_student_model"](experiment).to(device)
            student.load_state_dict(payload["state_dict"], strict=True)
            student.eval()
            shape = experiment.model_input_shape  # e.g. (1, 40, 49)
            with torch.no_grad():
                dummy = torch.zeros((1, *shape), dtype=torch.float32, device=device)
                logits = student(dummy)
            print(
                f"[{name}] OK  tag={experiment.tag}  input_shape={tuple(shape)}  "
                f"logits_shape={tuple(logits.shape)}  recorded_test_top1="
                f"{payload.get('test_metrics', {}).get('accuracy', 'n/a')}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAIL: {exc}")
            failures += 1
    if failures:
        print(f"\n{failures} bundle(s) failed to verify.")
        return 1
    print("\nAll bundles loaded and forward-pass clean.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_bundle_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--bundles", nargs=3, required=True,
                   help="Three student_best.pt paths in order ens_a ens_b ens_c (or your own order).")
    p.add_argument("--variant-names", nargs=3, default=["ens_a", "ens_b", "ens_c"],
                   help="Variant names matching --bundles order.")
    p.add_argument("--device", default="cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description="Host-side hash KWS ensemble simulator")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_eval = sub.add_parser("eval", help="Run full test split through 3 students + aggregators")
    _add_bundle_args(p_eval)
    p_eval.add_argument("--output", default="", help="Optional path to write a JSON report")
    p_eval.set_defaults(func=cmd_eval)

    p_demo = sub.add_parser("demo", help="Replay test samples and stream JSONL events for the dashboard")
    _add_bundle_args(p_demo)
    p_demo.add_argument("--samples", type=int, default=64)
    p_demo.add_argument("--seed", type=int, default=0)
    p_demo.add_argument("--aggregator", choices=("mean_logits", "temperature_scaled", "learned_weights"),
                        default="mean_logits")
    p_demo.add_argument("--params-json", default="",
                        help="Optional ensemble_results.json with calibrated temps/weights.")
    p_demo.add_argument("--reset-streams", action="store_true")
    p_demo.add_argument("--verbose", action="store_true")
    p_demo.set_defaults(func=cmd_demo)

    p_smoke = sub.add_parser("smoke", help="Synthetic 3-model demo (no torch / no checkpoints)")
    p_smoke.add_argument("--samples", type=int, default=20)
    p_smoke.add_argument("--reset-streams", action="store_true")
    p_smoke.set_defaults(func=cmd_smoke)

    p_verify = sub.add_parser("verify", help="Load each .pt and run one dummy forward pass — no TF / Speech Commands needed")
    _add_bundle_args(p_verify)
    p_verify.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
