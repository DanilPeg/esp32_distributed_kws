"""Cross-check C++ MCU aggregator vs the numpy reference.

Builds and runs the C++ harness, then runs the same vectors through
``aggregation.py``. Verifies that the predicted labels (and ranking-relevant
scores) match across implementations.

This is a smoke test, not a numeric audit — fixed-point Q8.8 transport in C++
introduces small quantisation that we don't try to reproduce in numpy.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
AGG_DIR = REPO_ROOT / "code" / "firmware" / "hash_kws_aggregator"

sys.path.insert(0, str(REPO_ROOT / "code" / "training" / "hash_ensemble"))
import aggregation as agg  # noqa: E402


VECTORS = {
    "case1_meanlogits_3voters": dict(
        mode="mean_logits",
        logits=np.array([
            [12, -8, 3, -2, 20, -5, 1, 0, -7, 2, 4, -3],
            [10, -5, 2, -1, 18, -4, 0, 1, -6, 3, 5, -2],
            [14, -9, 4, -3, 22, -6, 2, 0, -8, 1, 3, -4],
        ], dtype=np.float64),
    ),
    "case2_meanlogits_2voters_paired": dict(
        mode="mean_logits",
        logits=np.array([
            [12, -8, 3, -2, 20, -5, 1, 0, -7, 2, 4, -3],
            [10, -5, 2, -1, 18, -4, 0, 1, -6, 3, 5, -2],
        ], dtype=np.float64),
    ),
    "case3_temperature_scaled": dict(
        mode="temperature_scaled",
        temperatures=np.array([1.2, 0.9, 1.4], dtype=np.float64),
        logits=np.array([
            [12, -8, 3, -2, 20, -5, 1, 0, -7, 2, 4, -3],
            [10, -5, 2, -1, 18, -4, 0, 1, -6, 3, 5, -2],
            [14, -9, 4, -3, 22, -6, 2, 0, -8, 1, 3, -4],
        ], dtype=np.float64),
    ),
    "case4_learned_weights": dict(
        mode="learned_weights",
        weights=np.array([0.20, 0.55, 0.25], dtype=np.float64),
        logits=np.array([
            [12, -8, 3, -2, 20, -5, 1, 0, -7, 2, 4, -3],
            [10, -5, 2, -1, 18, -4, 0, 1, -6, 3, 5, -2],
            [14, -9, 4, -3, 22, -6, 2, 0, -8, 1, 3, -4],
        ], dtype=np.float64),
    ),
    "case5_disagreement_meanlogits": dict(
        mode="mean_logits",
        logits=np.array([
            [30, -8, 3, -2, 20, -5, 1, 0, -7, 2, 4, -3],
            [10, -5, 2, -1, 35, -4, 0, 1, -6, 3, 5, -2],
            [28, -9, 4, -3, 22, -6, 2, 0, -8, 1, 3, -4],
        ], dtype=np.float64),
    ),
    "case6_disagreement_learned": dict(
        mode="learned_weights",
        weights=np.array([0.20, 0.55, 0.25], dtype=np.float64),
        logits=np.array([
            [30, -8, 3, -2, 20, -5, 1, 0, -7, 2, 4, -3],
            [10, -5, 2, -1, 35, -4, 0, 1, -6, 3, 5, -2],
            [28, -9, 4, -3, 22, -6, 2, 0, -8, 1, 3, -4],
        ], dtype=np.float64),
    ),
}


def numpy_predict(case: dict) -> int:
    logits = case["logits"][:, np.newaxis, :]  # [N, B=1, C]
    if case["mode"] == "mean_logits":
        out = agg.mean_logits(logits)
    elif case["mode"] == "temperature_scaled":
        out = agg.temperature_scaled_mean(logits, case["temperatures"])
    elif case["mode"] == "learned_weights":
        out = agg.learned_weights_mean(logits, case["weights"])
    else:
        raise ValueError(f"Unknown mode: {case['mode']}")
    return int(out.argmax(axis=-1)[0])


def parse_cpp_label(line: str) -> tuple[str, int]:
    name, payload = line.split(maxsplit=1)
    fields = dict(item.split("=", 1) for item in payload.split())
    return name, int(fields["label"])


def main() -> int:
    if shutil.which("g++") is None:
        print("g++ not found — skipping cross-check.")
        return 0
    suffix = ".exe" if sys.platform.startswith("win") else ""
    bin_path = Path(tempfile.gettempdir()) / f"agg_test_match{suffix}"
    cmd = [
        "g++", "-std=c++14", "-O2", "-Wall", "-Wextra",
        str(AGG_DIR / "hash_ensemble_aggregator.cpp"),
        str(AGG_DIR / "test_aggregator_main.cpp"),
        "-o", str(bin_path),
    ]
    subprocess.run(cmd, check=True)
    proc = subprocess.run([str(bin_path)], capture_output=True, text=True, check=True)
    cpp_labels: dict[str, int] = {}
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        name, label = parse_cpp_label(line)
        cpp_labels[name] = label

    mismatches: list[str] = []
    for name, case in VECTORS.items():
        py_label = numpy_predict(case)
        cpp_label = cpp_labels.get(name)
        ok = (cpp_label == py_label)
        flag = "OK" if ok else "MISMATCH"
        print(f"{name}: numpy={py_label} cpp={cpp_label} -> {flag}")
        if not ok:
            mismatches.append(name)

    if mismatches:
        print(f"\n{len(mismatches)} mismatch(es). FAIL.")
        return 1
    print("\nAll cases match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
