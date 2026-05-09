"""Pure-numpy aggregation utilities for the 3-model hash KWS ensemble.

All functions operate on stacked logits with shape ``[N, B, C]`` where
``N`` is the number of ensemble members, ``B`` is the batch size and ``C``
is the class count. Outputs are ``[B, C]`` (logits or probabilities) or
``[B]`` (predicted labels) depending on the aggregator.

The two new-in-v2 aggregators (``temperature_scaled_mean`` and
``learned_weights_mean``) accept a tiny set of fitted parameters that fit
into a few floats and can be hard-coded into the firmware aggregator.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _check_logits(logits: np.ndarray) -> None:
    if logits.ndim != 3:
        raise ValueError(
            f"Expected logits with shape [N, B, C], got shape {tuple(logits.shape)}"
        )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def _entropy_per_row(probs: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    return -np.sum(probs * np.log(probs + eps), axis=axis)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(f"Weights must sum to a positive number, got {total}")
    return weights / total


# ---------------------------------------------------------------------------
# Basic aggregators (the ones the research already tested)
# ---------------------------------------------------------------------------


def mean_logits(logits: np.ndarray) -> np.ndarray:
    """Headline aggregator from the research (`A4 best on KWS = mean_logits`)."""

    _check_logits(logits)
    return logits.mean(axis=0)


def mean_probs(logits: np.ndarray) -> np.ndarray:
    _check_logits(logits)
    probs = _softmax(logits, axis=-1)
    return probs.mean(axis=0)


def conf_weighted(logits: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-input confidence-weighted mean of softmax probs.

    Weight per (b, k) is ``1 / H(softmax(logits_k[b]))``. Refuted by the
    research as a system-level upgrade, kept for parity with §5 of NOTES.md.
    """

    _check_logits(logits)
    probs = _softmax(logits, axis=-1)
    entropy = _entropy_per_row(probs, axis=-1)  # [N, B]
    weights = 1.0 / (entropy + eps)
    weights = weights / weights.sum(axis=0, keepdims=True)
    return np.einsum("nb,nbc->bc", weights, probs)


def trimmed_mean(logits: np.ndarray, drop: int = 1) -> np.ndarray:
    """Per-(b, c) trimmed mean: drop the highest and lowest ``drop`` values."""

    _check_logits(logits)
    n = logits.shape[0]
    if drop <= 0 or 2 * drop >= n:
        # With N=3 and drop=1 this falls through to the median.
        return np.median(logits, axis=0)
    sorted_logits = np.sort(logits, axis=0)
    return sorted_logits[drop : n - drop].mean(axis=0)


def majority_vote(logits: np.ndarray) -> np.ndarray:
    """Hard-vote labels from per-model argmax. Returns shape ``[B]`` int labels."""

    _check_logits(logits)
    n, b, c = logits.shape
    preds = logits.argmax(axis=-1)  # [N, B]
    out = np.empty(b, dtype=np.int64)
    for i in range(b):
        counts = np.bincount(preds[:, i], minlength=c)
        max_count = counts.max()
        winners = np.flatnonzero(counts == max_count)
        if winners.size == 1:
            out[i] = int(winners[0])
        else:
            # Tie-break: pick the candidate with the highest summed logit
            tied_logits = np.array(
                [logits[:, i, label].sum() for label in winners], dtype=np.float64
            )
            out[i] = int(winners[int(tied_logits.argmax())])
    return out


# ---------------------------------------------------------------------------
# Per-model temperature scaling (new in v2)
# ---------------------------------------------------------------------------


def _nll_at_temperature(
    logits_one_model: np.ndarray, labels: np.ndarray, temperature: float
) -> float:
    log_probs = _log_softmax(logits_one_model / max(temperature, 1e-6), axis=-1)
    return float(-log_probs[np.arange(labels.shape[0]), labels].mean())


def fit_per_model_temperatures(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    coarse_grid: tuple[float, ...] = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0),
    refine_iters: int = 30,
    refine_tol: float = 1e-3,
) -> np.ndarray:
    """Fit one temperature per model by NLL line-search on validation set.

    Coarse grid first, then golden-section refinement. Returns ``[N]``.
    Calibrated logits are obtained as ``logits_k / T_k``.
    """

    _check_logits(val_logits)
    if val_labels.ndim != 1 or val_labels.shape[0] != val_logits.shape[1]:
        raise ValueError("val_labels must be 1D with length equal to batch size")

    temps = np.empty(val_logits.shape[0], dtype=np.float64)
    for k in range(val_logits.shape[0]):
        per_model = val_logits[k]
        best_t = min(
            coarse_grid,
            key=lambda t: _nll_at_temperature(per_model, val_labels, t),
        )
        # Golden-section search around the best coarse value.
        lo = max(0.1, best_t / 2.0)
        hi = best_t * 2.0
        phi = (1.0 + 5.0 ** 0.5) / 2.0
        rho = 2.0 - phi
        a, b = lo, hi
        c = a + rho * (b - a)
        d = b - rho * (b - a)
        f_c = _nll_at_temperature(per_model, val_labels, c)
        f_d = _nll_at_temperature(per_model, val_labels, d)
        for _ in range(refine_iters):
            if f_c < f_d:
                b, d, f_d = d, c, f_c
                c = a + rho * (b - a)
                f_c = _nll_at_temperature(per_model, val_labels, c)
            else:
                a, c, f_c = c, d, f_d
                d = b - rho * (b - a)
                f_d = _nll_at_temperature(per_model, val_labels, d)
            if (b - a) < refine_tol:
                break
        temps[k] = 0.5 * (a + b)
    return temps


def temperature_scaled_mean(
    logits: np.ndarray, temperatures: np.ndarray
) -> np.ndarray:
    """Mean of softmax(logits_k / T_k) across models. Returns probs ``[B, C]``."""

    _check_logits(logits)
    temperatures = np.asarray(temperatures, dtype=np.float64)
    if temperatures.shape != (logits.shape[0],):
        raise ValueError(
            f"Expected temperatures shape [{logits.shape[0]}], got {tuple(temperatures.shape)}"
        )
    scaled = logits / temperatures.reshape(-1, 1, 1)
    probs = _softmax(scaled, axis=-1)
    return probs.mean(axis=0)


# ---------------------------------------------------------------------------
# Learned 3-weight aggregator (new in v2)
# ---------------------------------------------------------------------------


def fit_learned_weights(
    val_logits: np.ndarray,
    val_labels: np.ndarray,
    n_iters: int = 200,
    lr: float = 0.05,
    init_z: tuple[float, ...] | None = None,
) -> dict[str, Any]:
    """Fit non-negative softmax-parametrized weights ``w = softmax(z)``.

    Loss: cross-entropy on the weighted-sum logits ``Σ w_k · logits_k``.
    Returns ``{"weights": [N], "z": [N], "history": [...]}``.
    Trained on validation only — no leakage into the test report.
    """

    _check_logits(val_logits)
    n_models = val_logits.shape[0]
    if val_labels.ndim != 1 or val_labels.shape[0] != val_logits.shape[1]:
        raise ValueError("val_labels must be 1D with length equal to batch size")

    z = np.zeros(n_models, dtype=np.float64) if init_z is None else np.asarray(init_z, dtype=np.float64)
    history: list[dict[str, float]] = []
    labels = val_labels.astype(np.int64)
    batch = val_logits.shape[1]

    for step in range(1, n_iters + 1):
        w = _softmax(z, axis=-1)                              # [N]
        agg = np.einsum("n,nbc->bc", w, val_logits)           # [B, C]
        log_p = _log_softmax(agg, axis=-1)                    # [B, C]
        loss = float(-log_p[np.arange(batch), labels].mean())

        # ∂loss/∂agg = (softmax(agg) - one_hot(labels)) / B
        probs = np.exp(log_p)                                  # [B, C]
        grad_agg = probs.copy()
        grad_agg[np.arange(batch), labels] -= 1.0
        grad_agg /= batch                                      # [B, C]

        # ∂agg/∂w_k = sum_{b,c} logits_k * grad_agg
        grad_w = np.einsum("bc,nbc->n", grad_agg, val_logits)  # [N]

        # ∂w/∂z: Jacobian of softmax: J = diag(w) - w w^T
        jac = np.diag(w) - np.outer(w, w)
        grad_z = jac @ grad_w

        z -= lr * grad_z
        history.append({"step": float(step), "loss": loss})
        if step > 10 and abs(history[-1]["loss"] - history[-11]["loss"]) < 1e-6:
            break

    weights = _softmax(z, axis=-1)
    return {"weights": weights, "z": z, "history": history}


def learned_weights_mean(logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Sum_k w_k · logits_k. Returns ``[B, C]``."""

    _check_logits(logits)
    weights = _normalize_weights(weights)
    if weights.shape != (logits.shape[0],):
        raise ValueError(
            f"Expected weights shape [{logits.shape[0]}], got {tuple(weights.shape)}"
        )
    return np.einsum("n,nbc->bc", weights, logits)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def oracle_topk(logits: np.ndarray, labels: np.ndarray, k: int = 1) -> float:
    """Hit if any of the N models has the true label in its top-k."""

    _check_logits(logits)
    n, b, c = logits.shape
    if labels.shape[0] != b:
        raise ValueError("labels length must equal batch size")
    topk_idx = np.argsort(logits, axis=-1)[..., -k:]            # [N, B, k]
    label_col = labels.reshape(1, b, 1)
    hit_per_model = (topk_idx == label_col).any(axis=-1)        # [N, B]
    any_correct = hit_per_model.any(axis=0)
    return float(any_correct.mean())


def pairwise_disagreement(logits: np.ndarray) -> np.ndarray:
    """Symmetric N×N matrix of disagreement rates between models (0 on diag)."""

    _check_logits(logits)
    preds = logits.argmax(axis=-1)                              # [N, B]
    n = preds.shape[0]
    matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = float((preds[i] != preds[j]).mean())
    return matrix


def per_class_disagreement_rate(
    logits: np.ndarray, labels: np.ndarray, label_names: list[str] | None = None
) -> dict[str, float]:
    """Fraction of inputs of each true class where ≥2 models disagree."""

    _check_logits(logits)
    n, b, c = logits.shape
    preds = logits.argmax(axis=-1)                              # [N, B]
    out: dict[str, float] = {}
    unique = sorted(set(int(label) for label in labels.tolist()))
    for cls in unique:
        mask = labels == cls
        if not mask.any():
            continue
        cls_preds = preds[:, mask]                              # [N, n_cls]
        # disagreement = at least one pair of models predicts differently
        # equivalently: not all rows equal
        disagree = (cls_preds != cls_preds[0:1]).any(axis=0)
        rate = float(disagree.mean())
        key = label_names[cls] if (label_names and cls < len(label_names)) else str(cls)
        out[key] = rate
    return out


# ---------------------------------------------------------------------------
# Convenience: collect everything in one call (used by notebook + sim)
# ---------------------------------------------------------------------------


def evaluate_aggregators(
    test_logits: np.ndarray,
    test_labels: np.ndarray,
    val_logits: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Run all aggregators on ``test_logits`` and report metrics.

    Calibration / learned weights are fit on ``val_*`` if provided; otherwise
    those aggregators are skipped.
    """

    _check_logits(test_logits)
    if test_labels.shape[0] != test_logits.shape[1]:
        raise ValueError("test_labels length must equal test batch size")

    def top1(probs_or_logits: np.ndarray) -> float:
        preds = probs_or_logits.argmax(axis=-1)
        return float((preds == test_labels).mean())

    def topk_from_probs(probs: np.ndarray, k: int) -> float:
        topk_idx = np.argsort(probs, axis=-1)[..., -k:]
        label_col = test_labels.reshape(-1, 1)
        return float((topk_idx == label_col).any(axis=-1).mean())

    aggregators: dict[str, Any] = {}

    ml = mean_logits(test_logits)
    aggregators["mean_logits"] = {
        "top1": top1(ml),
        "top3": topk_from_probs(_softmax(ml, axis=-1), k=3),
    }

    mp = mean_probs(test_logits)
    aggregators["mean_probs"] = {"top1": top1(mp), "top3": topk_from_probs(mp, k=3)}

    cw = conf_weighted(test_logits)
    aggregators["conf_weighted"] = {"top1": top1(cw), "top3": topk_from_probs(cw, k=3)}

    tm = trimmed_mean(test_logits, drop=1)
    aggregators["trimmed"] = {
        "top1": top1(tm),
        "top3": topk_from_probs(_softmax(tm, axis=-1), k=3),
    }

    mv_preds = majority_vote(test_logits)
    aggregators["majority_vote"] = {
        "top1": float((mv_preds == test_labels).mean()),
    }

    if val_logits is not None and val_labels is not None:
        temps = fit_per_model_temperatures(val_logits, val_labels)
        ts = temperature_scaled_mean(test_logits, temps)
        aggregators["temperature_scaled"] = {
            "T": temps.tolist(),
            "top1": top1(ts),
            "top3": topk_from_probs(ts, k=3),
        }

        lw = fit_learned_weights(val_logits, val_labels)
        weights = lw["weights"]
        lwm = learned_weights_mean(test_logits, weights)
        aggregators["learned_weights"] = {
            "w": weights.tolist(),
            "z": lw["z"].tolist(),
            "loss_history_tail": lw["history"][-5:],
            "top1": top1(lwm),
            "top3": topk_from_probs(_softmax(lwm, axis=-1), k=3),
        }

    return {
        "aggregators": aggregators,
        "oracle_top1": oracle_topk(test_logits, test_labels, k=1),
        "pairwise_disagreement": pairwise_disagreement(test_logits).tolist(),
        "per_class_disagreement_rate": per_class_disagreement_rate(
            test_logits, test_labels, label_names=label_names
        ),
    }
