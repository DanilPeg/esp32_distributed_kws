from __future__ import annotations

from dataclasses import replace
from typing import Any

from hash_kws_lab.config import ExperimentConfig
from hashednet95.hashednet95_recipes import build_hashednet95_recipe_book


# ---------------------------------------------------------------------------
# Three near-homo variants of the baseline `hn95_kd128_layerwise_signed_residual`
# Differ only in:
#   - pointwise codebook sizes (slight perturbation around baseline)
#   - train.seed (init + dataloader shuffle)
# Architecture (channels, num_blocks, signed_hash, residual, stem/dw/linear codebooks)
# is held identical so all three share the same firmware runtime path and so the
# teacher logits cache (keyed on feature+dataset+teacher fingerprint) is reused
# across all three student runs.
# ---------------------------------------------------------------------------


_VARIANT_DEFINITIONS: tuple[tuple[str, int, tuple[int, int, int, int]], ...] = (
    # name        seed   pointwise_codebook_sizes (per block)
    ("ens_a",     13,    (1024, 1280, 1536, 1792)),  # -20% PW vs baseline
    ("ens_b",     29,    (1280, 1536, 1792, 2048)),  # baseline (= hn95_kd128_layerwise_signed_residual)
    ("ens_c",     47,    (1536, 1792, 2048, 2304)),  # +13% PW vs baseline
)

ENSEMBLE_TEACHER_VARIANT = "ens_b"


def variant_tag(base_tag: str, name: str, seed: int) -> str:
    return f"{base_tag}_hn95_{name}_s{seed}"


def build_ensemble_recipe_book(base: ExperimentConfig) -> dict[str, ExperimentConfig]:
    """Three near-homo HashedNet KWS recipes for the distributed ensemble track."""

    hn95_book = build_hashednet95_recipe_book(base)
    baseline = hn95_book["hn95_kd128_layerwise_signed_residual"]

    # Cache teacher logits so the second/third student reuse them.
    baseline_cached = replace(
        baseline,
        train=replace(
            baseline.train,
            cache_teacher_logits=True,
            teacher_logits_cache_dtype="float16",
        ),
    )

    teacher_anchor_tag = variant_tag(
        base.tag,
        ENSEMBLE_TEACHER_VARIANT,
        dict((name, seed) for name, seed, _ in _VARIANT_DEFINITIONS)[ENSEMBLE_TEACHER_VARIANT],
    )

    recipes: dict[str, ExperimentConfig] = {}
    for name, seed, pointwise_codebook_sizes in _VARIANT_DEFINITIONS:
        tag = variant_tag(base.tag, name, seed)
        recipes[name] = replace(
            baseline_cached,
            tag=tag,
            # Anchor every student to the teacher trained for ens_b. This is the
            # value used in run summaries; the actual checkpoint reuse goes
            # through TEACHER_CHECKPOINT_PATH / FORCE_TEACHER_RETRAIN at runtime.
            teacher_reuse_tag=teacher_anchor_tag,
            model=replace(
                baseline_cached.model,
                pointwise_codebook_sizes=pointwise_codebook_sizes,
            ),
            train=replace(
                baseline_cached.train,
                seed=seed,
            ),
        )
    return recipes


def describe_variant(experiment: ExperimentConfig) -> dict[str, Any]:
    return {
        "tag": experiment.tag,
        "seed": experiment.train.seed,
        "channels": experiment.model.channels,
        "num_blocks": experiment.model.num_blocks,
        "stem_codebook_size": experiment.model.stem_codebook_size,
        "depthwise_codebook_sizes": list(experiment.model.depthwise_codebook_sizes),
        "pointwise_codebook_sizes": list(experiment.model.pointwise_codebook_sizes),
        "linear_codebook_size": experiment.model.linear_codebook_size,
        "signed_hash": experiment.model.signed_hash,
        "use_residual": experiment.model.use_residual,
        "kd_alpha": experiment.train.kd_alpha,
        "kd_temperature": experiment.train.kd_temperature,
        "student_pretrain_epochs": experiment.train.student_pretrain_epochs,
        "student_epochs": experiment.train.student_epochs,
        "student_polish_epochs": experiment.train.student_polish_epochs,
        "cache_teacher_logits": experiment.train.cache_teacher_logits,
        "teacher_reuse_tag": experiment.teacher_reuse_tag,
    }
