from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any

from hash_kws_lab.config import ExperimentConfig, make_experiment
from hash_kws_lab.models import build_student_model, summarize_model
from hash_kws_lab.recipes import build_recipe_book as build_hash_recipe_book


def _tuned_exact_reference(base: ExperimentConfig) -> ExperimentConfig:
    return build_hash_recipe_book(base)["hash_deeper_fair_ce_exact_microfrontend_tuned"]


def _exact_cached(
    experiment: ExperimentConfig,
    *,
    batch_size: int,
    epochs: int,
    patience: int,
    label_smoothing: float = 0.02,
) -> ExperimentConfig:
    return replace(
        experiment,
        dataset=replace(
            experiment.dataset,
            batch_size=batch_size,
            num_workers=2,
            cache_features=True,
            cache_dtype="int8",
            time_shift_ms=0,
            gain_min=1.0,
            gain_max=1.0,
            noise_stddev=0.0,
        ),
        feature=replace(
            experiment.feature,
            frontend_name="exact_microfrontend",
            normalize_mode="none",
            require_exact_microfrontend=True,
        ),
        train=replace(
            experiment.train,
            student_epochs=epochs,
            student_scheduler_name="cosine",
            label_smoothing=label_smoothing,
            grad_clip_norm=1.0,
            student_early_stopping_patience=patience,
            use_ema=False,
            eval_with_ema=False,
        ),
    )


def build_hashednet95_recipe_book(base: ExperimentConfig) -> dict[str, ExperimentConfig]:
    """High-upside HashedNet-style recipes for the deploy-aligned KWS branch.

    These recipes keep the exact firmware microfrontend and current custom
    ESP32 hash-runtime format, while adding the missing HashedNets levers:
    signed hashing, per-layer codebook budgets, and virtual-width inflation.
    """

    reference = _tuned_exact_reference(base)

    hn95_paper_inflate96_fixedk = _exact_cached(
        replace(
            reference,
            tag=f"{base.tag}_hn95_paper_inflate96_fixedk",
            model=replace(
                reference.model,
                channels=96,
                num_blocks=4,
                stem_codebook_size=500,
                depthwise_codebook_size=500,
                pointwise_codebook_size=500,
                linear_codebook_size=500,
                depthwise_codebook_sizes=(),
                pointwise_codebook_sizes=(),
                signed_hash=True,
                use_residual=False,
            ),
        ),
        batch_size=128,
        epochs=48,
        patience=8,
    )

    hn95_layerwise96_signed_residual = _exact_cached(
        replace(
            reference,
            tag=f"{base.tag}_hn95_layerwise96_signed_residual",
            model=replace(
                reference.model,
                channels=96,
                num_blocks=4,
                stem_codebook_size=384,
                depthwise_codebook_size=192,
                pointwise_codebook_size=1024,
                linear_codebook_size=384,
                depthwise_codebook_sizes=(224, 192, 160, 128),
                pointwise_codebook_sizes=(896, 1024, 1152, 1280),
                signed_hash=True,
                use_residual=True,
            ),
        ),
        batch_size=128,
        epochs=64,
        patience=10,
    )

    hn95_layerwise96_signed_cached_specaug = replace(
        hn95_layerwise96_signed_residual,
        tag=f"{base.tag}_hn95_layerwise96_signed_cached_specaug",
        feature=replace(
            hn95_layerwise96_signed_residual.feature,
            specaugment_prob=0.25,
            time_mask_max=4,
            freq_mask_max=3,
        ),
        train=replace(
            hn95_layerwise96_signed_residual.train,
            label_smoothing=0.03,
        ),
    )

    hn95_inflate128_fixed_storage_paper = _exact_cached(
        replace(
            reference,
            tag=f"{base.tag}_hn95_inflate128_fixed_storage_paper",
            model=replace(
                reference.model,
                channels=128,
                num_blocks=4,
                stem_codebook_size=500,
                depthwise_codebook_size=500,
                pointwise_codebook_size=500,
                linear_codebook_size=500,
                depthwise_codebook_sizes=(),
                pointwise_codebook_sizes=(),
                signed_hash=True,
                use_residual=False,
            ),
        ),
        batch_size=96,
        epochs=64,
        patience=10,
    )

    hn95_inflate128_layerwise_signed_residual = _exact_cached(
        replace(
            reference,
            tag=f"{base.tag}_hn95_inflate128_layerwise_signed_residual",
            model=replace(
                reference.model,
                channels=128,
                num_blocks=4,
                stem_codebook_size=512,
                depthwise_codebook_size=224,
                pointwise_codebook_size=1536,
                linear_codebook_size=512,
                depthwise_codebook_sizes=(288, 256, 224, 192),
                pointwise_codebook_sizes=(1280, 1536, 1792, 2048),
                signed_hash=True,
                use_residual=True,
            ),
        ),
        batch_size=96,
        epochs=80,
        patience=12,
    )

    hn95_3block128_big_pointwise_signed = _exact_cached(
        replace(
            reference,
            tag=f"{base.tag}_hn95_3block128_big_pointwise_signed",
            model=replace(
                reference.model,
                channels=128,
                num_blocks=3,
                stem_codebook_size=512,
                depthwise_codebook_size=256,
                pointwise_codebook_size=2048,
                linear_codebook_size=512,
                depthwise_codebook_sizes=(288, 256, 224),
                pointwise_codebook_sizes=(1536, 2048, 2304),
                signed_hash=True,
                use_residual=True,
            ),
        ),
        batch_size=96,
        epochs=80,
        patience=12,
    )

    hn95_kd128_layerwise_signed_residual = replace(
        hn95_inflate128_layerwise_signed_residual,
        tag=f"{base.tag}_hn95_kd128_layerwise_signed_residual",
        teacher_reuse_tag=f"{base.tag}_hn95_kd128_layerwise_signed_residual",
        model=replace(
            hn95_inflate128_layerwise_signed_residual.model,
            teacher_name="dense_dscnn_teacher",
            teacher_channels=128,
            teacher_num_blocks=6,
            teacher_dropout=0.05,
        ),
        train=replace(
            hn95_inflate128_layerwise_signed_residual.train,
            teacher_epochs=48,
            student_pretrain_epochs=6,
            student_epochs=40,
            student_polish_epochs=4,
            teacher_lr=8e-4,
            student_lr=8e-4,
            polish_lr=2e-4,
            teacher_scheduler_name="cosine",
            student_scheduler_name="cosine",
            teacher_label_smoothing=0.02,
            label_smoothing=0.02,
            kd_alpha=0.55,
            kd_temperature=4.5,
            teacher_early_stopping_patience=8,
            student_early_stopping_patience=10,
        ),
    )

    hn95_kd128_cached_schedule = replace(
        hn95_kd128_layerwise_signed_residual,
        tag=f"{base.tag}_hn95_kd128_cached_schedule",
        teacher_reuse_tag=f"{base.tag}_hn95_kd128_layerwise_signed_residual",
        train=replace(
            hn95_kd128_layerwise_signed_residual.train,
            student_epochs=52,
            student_polish_epochs=8,
            label_smoothing=0.025,
            polish_label_smoothing=0.0,
            kd_alpha=0.65,
            kd_alpha_schedule="cosine",
            kd_alpha_final=0.35,
            kd_temperature=5.5,
            kd_temperature_schedule="cosine",
            kd_temperature_final=3.0,
            cache_teacher_logits=True,
            teacher_logits_cache_dtype="float16",
            student_early_stopping_patience=12,
        ),
    )

    hn95_kd128_rich_teacher160_cached_schedule = replace(
        hn95_kd128_cached_schedule,
        tag=f"{base.tag}_hn95_kd128_rich_teacher160_cached_schedule",
        teacher_reuse_tag=f"{base.tag}_hn95_teacher160_dense_reusable",
        model=replace(
            hn95_kd128_cached_schedule.model,
            teacher_channels=160,
            teacher_num_blocks=7,
            teacher_dropout=0.06,
        ),
        train=replace(
            hn95_kd128_cached_schedule.train,
            teacher_epochs=72,
            teacher_lr=6e-4,
            teacher_label_smoothing=0.035,
            teacher_early_stopping_patience=10,
        ),
    )

    hn95_kd112_latency_balanced_cached_schedule = replace(
        hn95_kd128_cached_schedule,
        tag=f"{base.tag}_hn95_kd112_latency_balanced_cached_schedule",
        model=replace(
            hn95_kd128_cached_schedule.model,
            channels=112,
            stem_codebook_size=448,
            depthwise_codebook_size=224,
            pointwise_codebook_size=1280,
            linear_codebook_size=448,
            depthwise_codebook_sizes=(256, 224, 192, 160),
            pointwise_codebook_sizes=(1024, 1280, 1536, 1792),
        ),
        dataset=replace(
            hn95_kd128_cached_schedule.dataset,
            batch_size=112,
        ),
        train=replace(
            hn95_kd128_cached_schedule.train,
            student_epochs=50,
            student_polish_epochs=6,
        ),
    )

    hn95_kd96_latency_guard_cached_schedule = replace(
        hn95_kd128_cached_schedule,
        tag=f"{base.tag}_hn95_kd96_latency_guard_cached_schedule",
        model=replace(
            hn95_kd128_cached_schedule.model,
            channels=96,
            stem_codebook_size=384,
            depthwise_codebook_size=192,
            pointwise_codebook_size=1024,
            linear_codebook_size=384,
            depthwise_codebook_sizes=(224, 192, 160, 128),
            pointwise_codebook_sizes=(896, 1024, 1152, 1280),
        ),
        dataset=replace(
            hn95_kd128_cached_schedule.dataset,
            batch_size=128,
        ),
        train=replace(
            hn95_kd128_cached_schedule.train,
            student_epochs=48,
            student_polish_epochs=6,
            student_early_stopping_patience=10,
        ),
    )

    hn95_kd128_pw2048_budget_cached_schedule = replace(
        hn95_kd128_cached_schedule,
        tag=f"{base.tag}_hn95_kd128_pw2048_budget_cached_schedule",
        model=replace(
            hn95_kd128_cached_schedule.model,
            stem_codebook_size=448,
            depthwise_codebook_size=192,
            pointwise_codebook_size=2048,
            linear_codebook_size=640,
            depthwise_codebook_sizes=(256, 224, 192, 160),
            pointwise_codebook_sizes=(1536, 1792, 2048, 2304),
        ),
        train=replace(
            hn95_kd128_cached_schedule.train,
            student_epochs=56,
            student_polish_epochs=8,
        ),
    )

    hn95_kd128_hard_polish_cached = replace(
        hn95_kd128_cached_schedule,
        tag=f"{base.tag}_hn95_kd128_hard_polish_cached",
        train=replace(
            hn95_kd128_cached_schedule.train,
            label_smoothing=0.015,
            student_epochs=50,
            student_polish_epochs=10,
            polish_lr=1.2e-4,
            polish_label_smoothing=0.0,
        ),
    )

    hn95_kd128_cached_schedule_s29 = replace(
        hn95_kd128_cached_schedule,
        tag=f"{base.tag}_hn95_kd128_cached_schedule_s29",
        train=replace(
            hn95_kd128_cached_schedule.train,
            seed=29,
        ),
    )

    hn95_kd128_cached_schedule_s47 = replace(
        hn95_kd128_cached_schedule,
        tag=f"{base.tag}_hn95_kd128_cached_schedule_s47",
        train=replace(
            hn95_kd128_cached_schedule.train,
            seed=47,
        ),
    )

    return {
        "hn95_paper_inflate96_fixedk": hn95_paper_inflate96_fixedk,
        "hn95_layerwise96_signed_residual": hn95_layerwise96_signed_residual,
        "hn95_layerwise96_signed_cached_specaug": hn95_layerwise96_signed_cached_specaug,
        "hn95_inflate128_fixed_storage_paper": hn95_inflate128_fixed_storage_paper,
        "hn95_inflate128_layerwise_signed_residual": hn95_inflate128_layerwise_signed_residual,
        "hn95_3block128_big_pointwise_signed": hn95_3block128_big_pointwise_signed,
        "hn95_kd128_layerwise_signed_residual": hn95_kd128_layerwise_signed_residual,
        "hn95_kd128_cached_schedule": hn95_kd128_cached_schedule,
        "hn95_kd128_rich_teacher160_cached_schedule": hn95_kd128_rich_teacher160_cached_schedule,
        "hn95_kd112_latency_balanced_cached_schedule": hn95_kd112_latency_balanced_cached_schedule,
        "hn95_kd96_latency_guard_cached_schedule": hn95_kd96_latency_guard_cached_schedule,
        "hn95_kd128_pw2048_budget_cached_schedule": hn95_kd128_pw2048_budget_cached_schedule,
        "hn95_kd128_hard_polish_cached": hn95_kd128_hard_polish_cached,
        "hn95_kd128_cached_schedule_s29": hn95_kd128_cached_schedule_s29,
        "hn95_kd128_cached_schedule_s47": hn95_kd128_cached_schedule_s47,
    }


def with_drive_cache_paths(
    experiment: ExperimentConfig,
    drive_cache_root: str | Path,
) -> ExperimentConfig:
    """Persist Speech Commands, exact-feature cache, and exports on Drive."""

    root = Path(drive_cache_root)
    data_root = Path(os.environ.get("SPEECHCOMMANDS_DATA_ROOT", root / "speechcommands_v2"))
    feature_cache_root = root / "hash_feature_cache"
    teacher_logits_cache_root = root / "teacher_logits_cache"
    artifact_root = root / "hash_artifacts"
    data_root.mkdir(parents=True, exist_ok=True)
    feature_cache_root.mkdir(parents=True, exist_ok=True)
    teacher_logits_cache_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("SPEECHCOMMANDS_DATA_ROOT", str(data_root))
    os.environ.setdefault("HASH_KWS_FEATURE_CACHE_ROOT", str(feature_cache_root))
    os.environ.setdefault("HASH_KWS_TEACHER_LOGITS_CACHE_ROOT", str(teacher_logits_cache_root))
    return replace(
        experiment,
        train=replace(
            experiment.train,
            teacher_logits_cache_dir=str(teacher_logits_cache_root),
        ),
        export=replace(
            experiment.export,
            artifacts_dir=str(artifact_root),
        ),
    )


def describe_hashednet95_recipe(experiment: ExperimentConfig) -> dict[str, Any]:
    student = build_student_model(experiment)
    summary = summarize_model(student, experiment)
    reference = _tuned_exact_reference(
        make_experiment(tag="hash_kws12_iterlab_v1", vocabulary_preset=experiment.vocabulary_preset)
    )
    reference_summary = summarize_model(build_student_model(reference), reference)
    virtual = int(summary["virtual_dense_parameters"])
    compact = int(summary["hash_compact_parameters"])
    ref_virtual = int(reference_summary["virtual_dense_parameters"])
    ref_compact = int(reference_summary["hash_compact_parameters"])
    return {
        "tag": experiment.tag,
        "labels": experiment.all_labels,
        "model_input_shape": experiment.model_input_shape,
        "channels": experiment.model.channels,
        "num_blocks": experiment.model.num_blocks,
        "signed_hash": experiment.model.signed_hash,
        "use_residual": experiment.model.use_residual,
        "train_seed": experiment.train.seed,
        "teacher_reuse_tag": experiment.teacher_reuse_tag,
        "teacher_name": experiment.model.teacher_name,
        "teacher_channels": experiment.model.teacher_channels,
        "teacher_num_blocks": experiment.model.teacher_num_blocks,
        "teacher_dropout": experiment.model.teacher_dropout,
        "kd_alpha": experiment.train.kd_alpha,
        "kd_alpha_schedule": experiment.train.kd_alpha_schedule,
        "kd_alpha_final": experiment.train.kd_alpha_final,
        "kd_temperature": experiment.train.kd_temperature,
        "kd_temperature_schedule": experiment.train.kd_temperature_schedule,
        "kd_temperature_final": experiment.train.kd_temperature_final,
        "cache_teacher_logits": experiment.train.cache_teacher_logits,
        "student_polish_epochs": experiment.train.student_polish_epochs,
        "polish_label_smoothing": experiment.train.polish_label_smoothing,
        "depthwise_codebook_sizes": list(experiment.model.depthwise_codebook_sizes),
        "pointwise_codebook_sizes": list(experiment.model.pointwise_codebook_sizes),
        "student_summary": summary,
        "reference_summary": reference_summary,
        "virtual_inflation_vs_reference": virtual / max(ref_virtual, 1),
        "compact_growth_vs_reference": compact / max(ref_compact, 1),
        "virtual_per_compact_parameter": virtual / max(compact, 1),
        "reference_virtual_per_compact_parameter": ref_virtual / max(ref_compact, 1),
    }
