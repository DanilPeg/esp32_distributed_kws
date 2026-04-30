from __future__ import annotations

from dataclasses import replace

from .config import ExperimentConfig


def build_recipe_book(base: ExperimentConfig) -> dict[str, ExperimentConfig]:
    hash_deeper_usermix_ce = replace(
        base,
        tag=f"{base.tag}_hash_deeper_usermix_ce",
        dataset=replace(
            base.dataset,
            unknown_fraction=0.10,
            silence_fraction=0.10,
            silence_reference="known",
            time_shift_ms=0,
            gain_min=1.0,
            gain_max=1.0,
            noise_stddev=0.0,
        ),
        feature=replace(
            base.feature,
            specaugment_prob=0.0,
            time_mask_max=0,
            freq_mask_max=0,
        ),
        model=replace(
            base.model,
            teacher_name="",
            student_name="hash_dscnn_deeper",
            channels=64,
            num_blocks=4,
            codebook_size=500,
            stem_codebook_size=500,
            depthwise_codebook_size=500,
            pointwise_codebook_size=500,
            linear_codebook_size=500,
            signed_hash=False,
            hash_only_pointwise=False,
        ),
        train=replace(
            base.train,
            teacher_epochs=0,
            student_pretrain_epochs=0,
            student_epochs=20,
            student_polish_epochs=0,
            student_lr=1e-3,
            weight_decay=1e-4,
            student_scheduler_name="none",
            label_smoothing=0.0,
            kd_alpha=0.0,
            grad_clip_norm=0.0,
            student_early_stopping_patience=0,
            use_ema=False,
        ),
    )

    hash_deeper_fair_ce = replace(
        hash_deeper_usermix_ce,
        tag=f"{base.tag}_hash_deeper_fair_ce",
        dataset=replace(
            hash_deeper_usermix_ce.dataset,
            unknown_fraction=1.0,
            silence_fraction=0.12,
            silence_reference="known",
        ),
    )

    hash_deeper_fair_augmented = replace(
        hash_deeper_fair_ce,
        tag=f"{base.tag}_hash_deeper_fair_augmented",
        dataset=replace(
            hash_deeper_fair_ce.dataset,
            time_shift_ms=100,
            gain_min=0.8,
            gain_max=1.2,
            noise_stddev=0.004,
        ),
        feature=replace(
            hash_deeper_fair_ce.feature,
            specaugment_prob=0.35,
            time_mask_max=4,
            freq_mask_max=3,
        ),
        train=replace(
            hash_deeper_fair_ce.train,
            student_epochs=24,
            student_scheduler_name="cosine",
            label_smoothing=0.05,
            grad_clip_norm=1.0,
            student_early_stopping_patience=6,
            use_ema=True,
            ema_decay=0.995,
            eval_with_ema=True,
        ),
    )

    hash_deeper_fair_signed = replace(
        hash_deeper_fair_augmented,
        tag=f"{base.tag}_hash_deeper_fair_signed",
        model=replace(
            hash_deeper_fair_augmented.model,
            signed_hash=True,
        ),
    )

    hash_deeper_fair_ce_exact_microfrontend = replace(
        hash_deeper_fair_ce,
        tag=f"{base.tag}_hash_deeper_fair_ce_exact_microfrontend",
        dataset=replace(
            hash_deeper_fair_ce.dataset,
            num_workers=2,
            cache_features=True,
        ),
        feature=replace(
            hash_deeper_fair_ce.feature,
            frontend_name="exact_microfrontend",
            normalize_mode="none",
            require_exact_microfrontend=True,
            specaugment_prob=0.0,
            time_mask_max=0,
            freq_mask_max=0,
        ),
    )

    hash_deeper_fair_ce_exact_microfrontend_tuned = replace(
        hash_deeper_fair_ce_exact_microfrontend,
        tag=f"{base.tag}_hash_deeper_fair_ce_exact_microfrontend_tuned",
        dataset=replace(
            hash_deeper_fair_ce_exact_microfrontend.dataset,
            batch_size=128,
            num_workers=2,
            cache_features=True,
        ),
        train=replace(
            hash_deeper_fair_ce_exact_microfrontend.train,
            student_epochs=28,
            student_scheduler_name="cosine",
            label_smoothing=0.02,
            grad_clip_norm=1.0,
            student_early_stopping_patience=6,
            use_ema=False,
            eval_with_ema=False,
        ),
    )

    hash_deeper_fair_specaug_exact_microfrontend = replace(
        hash_deeper_fair_ce_exact_microfrontend_tuned,
        tag=f"{base.tag}_hash_deeper_fair_specaug_exact_microfrontend",
        feature=replace(
            hash_deeper_fair_ce_exact_microfrontend_tuned.feature,
            specaugment_prob=0.35,
            time_mask_max=6,
            freq_mask_max=4,
        ),
    )

    hash_deeper_fair_pointwise_budget_exact_microfrontend = replace(
        hash_deeper_fair_specaug_exact_microfrontend,
        tag=f"{base.tag}_hash_deeper_fair_pointwise_budget_exact_microfrontend",
        model=replace(
            hash_deeper_fair_specaug_exact_microfrontend.model,
            stem_codebook_size=192,
            depthwise_codebook_size=96,
            pointwise_codebook_size=1024,
            linear_codebook_size=256,
        ),
    )

    hash_deeper_fair_balanced_budget_exact_microfrontend = replace(
        hash_deeper_fair_ce_exact_microfrontend_tuned,
        tag=f"{base.tag}_hash_deeper_fair_balanced_budget_exact_microfrontend",
        model=replace(
            hash_deeper_fair_ce_exact_microfrontend_tuned.model,
            stem_codebook_size=256,
            depthwise_codebook_size=192,
            pointwise_codebook_size=864,
            linear_codebook_size=520,
        ),
        train=replace(
            hash_deeper_fair_ce_exact_microfrontend_tuned.train,
            student_epochs=60,
        ),
    )

    hash_deeper_fair_3block_big_pointwise_exact_microfrontend = replace(
        hash_deeper_fair_ce_exact_microfrontend_tuned,
        tag=f"{base.tag}_hash_deeper_fair_3block_big_pointwise_exact_microfrontend",
        model=replace(
            hash_deeper_fair_ce_exact_microfrontend_tuned.model,
            num_blocks=3,
            stem_codebook_size=500,
            depthwise_codebook_size=500,
            pointwise_codebook_size=1000,
            linear_codebook_size=384,
        ),
        train=replace(
            hash_deeper_fair_ce_exact_microfrontend_tuned.train,
            student_epochs=60,
        ),
    )

    hash_deeper_fair_augmented_exact_microfrontend = replace(
        hash_deeper_fair_augmented,
        tag=f"{base.tag}_hash_deeper_fair_augmented_exact_microfrontend",
        dataset=replace(
            hash_deeper_fair_augmented.dataset,
            num_workers=2,
            cache_features=False,
        ),
        feature=replace(
            hash_deeper_fair_augmented.feature,
            frontend_name="exact_microfrontend",
            normalize_mode="none",
            require_exact_microfrontend=True,
        ),
    )

    hash_deeper_fair_signed_exact_microfrontend = replace(
        hash_deeper_fair_augmented_exact_microfrontend,
        tag=f"{base.tag}_hash_deeper_fair_signed_exact_microfrontend",
        model=replace(
            hash_deeper_fair_augmented_exact_microfrontend.model,
            signed_hash=True,
        ),
    )

    hash_deeper_fair_signed_cached_exact_microfrontend = replace(
        hash_deeper_fair_specaug_exact_microfrontend,
        tag=f"{base.tag}_hash_deeper_fair_signed_cached_exact_microfrontend",
        model=replace(
            hash_deeper_fair_specaug_exact_microfrontend.model,
            signed_hash=True,
        ),
    )

    hash_deeper_fair_residual_exact_microfrontend = replace(
        hash_deeper_fair_ce_exact_microfrontend_tuned,
        tag=f"{base.tag}_hash_deeper_fair_residual_exact_microfrontend",
        model=replace(
            hash_deeper_fair_ce_exact_microfrontend_tuned.model,
            use_residual=True,
        ),
        train=replace(
            hash_deeper_fair_ce_exact_microfrontend_tuned.train,
            student_epochs=60,
        ),
    )

    hash_deeper_fair_residual_specaug_exact_microfrontend = replace(
        hash_deeper_fair_specaug_exact_microfrontend,
        tag=f"{base.tag}_hash_deeper_fair_residual_specaug_exact_microfrontend",
        model=replace(
            hash_deeper_fair_specaug_exact_microfrontend.model,
            use_residual=True,
        ),
        train=replace(
            hash_deeper_fair_specaug_exact_microfrontend.train,
            student_epochs=60,
        ),
    )

    hash_pointwise_only_fair = replace(
        hash_deeper_fair_augmented,
        tag=f"{base.tag}_hash_pointwise_only_fair",
        model=replace(
            hash_deeper_fair_augmented.model,
            hash_only_pointwise=True,
            pointwise_codebook_size=384,
            linear_codebook_size=384,
        ),
    )

    dense_teacher_fair = replace(
        hash_deeper_fair_augmented,
        tag=f"{base.tag}_dense_teacher_fair",
        model=replace(
            hash_deeper_fair_augmented.model,
            teacher_name="",
            student_name="dense_dscnn_teacher",
            teacher_channels=96,
            teacher_num_blocks=5,
            student_dropout=0.10,
        ),
        train=replace(
            hash_deeper_fair_augmented.train,
            student_epochs=18,
            label_smoothing=0.05,
            kd_alpha=0.0,
        ),
        export=replace(
            hash_deeper_fair_augmented.export,
            model_stem="dense_hash_teacher",
        ),
    )

    hash_deeper_fair_kd = replace(
        hash_deeper_fair_augmented,
        tag=f"{base.tag}_hash_deeper_fair_kd",
        teacher_reuse_tag=f"{base.tag}_dense_teacher_fair",
        model=replace(
            hash_deeper_fair_augmented.model,
            teacher_name="dense_dscnn_teacher",
            student_name="hash_dscnn_deeper",
            teacher_channels=96,
            teacher_num_blocks=5,
        ),
        train=replace(
            hash_deeper_fair_augmented.train,
            teacher_epochs=18,
            student_pretrain_epochs=6,
            student_epochs=14,
            student_polish_epochs=4,
            teacher_lr=8e-4,
            student_lr=9e-4,
            polish_lr=2e-4,
            teacher_scheduler_name="cosine",
            student_scheduler_name="cosine",
            teacher_label_smoothing=0.05,
            label_smoothing=0.02,
            kd_alpha=0.60,
            kd_temperature=4.0,
            teacher_early_stopping_patience=5,
            student_early_stopping_patience=6,
        ),
    )

    return {
        "hash_deeper_usermix_ce": hash_deeper_usermix_ce,
        "hash_deeper_fair_ce": hash_deeper_fair_ce,
        "hash_deeper_fair_augmented": hash_deeper_fair_augmented,
        "hash_deeper_fair_signed": hash_deeper_fair_signed,
        "hash_deeper_fair_ce_exact_microfrontend": hash_deeper_fair_ce_exact_microfrontend,
        "hash_deeper_fair_ce_exact_microfrontend_tuned": hash_deeper_fair_ce_exact_microfrontend_tuned,
        "hash_deeper_fair_specaug_exact_microfrontend": hash_deeper_fair_specaug_exact_microfrontend,
        "hash_deeper_fair_pointwise_budget_exact_microfrontend": hash_deeper_fair_pointwise_budget_exact_microfrontend,
        "hash_deeper_fair_balanced_budget_exact_microfrontend": hash_deeper_fair_balanced_budget_exact_microfrontend,
        "hash_deeper_fair_3block_big_pointwise_exact_microfrontend": hash_deeper_fair_3block_big_pointwise_exact_microfrontend,
        "hash_deeper_fair_augmented_exact_microfrontend": hash_deeper_fair_augmented_exact_microfrontend,
        "hash_deeper_fair_signed_exact_microfrontend": hash_deeper_fair_signed_exact_microfrontend,
        "hash_deeper_fair_signed_cached_exact_microfrontend": hash_deeper_fair_signed_cached_exact_microfrontend,
        "hash_deeper_fair_residual_exact_microfrontend": hash_deeper_fair_residual_exact_microfrontend,
        "hash_deeper_fair_residual_specaug_exact_microfrontend": hash_deeper_fair_residual_specaug_exact_microfrontend,
        "hash_pointwise_only_fair": hash_pointwise_only_fair,
        "dense_teacher_fair": dense_teacher_fair,
        "hash_deeper_fair_kd": hash_deeper_fair_kd,
    }
