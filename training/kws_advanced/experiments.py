from __future__ import annotations

from dataclasses import replace

from .config import ExperimentConfig


def build_recipe_book(base: ExperimentConfig) -> dict[str, ExperimentConfig]:
    baseline = replace(
        base,
        tag=f"{base.tag}_baseline",
        dataset=replace(
            base.dataset,
            batch_size=128,
            mixup_alpha=0.0,
            specaugment_prob=0.3,
            time_mask_max=2,
            freq_mask_max=2,
        ),
        model=replace(
            base.model,
            teacher_name="baseline_cnn",
            student_name="baseline_cnn",
            teacher_width=1.0,
            student_width=1.0,
            teacher_dropout=0.15,
            student_dropout=0.15,
            teacher_label_smoothing=0.01,
        ),
        train=replace(
            base.train,
            ce_loss_weight=1.0,
            kd_loss_weight=0.0,
            use_qat=False,
            teacher_epochs=12,
            student_epochs=12,
            teacher_early_stopping_patience=4,
            teacher_early_stopping_min_delta=0.0,
            optimizer_name="adam",
            weight_decay=0.0,
        ),
    )

    fast_student = replace(
        base,
        tag=f"{base.tag}_fast_student",
        model=replace(
            base.model,
            student_name="student_factorized_dscnn",
            student_width=0.65,
            student_dropout=0.08,
        ),
        train=replace(
            base.train,
            ce_loss_weight=1.0,
            kd_loss_weight=0.0,
            use_qat=False,
            student_epochs=14,
        ),
    )

    distilled_student = replace(
        base,
        tag=f"{base.tag}_distilled_v1ref",
        dataset=replace(
            base.dataset,
            batch_size=128,
            noise_stddev=0.005,
            mixup_alpha=0.2,
            specaugment_prob=0.7,
            time_mask_max=5,
            freq_mask_max=4,
        ),
        model=replace(
            base.model,
            teacher_name="teacher_factorized_dscnn",
            student_name="student_factorized_dscnn",
            teacher_width=1.0,
            student_width=0.75,
            student_dropout=0.12,
            teacher_label_smoothing=0.01,
            label_smoothing=0.05,
        ),
        train=replace(
            base.train,
            teacher_epochs=18,
            student_epochs=24,
            ce_loss_weight=0.35,
            kd_loss_weight=0.65,
            distill_temperature=4.0,
            distillation_mode="logits",
            use_qat=False,
            teacher_early_stopping_patience=6,
            teacher_early_stopping_min_delta=0.0,
            early_stopping_patience=6,
            optimizer_name="adam",
            weight_decay=0.0,
        ),
    )

    distilled_qat = replace(
        distilled_student,
        tag=f"{base.tag}_distilled_qat_v1ref",
        train=replace(
            distilled_student.train,
            use_qat=True,
            qat_epochs=6,
        ),
    )

    distilled_student_v2 = replace(
        base,
        tag=f"{base.tag}_distilled_v2",
        model=replace(
            base.model,
            teacher_name="teacher_factorized_dscnn",
            student_name="student_factorized_dscnn_v2",
            teacher_width=1.0,
            student_width=0.875,
            student_dropout=0.08,
            teacher_label_smoothing=0.01,
            label_smoothing=0.02,
        ),
        train=replace(
            base.train,
            use_qat=False,
            ce_loss_weight=0.2,
            kd_loss_weight=0.8,
            distill_temperature=5.0,
            distillation_mode="logits",
            teacher_epochs=14,
            student_epochs=18,
            teacher_early_stopping_patience=4,
            teacher_early_stopping_min_delta=0.0,
            early_stopping_patience=4,
            optimizer_name="adamw",
            weight_decay=1.5e-4,
        ),
    )

    distilled_student_v3 = replace(
        distilled_student_v2,
        tag=f"{base.tag}_distilled_v3",
        train=replace(
            distilled_student_v2.train,
            teacher_epochs=10,
            student_epochs=10,
            student_polish_epochs=1,
            ce_loss_weight=0.28,
            kd_loss_weight=0.5,
            distill_temperature=6.0,
            distillation_mode="multilevel",
            hint_warmup_epochs=2,
            hint_warmup_ce_weight=0.35,
            hint_warmup_hint_weight=0.65,
            hint_loss_weight=0.14,
            attention_loss_weight=0.08,
            similarity_loss_weight=0.04,
            hint_layer_pair=("stage3_block1_out", "stage3_block2_out"),
            attention_layer_pairs=(
                ("stage2_block2_out", "stage2_block2_out"),
                ("stage3_block1_out", "stage3_block2_out"),
            ),
            similarity_layer_pair=("stage3_block1_out", "stage3_block2_out"),
            teacher_early_stopping_patience=4,
            teacher_early_stopping_min_delta=0.0,
            early_stopping_patience=2,
            early_stopping_min_delta=0.002,
        ),
    )

    distilled_student_v3_teacher_plus = replace(
        distilled_student_v3,
        tag=f"{base.tag}_distilled_v3_teacher_plus",
        model=replace(
            distilled_student_v3.model,
            teacher_name="teacher_factorized_dscnn_xl",
            teacher_width=1.0,
            teacher_dropout=0.15,
            teacher_label_smoothing=0.01,
        ),
        train=replace(
            distilled_student_v3.train,
            teacher_epochs=16,
            teacher_early_stopping_patience=5,
            teacher_early_stopping_min_delta=0.0,
        ),
    )

    distilled_student_v3_bcresnet_teacher = replace(
        distilled_student_v3,
        tag=f"{base.tag}_distilled_v3_bcresnet_teacher",
        model=replace(
            distilled_student_v3.model,
            teacher_name="teacher_bcresnet",
            teacher_width=1.0,
            teacher_dropout=0.0,
            teacher_label_smoothing=0.0,
        ),
        train=replace(
            distilled_student_v3.train,
            teacher_epochs=20,
            teacher_early_stopping_patience=6,
            teacher_early_stopping_min_delta=0.0,
        ),
    )

    distilled_student_v3_bcresnet_teacher_seq = replace(
        distilled_student_v3_bcresnet_teacher,
        tag=f"{base.tag}_distilled_v3_bcresnet_teacher_seq",
        dataset=replace(
            distilled_student_v3_bcresnet_teacher.dataset,
            batch_size=96,
            mixup_alpha=0.08,
            specaugment_prob=0.45,
            time_mask_max=5,
            freq_mask_max=4,
        ),
        train=replace(
            distilled_student_v3_bcresnet_teacher.train,
            student_pretrain_epochs=6,
            student_epochs=10,
            student_polish_epochs=1,
            ce_loss_weight=0.45,
            kd_loss_weight=0.55,
            distill_temperature=4.5,
            distillation_mode="logits",
            hint_warmup_epochs=0,
            hint_warmup_ce_weight=0.0,
            hint_warmup_hint_weight=0.0,
            hint_loss_weight=0.0,
            attention_loss_weight=0.0,
            similarity_loss_weight=0.0,
            hint_layer_pair=("", ""),
            attention_layer_pairs=(),
            similarity_layer_pair=("", ""),
            early_stopping_patience=4,
            early_stopping_min_delta=0.001,
        ),
    )

    distilled_student_v3_kwt_teacher = replace(
        distilled_student_v3,
        tag=f"{base.tag}_distilled_v3_kwt_teacher",
        dataset=replace(
            distilled_student_v3.dataset,
            batch_size=128,
            mixup_alpha=0.0,
            specaugment_prob=0.7,
            time_mask_max=6,
            freq_mask_max=4,
        ),
        model=replace(
            distilled_student_v3.model,
            teacher_name="teacher_kwt",
            teacher_width=1.0,
            teacher_dropout=0.0,
            teacher_label_smoothing=0.1,
        ),
        train=replace(
            distilled_student_v3.train,
            teacher_epochs=40,
            teacher_warmup_epochs=8,
            teacher_lr=5e-4,
            student_epochs=12,
            student_polish_epochs=1,
            ce_loss_weight=0.25,
            kd_loss_weight=0.75,
            distill_temperature=6.0,
            distillation_mode="logits",
            hint_warmup_epochs=0,
            hint_warmup_ce_weight=0.0,
            hint_warmup_hint_weight=0.0,
            hint_loss_weight=0.0,
            attention_loss_weight=0.0,
            similarity_loss_weight=0.0,
            hint_layer_pair=("", ""),
            attention_layer_pairs=(),
            similarity_layer_pair=("", ""),
            teacher_early_stopping_patience=8,
            teacher_early_stopping_min_delta=0.0,
            early_stopping_patience=4,
            early_stopping_min_delta=0.001,
            optimizer_name="adamw",
            weight_decay=5e-4,
        ),
    )

    distilled_student_v3_kwt_rich_teacher = replace(
        distilled_student_v3_kwt_teacher,
        tag=f"{base.tag}_distilled_v3_kwt_rich_teacher",
        teacher_frontend=replace(
            distilled_student_v3.frontend,
            enable_exact_microfrontend=False,
            use_pcen_fallback=False,
            frame_length_ms=25,
            frame_step_ms=10,
            num_channels=64,
            lower_band_limit=60.0,
            upper_band_limit=7_800.0,
            fallback_feature_clip=10.0,
        ),
        student_frontend=base.frontend,
        dataset=replace(
            distilled_student_v3_kwt_teacher.dataset,
            batch_size=96,
            specaugment_prob=0.5,
            time_mask_max=8,
            freq_mask_max=6,
        ),
        model=replace(
            distilled_student_v3_kwt_teacher.model,
            teacher_label_smoothing=0.05,
        ),
        train=replace(
            distilled_student_v3_kwt_teacher.train,
            teacher_epochs=36,
            teacher_warmup_epochs=8,
            teacher_lr=4e-4,
            teacher_early_stopping_patience=8,
            teacher_early_stopping_min_delta=0.0,
        ),
    )

    distilled_student_v3_kwt_rich_teacher_seq = replace(
        distilled_student_v3_kwt_rich_teacher,
        tag=f"{base.tag}_distilled_v3_kwt_rich_teacher_seq",
        teacher_reuse_tag=distilled_student_v3_kwt_rich_teacher.tag,
        dataset=replace(
            distilled_student_v3_kwt_rich_teacher.dataset,
            mixup_alpha=0.05,
            specaugment_prob=0.4,
            time_mask_max=6,
            freq_mask_max=4,
        ),
        train=replace(
            distilled_student_v3_kwt_rich_teacher.train,
            student_pretrain_epochs=6,
            student_epochs=8,
            student_polish_epochs=1,
            ce_loss_weight=0.45,
            kd_loss_weight=0.55,
            distill_temperature=4.5,
            early_stopping_patience=3,
            early_stopping_min_delta=0.001,
        ),
    )

    assistant_bcresnet_from_kwt_rich_teacher = replace(
        distilled_student_v3_kwt_rich_teacher,
        tag=f"{base.tag}_assistant_bcresnet_from_kwt_rich_teacher",
        teacher_reuse_tag=distilled_student_v3_kwt_rich_teacher.tag,
        dataset=replace(
            distilled_student_v3_kwt_rich_teacher.dataset,
            batch_size=96,
            mixup_alpha=0.05,
            specaugment_prob=0.35,
            time_mask_max=5,
            freq_mask_max=4,
        ),
        model=replace(
            distilled_student_v3_kwt_rich_teacher.model,
            student_name="teacher_bcresnet",
            student_width=1.0,
            student_dropout=0.0,
            label_smoothing=0.01,
        ),
        train=replace(
            distilled_student_v3_kwt_rich_teacher.train,
            student_pretrain_epochs=4,
            student_epochs=10,
            student_polish_epochs=1,
            ce_loss_weight=0.40,
            kd_loss_weight=0.60,
            distill_temperature=4.5,
            early_stopping_patience=4,
            early_stopping_min_delta=0.001,
        ),
    )

    distilled_qat_v2 = replace(
        distilled_student_v2,
        tag=f"{base.tag}_distilled_qat_v2",
        train=replace(
            distilled_student_v2.train,
            use_qat=True,
            qat_epochs=4,
        ),
    )

    kws20_probe = replace(
        base,
        tag="kws20_probe",
        vocabulary_preset="kws20",
        dataset=replace(
            base.dataset,
            batch_size=96,
            mixup_alpha=0.15,
        ),
        train=replace(
            base.train,
            teacher_epochs=22,
            student_epochs=26,
        ),
    )

    return {
        "baseline": baseline,
        "fast_student": fast_student,
        "distilled_student": distilled_student,
        "distilled_qat": distilled_qat,
        "distilled_student_v2": distilled_student_v2,
        "distilled_student_v3": distilled_student_v3,
        "distilled_student_v3_teacher_plus": distilled_student_v3_teacher_plus,
        "distilled_student_v3_bcresnet_teacher": distilled_student_v3_bcresnet_teacher,
        "distilled_student_v3_bcresnet_teacher_seq": distilled_student_v3_bcresnet_teacher_seq,
        "distilled_student_v3_kwt_teacher": distilled_student_v3_kwt_teacher,
        "distilled_student_v3_kwt_rich_teacher": distilled_student_v3_kwt_rich_teacher,
        "distilled_student_v3_kwt_rich_teacher_seq": distilled_student_v3_kwt_rich_teacher_seq,
        "assistant_bcresnet_from_kwt_rich_teacher": assistant_bcresnet_from_kwt_rich_teacher,
        "distilled_qat_v2": distilled_qat_v2,
        "kws20_probe": kws20_probe,
    }


def recipe_names(base: ExperimentConfig) -> list[str]:
    return list(build_recipe_book(base).keys())
