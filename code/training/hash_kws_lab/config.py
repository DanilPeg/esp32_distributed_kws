from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VOCABULARY_PRESETS: dict[str, list[str]] = {
    "kws12": [
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
    ],
}


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    clip_samples: int = 16_000
    window_ms: int = 30
    hop_ms: int = 20
    n_mels: int = 40
    center: bool = False
    frontend_name: str = "log_mel"
    normalize_mode: str = "instance"
    log_offset: float = 1e-6
    require_exact_microfrontend: bool = False
    lower_band_limit: float = 125.0
    upper_band_limit: float = 7_500.0
    smoothing_bits: int = 10
    even_smoothing: float = 0.025
    odd_smoothing: float = 0.06
    min_signal_remaining: float = 0.05
    enable_pcan: bool = True
    pcan_strength: float = 0.95
    pcan_offset: float = 80.0
    gain_bits: int = 21
    enable_log: bool = True
    scale_shift: int = 6
    exact_value_divisor: float = 665.6
    specaugment_prob: float = 0.0
    time_mask_max: int = 0
    freq_mask_max: int = 0

    @property
    def window_samples(self) -> int:
        return (self.sample_rate * self.window_ms) // 1000

    @property
    def hop_samples(self) -> int:
        return (self.sample_rate * self.hop_ms) // 1000

    @property
    def frame_count(self) -> int:
        usable = max(0, self.clip_samples - self.window_samples)
        return 1 + (usable // self.hop_samples)

    @property
    def feature_shape(self) -> tuple[int, int]:
        return (self.n_mels, self.frame_count)


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: str = "data"
    batch_size: int = 256
    num_workers: int = 0
    persistent_workers: bool = True
    prefetch_factor: int = 2
    seed: int = 13
    unknown_fraction: float = 1.0
    silence_fraction: float = 0.12
    silence_reference: str = "known"
    train_limit: int | None = None
    val_limit: int | None = None
    test_limit: int | None = None
    time_shift_ms: int = 0
    gain_min: float = 1.0
    gain_max: float = 1.0
    noise_stddev: float = 0.0
    cache_features: bool = False
    cache_dtype: str = "int8"


@dataclass(frozen=True)
class ModelConfig:
    teacher_name: str = ""
    student_name: str = "hash_dscnn_deeper"
    channels: int = 64
    teacher_channels: int = 96
    num_blocks: int = 4
    teacher_num_blocks: int = 5
    codebook_size: int = 500
    stem_codebook_size: int = 500
    depthwise_codebook_size: int = 500
    pointwise_codebook_size: int = 500
    linear_codebook_size: int = 500
    depthwise_codebook_sizes: tuple[int, ...] = ()
    pointwise_codebook_sizes: tuple[int, ...] = ()
    signed_hash: bool = False
    hash_only_pointwise: bool = False
    use_residual: bool = False
    teacher_dropout: float = 0.10
    student_dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 13
    teacher_epochs: int = 18
    student_pretrain_epochs: int = 0
    student_epochs: int = 20
    student_polish_epochs: int = 0
    teacher_lr: float = 1e-3
    student_lr: float = 1e-3
    polish_lr: float = 2.5e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    teacher_scheduler_name: str = "none"
    student_scheduler_name: str = "none"
    label_smoothing: float = 0.0
    teacher_label_smoothing: float = 0.0
    kd_alpha: float = 0.0
    kd_alpha_schedule: str = "constant"
    kd_alpha_final: float | None = None
    kd_temperature: float = 4.0
    kd_temperature_schedule: str = "constant"
    kd_temperature_final: float | None = None
    cache_teacher_logits: bool = False
    teacher_logits_cache_dir: str = ""
    teacher_logits_cache_dtype: str = "float16"
    teacher_logits_cache_rebuild: bool = False
    polish_label_smoothing: float | None = None
    grad_clip_norm: float = 0.0
    teacher_early_stopping_patience: int = 0
    student_early_stopping_patience: int = 0
    use_amp: bool = True
    use_ema: bool = False
    ema_decay: float = 0.999
    eval_with_ema: bool = True
    top_k: int = 3

    @property
    def uses_distillation(self) -> bool:
        return self.kd_alpha > 0.0


@dataclass(frozen=True)
class ExportConfig:
    artifacts_dir: str = "code/training/hash_artifacts"
    model_stem: str = "hash_kws_student"


@dataclass(frozen=True)
class ExperimentConfig:
    tag: str
    vocabulary_preset: str
    teacher_reuse_tag: str = ""
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    @property
    def commands(self) -> list[str]:
        if self.vocabulary_preset not in VOCABULARY_PRESETS:
            raise KeyError(f"Unknown vocabulary preset: {self.vocabulary_preset}")
        return list(VOCABULARY_PRESETS[self.vocabulary_preset])

    @property
    def all_labels(self) -> list[str]:
        return [*self.commands, "unknown", "silence"]

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.all_labels)}

    @property
    def num_labels(self) -> int:
        return len(self.all_labels)

    @property
    def feature_shape(self) -> tuple[int, int]:
        return self.feature.feature_shape

    @property
    def model_input_shape(self) -> tuple[int, int, int]:
        mel_bins, frames = self.feature_shape
        return (1, mel_bins, frames)

    @property
    def uses_teacher(self) -> bool:
        return bool(self.model.teacher_name)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["commands"] = self.commands
        payload["all_labels"] = self.all_labels
        payload["num_labels"] = self.num_labels
        payload["feature_shape"] = list(self.feature_shape)
        payload["model_input_shape"] = list(self.model_input_shape)
        payload["uses_teacher"] = self.uses_teacher
        payload["uses_distillation"] = self.train.uses_distillation
        return payload


def make_experiment(
    tag: str = "hash_kws12_iterlab_v1",
    vocabulary_preset: str = "kws12",
) -> ExperimentConfig:
    return ExperimentConfig(
        tag=tag,
        vocabulary_preset=vocabulary_preset,
    )


def experiment_from_dict(payload: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        tag=str(payload["tag"]),
        vocabulary_preset=str(payload["vocabulary_preset"]),
        teacher_reuse_tag=str(payload.get("teacher_reuse_tag", "")),
        feature=FeatureConfig(**payload.get("feature", {})),
        dataset=DatasetConfig(**payload.get("dataset", {})),
        model=ModelConfig(**payload.get("model", {})),
        train=TrainConfig(**payload.get("train", {})),
        export=ExportConfig(**payload.get("export", {})),
    )
