from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


VOCABULARY_PRESETS: dict[str, list[str]] = {
    "kws4": ["yes", "no"],
    "kws12": ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"],
    "kws20": [
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
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
    ],
}


@dataclass(frozen=True)
class FrontendConfig:
    sample_rate: int = 16_000
    clip_ms: int = 1_000
    frame_length_ms: int = 30
    frame_step_ms: int = 20
    fft_length: int = 512
    num_channels: int = 40
    lower_band_limit: float = 125.0
    upper_band_limit: float = 7_500.0
    enable_exact_microfrontend: bool = True
    use_pcen_fallback: bool = True
    smoothing_bits: int = 10
    even_smoothing: float = 0.025
    odd_smoothing: float = 0.06
    min_signal_remaining: float = 0.05
    pcan_strength: float = 0.95
    pcan_offset: float = 80.0
    gain_bits: int = 21
    scale_shift: int = 6
    fallback_feature_clip: float = 8.0
    pcen_alpha: float = 0.98
    pcen_delta: float = 2.0
    pcen_root: float = 0.5
    pcen_floor: float = 1e-6
    pcen_smoothing_coef: float = 0.04

    @property
    def desired_samples(self) -> int:
        return (self.sample_rate * self.clip_ms) // 1000

    @property
    def frame_length_samples(self) -> int:
        return (self.sample_rate * self.frame_length_ms) // 1000

    @property
    def frame_step_samples(self) -> int:
        return (self.sample_rate * self.frame_step_ms) // 1000

    @property
    def frame_count(self) -> int:
        usable = self.desired_samples - self.frame_length_samples
        return 1 + max(0, usable // self.frame_step_samples)

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return (self.frame_count, self.num_channels, 1)


@dataclass(frozen=True)
class DatasetConfig:
    data_dir: str = "data"
    dataset_root_override: str = ""
    archive_path_override: str = ""
    batch_size: int = 96
    shuffle_buffer: int = 6_000
    seed: int = 42
    unknown_ratio: float = 1.0
    silence_ratio: float = 0.12
    train_fraction: float = 1.0
    eval_fraction: float = 1.0
    cache_validation: bool = True
    time_shift_ms: int = 100
    gain_min: float = 0.8
    gain_max: float = 1.2
    noise_stddev: float = 0.004
    mixup_alpha: float = 0.1
    specaugment_prob: float = 0.5
    time_mask_max: int = 4
    freq_mask_max: int = 3


@dataclass(frozen=True)
class ModelConfig:
    teacher_name: str = "teacher_factorized_dscnn"
    student_name: str = "student_factorized_dscnn_v2"
    teacher_width: float = 1.0
    student_width: float = 0.875
    teacher_dropout: float = 0.20
    student_dropout: float = 0.08
    teacher_label_smoothing: float = 0.01
    label_smoothing: float = 0.02


@dataclass(frozen=True)
class TrainConfig:
    teacher_epochs: int = 14
    student_pretrain_epochs: int = 0
    student_epochs: int = 18
    qat_epochs: int = 4
    student_polish_epochs: int = 0
    teacher_warmup_epochs: int = 0
    teacher_lr: float = 1e-3
    student_lr: float = 9e-4
    qat_lr: float = 1e-4
    ce_loss_weight: float = 0.2
    kd_loss_weight: float = 0.8
    distill_temperature: float = 5.0
    distillation_mode: str = "logits"
    hint_warmup_epochs: int = 0
    hint_warmup_ce_weight: float = 0.4
    hint_warmup_hint_weight: float = 0.6
    hint_loss_weight: float = 0.0
    attention_loss_weight: float = 0.0
    similarity_loss_weight: float = 0.0
    hint_layer_pair: tuple[str, str] = ("stage3_block1_out", "stage3_block2_out")
    attention_layer_pairs: tuple[tuple[str, str], ...] = ()
    similarity_layer_pair: tuple[str, str] = ("", "")
    use_qat: bool = False
    teacher_early_stopping_patience: int = 4
    teacher_early_stopping_min_delta: float = 0.0
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 0.0
    top_k: int = 3
    optimizer_name: str = "adamw"
    weight_decay: float = 1.5e-4
    clipnorm: float = 1.0

    @property
    def uses_distillation(self) -> bool:
        return any(
            weight > 0.0
            for weight in (
                self.kd_loss_weight,
                self.hint_loss_weight,
                self.attention_loss_weight,
                self.similarity_loss_weight,
            )
        )


@dataclass(frozen=True)
class ExportConfig:
    artifacts_dir: str = "code/training/artifacts"
    model_stem: str = "kws_esp32_advanced"
    representative_batches: int = 200
    export_c_array: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    tag: str
    vocabulary_preset: str
    teacher_reuse_tag: str = ""
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    teacher_frontend: FrontendConfig | None = None
    student_frontend: FrontendConfig | None = None
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
        return ["silence", "unknown", *self.commands]

    @property
    def label_to_index(self) -> dict[str, int]:
        return {label: index for index, label in enumerate(self.all_labels)}

    @property
    def num_labels(self) -> int:
        return len(self.all_labels)

    @property
    def teacher_frontend_config(self) -> FrontendConfig:
        return self.teacher_frontend or self.frontend

    @property
    def student_frontend_config(self) -> FrontendConfig:
        return self.student_frontend or self.frontend

    @property
    def teacher_feature_shape(self) -> tuple[int, int, int]:
        return self.teacher_frontend_config.feature_shape

    @property
    def student_feature_shape(self) -> tuple[int, int, int]:
        return self.student_frontend_config.feature_shape

    @property
    def feature_shape(self) -> tuple[int, int, int]:
        return self.student_feature_shape

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["commands"] = self.commands
        data["all_labels"] = self.all_labels
        data["feature_shape"] = list(self.feature_shape)
        data["teacher_feature_shape"] = list(self.teacher_feature_shape)
        data["student_feature_shape"] = list(self.student_feature_shape)
        data["num_labels"] = self.num_labels
        return data


def make_experiment(
    tag: str = "kws12_iterlab_v1",
    vocabulary_preset: str = "kws12",
) -> ExperimentConfig:
    return ExperimentConfig(
        tag=tag,
        vocabulary_preset=vocabulary_preset,
    )
