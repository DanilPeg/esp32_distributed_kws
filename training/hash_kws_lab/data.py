from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm.auto import tqdm

from .config import ExperimentConfig

try:
    import torchaudio
    from torchaudio.datasets import SPEECHCOMMANDS
except ImportError:  # pragma: no cover
    torchaudio = None
    SPEECHCOMMANDS = None

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover
    tf = None


def ensure_torchaudio_available() -> None:
    if torchaudio is None or SPEECHCOMMANDS is None:
        raise ImportError("torchaudio is required for hash KWS experiments")


def ensure_exact_microfrontend_available() -> None:
    if tf is None:
        raise ImportError(
            "tensorflow is required for exact microfrontend hash-KWS experiments"
        )
    if _microfrontend_module() is None:
        raise ImportError(
            "tensorflow microfrontend op is unavailable. "
            "Install a TensorFlow build that provides "
            "tensorflow.lite.experimental.microfrontend.python.ops.audio_microfrontend_op."
        )


@lru_cache(maxsize=1)
def _microfrontend_module():
    if tf is None:
        return None
    try:
        from tensorflow.lite.experimental.microfrontend.python.ops import (  # type: ignore
            audio_microfrontend_op,
        )
    except Exception:
        return None
    return audio_microfrontend_op


def _waveform_to_int16(waveform: torch.Tensor) -> Any:
    if tf is None:
        raise ImportError("tensorflow is required for exact microfrontend extraction")
    flattened = waveform.detach().cpu().to(torch.float32).view(-1).clamp_(-1.0, 1.0)
    tensor = tf.convert_to_tensor(flattened.numpy(), dtype=tf.float32)
    return tf.cast(tf.round(tensor * 32767.0), tf.int16)


def _fit_exact_feature_frames(features: Any, experiment: ExperimentConfig) -> Any:
    if tf is None:
        raise ImportError("tensorflow is required for exact microfrontend extraction")
    config = experiment.feature
    features = tf.cast(features, tf.float32)
    features = features[: config.frame_count]
    pad_frames = tf.maximum(0, config.frame_count - tf.shape(features)[0])
    features = tf.pad(features, [[0, pad_frames], [0, 0]])
    features.set_shape([config.frame_count, config.n_mels])
    return features


def _extract_exact_microfrontend_feature_map(
    waveform: torch.Tensor,
    experiment: ExperimentConfig,
) -> torch.Tensor:
    ensure_exact_microfrontend_available()
    if tf is None:
        raise ImportError("tensorflow is required for exact microfrontend extraction")

    config = experiment.feature
    module = _microfrontend_module()
    if module is None:
        raise ImportError("Exact microfrontend op is unavailable")

    audio = _waveform_to_int16(waveform)
    common_kwargs = {
        "sample_rate": config.sample_rate,
        "num_channels": config.n_mels,
        "lower_band_limit": config.lower_band_limit,
        "upper_band_limit": config.upper_band_limit,
        "smoothing_bits": config.smoothing_bits,
        "even_smoothing": config.even_smoothing,
        "odd_smoothing": config.odd_smoothing,
        "min_signal_remaining": config.min_signal_remaining,
        "enable_pcan": config.enable_pcan,
        "pcan_strength": config.pcan_strength,
        "pcan_offset": config.pcan_offset,
        "gain_bits": config.gain_bits,
        "enable_log": config.enable_log,
        "scale_shift": config.scale_shift,
        "left_context": 0,
        "right_context": 0,
        "frame_stride": 1,
        "zero_padding": False,
    }
    signature_variants = (
        {
            "window_size": config.window_ms,
            "window_step": config.hop_ms,
        },
        {
            "window_size_ms": config.window_ms,
            "window_step_ms": config.hop_ms,
        },
    )

    features = None
    for variant in signature_variants:
        try:
            features = module.audio_microfrontend(audio, **common_kwargs, **variant)
            break
        except TypeError:
            continue
    if features is None:
        raise RuntimeError("Failed to invoke the TensorFlow microfrontend op")

    divisor = max(float(config.exact_value_divisor), 1e-6)
    quantized = tf.round((tf.cast(features, tf.float32) * 256.0) / divisor) - 128.0
    quantized = tf.clip_by_value(quantized, -128.0, 127.0)
    quantized = _fit_exact_feature_frames(quantized, experiment)
    tensor = torch.from_numpy(quantized.numpy()).to(torch.float32)
    return tensor.transpose(0, 1).unsqueeze(0)


def _cache_signature(experiment: ExperimentConfig, subset: str, training: bool) -> str:
    dataset = experiment.dataset
    payload = {
        "version": 2,
        "subset": subset,
        "training": training,
        "vocabulary_preset": experiment.vocabulary_preset,
        "commands": experiment.commands,
        "all_labels": experiment.all_labels,
        "feature": asdict(experiment.feature),
        "dataset_mix_and_waveform": {
            "seed": dataset.seed,
            "unknown_fraction": dataset.unknown_fraction,
            "silence_fraction": dataset.silence_fraction,
            "silence_reference": dataset.silence_reference,
            "time_shift_ms": dataset.time_shift_ms,
            "gain_min": dataset.gain_min,
            "gain_max": dataset.gain_max,
            "noise_stddev": dataset.noise_stddev,
            "cache_dtype": dataset.cache_dtype,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


class AudioFeatureExtractor(torch.nn.Module):
    def __init__(self, experiment: ExperimentConfig) -> None:
        super().__init__()
        ensure_torchaudio_available()
        self.config = experiment.feature
        self.experiment = experiment
        self.mel = None
        if self.config.frontend_name == "log_mel":
            self.mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.sample_rate,
                n_fft=self.config.window_samples,
                win_length=self.config.window_samples,
                hop_length=self.config.hop_samples,
                n_mels=self.config.n_mels,
                center=self.config.center,
                power=2.0,
            )
        elif self.config.frontend_name == "exact_microfrontend":
            if self.config.require_exact_microfrontend:
                ensure_exact_microfrontend_available()
        else:
            raise ValueError(f"Unsupported frontend_name: {self.config.frontend_name}")

    def _normalize(self, spec: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_mode == "none":
            return spec
        if self.config.normalize_mode == "instance":
            return (spec - spec.mean()) / (spec.std() + 1e-5)
        if self.config.normalize_mode == "per_frequency":
            mean = spec.mean(dim=-1, keepdim=True)
            std = spec.std(dim=-1, keepdim=True)
            return (spec - mean) / (std + 1e-5)
        raise ValueError(f"Unsupported normalize_mode: {self.config.normalize_mode}")

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        waveform = waveform.to(torch.float32)
        if self.config.frontend_name == "log_mel":
            if self.mel is None:
                raise RuntimeError("log-mel frontend is not initialized")
            spec = self.mel(waveform)
            spec = torch.log(spec + self.config.log_offset)
        elif self.config.frontend_name == "exact_microfrontend":
            spec = _extract_exact_microfrontend_feature_map(waveform, self.experiment)
        else:
            raise ValueError(f"Unsupported frontend_name: {self.config.frontend_name}")
        return self._normalize(spec)


def _apply_spec_augment(feature_map: torch.Tensor, experiment: ExperimentConfig, rng: random.Random) -> torch.Tensor:
    config = experiment.feature
    if config.specaugment_prob <= 0.0 or rng.random() > config.specaugment_prob:
        return feature_map
    augmented = feature_map.clone()
    if config.time_mask_max > 0:
        width = rng.randint(0, config.time_mask_max)
        if width > 0 and augmented.shape[-1] > width:
            start = rng.randint(0, augmented.shape[-1] - width)
            augmented[:, :, start : start + width] = 0.0
    if config.freq_mask_max > 0:
        height = rng.randint(0, config.freq_mask_max)
        if height > 0 and augmented.shape[-2] > height:
            start = rng.randint(0, augmented.shape[-2] - height)
            augmented[:, start : start + height, :] = 0.0
    return augmented


class SpeechCommandsHashDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, root: Path, experiment: ExperimentConfig, subset: str, training: bool) -> None:
        ensure_torchaudio_available()
        self.experiment = experiment
        self.subset = subset
        self.training = training
        self.sample_rate = experiment.feature.sample_rate
        self.clip_samples = experiment.feature.clip_samples
        self.feature_extractor = AudioFeatureExtractor(experiment)
        self.base = SPEECHCOMMANDS(root=str(root), download=True, subset=subset)
        self.base_path = Path(getattr(self.base, "_path", root))

        walker = list(getattr(self.base, "_walker", []))
        if not walker:
            walker = list(range(len(self.base)))

        known_indices: list[int] = []
        unknown_indices: list[int] = []
        discovered_labels: Counter[str] = Counter()
        wanted_words = set(experiment.commands)

        for index, item in enumerate(walker):
            label = self._label_from_item(index, item)
            discovered_labels[label] += 1
            if label in wanted_words:
                known_indices.append(index)
            elif label != "_background_noise_":
                unknown_indices.append(index)

        if not known_indices:
            preview = dict(discovered_labels.most_common(20))
            raise ValueError(
                f"No wanted words found for subset={subset!r}. Discovered labels sample={preview}"
            )

        offset = {"training": 0, "validation": 1, "testing": 2}[subset]
        self.rng = random.Random(experiment.dataset.seed + offset)
        self.rng.shuffle(unknown_indices)

        unknown_target = 0
        if experiment.dataset.unknown_fraction > 0 and unknown_indices:
            unknown_target = max(1, int(round(len(known_indices) * experiment.dataset.unknown_fraction)))
            unknown_target = min(len(unknown_indices), unknown_target)
        selected_unknown = unknown_indices[:unknown_target]

        silence_reference_count = len(known_indices)
        if experiment.dataset.silence_reference == "mixed":
            silence_reference_count = len(known_indices) + len(selected_unknown)
        silence_target = 0
        if experiment.dataset.silence_fraction > 0:
            silence_target = max(1, int(round(silence_reference_count * experiment.dataset.silence_fraction)))

        self.entries: list[tuple[str, int | None, int]] = []
        for index in known_indices:
            label = self._label_from_item(index, walker[index] if isinstance(walker[index], str) else index)
            self.entries.append(("speech", index, experiment.label_to_index[label]))
        for index in selected_unknown:
            self.entries.append(("speech", index, experiment.label_to_index["unknown"]))
        for silence_index in range(silence_target):
            self.entries.append(("silence", silence_index, experiment.label_to_index["silence"]))

        self.rng.shuffle(self.entries)
        self.background_noises = self._load_background_noise()
        self.cache_enabled = bool(experiment.dataset.cache_features)
        self.cache_dtype = experiment.dataset.cache_dtype
        self.cached_features: torch.Tensor | None = None
        self.cached_labels: torch.Tensor | None = None
        self.cache_status: dict[str, Any] | None = None
        signature = _cache_signature(experiment, subset=subset, training=training)
        feature_cache_root = os.environ.get("HASH_KWS_FEATURE_CACHE_ROOT", "").strip()
        cache_root = Path(feature_cache_root) if feature_cache_root else root / "hash_feature_cache"
        self.cache_path = cache_root / f"{signature}.pt"

    def _label_from_item(self, index: int, item: Any) -> str:
        if isinstance(item, str):
            path = Path(item)
            if path.parent.name:
                return path.parent.name
        _, _, label, *_ = self.base[index]
        return str(label)

    def _load_background_noise(self) -> list[torch.Tensor]:
        noises: list[torch.Tensor] = []
        candidate_dirs = [
            self.base_path / "_background_noise_",
            self.base_path.parent / "_background_noise_",
        ]
        data_root = os.environ.get("SPEECHCOMMANDS_DATA_ROOT", "").strip()
        if data_root:
            candidate_dirs.append(Path(data_root) / "_background_noise_")

        noise_dir = None
        for candidate in candidate_dirs:
            if candidate.exists():
                noise_dir = candidate
                break
        if noise_dir is None:
            return noises

        for noise_path in sorted(noise_dir.glob("*.wav")):
            waveform, sample_rate = torchaudio.load(str(noise_path))
            waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
            noises.append(waveform)
        return noises

    def _prepare_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        if waveform.shape[1] < self.clip_samples:
            waveform = F.pad(waveform, (0, self.clip_samples - waveform.shape[1]))
        elif waveform.shape[1] > self.clip_samples:
            waveform = waveform[:, : self.clip_samples]
        return waveform

    def _augment_waveform(self, waveform: torch.Tensor, index: int) -> torch.Tensor:
        dataset_cfg = self.experiment.dataset
        if dataset_cfg.time_shift_ms > 0:
            shift_samples = (self.sample_rate * dataset_cfg.time_shift_ms) // 1000
            local_rng = random.Random(dataset_cfg.seed * 1_000_003 + index)
            shift = local_rng.randint(-shift_samples, shift_samples)
            if shift > 0:
                waveform = F.pad(waveform, (shift, 0))[:, : self.clip_samples]
            elif shift < 0:
                waveform = F.pad(waveform, (0, -shift))[:, -shift : -shift + self.clip_samples]
        if dataset_cfg.gain_min != 1.0 or dataset_cfg.gain_max != 1.0:
            local_rng = random.Random(dataset_cfg.seed * 1_000_033 + index)
            gain = local_rng.uniform(dataset_cfg.gain_min, dataset_cfg.gain_max)
            waveform = waveform * gain
        if dataset_cfg.noise_stddev > 0.0:
            waveform = waveform + torch.randn_like(waveform) * dataset_cfg.noise_stddev
        return waveform.clamp_(-1.0, 1.0)

    def _make_silence(self, index: int) -> torch.Tensor:
        if not self.background_noises:
            return torch.zeros(1, self.clip_samples)
        noise = self.background_noises[index % len(self.background_noises)]
        if noise.shape[1] <= self.clip_samples:
            waveform = F.pad(noise, (0, max(self.clip_samples - noise.shape[1], 0)))
            return waveform[:, : self.clip_samples] * 0.1
        start = (index * 9973) % (noise.shape[1] - self.clip_samples + 1)
        return noise[:, start : start + self.clip_samples] * 0.1

    def _can_store_int8_cache(self) -> bool:
        return (
            self.experiment.feature.frontend_name == "exact_microfrontend"
            and self.experiment.feature.normalize_mode == "none"
            and self.cache_dtype == "int8"
        )

    def _compute_item(self, index: int) -> tuple[torch.Tensor, int]:
        kind, base_index, label_id = self.entries[index]
        if kind == "speech":
            waveform, sample_rate, _, *_ = self.base[int(base_index)]
            waveform = self._prepare_waveform(waveform, int(sample_rate))
        else:
            waveform = self._make_silence(index)

        if self.training:
            waveform = self._augment_waveform(waveform, index=index)

        features = self.feature_extractor(waveform)
        if self.training:
            local_rng = random.Random(self.experiment.dataset.seed * 65_537 + index)
            features = _apply_spec_augment(features, self.experiment, local_rng)
        return features.squeeze(0), label_id

    def _cache_tensor(self, features: torch.Tensor) -> torch.Tensor:
        if self._can_store_int8_cache():
            return features.to(torch.int8)
        return features.to(torch.float32)

    def _restore_cached_tensor(self, features: torch.Tensor) -> torch.Tensor:
        return features.to(torch.float32)

    def materialize_feature_cache(self) -> dict[str, Any]:
        if not self.cache_enabled:
            self.cache_status = {
                "enabled": False,
                "status": "disabled",
                "path": str(self.cache_path),
            }
            return self.cache_status

        if self.cached_features is not None and self.cached_labels is not None:
            self.cache_status = {
                "enabled": True,
                "status": "memory",
                "path": str(self.cache_path),
                "items": int(self.cached_labels.numel()),
            }
            return self.cache_status

        if self.cache_path.exists():
            payload = torch.load(self.cache_path, map_location="cpu")
            self.cached_features = payload["features"].cpu()
            self.cached_labels = payload["labels"].cpu()
            self.cache_status = {
                "enabled": True,
                "status": "loaded",
                "path": str(self.cache_path),
                "items": int(self.cached_labels.numel()),
                "dtype": str(self.cached_features.dtype),
            }
            return self.cache_status

        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        cached_features: list[torch.Tensor] = []
        cached_labels: list[int] = []
        progress = tqdm(
            range(len(self.entries)),
            desc=f"cache {self.subset} {self.experiment.feature.frontend_name}",
            leave=False,
        )
        for index in progress:
            features, label_id = self._compute_item(index)
            cached_features.append(self._cache_tensor(features))
            cached_labels.append(int(label_id))

        self.cached_features = torch.stack(cached_features, dim=0).cpu()
        self.cached_labels = torch.tensor(cached_labels, dtype=torch.int64)
        torch.save(
            {
                "features": self.cached_features,
                "labels": self.cached_labels,
                "feature_shape": list(self.cached_features.shape),
                "frontend_name": self.experiment.feature.frontend_name,
                "subset": self.subset,
                "training": self.training,
            },
            self.cache_path,
        )
        self.cache_status = {
            "enabled": True,
            "status": "built",
            "path": str(self.cache_path),
            "items": int(self.cached_labels.numel()),
            "dtype": str(self.cached_features.dtype),
        }
        return self.cache_status

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if self.cached_features is not None and self.cached_labels is not None:
            features = self._restore_cached_tensor(self.cached_features[index])
            label_id = int(self.cached_labels[index].item())
            return features, label_id
        return self._compute_item(index)


def describe_dataset(dataset: Dataset[tuple[torch.Tensor, int]], labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    source = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(source))
    for index in indices:
        _, _, label_id = source.entries[index]
        counts[labels[int(label_id)]] += 1
    return counts


def maybe_limit(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError(f"Dataset limit must be positive, got {limit}")
    return Subset(dataset, list(range(limit)))


def ensure_non_empty(name: str, dataset: Dataset[Any]) -> None:
    if len(dataset) == 0:
        raise ValueError(f"{name} dataset is empty")


def _project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "code").exists() and (candidate / "notes").exists():
            return candidate
    return start.resolve()


def prepare_dataloaders(
    project_root: Path | None,
    experiment: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    ensure_torchaudio_available()
    if project_root is None:
        project_root = _project_root(Path.cwd())

    data_root_override = os.environ.get("SPEECHCOMMANDS_DATA_ROOT", "").strip()
    data_root = Path(data_root_override) if data_root_override else project_root / experiment.dataset.data_dir
    data_root.mkdir(parents=True, exist_ok=True)

    train_dataset = maybe_limit(
        SpeechCommandsHashDataset(data_root, experiment=experiment, subset="training", training=True),
        experiment.dataset.train_limit,
    )
    val_dataset = maybe_limit(
        SpeechCommandsHashDataset(data_root, experiment=experiment, subset="validation", training=False),
        experiment.dataset.val_limit,
    )
    test_dataset = maybe_limit(
        SpeechCommandsHashDataset(data_root, experiment=experiment, subset="testing", training=False),
        experiment.dataset.test_limit,
    )

    ensure_non_empty("train", train_dataset)
    ensure_non_empty("validation", val_dataset)
    ensure_non_empty("test", test_dataset)

    cache_summary: dict[str, Any] = {}
    for split_name, dataset in (
        ("train", train_dataset),
        ("validation", val_dataset),
        ("test", test_dataset),
    ):
        if isinstance(dataset, Subset):
            source = dataset.dataset
            cache_summary[split_name] = {
                "enabled": False,
                "status": "skipped_for_limited_subset",
                "items": len(dataset),
                "source_items": len(source),
            }
            print(
                f"Skipping full feature-cache materialization for limited {split_name}: "
                f"{len(dataset)} of {len(source)} items.",
                flush=True,
            )
            continue
        source = dataset.dataset if isinstance(dataset, Subset) else dataset
        if hasattr(source, "materialize_feature_cache"):
            print(
                f"Preparing feature cache for {split_name}: "
                f"{len(source)} source items -> {source.cache_path}",
                flush=True,
            )
            cache_summary[split_name] = source.materialize_feature_cache()

    loader_kwargs = {
        "batch_size": experiment.dataset.batch_size,
        "num_workers": experiment.dataset.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if experiment.dataset.num_workers > 0:
        loader_kwargs["persistent_workers"] = experiment.dataset.persistent_workers
        loader_kwargs["prefetch_factor"] = experiment.dataset.prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    feature_preview, _ = next(iter(train_loader))
    return {
        "project_root": project_root,
        "data_root": data_root,
        "loaders": {
            "train": train_loader,
            "validation": val_loader,
            "test": test_loader,
        },
        "summary": {
            "train": describe_dataset(train_dataset, experiment.all_labels),
            "validation": describe_dataset(val_dataset, experiment.all_labels),
            "test": describe_dataset(test_dataset, experiment.all_labels),
        },
        "cache_summary": cache_summary,
        "feature_preview_shape": list(feature_preview.shape),
    }
