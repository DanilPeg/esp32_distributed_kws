from __future__ import annotations

import os
import random
import tarfile
from pathlib import Path
from typing import Any

import tensorflow as tf

from .config import ExperimentConfig
from .features import (
    apply_spec_augment,
    describe_frontend_runtime,
    extract_feature_map,
    pad_or_trim_waveform,
)


AUTOTUNE = tf.data.AUTOTUNE
RECORD_AUDIO = 0
RECORD_SILENCE = 1


def _looks_like_speech_commands_root(path: Path) -> bool:
    required = [
        path / "validation_list.txt",
        path / "testing_list.txt",
    ]
    command_dirs = [
        path / "yes",
        path / "no",
    ]
    return all(item.exists() for item in required) and any(item.exists() for item in command_dirs)


def _project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "code").exists() and (candidate / "notes").exists():
            return candidate
    return start.resolve()


def download_speech_commands(project_root: Path, experiment: ExperimentConfig) -> Path:
    env_dataset_root = os.environ.get("SPEECH_COMMANDS_DATASET_ROOT", "").strip()
    env_archive_path = os.environ.get("SPEECH_COMMANDS_ARCHIVE_PATH", "").strip()
    dataset_root_override = experiment.dataset.dataset_root_override.strip()
    archive_path_override = experiment.dataset.archive_path_override.strip()

    explicit_dataset_roots = [
        Path(path)
        for path in [env_dataset_root, dataset_root_override]
        if path
    ]
    for candidate in explicit_dataset_roots:
        if candidate.exists():
            return candidate.resolve()

    data_dir = project_root / experiment.dataset.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = data_dir / "speech_commands_v0.02"
    if extracted_dir.exists():
        return extracted_dir.resolve()
    if _looks_like_speech_commands_root(data_dir):
        extracted_dir.mkdir(parents=True, exist_ok=True)
        for item in list(data_dir.iterdir()):
            if item == extracted_dir:
                continue
            item.rename(extracted_dir / item.name)
        return extracted_dir.resolve()

    explicit_archives = [
        Path(path)
        for path in [env_archive_path, archive_path_override]
        if path
    ]
    for archive_path in explicit_archives:
        if archive_path.exists():
            extracted_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(path=extracted_dir)
            if extracted_dir.exists():
                return extracted_dir.resolve()

    try:
        archive_path = Path(
            tf.keras.utils.get_file(
                fname="speech_commands_v0.02.tar.gz",
                origin="https://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                extract=True,
                cache_dir=str(data_dir),
                cache_subdir="",
            )
        )
        extracted_candidate = archive_path.parent / "speech_commands_v0.02"
        if extracted_candidate.exists():
            return extracted_candidate.resolve()
    except Exception as exc:
        raise RuntimeError(
            "Could not prepare Speech Commands dataset. "
            "Use one of these options: "
            "1) set SPEECH_COMMANDS_DATASET_ROOT to an already extracted speech_commands_v0.02 folder; "
            "2) set SPEECH_COMMANDS_ARCHIVE_PATH to a local speech_commands_v0.02.tar.gz file; "
            "3) place speech_commands_v0.02 under <project>/data/ . "
            f"Original download error: {exc}"
        ) from exc

    raise RuntimeError(
        "Speech Commands dataset was not found after download/extract. "
        "Set SPEECH_COMMANDS_DATASET_ROOT or SPEECH_COMMANDS_ARCHIVE_PATH explicitly."
    )


def _split_name_for(
    relative_path: str,
    validation_list: set[str],
    testing_list: set[str],
) -> str:
    if relative_path in validation_list:
        return "validation"
    if relative_path in testing_list:
        return "test"
    return "train"


def _make_record(path: str, label_id: int, kind: int) -> dict[str, Any]:
    return {"path": path, "label_id": label_id, "kind": kind}


def _maybe_take_fraction(
    records: list[dict[str, Any]],
    fraction: float,
    seed: int,
) -> list[dict[str, Any]]:
    if fraction >= 1.0:
        return list(records)
    fraction = max(fraction, 0.01)
    rng = random.Random(seed)
    copied = list(records)
    rng.shuffle(copied)
    keep = max(1, int(len(copied) * fraction))
    return copied[:keep]


def build_record_index(
    project_root: Path,
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    dataset_root = download_speech_commands(project_root, experiment)
    validation_list = set((dataset_root / "validation_list.txt").read_text().splitlines())
    testing_list = set((dataset_root / "testing_list.txt").read_text().splitlines())

    per_split = {
        "train": {"targets": [], "unknown": []},
        "validation": {"targets": [], "unknown": []},
        "test": {"targets": [], "unknown": []},
    }

    target_commands = set(experiment.commands)
    labels = experiment.label_to_index
    for wav_path in dataset_root.glob("*/*.wav"):
        if wav_path.parent.name == "_background_noise_":
            continue
        relative = wav_path.relative_to(dataset_root).as_posix()
        split_name = _split_name_for(relative, validation_list, testing_list)
        word = wav_path.parent.name
        if word in target_commands:
            per_split[split_name]["targets"].append(
                _make_record(str(wav_path), labels[word], RECORD_AUDIO)
            )
        else:
            per_split[split_name]["unknown"].append(
                _make_record(str(wav_path), labels["unknown"], RECORD_AUDIO)
            )

    seed = experiment.dataset.seed
    merged_records: dict[str, list[dict[str, Any]]] = {}
    split_summary: dict[str, dict[str, int]] = {}

    for offset, split_name in enumerate(("train", "validation", "test")):
        rng = random.Random(seed + offset)
        targets = list(per_split[split_name]["targets"])
        unknown = list(per_split[split_name]["unknown"])
        rng.shuffle(targets)
        rng.shuffle(unknown)

        unknown_cap = int(len(targets) * experiment.dataset.unknown_ratio)
        mixed = targets + unknown[:unknown_cap]
        silence_count = int(len(mixed) * experiment.dataset.silence_ratio)
        mixed.extend(
            _make_record("", labels["silence"], RECORD_SILENCE) for _ in range(silence_count)
        )
        rng.shuffle(mixed)

        fraction = (
            experiment.dataset.train_fraction
            if split_name == "train"
            else experiment.dataset.eval_fraction
        )
        mixed = _maybe_take_fraction(mixed, fraction=fraction, seed=seed + 100 + offset)
        merged_records[split_name] = mixed
        split_summary[split_name] = describe_records(mixed, experiment.all_labels)

    return {
        "dataset_root": dataset_root,
        "records": merged_records,
        "summary": split_summary,
    }


def describe_records(records: list[dict[str, Any]], labels: list[str]) -> dict[str, int]:
    counts = {label: 0 for label in labels}
    for record in records:
        label_id = int(record["label_id"])
        counts[labels[label_id]] += 1
    return counts


def _decode_wav(path: tf.Tensor) -> tf.Tensor:
    audio_binary = tf.io.read_file(path)
    waveform, _ = tf.audio.decode_wav(audio_binary, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    return waveform


def _random_time_shift(waveform: tf.Tensor, shift_samples: int) -> tf.Tensor:
    if shift_samples <= 0:
        return waveform
    padded = tf.pad(waveform, [[shift_samples, shift_samples]])
    start = tf.random.uniform([], 0, 2 * shift_samples + 1, dtype=tf.int32)
    shifted = padded[start : start + tf.shape(waveform)[0]]
    shifted.set_shape(waveform.shape)
    return shifted


def _augment_waveform(waveform: tf.Tensor, experiment: ExperimentConfig) -> tf.Tensor:
    waveform = tf.cast(waveform, tf.float32)
    shift_samples = (experiment.frontend.sample_rate * experiment.dataset.time_shift_ms) // 1000
    waveform = _random_time_shift(waveform, shift_samples=shift_samples)
    gain = tf.random.uniform(
        [],
        experiment.dataset.gain_min,
        experiment.dataset.gain_max,
        dtype=tf.float32,
    )
    waveform = waveform * gain
    if experiment.dataset.noise_stddev > 0.0:
        waveform = waveform + tf.random.normal(
            tf.shape(waveform),
            stddev=experiment.dataset.noise_stddev,
            dtype=tf.float32,
        )
    return tf.clip_by_value(waveform, -1.0, 1.0)


def _prepare_example(
    sample: dict[str, tf.Tensor],
    experiment: ExperimentConfig,
    training: bool,
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    kind = sample["kind"]

    def make_silence() -> tf.Tensor:
        return tf.zeros([experiment.frontend.desired_samples], dtype=tf.float32)

    def decode_audio() -> tf.Tensor:
        return _decode_wav(sample["path"])

    waveform = tf.cond(
        tf.equal(kind, RECORD_SILENCE),
        make_silence,
        decode_audio,
    )
    waveform = pad_or_trim_waveform(waveform, experiment.student_frontend_config)
    if training:
        waveform = _augment_waveform(waveform, experiment)

    student_feature_map = extract_feature_map(waveform, experiment.student_frontend_config)
    teacher_feature_map = extract_feature_map(waveform, experiment.teacher_frontend_config)

    if training:
        student_feature_map = apply_spec_augment(
            student_feature_map,
            specaugment_prob=experiment.dataset.specaugment_prob,
            time_mask_max=experiment.dataset.time_mask_max,
            freq_mask_max=experiment.dataset.freq_mask_max,
        )
        teacher_feature_map = apply_spec_augment(
            teacher_feature_map,
            specaugment_prob=experiment.dataset.specaugment_prob,
            time_mask_max=experiment.dataset.time_mask_max,
            freq_mask_max=experiment.dataset.freq_mask_max,
        )

    label = tf.one_hot(sample["label_id"], depth=experiment.num_labels, dtype=tf.float32)
    return {
        "student": student_feature_map,
        "teacher": teacher_feature_map,
    }, label


def _mixup_batch(
    features: dict[str, tf.Tensor],
    labels: tf.Tensor,
    alpha: float,
) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
    if alpha <= 0.0:
        return features, labels

    student_features = features["student"]
    teacher_features = features["teacher"]
    batch_size = tf.shape(student_features)[0]
    gamma_a = tf.random.gamma([batch_size], alpha=alpha, dtype=tf.float32)
    gamma_b = tf.random.gamma([batch_size], alpha=alpha, dtype=tf.float32)
    lam = gamma_a / (gamma_a + gamma_b)
    permutation = tf.random.shuffle(tf.range(batch_size))

    mixed_student_features = (
        student_features * lam[:, tf.newaxis, tf.newaxis, tf.newaxis]
        + tf.gather(student_features, permutation)
        * (1.0 - lam)[:, tf.newaxis, tf.newaxis, tf.newaxis]
    )
    mixed_teacher_features = (
        teacher_features * lam[:, tf.newaxis, tf.newaxis, tf.newaxis]
        + tf.gather(teacher_features, permutation)
        * (1.0 - lam)[:, tf.newaxis, tf.newaxis, tf.newaxis]
    )
    mixed_labels = (
        labels * lam[:, tf.newaxis]
        + tf.gather(labels, permutation) * (1.0 - lam)[:, tf.newaxis]
    )
    return {
        "student": mixed_student_features,
        "teacher": mixed_teacher_features,
    }, mixed_labels


def _records_to_tensor_slices(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        "path": [record["path"] for record in records],
        "label_id": [int(record["label_id"]) for record in records],
        "kind": [int(record["kind"]) for record in records],
    }


def make_dataset(
    records: list[dict[str, Any]],
    experiment: ExperimentConfig,
    training: bool,
) -> tf.data.Dataset:
    def map_example(sample: dict[str, tf.Tensor]) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        return _prepare_example(sample, experiment=experiment, training=training)

    def map_mixup(features: dict[str, tf.Tensor], labels: tf.Tensor) -> tuple[dict[str, tf.Tensor], tf.Tensor]:
        return _mixup_batch(
            features,
            labels,
            alpha=experiment.dataset.mixup_alpha,
        )

    slices = _records_to_tensor_slices(records)
    dataset = tf.data.Dataset.from_tensor_slices(slices)
    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(records), experiment.dataset.shuffle_buffer),
            seed=experiment.dataset.seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(map_example, num_parallel_calls=AUTOTUNE)
    if not training and experiment.dataset.cache_validation:
        dataset = dataset.cache()
    dataset = dataset.batch(experiment.dataset.batch_size)
    if training and experiment.dataset.mixup_alpha > 0.0:
        dataset = dataset.map(map_mixup, num_parallel_calls=AUTOTUNE)
    return dataset.prefetch(AUTOTUNE)


def _select_feature_view(
    dataset: tf.data.Dataset,
    key: str,
) -> tf.data.Dataset:
    def select_view(features: dict[str, tf.Tensor], labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return features[key], labels

    return dataset.map(select_view, num_parallel_calls=AUTOTUNE)


def prepare_datasets(
    project_root: Path | None,
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    if project_root is None:
        project_root = _project_root(Path.cwd())
    index = build_record_index(project_root, experiment)
    dual_datasets = {
        split_name: make_dataset(records, experiment=experiment, training=(split_name == "train"))
        for split_name, records in index["records"].items()
    }
    datasets: dict[str, tf.data.Dataset] = {}
    for split_name, dataset in dual_datasets.items():
        datasets[f"distillation_{split_name}"] = dataset
        datasets[f"teacher_{split_name}"] = _select_feature_view(dataset, "teacher")
        datasets[f"student_{split_name}"] = _select_feature_view(dataset, "student")
    train_size = len(index["records"]["train"])
    steps_per_epoch = max(1, train_size // experiment.dataset.batch_size)
    return {
        "project_root": project_root,
        "dataset_root": index["dataset_root"],
        "records": index["records"],
        "summary": index["summary"],
        "frontend": {
            "student": describe_frontend_runtime(experiment.student_frontend_config),
            "teacher": describe_frontend_runtime(experiment.teacher_frontend_config),
            "teacher_student_frontend_match": experiment.teacher_frontend_config == experiment.student_frontend_config,
        },
        "datasets": datasets,
        "steps_per_epoch": steps_per_epoch,
    }
