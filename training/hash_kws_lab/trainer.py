from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .config import ExperimentConfig


def _autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def _build_grad_scaler(device: torch.device, enabled: bool):
    scaler_enabled = bool(enabled and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler(device.type, enabled=scaler_enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=scaler_enabled)
    return torch.cuda.amp.GradScaler(enabled=scaler_enabled)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.parameter_keys = {key for key, _ in model.named_parameters()}
        self.shadow = {key: value.detach().clone() for key, value in model.state_dict().items()}
        self.backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, value in model.state_dict().items():
            if key not in self.parameter_keys:
                self.shadow[key] = value.detach().clone()
                continue
            if not torch.is_floating_point(value):
                self.shadow[key] = value.detach().clone()
                continue
            self.shadow[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model: nn.Module) -> None:
        self.backup = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=True)
            self.backup = None


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, optimizer_name: str) -> torch.optim.Optimizer:
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if optimizer_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise KeyError(f"Unknown optimizer_name: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_name: str,
    epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    raise KeyError(f"Unknown scheduler_name: {scheduler_name}")


def evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    device: torch.device,
    label_smoothing: float = 0.0,
    top_k: int = 3,
    use_amp: bool = True,
    desc: str = "eval",
) -> dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    total_loss = 0.0
    total_correct = 0
    total_topk = 0
    total_items = 0

    progress = tqdm(loader, desc=desc, leave=False)
    for batch in progress:
        features, targets, _ = _unpack_batch(batch)
        features = features.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _autocast_context(device, enabled=use_amp):
            logits = model(features)
            loss = criterion(logits, targets)

        batch_size = int(features.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == targets).sum().item())
        topk = logits.topk(min(top_k, logits.shape[1]), dim=1).indices
        total_topk += int((topk == targets.unsqueeze(1)).any(dim=1).sum().item())
        total_items += batch_size
        progress.set_postfix(
            loss=f"{total_loss / max(total_items, 1):.4f}",
            acc=f"{total_correct / max(total_items, 1):.4f}",
        )

    return {
        "loss": total_loss / max(total_items, 1),
        "accuracy": total_correct / max(total_items, 1),
        f"top{top_k}_accuracy": total_topk / max(total_items, 1),
    }


def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    return F.cross_entropy(logits, targets, label_smoothing=label_smoothing)


def _kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


def _scheduled_value(
    start: float,
    final: float | None,
    schedule_name: str,
    epoch: int,
    epochs: int,
) -> float:
    name = schedule_name.strip().lower()
    if final is None or name in ("", "none", "constant"):
        return float(start)

    progress = 0.0 if epochs <= 1 else float(epoch - 1) / float(epochs - 1)
    if name in ("linear", "linear_decay", "linear_ramp"):
        factor = progress
    elif name in ("cosine", "cosine_decay"):
        factor = 0.5 - (0.5 * math.cos(math.pi * progress))
    else:
        raise KeyError(f"Unknown schedule_name: {schedule_name}")
    return float(start + ((final - start) * factor))


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if not isinstance(batch, (tuple, list)):
        raise TypeError(f"Expected dataloader batch tuple/list, got {type(batch)!r}")
    if len(batch) == 2:
        features, targets = batch
        return features, targets, None
    if len(batch) == 3:
        features, targets, teacher_logits = batch
        return features, targets, teacher_logits
    raise ValueError(f"Expected 2 or 3 batch items, got {len(batch)}")


def _state_dict_fingerprint(model: nn.Module) -> str:
    digest = hashlib.sha1()
    for key, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()[:16]


def _teacher_logits_cache_path(
    teacher: nn.Module,
    loader: DataLoader[Any],
    experiment: ExperimentConfig,
    split_name: str,
) -> tuple[Path, dict[str, Any]]:
    configured = experiment.train.teacher_logits_cache_dir.strip()
    env_root = os.environ.get("HASH_KWS_TEACHER_LOGITS_CACHE_ROOT", "").strip()
    root = Path(configured or env_root or (Path(experiment.export.artifacts_dir) / "teacher_logits"))
    metadata = {
        "version": 1,
        "split_name": split_name,
        "items": len(loader.dataset),
        "labels": experiment.all_labels,
        "feature": asdict(experiment.feature),
        "dataset": {
            "seed": experiment.dataset.seed,
            "unknown_fraction": experiment.dataset.unknown_fraction,
            "silence_fraction": experiment.dataset.silence_fraction,
            "silence_reference": experiment.dataset.silence_reference,
            "time_shift_ms": experiment.dataset.time_shift_ms,
            "gain_min": experiment.dataset.gain_min,
            "gain_max": experiment.dataset.gain_max,
            "noise_stddev": experiment.dataset.noise_stddev,
            "train_limit": experiment.dataset.train_limit,
            "val_limit": experiment.dataset.val_limit,
            "test_limit": experiment.dataset.test_limit,
        },
        "teacher": {
            "name": experiment.model.teacher_name,
            "channels": experiment.model.teacher_channels,
            "num_blocks": experiment.model.teacher_num_blocks,
            "dropout": experiment.model.teacher_dropout,
            "state_fingerprint": _state_dict_fingerprint(teacher),
        },
    }
    signature = hashlib.sha1(
        json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return root / f"{signature}_{split_name}_teacher_logits.pt", metadata


class TeacherLogitsDataset(Dataset[tuple[torch.Tensor, int, torch.Tensor]]):
    def __init__(self, source: Dataset[Any], logits: torch.Tensor) -> None:
        if len(source) != int(logits.shape[0]):
            raise ValueError(f"Dataset/logit length mismatch: {len(source)} vs {int(logits.shape[0])}")
        self.source = source
        self.logits = logits.cpu()

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        features, target = self.source[index]
        return features, int(target), self.logits[index]


def _loader_like(loader: DataLoader[Any], dataset: Dataset[Any], shuffle: bool) -> DataLoader[Any]:
    kwargs: dict[str, Any] = {
        "batch_size": loader.batch_size or 1,
        "shuffle": shuffle,
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "collate_fn": loader.collate_fn,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = False
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(dataset, **kwargs)


def materialize_teacher_logits(
    teacher: nn.Module,
    loader: DataLoader[Any],
    experiment: ExperimentConfig,
    device: torch.device,
    split_name: str = "train",
) -> dict[str, Any]:
    """Precompute teacher logits and return a train loader that serves them.

    The cache key includes the exact feature/dataset setup and a fingerprint of
    the teacher state, so multiple student recipes can reuse logits only when
    the supervising model is actually identical.
    """

    cache_path, metadata = _teacher_logits_cache_path(teacher, loader, experiment, split_name)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dtype_name = experiment.train.teacher_logits_cache_dtype.strip().lower()
    stored_dtype = torch.float16 if dtype_name == "float16" else torch.float32

    cache_status = "built"
    logits: torch.Tensor | None = None
    if cache_path.exists() and not experiment.train.teacher_logits_cache_rebuild:
        payload = torch.load(cache_path, map_location="cpu")
        cached_metadata = payload.get("metadata", {})
        cached_logits = payload.get("logits")
        if (
            isinstance(cached_logits, torch.Tensor)
            and int(cached_logits.shape[0]) == len(loader.dataset)
            and cached_metadata == metadata
        ):
            logits = cached_logits.cpu()
            cache_status = "loaded"

    if logits is None:
        was_training = teacher.training
        teacher.eval()
        teacher.to(device)
        sequential_loader = _loader_like(loader, loader.dataset, shuffle=False)
        chunks: list[torch.Tensor] = []
        progress = tqdm(sequential_loader, desc=f"cache {split_name} teacher logits", leave=False)
        with torch.no_grad():
            for batch in progress:
                features, _, _ = _unpack_batch(batch)
                features = features.to(device, non_blocking=True)
                with _autocast_context(device, enabled=experiment.train.use_amp):
                    batch_logits = teacher(features)
                chunks.append(batch_logits.detach().to("cpu", dtype=stored_dtype))
        if was_training:
            teacher.train()
        logits = torch.cat(chunks, dim=0)
        torch.save(
            {
                "logits": logits,
                "metadata": metadata,
                "dtype": str(logits.dtype),
            },
            cache_path,
        )

    cached_dataset = TeacherLogitsDataset(loader.dataset, logits)
    cached_loader = _loader_like(loader, cached_dataset, shuffle=True)
    return {
        "loader": cached_loader,
        "path": str(cache_path),
        "status": cache_status,
        "items": int(logits.shape[0]),
        "dtype": str(logits.dtype),
        "metadata": metadata,
    }


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    payload = torch.load(path, map_location=device)
    state_dict: dict[str, torch.Tensor]
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and isinstance(payload.get("best_state"), dict):
        state_dict = payload["best_state"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(path),
        "state_keys": len(state_dict),
        "payload_keys": sorted(payload.keys()) if isinstance(payload, dict) else [],
    }


def _train_stage(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    device: torch.device,
    epochs: int,
    stage_name: str,
    lr: float,
    optimizer_name: str,
    scheduler_name: str,
    weight_decay: float,
    label_smoothing: float,
    grad_clip_norm: float,
    use_amp: bool,
    use_ema: bool,
    ema_decay: float,
    eval_with_ema: bool,
    top_k: int,
    early_stopping_patience: int,
    teacher: nn.Module | None = None,
    kd_alpha: float = 0.0,
    kd_alpha_schedule: str = "constant",
    kd_alpha_final: float | None = None,
    kd_temperature: float = 4.0,
    kd_temperature_schedule: str = "constant",
    kd_temperature_final: float | None = None,
) -> dict[str, Any]:
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay, optimizer_name=optimizer_name)
    scheduler = build_scheduler(optimizer, scheduler_name=scheduler_name, epochs=epochs)
    scaler = _build_grad_scaler(device, enabled=use_amp)
    ema = ModelEMA(model, decay=ema_decay) if use_ema else None
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_accuracy = -1.0
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    if teacher is not None:
        teacher.eval()
        teacher.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        epoch_kd_alpha = _scheduled_value(
            kd_alpha,
            kd_alpha_final,
            kd_alpha_schedule,
            epoch,
            epochs,
        )
        epoch_kd_temperature = _scheduled_value(
            kd_temperature,
            kd_temperature_final,
            kd_temperature_schedule,
            epoch,
            epochs,
        )

        progress = tqdm(train_loader, desc=f"{stage_name} | epoch {epoch}/{epochs}", leave=True)
        for batch in progress:
            features, targets, cached_teacher_logits = _unpack_batch(batch)
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(device, enabled=use_amp):
                logits = model(features)
                loss = _cross_entropy(logits, targets, label_smoothing=label_smoothing)
                if teacher is not None and epoch_kd_alpha > 0.0:
                    if cached_teacher_logits is None:
                        with torch.no_grad():
                            teacher_logits = teacher(features)
                    else:
                        teacher_logits = cached_teacher_logits.to(
                            device,
                            non_blocking=True,
                            dtype=logits.dtype,
                        )
                    loss = (1.0 - epoch_kd_alpha) * loss + epoch_kd_alpha * _kd_loss(
                        logits,
                        teacher_logits,
                        temperature=epoch_kd_temperature,
                    )

            scaler.scale(loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            batch_size = int(features.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_items += batch_size
            progress.set_postfix(
                loss=f"{total_loss / max(total_items, 1):.4f}",
                acc=f"{total_correct / max(total_items, 1):.4f}",
            )

        if scheduler is not None:
            scheduler.step()

        train_metrics = {
            "loss": total_loss / max(total_items, 1),
            "accuracy": total_correct / max(total_items, 1),
        }
        if ema is not None and eval_with_ema:
            ema.apply_to(model)
        val_metrics = evaluate(
            model,
            val_loader,
            device=device,
            label_smoothing=0.0,
            top_k=top_k,
            use_amp=use_amp,
            desc=f"{stage_name} | val {epoch}/{epochs}",
        )
        if ema is not None and eval_with_ema:
            ema.restore(model)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "lr": float(optimizer.param_groups[0]["lr"]),
                "kd_alpha": float(epoch_kd_alpha if teacher is not None else 0.0),
                "kd_temperature": float(epoch_kd_temperature if teacher is not None else 0.0),
            }
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = copy.deepcopy(ema.shadow if ema is not None and eval_with_ema else model.state_dict())
        else:
            stale_epochs += 1

        if early_stopping_patience > 0 and stale_epochs >= early_stopping_patience:
            break

    model.load_state_dict(best_state, strict=True)
    return {
        "history": history,
        "best_state": best_state,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "elapsed_sec": time.perf_counter() - started,
    }


def train_teacher(
    teacher: nn.Module,
    loaders: dict[str, DataLoader[Any]],
    experiment: ExperimentConfig,
    device: torch.device,
) -> dict[str, Any]:
    return _train_stage(
        model=teacher.to(device),
        train_loader=loaders["train"],
        val_loader=loaders["validation"],
        device=device,
        epochs=experiment.train.teacher_epochs,
        stage_name="hash_teacher",
        lr=experiment.train.teacher_lr,
        optimizer_name=experiment.train.optimizer_name,
        scheduler_name=experiment.train.teacher_scheduler_name,
        weight_decay=experiment.train.weight_decay,
        label_smoothing=experiment.train.teacher_label_smoothing,
        grad_clip_norm=experiment.train.grad_clip_norm,
        use_amp=experiment.train.use_amp,
        use_ema=experiment.train.use_ema,
        ema_decay=experiment.train.ema_decay,
        eval_with_ema=experiment.train.eval_with_ema,
        top_k=experiment.train.top_k,
        early_stopping_patience=experiment.train.teacher_early_stopping_patience,
    )


def train_student(
    student: nn.Module,
    loaders: dict[str, DataLoader[Any]],
    experiment: ExperimentConfig,
    device: torch.device,
    teacher: nn.Module | None = None,
) -> dict[str, Any]:
    student = student.to(device)
    full_history: list[dict[str, float]] = []
    stage_summaries: list[dict[str, Any]] = []
    best_state = copy.deepcopy(student.state_dict())
    teacher_logits_cache: dict[str, Any] | None = None

    if experiment.train.student_pretrain_epochs > 0:
        pretrain = _train_stage(
            model=student,
            train_loader=loaders["train"],
            val_loader=loaders["validation"],
            device=device,
            epochs=experiment.train.student_pretrain_epochs,
            stage_name="hash_student_pretrain",
            lr=experiment.train.student_lr,
            optimizer_name=experiment.train.optimizer_name,
            scheduler_name=experiment.train.student_scheduler_name,
            weight_decay=experiment.train.weight_decay,
            label_smoothing=experiment.train.label_smoothing,
            grad_clip_norm=experiment.train.grad_clip_norm,
            use_amp=experiment.train.use_amp,
            use_ema=experiment.train.use_ema,
            ema_decay=experiment.train.ema_decay,
            eval_with_ema=experiment.train.eval_with_ema,
            top_k=experiment.train.top_k,
            early_stopping_patience=experiment.train.student_early_stopping_patience,
        )
        best_state = copy.deepcopy(student.state_dict())
        full_history.extend(pretrain["history"])
        stage_summaries.append(
            {
                "stage": "pretrain",
                "best_val_accuracy": pretrain["best_val_accuracy"],
                "best_epoch": pretrain["best_epoch"],
                "elapsed_sec": pretrain["elapsed_sec"],
            }
        )

    main_train_loader = loaders["train"]
    if (
        teacher is not None
        and experiment.train.uses_distillation
        and experiment.train.cache_teacher_logits
    ):
        teacher_logits_cache = materialize_teacher_logits(
            teacher=teacher,
            loader=loaders["train"],
            experiment=experiment,
            device=device,
            split_name="train",
        )
        main_train_loader = teacher_logits_cache["loader"]

    main_train = _train_stage(
        model=student,
        train_loader=main_train_loader,
        val_loader=loaders["validation"],
        device=device,
        epochs=experiment.train.student_epochs,
        stage_name="hash_student",
        lr=experiment.train.student_lr,
        optimizer_name=experiment.train.optimizer_name,
        scheduler_name=experiment.train.student_scheduler_name,
        weight_decay=experiment.train.weight_decay,
        label_smoothing=experiment.train.label_smoothing,
        grad_clip_norm=experiment.train.grad_clip_norm,
        use_amp=experiment.train.use_amp,
        use_ema=experiment.train.use_ema,
        ema_decay=experiment.train.ema_decay,
        eval_with_ema=experiment.train.eval_with_ema,
        top_k=experiment.train.top_k,
        early_stopping_patience=experiment.train.student_early_stopping_patience,
        teacher=teacher,
        kd_alpha=experiment.train.kd_alpha,
        kd_alpha_schedule=experiment.train.kd_alpha_schedule,
        kd_alpha_final=experiment.train.kd_alpha_final,
        kd_temperature=experiment.train.kd_temperature,
        kd_temperature_schedule=experiment.train.kd_temperature_schedule,
        kd_temperature_final=experiment.train.kd_temperature_final,
    )
    best_state = copy.deepcopy(student.state_dict())
    full_history.extend(main_train["history"])
    stage_summaries.append(
        {
            "stage": "student",
            "best_val_accuracy": main_train["best_val_accuracy"],
            "best_epoch": main_train["best_epoch"],
            "elapsed_sec": main_train["elapsed_sec"],
            "distillation_enabled": bool(teacher is not None and experiment.train.uses_distillation),
            "teacher_logits_cache": {
                key: value
                for key, value in (teacher_logits_cache or {}).items()
                if key not in ("loader", "metadata")
            },
            "kd_alpha_schedule": experiment.train.kd_alpha_schedule,
            "kd_alpha_final": experiment.train.kd_alpha_final,
            "kd_temperature_schedule": experiment.train.kd_temperature_schedule,
            "kd_temperature_final": experiment.train.kd_temperature_final,
        }
    )

    if experiment.train.student_polish_epochs > 0:
        polish_label_smoothing = (
            experiment.train.polish_label_smoothing
            if experiment.train.polish_label_smoothing is not None
            else max(0.0, experiment.train.label_smoothing * 0.5)
        )
        polish = _train_stage(
            model=student,
            train_loader=loaders["train"],
            val_loader=loaders["validation"],
            device=device,
            epochs=experiment.train.student_polish_epochs,
            stage_name="hash_student_polish",
            lr=experiment.train.polish_lr,
            optimizer_name=experiment.train.optimizer_name,
            scheduler_name="none",
            weight_decay=experiment.train.weight_decay,
            label_smoothing=polish_label_smoothing,
            grad_clip_norm=experiment.train.grad_clip_norm,
            use_amp=experiment.train.use_amp,
            use_ema=experiment.train.use_ema,
            ema_decay=experiment.train.ema_decay,
            eval_with_ema=experiment.train.eval_with_ema,
            top_k=experiment.train.top_k,
            early_stopping_patience=max(0, experiment.train.student_early_stopping_patience // 2),
        )
        best_state = copy.deepcopy(student.state_dict())
        full_history.extend(polish["history"])
        stage_summaries.append(
            {
                "stage": "polish",
                "best_val_accuracy": polish["best_val_accuracy"],
                "best_epoch": polish["best_epoch"],
                "elapsed_sec": polish["elapsed_sec"],
                "label_smoothing": polish_label_smoothing,
            }
        )

    student.load_state_dict(best_state, strict=True)
    test_metrics = evaluate(
        student,
        loaders["test"],
        device=device,
        label_smoothing=0.0,
        top_k=experiment.train.top_k,
        use_amp=experiment.train.use_amp,
        desc="hash_student | test",
    )
    return {
        "history": full_history,
        "test_metrics": test_metrics,
        "best_state": best_state,
        "stage_summaries": stage_summaries,
    }
