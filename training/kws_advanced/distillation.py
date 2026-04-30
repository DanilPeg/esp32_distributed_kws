from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from .config import ExperimentConfig


@tf.keras.utils.register_keras_serializable(package="kws_advanced")
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self,
        initial_learning_rate: float,
        decay_steps: int,
        warmup_steps: int = 0,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.initial_learning_rate = float(initial_learning_rate)
        self.decay_steps = int(max(1, decay_steps))
        self.warmup_steps = int(max(0, warmup_steps))
        self.alpha = float(alpha)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        initial_lr = tf.cast(self.initial_learning_rate, tf.float32)
        warmup_steps = tf.cast(max(1, self.warmup_steps), tf.float32)

        def warmup() -> tf.Tensor:
            return initial_lr * (step + 1.0) / warmup_steps

        def cosine() -> tf.Tensor:
            adjusted_step = tf.maximum(0.0, step - float(self.warmup_steps))
            adjusted_decay_steps = float(max(1, self.decay_steps - self.warmup_steps))
            cosine_decay = 0.5 * (
                1.0
                + tf.cos(
                    tf.constant(3.141592653589793, dtype=tf.float32)
                    * adjusted_step
                    / adjusted_decay_steps
                )
            )
            decayed = (1.0 - self.alpha) * cosine_decay + self.alpha
            return initial_lr * decayed

        if self.warmup_steps <= 0:
            return cosine()
        return tf.cond(step < float(self.warmup_steps), warmup, cosine)

    def get_config(self) -> dict:
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_steps": self.decay_steps,
            "warmup_steps": self.warmup_steps,
            "alpha": self.alpha,
        }


def _build_optimizer(
    experiment: ExperimentConfig,
    learning_rate: float,
    total_steps: int,
    warmup_steps: int = 0,
) -> tf.keras.optimizers.Optimizer:
    total_steps = max(1, total_steps)
    schedule = WarmupCosineDecay(
        initial_learning_rate=learning_rate,
        decay_steps=total_steps,
        warmup_steps=warmup_steps,
        alpha=0.1,
    )
    optimizer_name = experiment.train.optimizer_name.lower()
    common_kwargs = {
        "learning_rate": schedule,
        "clipnorm": experiment.train.clipnorm,
    }
    if optimizer_name == "adamw":
        return tf.keras.optimizers.AdamW(
            weight_decay=experiment.train.weight_decay,
            **common_kwargs,
        )
    if optimizer_name == "adam":
        return tf.keras.optimizers.Adam(**common_kwargs)
    raise KeyError(f"Unsupported optimizer_name: {experiment.train.optimizer_name}")


def compile_classifier(
    model: tf.keras.Model,
    experiment: ExperimentConfig,
    learning_rate: float,
    epochs: int,
    steps_per_epoch: int,
    label_smoothing: float | None = None,
    warmup_epochs: int = 0,
) -> tf.keras.Model:
    optimizer = _build_optimizer(
        experiment=experiment,
        learning_rate=learning_rate,
        total_steps=epochs * steps_per_epoch,
        warmup_steps=warmup_epochs * steps_per_epoch,
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=experiment.model.label_smoothing if label_smoothing is None else label_smoothing,
            from_logits=True,
        ),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=experiment.train.top_k,
                name=f"top{experiment.train.top_k}_accuracy",
            ),
        ],
    )
    return model


def build_callbacks(output_dir: Path, patience: int) -> list[tf.keras.callbacks.Callback]:
    return build_callbacks_for_monitor(output_dir, patience=patience)


def build_callbacks_for_monitor(
    output_dir: Path,
    patience: int,
    monitor: str = "val_accuracy",
    mode: str = "max",
    include_checkpoint: bool = True,
    save_weights_only: bool = False,
    csv_log_filename: str | None = "epoch_log.csv",
    min_delta: float = 0.0,
) -> list[tf.keras.callbacks.Callback]:
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    if csv_log_filename:
        callbacks.append(
            tf.keras.callbacks.CSVLogger(
                str(output_dir / csv_log_filename),
                append=False,
            )
        )
    if include_checkpoint:
        filename = "best.weights.h5" if save_weights_only else "best.keras"
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / filename),
                monitor=monitor,
                mode=mode,
                save_best_only=True,
                save_weights_only=save_weights_only,
            )
        )
    return callbacks


@dataclass
class HistoryBundle:
    history: dict[str, list[float]]


def _merge_histories(stage_histories: list[tuple[str, tf.keras.callbacks.History | None]]) -> HistoryBundle | None:
    merged: dict[str, list[float]] = {}
    saw_history = False
    for _, history in stage_histories:
        if history is None:
            continue
        saw_history = True
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(list(values))
    if not saw_history:
        return None
    return HistoryBundle(history=merged)


def _student_only_dataset(dataset):
    element_spec = getattr(dataset, "element_spec", None)
    if not isinstance(element_spec, tuple) or len(element_spec) != 2:
        return dataset
    feature_spec, _ = element_spec
    if not isinstance(feature_spec, dict) or "student" not in feature_spec:
        return dataset

    def select_student(features, labels):
        return features["student"], labels

    return dataset.map(select_student, num_parallel_calls=tf.data.AUTOTUNE)


def _valid_pair(pair: tuple[str, str] | None) -> tuple[str, str] | None:
    if not pair or len(pair) != 2:
        return None
    if not pair[0] or not pair[1]:
        return None
    return pair


def _valid_pairs(pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]]) -> list[tuple[str, str]]:
    valid: list[tuple[str, str]] = []
    for pair in pairs:
        checked = _valid_pair(pair)
        if checked is not None:
            valid.append(checked)
    return valid


def _unique_names(names: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.add(name)
            unique.append(name)
    return unique


def _build_probe(model: tf.keras.Model, layer_names: list[str]) -> tf.keras.Model | None:
    if not layer_names:
        return None
    outputs = [model.output, *[model.get_layer(name).output for name in layer_names]]
    return tf.keras.Model(inputs=model.inputs, outputs=outputs, name=f"{model.name}_probe")


def _unpack_probe_outputs(outputs, feature_names: list[str]) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    if not feature_names:
        return outputs, {}
    if not isinstance(outputs, (list, tuple)):
        raise TypeError("Probe outputs must be a list or tuple when feature names are provided.")
    logits = outputs[0]
    feature_values = outputs[1:]
    return logits, {name: tensor for name, tensor in zip(feature_names, feature_values)}


def _align_teacher_map(student_map: tf.Tensor, teacher_map: tf.Tensor) -> tf.Tensor:
    student_hw = tf.shape(student_map)[1:3]
    return tf.image.resize(teacher_map, size=student_hw, method="bilinear")


def _normalize_feature_map(feature_map: tf.Tensor) -> tf.Tensor:
    return tf.nn.l2_normalize(feature_map, axis=-1)


def _attention_vector(feature_map: tf.Tensor) -> tf.Tensor:
    attention = tf.reduce_mean(tf.square(feature_map), axis=-1)
    attention = tf.reshape(attention, [tf.shape(attention)[0], -1])
    return tf.nn.l2_normalize(attention, axis=-1)


def _frame_similarity_matrix(feature_map: tf.Tensor) -> tf.Tensor:
    time_embeddings = tf.reduce_mean(feature_map, axis=2)
    time_embeddings = tf.nn.l2_normalize(time_embeddings, axis=-1)
    return tf.matmul(time_embeddings, time_embeddings, transpose_b=True)


@tf.keras.utils.register_keras_serializable(package="kws_advanced")
class Distiller(tf.keras.Model):
    def __init__(
        self,
        student: tf.keras.Model,
        teacher: tf.keras.Model,
        hint_layer_pair: tuple[str, str] | None = None,
        attention_layer_pairs: tuple[tuple[str, str], ...] | list[tuple[str, str]] = (),
        similarity_layer_pair: tuple[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.hint_layer_pair = _valid_pair(hint_layer_pair)
        self.attention_layer_pairs = tuple(_valid_pairs(attention_layer_pairs))
        self.similarity_layer_pair = _valid_pair(similarity_layer_pair)

        student_feature_names = []
        teacher_feature_names = []
        for student_name, teacher_name in self.attention_layer_pairs:
            student_feature_names.append(student_name)
            teacher_feature_names.append(teacher_name)
        if self.hint_layer_pair:
            student_feature_names.append(self.hint_layer_pair[0])
            teacher_feature_names.append(self.hint_layer_pair[1])
        if self.similarity_layer_pair:
            student_feature_names.append(self.similarity_layer_pair[0])
            teacher_feature_names.append(self.similarity_layer_pair[1])

        self.student_feature_names = _unique_names(student_feature_names)
        self.teacher_feature_names = _unique_names(teacher_feature_names)
        self.student_probe = _build_probe(self.student, self.student_feature_names)
        self.teacher_probe = _build_probe(self.teacher, self.teacher_feature_names)

        self.student_loss_fn: tf.keras.losses.Loss | None = None
        self.distillation_loss_fn: tf.keras.losses.Loss | None = None
        self.ce_weight = 0.2
        self.kd_weight = 0.8
        self.hint_weight = 0.0
        self.attention_weight = 0.0
        self.similarity_weight = 0.0
        self.temperature = 4.0
        self.metric_list: list[tf.keras.metrics.Metric] = []

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.student_loss_tracker = tf.keras.metrics.Mean(name="student_loss")
        self.distillation_loss_tracker = tf.keras.metrics.Mean(name="distillation_loss")
        self.hint_loss_tracker = tf.keras.metrics.Mean(name="hint_loss")
        self.attention_loss_tracker = tf.keras.metrics.Mean(name="attention_loss")
        self.similarity_loss_tracker = tf.keras.metrics.Mean(name="similarity_loss")

        self.hint_projector: tf.keras.layers.Layer | None = None
        if self.hint_layer_pair:
            teacher_hint_layer = self.teacher.get_layer(self.hint_layer_pair[1])
            teacher_channels = int(teacher_hint_layer.output.shape[-1])
            self.hint_projector = tf.keras.layers.Conv2D(
                teacher_channels,
                kernel_size=(1, 1),
                padding="same",
                use_bias=False,
                name="hint_projector",
            )

    @property
    def metrics(self):
        return [
            self.loss_tracker,
            self.student_loss_tracker,
            self.distillation_loss_tracker,
            self.hint_loss_tracker,
            self.attention_loss_tracker,
            self.similarity_loss_tracker,
            *self.metric_list,
        ]

    def compile(
        self,
        optimizer: tf.keras.optimizers.Optimizer,
        metrics: list[tf.keras.metrics.Metric],
        student_loss_fn: tf.keras.losses.Loss,
        distillation_loss_fn: tf.keras.losses.Loss,
        ce_weight: float,
        kd_weight: float,
        hint_weight: float,
        attention_weight: float,
        similarity_weight: float,
        temperature: float,
    ) -> None:
        super().compile(optimizer=optimizer)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.ce_weight = ce_weight
        self.kd_weight = kd_weight
        self.hint_weight = hint_weight
        self.attention_weight = attention_weight
        self.similarity_weight = similarity_weight
        self.temperature = temperature
        self.metric_list = metrics

    def call(self, inputs, training=False):
        student_features, _ = self._split_features(inputs)
        return self.student(student_features, training=training)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "student": tf.keras.utils.serialize_keras_object(self.student),
                "teacher": tf.keras.utils.serialize_keras_object(self.teacher),
                "hint_layer_pair": self.hint_layer_pair,
                "attention_layer_pairs": list(self.attention_layer_pairs),
                "similarity_layer_pair": self.similarity_layer_pair,
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        config = dict(config)
        student = tf.keras.utils.deserialize_keras_object(config.pop("student"))
        teacher = tf.keras.utils.deserialize_keras_object(config.pop("teacher"))
        return cls(student=student, teacher=teacher, **config)

    def _forward_student(
        self,
        features: tf.Tensor,
        training: bool,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        if self.student_probe is None:
            logits = self.student(features, training=training)
            return logits, {}
        return _unpack_probe_outputs(self.student_probe(features, training=training), self.student_feature_names)

    def _forward_teacher(self, features: tf.Tensor) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        if self.teacher_probe is None:
            logits = self.teacher(features, training=False)
            return logits, {}
        return _unpack_probe_outputs(self.teacher_probe(features, training=False), self.teacher_feature_names)

    def _split_features(self, features) -> tuple[tf.Tensor, tf.Tensor]:
        if isinstance(features, dict):
            student_features = features["student"]
            teacher_features = features.get("teacher", student_features)
            return student_features, teacher_features
        return features, features

    def _compute_distillation_terms(
        self,
        labels: tf.Tensor,
        student_logits: tf.Tensor,
        teacher_logits: tf.Tensor,
        student_features: dict[str, tf.Tensor],
        teacher_features: dict[str, tf.Tensor],
        training: bool,
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
        student_loss = self.student_loss_fn(labels, student_logits)
        zero = tf.constant(0.0, dtype=student_logits.dtype)

        kd_loss = zero
        if self.kd_weight > 0.0:
            kd_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_logits / self.temperature, axis=1),
                tf.nn.softmax(student_logits / self.temperature, axis=1),
            ) * (self.temperature ** 2)

        hint_loss = zero
        if self.hint_weight > 0.0 and self.hint_layer_pair and self.hint_projector is not None:
            student_map = student_features[self.hint_layer_pair[0]]
            teacher_map = teacher_features[self.hint_layer_pair[1]]
            teacher_map = tf.stop_gradient(_align_teacher_map(student_map, teacher_map))
            projected_student = self.hint_projector(student_map, training=training)
            projected_student = _normalize_feature_map(projected_student)
            teacher_map = _normalize_feature_map(teacher_map)
            hint_loss = tf.reduce_mean(tf.square(projected_student - teacher_map))

        attention_loss = zero
        if self.attention_weight > 0.0 and self.attention_layer_pairs:
            attention_terms = []
            for student_name, teacher_name in self.attention_layer_pairs:
                student_map = student_features[student_name]
                teacher_map = teacher_features[teacher_name]
                teacher_map = tf.stop_gradient(_align_teacher_map(student_map, teacher_map))
                student_attention = _attention_vector(student_map)
                teacher_attention = _attention_vector(teacher_map)
                attention_terms.append(tf.reduce_mean(tf.square(student_attention - teacher_attention)))
            attention_loss = tf.add_n(attention_terms) / float(len(attention_terms))

        similarity_loss = zero
        if self.similarity_weight > 0.0 and self.similarity_layer_pair:
            student_map = student_features[self.similarity_layer_pair[0]]
            teacher_map = teacher_features[self.similarity_layer_pair[1]]
            teacher_map = tf.stop_gradient(_align_teacher_map(student_map, teacher_map))
            student_similarity = _frame_similarity_matrix(student_map)
            teacher_similarity = _frame_similarity_matrix(teacher_map)
            similarity_loss = tf.reduce_mean(tf.square(student_similarity - teacher_similarity))

        total_loss = (
            self.ce_weight * student_loss
            + self.kd_weight * kd_loss
            + self.hint_weight * hint_loss
            + self.attention_weight * attention_loss
            + self.similarity_weight * similarity_loss
        )

        return total_loss, {
            "student_loss": student_loss,
            "distillation_loss": kd_loss,
            "hint_loss": hint_loss,
            "attention_loss": attention_loss,
            "similarity_loss": similarity_loss,
        }

    def train_step(self, data):
        features, labels = data
        student_inputs, teacher_inputs = self._split_features(features)
        teacher_logits, teacher_features = self._forward_teacher(teacher_inputs)

        with tf.GradientTape() as tape:
            student_logits, student_features = self._forward_student(student_inputs, training=True)
            total_loss, losses = self._compute_distillation_terms(
                labels=labels,
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_features=student_features,
                teacher_features=teacher_features,
                training=True,
            )

        trainable_variables = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, trainable_variables))

        self.loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(losses["student_loss"])
        self.distillation_loss_tracker.update_state(losses["distillation_loss"])
        self.hint_loss_tracker.update_state(losses["hint_loss"])
        self.attention_loss_tracker.update_state(losses["attention_loss"])
        self.similarity_loss_tracker.update_state(losses["similarity_loss"])
        for metric in self.metric_list:
            metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}

    def test_step(self, data):
        features, labels = data
        student_inputs, teacher_inputs = self._split_features(features)
        teacher_logits, teacher_features = self._forward_teacher(teacher_inputs)
        student_logits, student_features = self._forward_student(student_inputs, training=False)
        total_loss, losses = self._compute_distillation_terms(
            labels=labels,
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_features=student_features,
            teacher_features=teacher_features,
            training=False,
        )
        self.loss_tracker.update_state(total_loss)
        self.student_loss_tracker.update_state(losses["student_loss"])
        self.distillation_loss_tracker.update_state(losses["distillation_loss"])
        self.hint_loss_tracker.update_state(losses["hint_loss"])
        self.attention_loss_tracker.update_state(losses["attention_loss"])
        self.similarity_loss_tracker.update_state(losses["similarity_loss"])
        for metric in self.metric_list:
            metric.update_state(labels, student_logits)
        return {metric.name: metric.result() for metric in self.metrics}


def build_distiller(
    student: tf.keras.Model,
    teacher: tf.keras.Model,
    experiment: ExperimentConfig,
    steps_per_epoch: int,
    stage: str = "main",
) -> Distiller:
    teacher.trainable = False
    train_cfg = experiment.train

    hint_layer_pair: tuple[str, str] | None = None
    attention_layer_pairs: tuple[tuple[str, str], ...] = ()
    similarity_layer_pair: tuple[str, str] | None = None
    ce_weight = train_cfg.ce_loss_weight
    kd_weight = train_cfg.kd_loss_weight
    hint_weight = train_cfg.hint_loss_weight
    attention_weight = train_cfg.attention_loss_weight
    similarity_weight = train_cfg.similarity_loss_weight
    stage_epochs = train_cfg.student_epochs

    if stage == "warmup":
        hint_layer_pair = _valid_pair(train_cfg.hint_layer_pair)
        ce_weight = train_cfg.hint_warmup_ce_weight
        kd_weight = 0.0
        hint_weight = train_cfg.hint_warmup_hint_weight
        attention_weight = 0.0
        similarity_weight = 0.0
        stage_epochs = train_cfg.hint_warmup_epochs
    elif stage == "main":
        if train_cfg.distillation_mode == "multilevel":
            hint_layer_pair = _valid_pair(train_cfg.hint_layer_pair)
            attention_layer_pairs = tuple(_valid_pairs(train_cfg.attention_layer_pairs))
            similarity_layer_pair = _valid_pair(train_cfg.similarity_layer_pair)
    else:
        raise KeyError(f"Unsupported distillation stage: {stage}")

    distiller = Distiller(
        student=student,
        teacher=teacher,
        hint_layer_pair=hint_layer_pair,
        attention_layer_pairs=attention_layer_pairs,
        similarity_layer_pair=similarity_layer_pair,
    )
    distiller.compile(
        optimizer=_build_optimizer(
            experiment=experiment,
            learning_rate=experiment.train.student_lr,
            total_steps=stage_epochs * steps_per_epoch,
        ),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(
                k=experiment.train.top_k,
                name=f"top{experiment.train.top_k}_accuracy",
            ),
        ],
        student_loss_fn=tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=experiment.model.label_smoothing,
            from_logits=True,
        ),
        distillation_loss_fn=tf.keras.losses.KLDivergence(),
        ce_weight=ce_weight,
        kd_weight=kd_weight,
        hint_weight=hint_weight,
        attention_weight=attention_weight,
        similarity_weight=similarity_weight,
        temperature=experiment.train.distill_temperature,
    )
    return distiller


def train_student_model(
    student: tf.keras.Model,
    teacher: tf.keras.Model,
    train_ds,
    val_ds,
    experiment: ExperimentConfig,
    steps_per_epoch: int,
    output_dir: Path,
) -> dict[str, object]:
    train_cfg = experiment.train
    stage_histories: list[tuple[str, tf.keras.callbacks.History | None]] = []
    stage_details: list[dict[str, object]] = []
    student_train_ds = _student_only_dataset(train_ds)
    student_val_ds = _student_only_dataset(val_ds)

    if train_cfg.student_pretrain_epochs > 0:
        pretrain_dir = output_dir / "pretrain"
        student = compile_classifier(
            student,
            experiment=experiment,
            learning_rate=train_cfg.student_lr,
            epochs=train_cfg.student_pretrain_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        pretrain_history = student.fit(
            student_train_ds,
            validation_data=student_val_ds,
            epochs=train_cfg.student_pretrain_epochs,
            callbacks=build_callbacks_for_monitor(
                pretrain_dir,
                patience=max(2, train_cfg.early_stopping_patience),
                monitor="val_accuracy",
                mode="max",
                include_checkpoint=True,
                min_delta=train_cfg.early_stopping_min_delta,
            ),
        )
        stage_histories.append(("pretrain", pretrain_history))
        stage_details.append(
            {
                "name": "pretrain",
                "epochs_requested": train_cfg.student_pretrain_epochs,
                "epoch_log_path": str(pretrain_dir / "epoch_log.csv"),
                "checkpoint_path": str(pretrain_dir / "best.keras"),
            }
        )

    if train_cfg.uses_distillation:
        if train_cfg.distillation_mode == "multilevel" and train_cfg.hint_warmup_epochs > 0 and train_cfg.hint_warmup_hint_weight > 0.0:
            warmup_dir = output_dir / "hint_warmup"
            warmup_distiller = build_distiller(
                student=student,
                teacher=teacher,
                experiment=experiment,
                steps_per_epoch=steps_per_epoch,
                stage="warmup",
            )
            warmup_history = warmup_distiller.fit(
                train_ds,
                validation_data=val_ds,
                epochs=train_cfg.hint_warmup_epochs,
                callbacks=build_callbacks_for_monitor(
                    warmup_dir,
                    patience=1,
                    monitor="val_accuracy",
                    mode="max",
                    include_checkpoint=False,
                    min_delta=train_cfg.early_stopping_min_delta,
                ),
            )
            student = warmup_distiller.student
            stage_histories.append(("hint_warmup", warmup_history))
            stage_details.append(
                {
                    "name": "hint_warmup",
                    "epochs_requested": train_cfg.hint_warmup_epochs,
                    "epoch_log_path": str(warmup_dir / "epoch_log.csv"),
                }
            )

        distiller = build_distiller(
            student=student,
            teacher=teacher,
            experiment=experiment,
            steps_per_epoch=steps_per_epoch,
            stage="main",
        )
        distill_history = distiller.fit(
            train_ds,
            validation_data=val_ds,
            epochs=train_cfg.student_epochs,
            callbacks=build_callbacks_for_monitor(
                output_dir,
                patience=train_cfg.early_stopping_patience,
                monitor="val_accuracy",
                mode="max",
                include_checkpoint=False,
                min_delta=train_cfg.early_stopping_min_delta,
            ),
        )
        student = distiller.student
        stage_histories.append(("distillation", distill_history))
        stage_details.append(
            {
                "name": "distillation",
                "epochs_requested": train_cfg.student_epochs,
                "epoch_log_path": str(output_dir / "epoch_log.csv"),
                "mode": train_cfg.distillation_mode,
            }
        )
    else:
        student = compile_classifier(
            student,
            experiment=experiment,
            learning_rate=train_cfg.student_lr,
            epochs=train_cfg.student_epochs,
            steps_per_epoch=steps_per_epoch,
        )
        history = student.fit(
            student_train_ds,
            validation_data=student_val_ds,
            epochs=train_cfg.student_epochs,
            callbacks=build_callbacks_for_monitor(
                output_dir,
                patience=train_cfg.early_stopping_patience,
                monitor="val_accuracy",
                mode="max",
                include_checkpoint=True,
                min_delta=train_cfg.early_stopping_min_delta,
            ),
        )
        stage_histories.append(("student", history))
        stage_details.append(
            {
                "name": "student",
                "epochs_requested": train_cfg.student_epochs,
                "epoch_log_path": str(output_dir / "epoch_log.csv"),
                "checkpoint_path": str(output_dir / "best.keras"),
            }
        )

    if train_cfg.student_polish_epochs > 0:
        polish_dir = output_dir / "polish"
        student = compile_classifier(
            student,
            experiment=experiment,
            learning_rate=train_cfg.student_lr * 0.25,
            epochs=train_cfg.student_polish_epochs,
            steps_per_epoch=max(1, steps_per_epoch),
        )
        polish_history = student.fit(
            student_train_ds,
            validation_data=student_val_ds,
            epochs=train_cfg.student_polish_epochs,
            callbacks=build_callbacks_for_monitor(
                polish_dir,
                patience=1,
                monitor="val_accuracy",
                mode="max",
                include_checkpoint=False,
                min_delta=train_cfg.early_stopping_min_delta,
            ),
        )
        stage_histories.append(("polish", polish_history))
        stage_details.append(
            {
                "name": "polish",
                "epochs_requested": train_cfg.student_polish_epochs,
                "epoch_log_path": str(polish_dir / "epoch_log.csv"),
            }
        )

    merged_history = _merge_histories(stage_histories)
    return {
        "student": student,
        "history": merged_history,
        "extra": {
            "distillation_enabled": bool(train_cfg.uses_distillation),
            "distillation_mode": train_cfg.distillation_mode if train_cfg.uses_distillation else "none",
            "training_stages": stage_details,
            "main_epoch_log_path": str(output_dir / "epoch_log.csv"),
            "checkpoint_path": "" if train_cfg.uses_distillation else str(output_dir / "best.keras"),
        },
    }


def maybe_quantize_aware_clone(model: tf.keras.Model) -> tf.keras.Model | None:
    try:
        import tensorflow_model_optimization as tfmot
    except Exception:
        return None
    return tfmot.quantization.keras.quantize_model(model)
