from __future__ import annotations

from typing import Iterable

import tensorflow as tf

from .config import ExperimentConfig


def _round_filters(filters: int, width: float) -> int:
    scaled = int(round(filters * width))
    return max(8, int(round(scaled / 8.0) * 8))


def _conv_bn_relu(
    x: tf.Tensor,
    filters: int,
    kernel_size: tuple[int, int],
    strides: tuple[int, int],
    name: str,
) -> tf.Tensor:
    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_bn")(x)
    return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_relu")(x)


@tf.keras.utils.register_keras_serializable(package="kws_advanced")
class LearnableClassToken(tf.keras.layers.Layer):
    def build(self, input_shape) -> None:
        embedding_dim = input_shape[-1]
        if embedding_dim is None:
            raise ValueError("LearnableClassToken requires a known embedding dimension.")
        self.class_token = self.add_weight(
            name="class_token",
            shape=(1, 1, embedding_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        class_token = tf.broadcast_to(self.class_token, [batch_size, 1, tf.shape(inputs)[-1]])
        return tf.concat([class_token, inputs], axis=1)

    def get_config(self) -> dict:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="kws_advanced")
class LearnablePositionEmbedding(tf.keras.layers.Layer):
    def build(self, input_shape) -> None:
        sequence_length = input_shape[1]
        embedding_dim = input_shape[2]
        if sequence_length is None or embedding_dim is None:
            raise ValueError("LearnablePositionEmbedding requires known sequence and embedding dimensions.")
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(1, sequence_length, embedding_dim),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.position_embedding

    def get_config(self) -> dict:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="kws_advanced")
class TakeClassToken(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, 0]

    def get_config(self) -> dict:
        return super().get_config()


@tf.keras.utils.register_keras_serializable(package="kws_advanced")
class SubSpectralNormalization(tf.keras.layers.Layer):
    def __init__(
        self,
        spec_groups: int = 5,
        momentum: float = 0.99,
        epsilon: float = 1e-3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.spec_groups = spec_groups
        self.momentum = momentum
        self.epsilon = epsilon
        self._bn: tf.keras.layers.BatchNormalization | None = None

    def build(self, input_shape) -> None:
        channels = input_shape[-1]
        if channels is None:
            raise ValueError("SubSpectralNormalization requires a known channel dimension.")
        self._bn = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=self.momentum,
            epsilon=self.epsilon,
            name=f"{self.name}_bn",
        )
        self._bn.build((input_shape[0], None, input_shape[2], channels * self.spec_groups))
        super().build(input_shape)

    def call(self, inputs, training=None):
        if self._bn is None:
            raise RuntimeError("SubSpectralNormalization was not built.")
        shape = tf.shape(inputs)
        batch_size = shape[0]
        freq_bins = shape[1]
        time_bins = shape[2]
        channels = shape[3]
        tf.debugging.assert_equal(
            tf.math.floormod(freq_bins, self.spec_groups),
            0,
            message="Frequency bins must be divisible by spec_groups.",
        )
        reshaped = tf.reshape(
            inputs,
            [batch_size, freq_bins // self.spec_groups, time_bins, channels * self.spec_groups],
        )
        normalized = self._bn(reshaped, training=training)
        return tf.reshape(normalized, [batch_size, freq_bins, time_bins, channels])

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "spec_groups": self.spec_groups,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
            }
        )
        return config


def _factorized_residual_block(
    x: tf.Tensor,
    filters: int,
    temporal_kernel: int,
    freq_kernel: int,
    strides: tuple[int, int],
    residual: bool,
    name: str,
) -> tf.Tensor:
    shortcut = x
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(temporal_kernel, 1),
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_dw_time",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_dw_time_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_dw_time_relu")(x)

    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1, freq_kernel),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{name}_dw_freq",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_dw_freq_bn")(x)
    x = tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_dw_freq_relu")(x)

    x = tf.keras.layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{name}_pw",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_pw_bn")(x)

    if residual:
        if shortcut.shape[-1] != filters or strides != (1, 1):
            shortcut = tf.keras.layers.Conv2D(
                filters,
                kernel_size=(1, 1),
                strides=strides,
                padding="same",
                use_bias=False,
                name=f"{name}_skip_conv",
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(name=f"{name}_skip_bn")(shortcut)
        x = tf.keras.layers.Add(name=f"{name}_add")([x, shortcut])

    return tf.keras.layers.ReLU(max_value=6.0, name=f"{name}_out")(x)


def _bcresnet_head(x: tf.Tensor, base_channels: int, name: str) -> tf.Tensor:
    x = tf.keras.layers.Permute((2, 1, 3), name=f"{name}_permute_to_freq_time")(x)
    x = tf.keras.layers.Conv2D(
        base_channels * 2,
        kernel_size=(5, 5),
        strides=(2, 1),
        padding="same",
        use_bias=False,
        name=f"{name}_head_conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_head_bn")(x)
    return tf.keras.layers.ReLU(name=f"{name}_head_relu")(x)


def _bcresnet_block(
    x: tf.Tensor,
    out_channels: int,
    stage_index: int,
    use_stride: bool,
    name: str,
) -> tf.Tensor:
    shortcut = x
    in_channels = int(x.shape[-1])
    transition_block = in_channels != out_channels

    if transition_block:
        x = tf.keras.layers.Conv2D(
            out_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="same",
            use_bias=False,
            name=f"{name}_transition_conv",
        )(x)
        x = tf.keras.layers.BatchNormalization(name=f"{name}_transition_bn")(x)
        x = tf.keras.layers.ReLU(name=f"{name}_transition_relu")(x)

    stride = (2, 1) if use_stride else (1, 1)
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 1),
        strides=stride,
        padding="same",
        use_bias=False,
        name=f"{name}_f2_depthwise",
    )(x)
    x = SubSpectralNormalization(spec_groups=5, name=f"{name}_ssn")(x)
    aux_2d_res = x

    freq_bins = int(x.shape[1])
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(freq_bins, 1),
        name=f"{name}_freq_pool",
    )(x)
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(1, 3),
        strides=(1, 1),
        padding="same",
        dilation_rate=(1, 2 ** stage_index),
        use_bias=False,
        name=f"{name}_f1_depthwise",
    )(x)
    x = tf.keras.layers.BatchNormalization(name=f"{name}_f1_bn")(x)
    x = tf.keras.layers.Activation("swish", name=f"{name}_f1_swish")(x)
    x = tf.keras.layers.Conv2D(
        out_channels,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"{name}_f1_pointwise",
    )(x)
    x = tf.keras.layers.SpatialDropout2D(0.1, name=f"{name}_dropout")(x)
    x = tf.keras.layers.Add(name=f"{name}_broadcast_add")([x, aux_2d_res])

    if not transition_block:
        x = tf.keras.layers.Add(name=f"{name}_residual_add")([x, shortcut])

    return tf.keras.layers.ReLU(name=f"{name}_out")(x)


def _kwt_encoder_block(
    x: tf.Tensor,
    embedding_dim: int,
    num_heads: int,
    mlp_dim: int,
    dropout: float,
    name: str,
) -> tf.Tensor:
    attention = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim // num_heads,
        dropout=dropout,
        name=f"{name}_attention",
    )(x, x)
    x = tf.keras.layers.Add(name=f"{name}_attention_add")([x, attention])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_attention_norm")(x)

    mlp = tf.keras.layers.Dense(mlp_dim, activation=tf.nn.gelu, name=f"{name}_mlp_expand")(x)
    mlp = tf.keras.layers.Dropout(dropout, name=f"{name}_mlp_dropout1")(mlp)
    mlp = tf.keras.layers.Dense(embedding_dim, name=f"{name}_mlp_project")(mlp)
    mlp = tf.keras.layers.Dropout(dropout, name=f"{name}_mlp_dropout2")(mlp)

    x = tf.keras.layers.Add(name=f"{name}_mlp_add")([x, mlp])
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6, name=f"{name}_out")(x)
    return x


def build_bcresnet_teacher(
    input_shape: tuple[int, int, int],
    num_classes: int,
    width: float,
    dropout: float,
) -> tf.keras.Model:
    del dropout
    base_channels = max(8, int(round(16 * width)))
    channels = [
        base_channels * 2,
        base_channels,
        int(round(base_channels * 1.5)),
        base_channels * 2,
        int(round(base_channels * 2.5)),
        base_channels * 4,
    ]
    repeats = (2, 2, 4, 4)
    stride_stages = {1, 2}

    inputs = tf.keras.Input(shape=input_shape, name="input_features")
    x = _bcresnet_head(inputs, base_channels=base_channels, name="bcresnet")

    for stage_index, repeat_count in enumerate(repeats):
        out_channels = channels[stage_index + 1]
        for block_index in range(repeat_count):
            x = _bcresnet_block(
                x,
                out_channels=out_channels,
                stage_index=stage_index,
                use_stride=(stage_index in stride_stages and block_index == 0),
                name=f"bcresnet_stage{stage_index + 1}_block{block_index + 1}",
            )

    x = tf.keras.layers.ZeroPadding2D(padding=((0, 0), (2, 2)), name="bcresnet_classifier_pad")(x)
    x = tf.keras.layers.DepthwiseConv2D(
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        name="bcresnet_classifier_depthwise",
    )(x)
    x = tf.keras.layers.Conv2D(
        channels[-1],
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name="bcresnet_classifier_pointwise",
    )(x)
    x = tf.keras.layers.BatchNormalization(name="bcresnet_classifier_bn")(x)
    x = tf.keras.layers.ReLU(name="bcresnet_classifier_relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="bcresnet_global_pool")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, name="bcresnet_logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="teacher_bcresnet")


def build_kwt_teacher(
    input_shape: tuple[int, int, int],
    num_classes: int,
    width: float,
    dropout: float,
) -> tf.keras.Model:
    if input_shape[-1] != 1:
        raise ValueError("KWT teacher expects a single-channel feature map.")

    embedding_dim = max(64, int(round(192 * width / 64.0) * 64))
    num_heads = max(1, embedding_dim // 64)
    mlp_dim = embedding_dim * 4
    num_layers = 12

    inputs = tf.keras.Input(shape=input_shape, name="input_features")
    x = tf.keras.layers.Reshape((input_shape[0], input_shape[1]), name="kwt_frame_tokens")(inputs)
    x = tf.keras.layers.Dense(embedding_dim, use_bias=True, name="kwt_patch_projection")(x)
    x = LearnableClassToken(name="kwt_class_token")(x)
    x = LearnablePositionEmbedding(name="kwt_position_embedding")(x)
    x = tf.keras.layers.Dropout(dropout, name="kwt_input_dropout")(x)

    for block_index in range(num_layers):
        x = _kwt_encoder_block(
            x,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            name=f"kwt_block{block_index + 1}",
        )

    cls_token = TakeClassToken(name="kwt_cls_slice")(x)
    cls_token = tf.keras.layers.LayerNormalization(epsilon=1e-6, name="kwt_cls_norm")(cls_token)
    outputs = tf.keras.layers.Dense(num_classes, activation=None, name="kwt_logits")(cls_token)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="teacher_kwt")


def _pool_head(
    x: tf.Tensor,
    dropout: float,
    num_classes: int,
    name: str,
) -> tf.Tensor:
    head_filters = max(int(x.shape[-1]), 32)
    x = _conv_bn_relu(x, head_filters, kernel_size=(1, 1), strides=(1, 1), name=f"{name}_head")
    pool_height = int(x.shape[1])
    pool_width = int(x.shape[2])
    x = tf.keras.layers.AveragePooling2D(
        pool_size=(pool_height, pool_width),
        name=f"{name}_avgpool",
    )(x)
    x = tf.keras.layers.Flatten(name=f"{name}_flatten")(x)
    x = tf.keras.layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return tf.keras.layers.Dense(num_classes, activation=None, name=f"{name}_logits")(x)


def build_baseline_cnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    width: float,
    dropout: float,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input_features")
    x = _conv_bn_relu(inputs, _round_filters(24, width), (3, 3), (2, 2), name="stem")
    x = _conv_bn_relu(x, _round_filters(32, width), (3, 3), (2, 1), name="block1")
    x = _conv_bn_relu(x, _round_filters(48, width), (3, 3), (2, 2), name="block2")
    outputs = _pool_head(x, dropout=dropout, num_classes=num_classes, name="baseline")
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="baseline_cnn")


def build_factorized_dscnn(
    input_shape: tuple[int, int, int],
    num_classes: int,
    width: float,
    dropout: float,
    block_filters: Iterable[int],
    block_repeats: Iterable[int],
    name: str,
) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape, name="input_features")
    x = _conv_bn_relu(
        inputs,
        _round_filters(24, width),
        kernel_size=(5, 3),
        strides=(2, 2),
        name="stem",
    )

    for stage_index, (filters, repeats) in enumerate(zip(block_filters, block_repeats)):
        rounded_filters = _round_filters(filters, width)
        for block_index in range(repeats):
            strides = (1, 1)
            if stage_index > 0 and block_index == 0:
                strides = (2, 1) if stage_index == 1 else (2, 2)
            x = _factorized_residual_block(
                x,
                filters=rounded_filters,
                temporal_kernel=5 if stage_index < 2 else 3,
                freq_kernel=3,
                strides=strides,
                residual=True,
                name=f"stage{stage_index + 1}_block{block_index + 1}",
            )

    outputs = _pool_head(x, dropout=dropout, num_classes=num_classes, name=name)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def build_model_by_name(
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    width: float,
    dropout: float,
) -> tf.keras.Model:
    if model_name == "baseline_cnn":
        return build_baseline_cnn(input_shape, num_classes, width=width, dropout=dropout)
    if model_name == "teacher_factorized_dscnn":
        return build_factorized_dscnn(
            input_shape,
            num_classes,
            width=width,
            dropout=dropout,
            block_filters=(32, 48, 64),
            block_repeats=(2, 2, 2),
            name="teacher_factorized_dscnn",
        )
    if model_name == "teacher_factorized_dscnn_xl":
        return build_factorized_dscnn(
            input_shape,
            num_classes,
            width=width,
            dropout=dropout,
            block_filters=(48, 72, 96),
            block_repeats=(2, 3, 2),
            name="teacher_factorized_dscnn_xl",
        )
    if model_name == "teacher_bcresnet":
        return build_bcresnet_teacher(
            input_shape,
            num_classes,
            width=width,
            dropout=dropout,
        )
    if model_name == "teacher_kwt":
        return build_kwt_teacher(
            input_shape,
            num_classes,
            width=width,
            dropout=dropout,
        )
    if model_name == "student_factorized_dscnn":
        return build_factorized_dscnn(
            input_shape,
            num_classes,
            width=width,
            dropout=dropout,
            block_filters=(24, 32, 48),
            block_repeats=(1, 2, 1),
            name="student_factorized_dscnn",
        )
    if model_name == "student_factorized_dscnn_v2":
        return build_factorized_dscnn(
            input_shape,
            num_classes,
            width=width,
            dropout=dropout,
            block_filters=(32, 48, 64),
            block_repeats=(2, 2, 1),
            name="student_factorized_dscnn_v2",
        )
    raise KeyError(f"Unknown model name: {model_name}")


def build_teacher_model(experiment: ExperimentConfig) -> tf.keras.Model:
    return build_model_by_name(
        model_name=experiment.model.teacher_name,
        input_shape=experiment.teacher_feature_shape,
        num_classes=experiment.num_labels,
        width=experiment.model.teacher_width,
        dropout=experiment.model.teacher_dropout,
    )


def build_student_model(experiment: ExperimentConfig) -> tf.keras.Model:
    return build_model_by_name(
        model_name=experiment.model.student_name,
        input_shape=experiment.student_feature_shape,
        num_classes=experiment.num_labels,
        width=experiment.model.student_width,
        dropout=experiment.model.student_dropout,
    )


def _shape_list(value) -> list[int | None]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)) and value and not hasattr(value, "shape"):
        value = value[0]
    if hasattr(value, "shape"):
        shape = value.shape
    else:
        shape = value
    try:
        return list(tf.TensorShape(shape).as_list())
    except Exception:
        return []


def estimate_maccs(model: tf.keras.Model) -> int:
    total = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            output_shape = _shape_list(layer.output)
            input_shape = _shape_list(layer.input)
            if len(output_shape) != 4 or len(input_shape) != 4:
                continue
            _, out_h, out_w, out_c = output_shape
            kernel_h, kernel_w = layer.kernel_size
            in_c = input_shape[-1]
            if None in (out_h, out_w, out_c, in_c):
                continue
            total += int(out_h) * int(out_w) * kernel_h * kernel_w * in_c * int(out_c)
        elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            output_shape = _shape_list(layer.output)
            input_shape = _shape_list(layer.input)
            if len(output_shape) != 4 or len(input_shape) != 4:
                continue
            _, out_h, out_w, out_c = output_shape
            kernel_h, kernel_w = layer.kernel_size
            in_c = input_shape[-1]
            depth_multiplier = int(layer.depth_multiplier)
            if None in (out_h, out_w, out_c, in_c):
                continue
            total += int(out_h) * int(out_w) * kernel_h * kernel_w * in_c * depth_multiplier
        elif isinstance(layer, tf.keras.layers.Dense):
            input_shape = _shape_list(layer.input)
            if not input_shape or input_shape[-1] is None:
                continue
            total += int(input_shape[-1]) * int(layer.units)
        elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
            if not isinstance(layer.input, (list, tuple)) or len(layer.input) < 2:
                continue
            query_shape = _shape_list(layer.input[0])
            value_shape = _shape_list(layer.input[1])
            output_shape = _shape_list(layer.output)
            if len(query_shape) != 3 or len(value_shape) != 3 or len(output_shape) != 3:
                continue
            _, query_length, query_dim = query_shape
            _, value_length, value_dim = value_shape
            _, _, output_dim = output_shape
            key_dim = int(layer.key_dim)
            value_head_dim = int(layer.value_dim or layer.key_dim)
            num_heads = int(layer.num_heads)
            if None in (query_length, query_dim, value_length, value_dim, output_dim):
                continue
            total += int(query_length) * int(query_dim) * num_heads * key_dim
            total += int(value_length) * int(value_dim) * num_heads * key_dim
            total += int(value_length) * int(value_dim) * num_heads * value_head_dim
            total += int(query_length) * num_heads * value_head_dim * int(output_dim)
            total += num_heads * int(query_length) * int(value_length) * (key_dim + value_head_dim)
    return total


def required_tflite_ops(model_name: str) -> list[str]:
    if model_name not in {
        "baseline_cnn",
        "teacher_factorized_dscnn",
        "teacher_factorized_dscnn_xl",
        "teacher_bcresnet",
        "teacher_kwt",
        "student_factorized_dscnn",
        "student_factorized_dscnn_v2",
    }:
        raise KeyError(f"Unknown model name: {model_name}")
    if model_name == "teacher_kwt":
        return [
            "FULLY_CONNECTED",
            "BATCH_MATMUL",
            "SOFTMAX",
            "RESHAPE",
            "ADD",
            "MUL",
            "TRANSPOSE",
        ]
    return [
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "ADD",
        "AVERAGE_POOL_2D",
        "RESHAPE",
        "FULLY_CONNECTED",
    ]
