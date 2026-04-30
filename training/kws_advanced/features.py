from __future__ import annotations

from functools import lru_cache

import tensorflow as tf

from .config import FrontendConfig


def pad_or_trim_waveform(waveform: tf.Tensor, config: FrontendConfig) -> tf.Tensor:
    waveform = tf.cast(waveform, tf.float32)
    waveform = waveform[: config.desired_samples]
    padding = tf.maximum(0, config.desired_samples - tf.shape(waveform)[0])
    waveform = tf.pad(waveform, [[0, padding]])
    waveform.set_shape([config.desired_samples])
    return waveform


@lru_cache(maxsize=1)
def _microfrontend_module():
    try:
        from tensorflow.lite.experimental.microfrontend.python.ops import (  # type: ignore
            audio_microfrontend_op,
        )
    except Exception:
        return None
    return audio_microfrontend_op


def describe_frontend_runtime(config: FrontendConfig) -> dict[str, object]:
    module_available = _microfrontend_module() is not None
    using_exact = bool(module_available and config.enable_exact_microfrontend)
    return {
        "feature_shape": list(config.feature_shape),
        "enable_exact_microfrontend": config.enable_exact_microfrontend,
        "microfrontend_module_available": module_available,
        "using_exact_microfrontend": using_exact,
        "fallback_enabled": config.use_pcen_fallback,
        "fallback_feature_clip": config.fallback_feature_clip,
        "frame_length_ms": config.frame_length_ms,
        "frame_step_ms": config.frame_step_ms,
        "num_channels": config.num_channels,
    }


def _waveform_to_int16(waveform: tf.Tensor) -> tf.Tensor:
    waveform = tf.clip_by_value(tf.cast(waveform, tf.float32), -1.0, 1.0)
    return tf.cast(tf.round(waveform * 32767.0), tf.int16)


def _exact_microfrontend_features(
    waveform: tf.Tensor,
    config: FrontendConfig,
) -> tf.Tensor | None:
    module = _microfrontend_module()
    if module is None or not config.enable_exact_microfrontend:
        return None

    audio = _waveform_to_int16(waveform)
    common_kwargs = {
        "sample_rate": config.sample_rate,
        "num_channels": config.num_channels,
        "lower_band_limit": config.lower_band_limit,
        "upper_band_limit": config.upper_band_limit,
        "smoothing_bits": config.smoothing_bits,
        "even_smoothing": config.even_smoothing,
        "odd_smoothing": config.odd_smoothing,
        "min_signal_remaining": config.min_signal_remaining,
        "enable_pcan": True,
        "pcan_strength": config.pcan_strength,
        "pcan_offset": config.pcan_offset,
        "gain_bits": config.gain_bits,
        "enable_log": True,
        "scale_shift": config.scale_shift,
        "left_context": 0,
        "right_context": 0,
        "frame_stride": 1,
        "zero_padding": False,
    }
    signature_variants = (
        {
            "window_size": config.frame_length_ms,
            "window_step": config.frame_step_ms,
        },
        {
            "window_size_ms": config.frame_length_ms,
            "window_step_ms": config.frame_step_ms,
        },
    )
    for variant in signature_variants:
        try:
            features = module.audio_microfrontend(audio, **common_kwargs, **variant)
            return tf.cast(features, tf.float32)
        except TypeError:
            continue
    return None


def _mel_weight_matrix(config_key: tuple[int, int, int, float, float]) -> tf.Tensor:
    sample_rate, fft_length, num_channels, lower_band, upper_band = config_key
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=num_channels,
        num_spectrogram_bins=(fft_length // 2) + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=lower_band,
        upper_edge_hertz=upper_band,
    )


def _ema_over_time(values: tf.Tensor, coefficient: float) -> tf.Tensor:
    values = tf.cast(values, tf.float32)
    coefficient = tf.cast(coefficient, tf.float32)

    def step(prev: tf.Tensor, current: tf.Tensor) -> tf.Tensor:
        return (coefficient * prev) + ((1.0 - coefficient) * current)

    first = values[0]
    rest = values[1:]
    smoothed_rest = tf.scan(step, rest, initializer=first)
    return tf.concat([first[tf.newaxis, :], smoothed_rest], axis=0)


def _fallback_log_mel_or_pcen(
    waveform: tf.Tensor,
    config: FrontendConfig,
) -> tf.Tensor:
    stft = tf.signal.stft(
        waveform,
        frame_length=config.frame_length_samples,
        frame_step=config.frame_step_samples,
        fft_length=config.fft_length,
        window_fn=tf.signal.hann_window,
        pad_end=False,
    )
    power_spectrogram = tf.math.square(tf.abs(stft))
    mel_matrix = _mel_weight_matrix(
        (
            config.sample_rate,
            config.fft_length,
            config.num_channels,
            config.lower_band_limit,
            config.upper_band_limit,
        )
    )
    mel = tf.tensordot(power_spectrogram, mel_matrix, axes=1)
    mel = tf.cast(mel, tf.float32)
    mel.set_shape([None, config.num_channels])
    mel = tf.maximum(mel, config.pcen_floor)

    if config.use_pcen_fallback:
        smoother = _ema_over_time(mel, coefficient=config.pcen_smoothing_coef)
        normalized = mel / tf.pow(config.pcen_floor + smoother, config.pcen_alpha)
        pcen = tf.pow(normalized + config.pcen_delta, config.pcen_root) - (
            config.pcen_delta ** config.pcen_root
        )
        return tf.math.log1p(tf.maximum(pcen, 0.0))

    return tf.math.log1p(10.0 * mel)


def _quantize_exact_microfrontend(features: tf.Tensor) -> tf.Tensor:
    value_divisor = 25.6 * 26.0
    scaled = tf.round((features * 256.0) / value_divisor) - 128.0
    return tf.clip_by_value(scaled, -128.0, 127.0)


def _quantize_fallback_features(
    features: tf.Tensor,
    config: FrontendConfig,
) -> tf.Tensor:
    clipped = tf.clip_by_value(features, 0.0, config.fallback_feature_clip)
    scaled = (clipped / config.fallback_feature_clip) * 255.0 - 128.0
    return tf.clip_by_value(tf.round(scaled), -128.0, 127.0)


def _fit_feature_frames(features: tf.Tensor, config: FrontendConfig) -> tf.Tensor:
    features = features[: config.frame_count]
    pad_frames = tf.maximum(0, config.frame_count - tf.shape(features)[0])
    features = tf.pad(features, [[0, pad_frames], [0, 0]])
    features.set_shape([config.frame_count, config.num_channels])
    return features


def extract_feature_map(
    waveform: tf.Tensor,
    config: FrontendConfig,
) -> tf.Tensor:
    waveform = pad_or_trim_waveform(waveform, config)
    exact = _exact_microfrontend_features(waveform, config)
    if exact is not None:
        features = _quantize_exact_microfrontend(exact)
    else:
        fallback = _fallback_log_mel_or_pcen(waveform, config)
        features = _quantize_fallback_features(fallback, config)
    features = _fit_feature_frames(features, config)
    return tf.expand_dims(tf.cast(features, tf.float32), axis=-1)


def _mask_axis(
    feature_map: tf.Tensor,
    axis: int,
    mask_size: int,
) -> tf.Tensor:
    axis_length = tf.shape(feature_map)[axis]
    mask_size = tf.minimum(mask_size, axis_length)
    max_start = tf.maximum(axis_length - mask_size + 1, 1)
    start = tf.random.uniform([], 0, max_start, dtype=tf.int32)

    if axis == 0:
        mask = tf.concat(
            [
                tf.ones([start, tf.shape(feature_map)[1], 1], dtype=feature_map.dtype),
                tf.zeros([mask_size, tf.shape(feature_map)[1], 1], dtype=feature_map.dtype),
                tf.ones(
                    [axis_length - start - mask_size, tf.shape(feature_map)[1], 1],
                    dtype=feature_map.dtype,
                ),
            ],
            axis=0,
        )
    else:
        mask = tf.concat(
            [
                tf.ones([tf.shape(feature_map)[0], start, 1], dtype=feature_map.dtype),
                tf.zeros([tf.shape(feature_map)[0], mask_size, 1], dtype=feature_map.dtype),
                tf.ones(
                    [tf.shape(feature_map)[0], axis_length - start - mask_size, 1],
                    dtype=feature_map.dtype,
                ),
            ],
            axis=1,
        )
    return feature_map * mask


def apply_spec_augment(
    feature_map: tf.Tensor,
    specaugment_prob: float,
    time_mask_max: int,
    freq_mask_max: int,
) -> tf.Tensor:
    feature_map = tf.cast(feature_map, tf.float32)
    if specaugment_prob <= 0.0:
        return feature_map

    def unchanged() -> tf.Tensor:
        return feature_map

    def augmented() -> tf.Tensor:
        augmented_map = feature_map
        if time_mask_max > 0:
            time_size = tf.random.uniform([], 0, time_mask_max + 1, dtype=tf.int32)
            augmented_map = _mask_axis(augmented_map, axis=0, mask_size=time_size)
        if freq_mask_max > 0:
            freq_size = tf.random.uniform([], 0, freq_mask_max + 1, dtype=tf.int32)
            augmented_map = _mask_axis(augmented_map, axis=1, mask_size=freq_size)
        return augmented_map

    should_augment = tf.less_equal(
        tf.random.uniform([], dtype=tf.float32),
        tf.cast(specaugment_prob, tf.float32),
    )
    return tf.cond(should_augment, augmented, unchanged)
