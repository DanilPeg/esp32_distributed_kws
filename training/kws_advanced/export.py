from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tensorflow as tf

from .config import ExperimentConfig


def representative_dataset(
    dataset: tf.data.Dataset,
    batch_limit: int,
):
    for batch_features, _ in dataset.take(batch_limit):
        for item in batch_features:
            yield [tf.expand_dims(tf.cast(item, tf.float32), axis=0)]


def convert_to_int8_tflite(
    model: tf.keras.Model,
    representative_data: tf.data.Dataset,
    output_path: Path,
    representative_batches: int,
) -> bytes:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(
        representative_data,
        batch_limit=representative_batches,
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_bytes)
    return tflite_bytes


def inspect_tflite_model(model_path: Path) -> dict[str, Any]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]
    return {
        "input_shape": input_details["shape"].tolist(),
        "input_dtype": str(input_details["dtype"]),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_shape": output_details["shape"].tolist(),
        "output_dtype": str(output_details["dtype"]),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
        "tensor_count": len(interpreter.get_tensor_details()),
    }


def write_c_array(
    tflite_bytes: bytes,
    output_dir: Path,
    array_name: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    header_path = output_dir / f"{array_name}.h"
    source_path = output_dir / f"{array_name}.cc"
    include_guard = f"{array_name.upper()}_H_"

    header_text = "\n".join(
        [
            f"#ifndef {include_guard}",
            f"#define {include_guard}",
            "",
            "#include <cstddef>",
            "#include <cstdint>",
            "",
            f"extern const unsigned char {array_name}[];",
            f"extern const unsigned int {array_name}_len;",
            "",
            f"#endif  // {include_guard}",
            "",
        ]
    )

    hex_lines = []
    for offset in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[offset : offset + 12]
        hex_lines.append("  " + ", ".join(f"0x{byte:02x}" for byte in chunk))

    source_text = "\n".join(
        [
            f'#include "{header_path.name}"',
            "",
            f"const unsigned char {array_name}[] = {{",
            ",\n".join(hex_lines),
            "};",
            f"const unsigned int {array_name}_len = {len(tflite_bytes)};",
            "",
        ]
    )

    header_path.write_text(header_text, encoding="utf-8")
    source_path.write_text(source_text, encoding="utf-8")
    return {"header": header_path, "source": source_path}


def export_experiment_artifacts(
    model: tf.keras.Model,
    representative_data: tf.data.Dataset,
    experiment: ExperimentConfig,
    required_ops: list[str],
) -> dict[str, Any]:
    artifact_dir = Path(experiment.export.artifacts_dir) / experiment.tag
    model_path = artifact_dir / f"{experiment.export.model_stem}.tflite"
    tflite_bytes = convert_to_int8_tflite(
        model,
        representative_data=representative_data,
        output_path=model_path,
        representative_batches=experiment.export.representative_batches,
    )
    tflite_info = inspect_tflite_model(model_path)

    c_paths: dict[str, str] = {}
    if experiment.export.export_c_array:
        array_name = f"{experiment.export.model_stem}_model"
        written = write_c_array(tflite_bytes, output_dir=artifact_dir, array_name=array_name)
        c_paths = {key: str(path) for key, path in written.items()}

    metadata = {
        "experiment": experiment.to_dict(),
        "model_size_bytes": len(tflite_bytes),
        "required_tflite_ops": required_ops,
        "tflite": tflite_info,
        "labels": experiment.all_labels,
        "artifacts": {
            "tflite": str(model_path),
            **c_paths,
        },
    }
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata
