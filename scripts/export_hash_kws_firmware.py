from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPO_ROOT / "code" / "training"
if str(TRAINING_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAINING_ROOT))

from hash_kws_lab.config import ExperimentConfig, experiment_from_dict
from hash_kws_lab.models import (  # noqa: E402
    AnalyticHashConv2d,
    AnalyticHashDepthwiseConv2d,
    AnalyticHashLinear,
    HashDSCNN,
    build_student_model,
    summarize_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a trained hash-KWS PyTorch bundle into ESP32 firmware arrays.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        required=True,
        help="Path to the .pt bundle produced by the hash KWS lab.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "code" / "firmware" / "hash_kws_runtime",
        help="Directory where hash_model_data.cpp and metadata will be written.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root for resolving data paths.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for calibration batches.",
    )
    parser.add_argument(
        "--calibration-split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
        help="Dataset split used for activation calibration.",
    )
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=8,
        help="How many batches to use for activation scale calibration.",
    )
    parser.add_argument(
        "--firmware-export-snapshot",
        type=Path,
        default=None,
        help=(
            "Optional firmware_export_snapshot.json from a previous Colab run. "
            "When provided, activation_quant_params are reused and dataset calibration is skipped."
        ),
    )
    return parser.parse_args()


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for calibration, but it is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_bundle(bundle_path: Path) -> tuple[ExperimentConfig, dict[str, Any], nn.Module]:
    bundle = torch.load(bundle_path, map_location="cpu")
    experiment_payload = bundle.get("experiment")
    if not isinstance(experiment_payload, dict):
        raise ValueError("Bundle does not contain a serialized experiment payload.")

    experiment = experiment_from_dict(experiment_payload)
    model = build_student_model(experiment)
    state_dict = bundle.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Bundle does not contain a state_dict.")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return experiment, bundle, model


def _ensure_exportable_hash_model(model: nn.Module, experiment: ExperimentConfig) -> HashDSCNN:
    if not isinstance(model, HashDSCNN):
        raise TypeError(f"Expected HashDSCNN student model, got {type(model).__name__}")
    if experiment.model.hash_only_pointwise:
        raise NotImplementedError(
            "Firmware exporter currently supports fully hashed models only. "
            "The pointwise-only recipe needs dense-kernel support in the runtime."
        )
    if not isinstance(model.conv0, AnalyticHashConv2d):
        raise TypeError("Runtime exporter expects a hashed stem convolution.")
    for block in model.blocks:
        if not isinstance(block.depthwise, AnalyticHashDepthwiseConv2d):
            raise TypeError("Runtime exporter expects hashed depthwise blocks.")
        if not isinstance(block.pointwise, AnalyticHashConv2d):
            raise TypeError("Runtime exporter expects hashed pointwise blocks.")
    if not isinstance(model.fc, AnalyticHashLinear):
        raise TypeError("Runtime exporter expects a hashed linear classifier.")
    return model


def _bn_fold(conv_bias: torch.Tensor, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    gamma = bn.weight.detach().cpu().to(torch.float32)
    beta = bn.bias.detach().cpu().to(torch.float32)
    running_mean = bn.running_mean.detach().cpu().to(torch.float32)
    running_var = bn.running_var.detach().cpu().to(torch.float32)
    denom = torch.sqrt(running_var + float(bn.eps))
    post_scale = gamma / denom
    post_bias = beta + post_scale * (conv_bias.detach().cpu().to(torch.float32) - running_mean)
    return post_scale, post_bias


def _quantize_codebook(codebook: torch.Tensor) -> tuple[list[int], float]:
    values = codebook.detach().cpu().to(torch.float32).view(-1)
    max_abs = float(values.abs().max().item())
    if max_abs <= 1e-12:
        return [0 for _ in range(values.numel())], 1.0
    scale = max_abs / 127.0
    quantized = torch.clamp(torch.round(values / scale), min=-127, max=127).to(torch.int8)
    return [int(item) for item in quantized.tolist()], float(scale)


def _activation_forward(model: HashDSCNN, features: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    if features.dim() == 3:
        features = features.unsqueeze(1)
    stages: list[torch.Tensor] = []
    x = features
    x = torch.relu(model.bn0(model.conv0(x)))
    stages.append(x)
    for block in model.blocks:
        shortcut = x
        x = torch.relu(block.bn_dw(block.depthwise(x)))
        stages.append(x)
        x = block.bn_pw(block.pointwise(x))
        if block.residual:
            x = x + shortcut
        x = torch.relu(x)
        stages.append(x)
    return features, stages


@torch.no_grad()
def _collect_activation_quant_params(
    model: HashDSCNN,
    loader: torch.utils.data.DataLoader[Any],
    device: torch.device,
    max_batches: int,
) -> list[dict[str, float]]:
    if max_batches <= 0:
        raise ValueError("calibration-batches must be positive")

    stage_names = ["stem"]
    for block_index in range(len(model.blocks)):
        stage_names.append(f"block{block_index}_depthwise")
        stage_names.append(f"block{block_index}_pointwise")

    model = model.to(device)
    model.eval()
    input_abs_max = 0.0
    stage_abs_max = [0.0 for _ in stage_names]

    for batch_index, (features, _) in enumerate(loader):
        if batch_index >= max_batches:
            break
        features = features.to(device=device, dtype=torch.float32, non_blocking=(device.type == "cuda"))
        input_tensor, stages = _activation_forward(model, features)
        input_abs_max = max(input_abs_max, float(input_tensor.abs().amax().item()))
        for index, stage in enumerate(stages):
            stage_abs_max[index] = max(stage_abs_max[index], float(stage.abs().amax().item()))

    if input_abs_max <= 1e-12:
        input_abs_max = 1.0
    stage_output_scales = [max(value / 127.0, 1e-8) for value in stage_abs_max]
    input_scale = max(input_abs_max / 127.0, 1e-8)

    quant_params: list[dict[str, float]] = []
    previous_scale = input_scale
    for stage_name, output_scale in zip(stage_names, stage_output_scales):
        quant_params.append(
            {
                "stage": stage_name,
                "input_scale": float(previous_scale),
                "output_scale": float(output_scale),
            }
        )
        previous_scale = output_scale
    return quant_params


def _float_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().to(torch.float32).view(-1).tolist()]


def _format_cpp_float(value: float) -> str:
    text = f"{value:.9g}"
    if text in {"0", "-0"}:
        text = "0.0"
    elif all(marker not in text for marker in (".", "e", "E")):
        text = f"{text}.0"
    return f"{text}f"


def _format_cpp_array(name: str, ctype: str, values: list[str], values_per_line: int = 8) -> str:
    lines: list[str] = [f"constexpr {ctype} {name}[] = {{"]
    for start in range(0, len(values), values_per_line):
        chunk = values[start : start + values_per_line]
        lines.append("    " + ", ".join(chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def _emit_conv_layer(layer_name: str, layer: AnalyticHashConv2d, post_scale: list[float], post_bias: list[float]) -> tuple[str, str]:
    quantized_codebook, codebook_scale = _quantize_codebook(layer.codebook)
    codebook_name = f"k{layer_name}Codebook"
    scale_name = f"k{layer_name}PostScale"
    bias_name = f"k{layer_name}PostBias"
    arrays = [
        _format_cpp_array(codebook_name, "int8_t", [str(value) for value in quantized_codebook], values_per_line=16),
        _format_cpp_array(scale_name, "float", [_format_cpp_float(value) for value in post_scale]),
        _format_cpp_array(bias_name, "float", [_format_cpp_float(value) for value in post_bias]),
    ]
    initializer = "\n".join(
        [
            "{",
            f"    {codebook_name},",
            f"    {_format_cpp_float(codebook_scale)},",
            f"    {scale_name},",
            f"    {bias_name},",
            f"    {layer.codebook_size},",
            f"    {layer.in_channels},",
            f"    {layer.out_channels},",
            f"    {layer.kernel_size[0]},",
            f"    {layer.kernel_size[1]},",
            f"    {layer.stride[0]},",
            f"    {layer.stride[1]},",
            f"    {layer.padding[0]},",
            f"    {layer.padding[1]},",
            f"    {layer.layer_id},",
            f"    {'true' if layer.signed_hash else 'false'},",
            "}",
        ]
    )
    return "\n\n".join(arrays), initializer


def _emit_depthwise_layer(
    layer_name: str,
    layer: AnalyticHashDepthwiseConv2d,
    post_scale: list[float],
    post_bias: list[float],
) -> tuple[str, str]:
    quantized_codebook, codebook_scale = _quantize_codebook(layer.codebook)
    codebook_name = f"k{layer_name}Codebook"
    scale_name = f"k{layer_name}PostScale"
    bias_name = f"k{layer_name}PostBias"
    arrays = [
        _format_cpp_array(codebook_name, "int8_t", [str(value) for value in quantized_codebook], values_per_line=16),
        _format_cpp_array(scale_name, "float", [_format_cpp_float(value) for value in post_scale]),
        _format_cpp_array(bias_name, "float", [_format_cpp_float(value) for value in post_bias]),
    ]
    initializer = "\n".join(
        [
            "{",
            f"    {codebook_name},",
            f"    {_format_cpp_float(codebook_scale)},",
            f"    {scale_name},",
            f"    {bias_name},",
            f"    {layer.codebook_size},",
            f"    {layer.channels},",
            f"    {layer.kernel_size[0]},",
            f"    {layer.kernel_size[1]},",
            f"    {layer.stride[0]},",
            f"    {layer.stride[1]},",
            f"    {layer.padding[0]},",
            f"    {layer.padding[1]},",
            f"    {layer.layer_id},",
            f"    {'true' if layer.signed_hash else 'false'},",
            "}",
        ]
    )
    return "\n\n".join(arrays), initializer


def _emit_linear_layer(layer_name: str, layer: AnalyticHashLinear) -> tuple[str, str]:
    quantized_codebook, codebook_scale = _quantize_codebook(layer.codebook)
    codebook_name = f"k{layer_name}Codebook"
    bias_name = f"k{layer_name}Bias"
    arrays = [
        _format_cpp_array(codebook_name, "int8_t", [str(value) for value in quantized_codebook], values_per_line=16),
        _format_cpp_array(bias_name, "float", [_format_cpp_float(value) for value in _float_list(layer.bias)]),
    ]
    initializer = "\n".join(
        [
            "{",
            f"    {codebook_name},",
            f"    {_format_cpp_float(codebook_scale)},",
            f"    {bias_name},",
            f"    {layer.codebook_size},",
            f"    {layer.in_dim},",
            f"    {layer.out_dim},",
            f"    {layer.layer_id},",
            f"    {'true' if layer.signed_hash else 'false'},",
            "}",
        ]
    )
    return "\n\n".join(arrays), initializer


def _build_cpp_source(
    experiment: ExperimentConfig,
    model: HashDSCNN,
    quant_params: list[dict[str, float]],
    source_bundle: Path,
) -> str:
    sections: list[str] = []

    stem_post_scale, stem_post_bias = _bn_fold(model.conv0.bias, model.bn0)
    stem_arrays, stem_initializer = _emit_conv_layer(
        "Stem",
        model.conv0,
        _float_list(stem_post_scale),
        _float_list(stem_post_bias),
    )
    sections.append(stem_arrays)

    depthwise_initializers: list[str] = []
    pointwise_initializers: list[str] = []
    residual_values: list[str] = []
    for block_index, block in enumerate(model.blocks):
        dw_post_scale, dw_post_bias = _bn_fold(block.depthwise.bias, block.bn_dw)
        dw_arrays, dw_initializer = _emit_depthwise_layer(
            f"Block{block_index}Depthwise",
            block.depthwise,
            _float_list(dw_post_scale),
            _float_list(dw_post_bias),
        )
        pw_post_scale, pw_post_bias = _bn_fold(block.pointwise.bias, block.bn_pw)
        pw_arrays, pw_initializer = _emit_conv_layer(
            f"Block{block_index}Pointwise",
            block.pointwise,
            _float_list(pw_post_scale),
            _float_list(pw_post_bias),
        )
        sections.append(dw_arrays)
        sections.append(pw_arrays)
        depthwise_initializers.append(dw_initializer)
        pointwise_initializers.append(pw_initializer)
        residual_values.append("true" if block.residual else "false")

    classifier_arrays, classifier_initializer = _emit_linear_layer("Classifier", model.fc)
    sections.append(classifier_arrays)

    activation_values: list[str] = []
    for quant in quant_params:
        activation_values.append(
            "{"
            f"{_format_cpp_float(quant['input_scale'])}, "
            f"{_format_cpp_float(quant['output_scale'])}"
            "}"
        )
    activations_initializer = ",\n        ".join(activation_values)

    body = [
        '#include "hash_model_data.h"',
        "",
        "namespace hash_kws {",
        "",
        "namespace {",
        "",
        f"// Generated from bundle: {source_bundle.as_posix()}",
        f"// Experiment tag: {experiment.tag}",
        f"// Frontend in training bundle: {experiment.feature.frontend_name}",
        "// Note: runtime input semantics must match the training frontend for accuracy to hold.",
        "",
        "\n\n".join(sections),
        "",
        "}  // namespace",
        "",
        "const HashDscnnModelData g_hash_model = {",
        "    true,",
        f"    {experiment.feature.n_mels},",
        f"    {experiment.feature.frame_count},",
        "    1,",
        f"    {model.conv0.out_channels},",
        f"    {len(model.blocks)},",
        f"    {experiment.num_labels},",
        f"    {stem_initializer},",
        "    {",
        "        " + ",\n        ".join(depthwise_initializers),
        "    },",
        "    {",
        "        " + ",\n        ".join(pointwise_initializers),
        "    },",
        "    {",
        "        " + ",\n        ".join(residual_values),
        "    },",
        f"    {classifier_initializer},",
        "    {",
        f"        {activations_initializer}",
        "    },",
        "};",
        "",
        "}  // namespace hash_kws",
        "",
    ]
    return "\n".join(body)


def _bundle_flash_summary(model: HashDSCNN, experiment: ExperimentConfig, quant_params: list[dict[str, float]]) -> dict[str, Any]:
    model_summary = summarize_model(model, experiment)
    codebook_bytes = 0
    affine_bytes = 0

    codebook_bytes += int(model.conv0.codebook.numel())
    affine_bytes += 4 * (int(model.conv0.out_channels) * 2)
    for block in model.blocks:
        codebook_bytes += int(block.depthwise.codebook.numel())
        affine_bytes += 4 * (int(block.depthwise.channels) * 2)
        codebook_bytes += int(block.pointwise.codebook.numel())
        affine_bytes += 4 * (int(block.pointwise.out_channels) * 2)
    codebook_bytes += int(model.fc.codebook.numel())
    affine_bytes += 4 * int(model.fc.out_dim)

    activation_bytes = len(quant_params) * 8
    return {
        "hash_codebook_bytes_int8": codebook_bytes,
        "post_affine_and_classifier_bias_bytes_float32": affine_bytes,
        "activation_quant_bytes_float32": activation_bytes,
        "approx_runtime_model_bytes": codebook_bytes + affine_bytes + activation_bytes,
        "virtual_dense_int8_weight_bytes": model_summary["virtual_dense_parameters"],
    }


def _metadata_payload(
    experiment: ExperimentConfig,
    bundle: dict[str, Any],
    model: HashDSCNN,
    quant_params: list[dict[str, float]],
    source_bundle: Path,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "source_bundle": str(source_bundle),
        "experiment": experiment.to_dict(),
        "bundle_stage_name": bundle.get("stage_name", "student"),
        "frontend_warning": (
            "The exported runtime preserves model weights and activation scales, "
            "but on-device accuracy still depends on matching the training frontend."
        ),
        "activation_quant_params": quant_params,
        "model_summary": summarize_model(model, experiment),
        "firmware_flash_summary": _bundle_flash_summary(model, experiment, quant_params),
        "runtime_output_dir": str(output_dir),
    }


def export_bundle_to_firmware(
    bundle_path: Path,
    output_dir: Path,
    project_root: Path,
    device: torch.device,
    calibration_split: str = "validation",
    calibration_batches: int = 8,
    activation_quant_params: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    bundle_path = bundle_path.resolve()
    output_dir = output_dir.resolve()
    project_root = project_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiment, bundle, model = _load_bundle(bundle_path)
    model = _ensure_exportable_hash_model(model, experiment)

    if activation_quant_params is None:
        from hash_kws_lab.data import prepare_dataloaders

        calibration = prepare_dataloaders(
            project_root=project_root,
            experiment=experiment,
            device=device,
        )
        loader = calibration["loaders"][calibration_split]
        quant_params = _collect_activation_quant_params(
            model=model,
            loader=loader,
            device=device,
            max_batches=calibration_batches,
        )
    else:
        quant_params = activation_quant_params

    cpp_source = _build_cpp_source(
        experiment=experiment,
        model=model,
        quant_params=quant_params,
        source_bundle=bundle_path,
    )
    cpp_path = output_dir / "hash_model_data.cpp"
    cpp_path.write_text(cpp_source, encoding="utf-8")

    metadata = _metadata_payload(
        experiment=experiment,
        bundle=bundle,
        model=model,
        quant_params=quant_params,
        source_bundle=bundle_path,
        output_dir=output_dir,
    )
    metadata_path = output_dir / "hash_model_export_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "cpp_path": str(cpp_path),
        "metadata_path": str(metadata_path),
        "output_dir": str(output_dir),
        "activation_quant_params": quant_params,
        "firmware_flash_summary": metadata["firmware_flash_summary"],
    }


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    activation_quant_params = None
    if args.firmware_export_snapshot is not None:
        snapshot = json.loads(args.firmware_export_snapshot.read_text(encoding="utf-8"))
        activation_quant_params = snapshot.get("activation_quant_params")
        if not isinstance(activation_quant_params, list):
            raise ValueError(
                "Snapshot does not contain activation_quant_params as a list: "
                f"{args.firmware_export_snapshot}"
            )
    result = export_bundle_to_firmware(
        bundle_path=args.bundle,
        output_dir=args.output_dir,
        project_root=args.project_root,
        device=device,
        calibration_split=args.calibration_split,
        calibration_batches=args.calibration_batches,
        activation_quant_params=activation_quant_params,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
