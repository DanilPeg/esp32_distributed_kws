from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ExperimentConfig


CONV_HASH_OC = 1337
CONV_HASH_IC = 7919
CONV_HASH_KH = 2971
CONV_HASH_KW = 6151
CONV_HASH_LAYER = 104729

DW_HASH_CH = 1337
DW_HASH_KH = 7919
DW_HASH_KW = 2971
DW_HASH_LAYER = 104729

LINEAR_HASH_A = 1337
LINEAR_HASH_B = 7919
LINEAR_HASH_C = 2971

SIGN_HASH_A = 4099
SIGN_HASH_B = 6151
SIGN_HASH_C = 14887


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


def _layer_codebook_size(
    sizes: tuple[int, ...] | list[int],
    index: int,
    fallback: int,
) -> int:
    if index < len(sizes):
        selected = int(sizes[index])
        if selected <= 0:
            raise ValueError(f"Codebook size must be positive, got {selected}")
        return selected
    return int(fallback)


class AnalyticHashLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        codebook_size: int,
        layer_id: int,
        hash_a: int,
        hash_b: int,
        hash_c: int,
        signed_hash: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.codebook_size = codebook_size
        self.layer_id = layer_id
        self.hash_a = hash_a
        self.hash_b = hash_b
        self.hash_c = hash_c
        self.signed_hash = signed_hash

        self.codebook = nn.Parameter(torch.randn(codebook_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.register_buffer("i_idx", torch.arange(out_dim, dtype=torch.long).view(out_dim, 1), persistent=False)
        self.register_buffer("j_idx", torch.arange(in_dim, dtype=torch.long).view(1, in_dim), persistent=False)

    def hash_indices(self) -> torch.Tensor:
        return (
            (self.i_idx * self.hash_a + self.j_idx * self.hash_b + self.layer_id * self.hash_c)
            % self.codebook_size
        )

    def hash_signs(self) -> torch.Tensor:
        bits = (self.i_idx * SIGN_HASH_A + self.j_idx * SIGN_HASH_B + self.layer_id * SIGN_HASH_C) % 2
        return bits.to(torch.float32).mul_(2.0).sub_(1.0)

    def materialize_weight(self) -> torch.Tensor:
        weight = self.codebook[self.hash_indices()]
        if self.signed_hash:
            weight = weight * self.hash_signs().to(weight.device, dtype=weight.dtype)
        return weight

    def compact_parameter_count(self) -> int:
        return self.codebook.numel() + self.bias.numel()

    def virtual_parameter_count(self) -> int:
        return self.out_dim * self.in_dim

    def export_spec(self) -> dict[str, Any]:
        return {
            "type": "analytic_hash_linear",
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "codebook_size": self.codebook_size,
            "layer_id": self.layer_id,
            "signed_hash": self.signed_hash,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.materialize_weight(), self.bias)


class AnalyticHashConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        codebook_size: int,
        layer_id: int,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        signed_hash: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.codebook_size = codebook_size
        self.layer_id = layer_id
        self.signed_hash = signed_hash

        self.codebook = nn.Parameter(torch.randn(codebook_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.register_buffer("oc_idx", torch.arange(out_channels, dtype=torch.long).view(out_channels, 1, 1, 1), persistent=False)
        self.register_buffer("ic_idx", torch.arange(in_channels, dtype=torch.long).view(1, in_channels, 1, 1), persistent=False)
        self.register_buffer("kh_idx", torch.arange(self.kernel_size[0], dtype=torch.long).view(1, 1, self.kernel_size[0], 1), persistent=False)
        self.register_buffer("kw_idx", torch.arange(self.kernel_size[1], dtype=torch.long).view(1, 1, 1, self.kernel_size[1]), persistent=False)

    def hash_indices(self) -> torch.Tensor:
        return (
            (
                self.oc_idx * CONV_HASH_OC
                + self.ic_idx * CONV_HASH_IC
                + self.kh_idx * CONV_HASH_KH
                + self.kw_idx * CONV_HASH_KW
                + self.layer_id * CONV_HASH_LAYER
            )
            % self.codebook_size
        )

    def hash_signs(self) -> torch.Tensor:
        bits = (
            self.oc_idx * SIGN_HASH_A
            + self.ic_idx * SIGN_HASH_B
            + self.kh_idx * SIGN_HASH_C
            + self.kw_idx * (SIGN_HASH_A + SIGN_HASH_B)
            + self.layer_id * (SIGN_HASH_C + 11)
        ) % 2
        return bits.to(torch.float32).mul_(2.0).sub_(1.0)

    def materialize_weight(self) -> torch.Tensor:
        weight = self.codebook[self.hash_indices()]
        if self.signed_hash:
            weight = weight * self.hash_signs().to(weight.device, dtype=weight.dtype)
        return weight

    def compact_parameter_count(self) -> int:
        return self.codebook.numel() + self.bias.numel()

    def virtual_parameter_count(self) -> int:
        return self.out_channels * self.in_channels * self.kernel_size[0] * self.kernel_size[1]

    def export_spec(self) -> dict[str, Any]:
        return {
            "type": "analytic_hash_conv2d",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "kernel_size": list(self.kernel_size),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "codebook_size": self.codebook_size,
            "layer_id": self.layer_id,
            "signed_hash": self.signed_hash,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.materialize_weight(),
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )


class AnalyticHashDepthwiseConv2d(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int | tuple[int, int],
        codebook_size: int,
        layer_id: int,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        signed_hash: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.codebook_size = codebook_size
        self.layer_id = layer_id
        self.signed_hash = signed_hash

        self.codebook = nn.Parameter(torch.randn(codebook_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(channels))

        self.register_buffer("ch_idx", torch.arange(channels, dtype=torch.long).view(channels, 1, 1), persistent=False)
        self.register_buffer("kh_idx", torch.arange(self.kernel_size[0], dtype=torch.long).view(1, self.kernel_size[0], 1), persistent=False)
        self.register_buffer("kw_idx", torch.arange(self.kernel_size[1], dtype=torch.long).view(1, 1, self.kernel_size[1]), persistent=False)

    def hash_indices(self) -> torch.Tensor:
        return (
            (
                self.ch_idx * DW_HASH_CH
                + self.kh_idx * DW_HASH_KH
                + self.kw_idx * DW_HASH_KW
                + self.layer_id * DW_HASH_LAYER
            )
            % self.codebook_size
        )

    def hash_signs(self) -> torch.Tensor:
        bits = (
            self.ch_idx * SIGN_HASH_A
            + self.kh_idx * SIGN_HASH_B
            + self.kw_idx * SIGN_HASH_C
            + self.layer_id * (SIGN_HASH_A + 29)
        ) % 2
        return bits.to(torch.float32).mul_(2.0).sub_(1.0)

    def materialize_weight(self) -> torch.Tensor:
        weight = self.codebook[self.hash_indices()]
        if self.signed_hash:
            weight = weight * self.hash_signs().to(weight.device, dtype=weight.dtype)
        return weight.unsqueeze(1)

    def compact_parameter_count(self) -> int:
        return self.codebook.numel() + self.bias.numel()

    def virtual_parameter_count(self) -> int:
        return self.channels * self.kernel_size[0] * self.kernel_size[1]

    def export_spec(self) -> dict[str, Any]:
        return {
            "type": "analytic_hash_depthwise_conv2d",
            "channels": self.channels,
            "kernel_size": list(self.kernel_size),
            "stride": list(self.stride),
            "padding": list(self.padding),
            "codebook_size": self.codebook_size,
            "layer_id": self.layer_id,
            "signed_hash": self.signed_hash,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.materialize_weight(),
            self.bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.channels,
        )


class DSCNNBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        depthwise: nn.Module,
        pointwise: nn.Module,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.depthwise = depthwise
        self.bn_dw = nn.BatchNorm2d(channels)
        self.pointwise = pointwise
        self.bn_pw = nn.BatchNorm2d(channels)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = torch.relu(self.bn_dw(self.depthwise(x)))
        x = self.bn_pw(self.pointwise(x))
        if self.residual:
            x = x + shortcut
        x = torch.relu(x)
        return x


class HashDSCNN(nn.Module):
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        num_classes: int,
        stem_codebook_size: int,
        depthwise_codebook_size: int,
        pointwise_codebook_size: int,
        linear_codebook_size: int,
        depthwise_codebook_sizes: tuple[int, ...] | list[int],
        pointwise_codebook_sizes: tuple[int, ...] | list[int],
        signed_hash: bool,
        hash_only_pointwise: bool,
        use_residual: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        layer_id = 0
        self.use_residual = use_residual

        if hash_only_pointwise:
            self.conv0 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1, bias=True)
        else:
            self.conv0 = AnalyticHashConv2d(
                1,
                channels,
                kernel_size=3,
                codebook_size=stem_codebook_size,
                layer_id=layer_id,
                stride=2,
                padding=1,
                signed_hash=signed_hash,
            )
            layer_id += 1
        self.bn0 = nn.BatchNorm2d(channels)

        blocks: list[DSCNNBlock] = []
        for block_index in range(num_blocks):
            if hash_only_pointwise:
                depthwise = nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    groups=channels,
                    bias=True,
                )
            else:
                depthwise = AnalyticHashDepthwiseConv2d(
                    channels,
                    kernel_size=3,
                    codebook_size=_layer_codebook_size(
                        depthwise_codebook_sizes,
                        block_index,
                        depthwise_codebook_size,
                    ),
                    layer_id=layer_id,
                    padding=1,
                    signed_hash=signed_hash,
                )
                layer_id += 1

            pointwise = AnalyticHashConv2d(
                channels,
                channels,
                kernel_size=1,
                codebook_size=_layer_codebook_size(
                    pointwise_codebook_sizes,
                    block_index,
                    pointwise_codebook_size,
                ),
                layer_id=layer_id,
                signed_hash=signed_hash,
            )
            layer_id += 1
            blocks.append(
                DSCNNBlock(
                    channels=channels,
                    depthwise=depthwise,
                    pointwise=pointwise,
                    residual=use_residual,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = AnalyticHashLinear(
            channels,
            num_classes,
            codebook_size=linear_codebook_size,
            layer_id=layer_id,
            hash_a=LINEAR_HASH_A,
            hash_b=LINEAR_HASH_B,
            hash_c=LINEAR_HASH_C,
            signed_hash=signed_hash,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = torch.relu(self.bn0(self.conv0(x)))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.dropout(x)
        return self.fc(x)


class DenseDSCNN(nn.Module):
    def __init__(self, channels: int, num_blocks: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(channels)

        blocks: list[DSCNNBlock] = []
        for _ in range(num_blocks):
            depthwise = nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1,
                groups=channels,
                bias=False,
            )
            pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            blocks.append(DSCNNBlock(channels=channels, depthwise=depthwise, pointwise=pointwise))

        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = torch.relu(self.bn0(self.conv0(x)))
        for block in self.blocks:
            x = block(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.dropout(x)
        return self.fc(x)


def build_model_by_name(model_name: str, experiment: ExperimentConfig, role: str) -> nn.Module:
    if model_name == "hash_dscnn_deeper":
        return HashDSCNN(
            channels=experiment.model.channels,
            num_blocks=experiment.model.num_blocks,
            num_classes=experiment.num_labels,
            stem_codebook_size=experiment.model.stem_codebook_size or experiment.model.codebook_size,
            depthwise_codebook_size=experiment.model.depthwise_codebook_size or experiment.model.codebook_size,
            pointwise_codebook_size=experiment.model.pointwise_codebook_size or experiment.model.codebook_size,
            linear_codebook_size=experiment.model.linear_codebook_size or experiment.model.codebook_size,
            depthwise_codebook_sizes=experiment.model.depthwise_codebook_sizes,
            pointwise_codebook_sizes=experiment.model.pointwise_codebook_sizes,
            signed_hash=experiment.model.signed_hash,
            hash_only_pointwise=experiment.model.hash_only_pointwise,
            use_residual=experiment.model.use_residual,
            dropout=experiment.model.student_dropout,
        )
    if model_name == "dense_dscnn_teacher":
        channels = experiment.model.teacher_channels if role == "teacher" else experiment.model.channels
        num_blocks = experiment.model.teacher_num_blocks if role == "teacher" else experiment.model.num_blocks
        dropout = experiment.model.teacher_dropout if role == "teacher" else experiment.model.student_dropout
        return DenseDSCNN(channels=channels, num_blocks=num_blocks, num_classes=experiment.num_labels, dropout=dropout)
    raise KeyError(f"Unknown model name: {model_name}")


def build_teacher_model(experiment: ExperimentConfig) -> nn.Module | None:
    if not experiment.model.teacher_name:
        return None
    return build_model_by_name(experiment.model.teacher_name, experiment=experiment, role="teacher")


def build_student_model(experiment: ExperimentConfig) -> nn.Module:
    return build_model_by_name(experiment.model.student_name, experiment=experiment, role="student")


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def count_hash_compact_parameters(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if hasattr(module, "compact_parameter_count"):
            total += int(module.compact_parameter_count())
    return total


def count_virtual_dense_parameters(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if hasattr(module, "virtual_parameter_count"):
            total += int(module.virtual_parameter_count())
    return total


def collect_layer_inventory(model: nn.Module) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        if hasattr(module, "export_spec"):
            payload = dict(module.export_spec())
            payload["name"] = name
            payload["compact_parameters"] = int(module.compact_parameter_count())
            payload["virtual_parameters"] = int(module.virtual_parameter_count())
            inventory.append(payload)
        elif isinstance(module, nn.Conv2d):
            inventory.append(
                {
                    "name": name,
                    "type": "conv2d",
                    "in_channels": module.in_channels,
                    "out_channels": module.out_channels,
                    "kernel_size": list(module.kernel_size),
                    "stride": list(module.stride),
                    "padding": list(module.padding),
                    "groups": module.groups,
                }
            )
        elif isinstance(module, nn.Linear):
            inventory.append(
                {
                    "name": name,
                    "type": "linear",
                    "in_features": module.in_features,
                    "out_features": module.out_features,
                }
            )
    return inventory


def estimate_maccs(model: nn.Module, input_shape: tuple[int, int, int]) -> int:
    total = 0
    handles = []

    def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal total
        if not isinstance(output, torch.Tensor):
            return
        if isinstance(module, (nn.Conv2d, AnalyticHashConv2d)):
            in_tensor = inputs[0]
            out_h, out_w = int(output.shape[-2]), int(output.shape[-1])
            kh, kw = _pair(module.kernel_size)  # type: ignore[arg-type]
            groups = int(getattr(module, "groups", 1))
            out_channels = int(output.shape[1])
            in_channels = int(in_tensor.shape[1])
            total += out_h * out_w * out_channels * (in_channels // groups) * kh * kw
        elif isinstance(module, AnalyticHashDepthwiseConv2d):
            out_h, out_w = int(output.shape[-2]), int(output.shape[-1])
            kh, kw = module.kernel_size
            total += out_h * out_w * module.channels * kh * kw
        elif isinstance(module, nn.Linear):
            total += int(module.in_features) * int(module.out_features)
        elif isinstance(module, AnalyticHashLinear):
            total += int(module.in_dim) * int(module.out_dim)

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, AnalyticHashConv2d, AnalyticHashDepthwiseConv2d, AnalyticHashLinear)):
            handles.append(module.register_forward_hook(hook))

    device = next(model.parameters()).device
    dummy = torch.zeros((1, *input_shape), dtype=torch.float32, device=device)
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(dummy)
    if was_training:
        model.train()
    for handle in handles:
        handle.remove()
    return int(total)


def summarize_model(model: nn.Module, experiment: ExperimentConfig) -> dict[str, Any]:
    return {
        "trainable_parameters": count_parameters(model),
        "hash_compact_parameters": count_hash_compact_parameters(model),
        "virtual_dense_parameters": count_virtual_dense_parameters(model),
        "maccs_rough": estimate_maccs(model, experiment.model_input_shape),
        "layer_inventory": collect_layer_inventory(model),
    }
