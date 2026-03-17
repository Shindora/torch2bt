"""Shared data models and type mappings for torch2bt."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import torch.nn as nn


class Optimization(StrEnum):
    """Supported model optimization/quantization strategies."""

    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


DTYPE_DISPLAY: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float64: "float64",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.bool: "bool",
    torch.uint8: "uint8",
}

DTYPE_BT_MAP: dict[torch.dtype, str] = {
    torch.float32: "bt.Tensor",
    torch.float16: "bt.Tensor",
    torch.bfloat16: "bt.Tensor",
    torch.float64: "bt.Tensor",
    torch.int32: "bt.Tensor",
    torch.int64: "bt.Tensor",
    torch.bool: "bt.Tensor",
    torch.uint8: "bt.Tensor",
}

OPTIMIZATION_DTYPE: dict[Optimization, str] = {
    Optimization.FP32: "torch.float32",
    Optimization.FP16: "torch.float16",
    Optimization.BF16: "torch.bfloat16",
    Optimization.INT8: "torch.float16",
    Optimization.INT4: "torch.float16",
}


@dataclass
class TensorSpec:
    """Specification for a single tensor in the model's I/O signature."""

    name: str
    dtype: torch.dtype
    shape: tuple[int | None, ...]
    optional: bool = False

    @property
    def dtype_str(self) -> str:
        """Return a human-readable dtype label."""
        return DTYPE_DISPLAY.get(self.dtype, str(self.dtype))

    @property
    def bt_type(self) -> str:
        """Return the Bittensor type alias for this tensor."""
        return DTYPE_BT_MAP.get(self.dtype, "bt.Tensor")


@dataclass
class ModelSignature:
    """Extracted I/O signature from a PyTorch model's forward pass."""

    inputs: list[TensorSpec]
    outputs: list[TensorSpec]
    model_class_name: str
    model_module: str


@dataclass
class SubnetProtocol:
    """Protocol specification for a Bittensor subnet."""

    subnet_id: int
    name: str
    synapse_class: str
    description: str
    input_spec: dict[str, str]
    output_spec: dict[str, str]
    compatible_optimizations: list[Optimization]


@dataclass
class PackageConfig:
    """Configuration bundle for packaging a model for Bittensor."""

    model: nn.Module
    target_subnet: int
    optimization: Optimization
    wallet_name: str
    output_dir: Path = field(default_factory=lambda: Path("./torch2bt_output"))


@dataclass
class PackageResult:
    """Result of a successful t2b.package() call."""

    output_dir: Path
    miner_path: Path
    protocol_path: Path
    dockerfile_path: Path
    uv_project_path: Path
    success: bool
    warnings: list[str] = field(default_factory=list)
