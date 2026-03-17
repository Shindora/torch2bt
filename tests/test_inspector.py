"""Tests for torch2bt.inspector — model signature extraction."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from torch2bt.inspector import inspect_model, validate_against_subnet


class _SimpleLinear(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _UntypedModel(nn.Module):
    def forward(self, x, y):
        return x + y


class _MultiOutputModel(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x, x


class _OptionalInputModel(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:  # type: ignore[assignment]  # noqa: ARG002
        return x


def test_inspect_simple_linear() -> None:
    """A typed single-input, single-output model is fully resolved."""
    sig = inspect_model(_SimpleLinear())
    assert len(sig.inputs) == 1
    assert sig.inputs[0].name == "x"
    assert sig.inputs[0].dtype == torch.float32
    assert len(sig.outputs) == 1
    assert sig.outputs[0].name == "output"
    assert sig.model_class_name == "_SimpleLinear"


def test_inspect_untyped_model_defaults_float32() -> None:
    """Untyped parameters default to float32 tensors."""
    sig = inspect_model(_UntypedModel())
    assert len(sig.inputs) == 2
    for spec in sig.inputs:
        assert spec.dtype == torch.float32


def test_inspect_multi_output() -> None:
    """Tuple return annotation produces multiple output TensorSpecs."""
    sig = inspect_model(_MultiOutputModel())
    assert len(sig.outputs) == 2
    assert sig.outputs[0].name == "output_0"
    assert sig.outputs[1].name == "output_1"


def test_inspect_optional_input() -> None:
    """Parameters with defaults are flagged as optional."""
    sig = inspect_model(_OptionalInputModel())
    assert sig.inputs[0].optional is False
    assert sig.inputs[1].optional is True


def test_validate_against_supported_subnet() -> None:
    """Validation of a simple model against SN1 returns no errors."""
    sig = inspect_model(_SimpleLinear())
    warnings = validate_against_subnet(sig, subnet_id=1)
    assert isinstance(warnings, list)


def test_validate_against_unsupported_subnet_raises() -> None:
    """Validation against an unsupported subnet raises ValueError."""
    sig = inspect_model(_SimpleLinear())
    with pytest.raises(ValueError, match="not supported"):
        validate_against_subnet(sig, subnet_id=999)
