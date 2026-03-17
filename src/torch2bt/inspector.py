"""Inspect a PyTorch model's forward pass to extract its I/O signature."""

from __future__ import annotations

import inspect
import logging
from typing import Any, get_type_hints

import torch
import torch.nn as nn
from torch import Tensor  # type: ignore[attr-defined]

from torch2bt.models import ModelSignature, TensorSpec

logger = logging.getLogger(__name__)

_ANNOTATION_DTYPE_MAP: dict[Any, torch.dtype] = {
    Tensor: torch.float32,
    torch.FloatTensor: torch.float32,  # type: ignore[attr-defined]
    torch.HalfTensor: torch.float16,  # type: ignore[attr-defined]
    torch.DoubleTensor: torch.float64,  # type: ignore[attr-defined]
    torch.LongTensor: torch.int64,  # type: ignore[attr-defined]
    torch.IntTensor: torch.int32,  # type: ignore[attr-defined]
    torch.BoolTensor: torch.bool,  # type: ignore[attr-defined]
    torch.ByteTensor: torch.uint8,  # type: ignore[attr-defined]
}


def _resolve_dtype(annotation: Any) -> torch.dtype:  # noqa: ANN401
    """Resolve a type annotation to a torch.dtype.

    Args:
        annotation: A Python type annotation from a forward() parameter or return.

    Returns:
        The best-matching torch.dtype; defaults to float32 for unrecognized types.
    """
    if annotation in _ANNOTATION_DTYPE_MAP:
        return _ANNOTATION_DTYPE_MAP[annotation]

    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        for arg in getattr(annotation, "__args__", ()):
            if isinstance(arg, torch.dtype):
                return arg
            if arg in _ANNOTATION_DTYPE_MAP:
                return _ANNOTATION_DTYPE_MAP[arg]

    if hasattr(annotation, "__metadata__"):
        for meta in annotation.__metadata__:
            if isinstance(meta, torch.dtype):
                return meta

    return torch.float32


def _resolve_shape(annotation: Any) -> tuple[int | None, ...]:  # noqa: ANN401
    """Extract shape metadata from an Annotated[...] hint if present.

    Args:
        annotation: A Python type annotation potentially carrying shape metadata.

    Returns:
        A tuple of ints/None for each dimension; (None,) when unknown.
    """
    if hasattr(annotation, "__metadata__"):
        for meta in annotation.__metadata__:
            if isinstance(meta, tuple) and all(isinstance(d, (int, type(None))) for d in meta):
                return meta
    return (None,)


def inspect_model(model: nn.Module) -> ModelSignature:
    """Inspect a PyTorch model's forward() signature and return a ModelSignature.

    Args:
        model: The PyTorch model to inspect.

    Returns:
        A ModelSignature describing inputs and outputs of the model.

    Raises:
        ValueError: If forward() cannot be inspected.
    """
    try:
        sig = inspect.signature(model.forward)
    except (ValueError, TypeError) as exc:
        msg = f"Cannot inspect {type(model).__name__}.forward: {exc}"
        raise ValueError(msg) from exc

    try:
        hints = get_type_hints(model.forward, include_extras=True)
    except Exception:  # noqa: BLE001
        logger.warning(
            "Could not resolve type hints for %s.forward; assuming untyped tensors.",
            type(model).__name__,
        )
        hints = {}

    inputs: list[TensorSpec] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Tensor

        inputs.append(
            TensorSpec(
                name=name,
                dtype=_resolve_dtype(annotation),
                shape=_resolve_shape(annotation),
                optional=param.default is not inspect.Parameter.empty,
            ),
        )

    outputs: list[TensorSpec] = []
    return_hint = hints.get("return", sig.return_annotation)

    if return_hint not in (inspect.Parameter.empty, None):
        origin = getattr(return_hint, "__origin__", None)
        if origin is tuple:
            for idx, arg in enumerate(getattr(return_hint, "__args__", ())):
                outputs.append(
                    TensorSpec(
                        name=f"output_{idx}",
                        dtype=_resolve_dtype(arg),
                        shape=_resolve_shape(arg),
                    ),
                )
        else:
            outputs.append(
                TensorSpec(
                    name="output",
                    dtype=_resolve_dtype(return_hint),
                    shape=_resolve_shape(return_hint),
                ),
            )

    if not outputs:
        logger.warning(
            "No return annotation on %s.forward; assuming a single float32 Tensor output.",
            type(model).__name__,
        )
        outputs.append(TensorSpec(name="output", dtype=torch.float32, shape=(None,)))

    model_cls = type(model)
    return ModelSignature(
        inputs=inputs,
        outputs=outputs,
        model_class_name=model_cls.__name__,
        model_module=model_cls.__module__,
    )


def validate_against_subnet(signature: ModelSignature, subnet_id: int) -> list[str]:
    """Validate a model signature against a target subnet's protocol requirements.

    Args:
        signature: The inspected model signature.
        subnet_id: The target Bittensor subnet NetUID.

    Returns:
        A list of validation warnings; empty means fully compatible.

    Raises:
        ValueError: If the subnet_id is not supported in Phase A.
    """
    from torch2bt.subnets import get_subnet_protocol

    protocol = get_subnet_protocol(subnet_id)
    warnings: list[str] = []

    if not signature.inputs:
        warnings.append("Model has no detected inputs — synapse will inject raw tensors.")

    if not signature.outputs:
        warnings.append(
            "Model has no detected outputs — output synapse fields will be placeholders.",
        )

    logger.info(
        "Validated %s against %s (SN%d): %d warning(s).",
        signature.model_class_name,
        protocol.name,
        subnet_id,
        len(warnings),
    )
    return warnings
