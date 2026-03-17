"""torch2bt — PyTorch to Bittensor bridge (v0.1.0-alpha)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from torch2bt.models import Optimization, PackageConfig, PackageResult

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


def package(
    model: nn.Module,
    target_subnet: int,
    optimization: str = "fp32",
    wallet_name: str = "default",
    output_dir: str | Path = "./torch2bt_output",
) -> PackageResult:
    """Package a PyTorch model as a ready-to-deploy Bittensor miner.

    Inspects the model's forward() signature, validates it against the target
    subnet's protocol, then generates miner.py, protocol.py, Dockerfile, and
    pyproject.toml in the output directory.

    Args:
        model: The PyTorch nn.Module to package.
        target_subnet: Bittensor subnet NetUID to target (1 or 18 in Phase A).
        optimization: Precision strategy — "fp32", "fp16", "bf16", "int8", or "int4".
        wallet_name: Bittensor wallet name embedded in the generated miner.
        output_dir: Directory where generated files will be written.

    Returns:
        A PackageResult with paths to all generated artefacts.

    Raises:
        ValueError: If the target_subnet is unsupported or the model is incompatible.
    """
    from torch2bt import codegen, inspector
    from torch2bt.subnets import get_subnet_protocol

    opt = Optimization(optimization.lower())
    output_path = Path(output_dir)

    logger.info("Inspecting model: %s", type(model).__name__)
    signature = inspector.inspect_model(model)

    logger.info("Fetching protocol for SN%d", target_subnet)
    protocol = get_subnet_protocol(target_subnet)

    warnings = inspector.validate_against_subnet(signature, target_subnet)
    for w in warnings:
        logger.warning("[validation] %s", w)

    if not protocol.compatible_optimizations or opt not in protocol.compatible_optimizations:
        logger.warning(
            "[validation] Optimization %s is not recommended for SN%d.",
            opt.value,
            target_subnet,
        )
        warnings.append(
            f"Optimization '{opt.value}' is not in the recommended set for SN{target_subnet}.",
        )

    config = _make_config(model, target_subnet, opt, wallet_name, output_path)

    logger.info("Generating code...")
    protocol_src = codegen.generate_protocol(signature, protocol)
    miner_src = codegen.generate_miner(signature, protocol, config)
    dockerfile_src = codegen.generate_dockerfile(config, protocol)
    uv_project_src = codegen.generate_uv_project(protocol, opt)

    paths = codegen.write_package(
        output_path,
        protocol_src,
        miner_src,
        dockerfile_src,
        uv_project_src,
    )
    logger.info("Package written to: %s", output_path.resolve())

    return PackageResult(
        output_dir=output_path,
        miner_path=paths["miner"],
        protocol_path=paths["protocol"],
        dockerfile_path=paths["dockerfile"],
        uv_project_path=paths["uv_project"],
        success=True,
        warnings=warnings,
    )


def _make_config(
    model: nn.Module,
    target_subnet: int,
    opt: Optimization,
    wallet_name: str,
    output_dir: Path,
) -> PackageConfig:
    """Construct a PackageConfig.

    Args:
        model: The PyTorch model.
        target_subnet: The target subnet NetUID.
        opt: The resolved Optimization enum.
        wallet_name: The wallet name string.
        output_dir: The resolved output Path.

    Returns:
        A fully populated PackageConfig.
    """
    return PackageConfig(
        model=model,
        target_subnet=target_subnet,
        optimization=opt,
        wallet_name=wallet_name,
        output_dir=output_dir,
    )
