"""Tests for torch2bt.codegen — source file generation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch.nn as nn

from torch2bt import codegen
from torch2bt.inspector import inspect_model
from torch2bt.models import Optimization, PackageConfig
from torch2bt.subnets import get_subnet_protocol


class _DummyModel(nn.Module):
    def forward(self, x):
        return x


@pytest.fixture()
def sn1_protocol():
    return get_subnet_protocol(1)


@pytest.fixture()
def sn18_protocol():
    return get_subnet_protocol(18)


@pytest.fixture()
def dummy_signature():
    return inspect_model(_DummyModel())


@pytest.fixture()
def base_config(tmp_path: Path):
    return PackageConfig(
        model=_DummyModel(),
        target_subnet=1,
        optimization=Optimization.FP16,
        wallet_name="test_wallet",
        output_dir=tmp_path / "output",
    )


def test_generate_protocol_contains_synapse_class(dummy_signature, sn1_protocol) -> None:
    src = codegen.generate_protocol(dummy_signature, sn1_protocol)
    assert "class Prompting(bt.Synapse):" in src
    assert "roles: list[str] | None = None" in src
    assert "completion: str | None = None" in src
    assert "def deserialize" in src


def test_generate_protocol_sn18(dummy_signature, sn18_protocol) -> None:
    src = codegen.generate_protocol(dummy_signature, sn18_protocol)
    assert "class ImageResponse(bt.Synapse):" in src
    assert "prompt: str | None = None" in src
    assert "image_data: list[float] | None" in src


def test_generate_miner_contains_wallet(dummy_signature, sn1_protocol, base_config) -> None:
    src = codegen.generate_miner(dummy_signature, sn1_protocol, base_config)
    assert 'WALLET_NAME = "test_wallet"' in src
    assert "NETUID = 1" in src
    assert "PRECISION: torch.dtype = torch.float16" in src


def test_generate_miner_has_blacklist(dummy_signature, sn1_protocol, base_config) -> None:
    src = codegen.generate_miner(dummy_signature, sn1_protocol, base_config)
    assert "def blacklist" in src
    assert "MIN_VALIDATOR_STAKE" in src


def test_generate_dockerfile_contains_base_image(base_config, sn1_protocol) -> None:
    src = codegen.generate_dockerfile(base_config, sn1_protocol)
    assert "FROM pytorch/pytorch" in src
    assert "ENTRYPOINT" in src
    assert "NETUID=1" in src


def test_generate_uv_project_fp32(sn1_protocol) -> None:
    src = codegen.generate_uv_project(sn1_protocol, Optimization.FP32)
    assert "torch2bt-miner-sn1" in src
    assert "bitsandbytes" not in src


def test_generate_uv_project_int4_adds_bitsandbytes(sn1_protocol) -> None:
    src = codegen.generate_uv_project(sn1_protocol, Optimization.INT4)
    assert "bitsandbytes" in src


def test_write_package_creates_files(
    dummy_signature,
    sn1_protocol,
    base_config,
    tmp_path: Path,
) -> None:
    out = tmp_path / "pkg"
    protocol_src = codegen.generate_protocol(dummy_signature, sn1_protocol)
    miner_src = codegen.generate_miner(dummy_signature, sn1_protocol, base_config)
    dockerfile_src = codegen.generate_dockerfile(base_config, sn1_protocol)
    uv_src = codegen.generate_uv_project(sn1_protocol, Optimization.FP16)

    paths = codegen.write_package(out, protocol_src, miner_src, dockerfile_src, uv_src)

    assert paths["protocol"].exists()
    assert paths["miner"].exists()
    assert paths["dockerfile"].exists()
    assert paths["uv_project"].exists()
    assert paths["protocol"].read_text(encoding="utf-8") == protocol_src
