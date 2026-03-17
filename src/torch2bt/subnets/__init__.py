"""Subnet protocol registry for torch2bt Phase A."""

from __future__ import annotations

from torch2bt.models import SubnetProtocol
from torch2bt.subnets.base import BaseSubnet
from torch2bt.subnets.subnet1 import Subnet1
from torch2bt.subnets.subnet18 import Subnet18

_REGISTRY: dict[int, type[BaseSubnet]] = {
    1: Subnet1,
    18: Subnet18,
}

SUPPORTED_SUBNETS: list[int] = list(_REGISTRY.keys())


def get_subnet_protocol(subnet_id: int) -> SubnetProtocol:
    """Retrieve the SubnetProtocol for a given subnet ID.

    Args:
        subnet_id: The Bittensor subnet NetUID.

    Returns:
        The SubnetProtocol for the requested subnet.

    Raises:
        ValueError: If the subnet_id is not supported in Phase A.
    """
    if subnet_id not in _REGISTRY:
        msg = f"Subnet {subnet_id} is not supported in Phase A. Supported subnets: {SUPPORTED_SUBNETS}"  # noqa: E501
        raise ValueError(msg)
    return _REGISTRY[subnet_id]().protocol
