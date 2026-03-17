"""Mock Validator for local testing of torch2bt-generated miners."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

ForwardFn = Callable[["MockSynapse"], Coroutine[Any, Any, "MockSynapse"]]


@dataclass
class MockSynapse:
    """Generic synapse stand-in for local testing without a live Bittensor network.

    Attributes access falls through to the internal fields dict, mirroring how
    real bt.Synapse instances expose dynamic attributes.
    """

    fields: dict[str, Any] = field(default_factory=dict)
    dendrite_hotkey: str = "5FakeMockValidatorHotkey1111111111111111111"

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        """Proxy attribute reads to the fields dict."""
        if name in ("fields", "dendrite_hotkey"):
            return object.__getattribute__(self, name)
        try:
            return self.fields[name]
        except KeyError:
            return None

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Proxy attribute writes to the fields dict, except for core attrs."""
        if name in ("fields", "dendrite_hotkey"):
            object.__setattr__(self, name, value)
        else:
            self.fields[name] = value


class MockValidator:
    """Simulates a Bittensor validator for local harness testing.

    Example:
        async def my_forward(synapse: MockSynapse) -> MockSynapse:
            synapse.completion = "pong"
            return synapse

        validator = MockValidator("Prompting", subnet_id=1, forward_fn=my_forward)
        result = await validator.query({"roles": ["user"], "messages": ["ping"]})
        assert result.completion == "pong"
    """

    def __init__(
        self,
        synapse_class: str,
        subnet_id: int,
        forward_fn: ForwardFn | None = None,
        default_timeout: float = 10.0,
    ) -> None:
        """Initialise the mock validator.

        Args:
            synapse_class: Name of the synapse class under test.
            subnet_id: The subnet NetUID being tested against.
            forward_fn: Async callable that acts as the miner's forward function.
            default_timeout: Default per-query timeout in seconds.
        """
        self.synapse_class = synapse_class
        self.subnet_id = subnet_id
        self.forward_fn = forward_fn
        self.default_timeout = default_timeout
        self._query_count: int = 0
        self._results: list[dict[str, Any]] = []

    async def query(
        self,
        inputs: dict[str, Any],
        deadline: float | None = None,
    ) -> MockSynapse:
        """Send a single mock query through the miner's forward function.

        Args:
            inputs: Input field values to populate into the synapse.
            deadline: Per-query timeout override in seconds.

        Returns:
            The MockSynapse returned by the miner with output fields set.

        Raises:
            RuntimeError: If no forward_fn has been configured.
            asyncio.TimeoutError: If the forward call exceeds the timeout.
        """
        if self.forward_fn is None:
            msg = (
                "No forward_fn configured — pass an async miner forward function to MockValidator."
            )
            raise RuntimeError(msg)

        synapse = MockSynapse(fields=dict(inputs))
        self._query_count += 1
        logger.info(
            "[MockValidator] Query #%d → %s (SN%d) inputs=%s",
            self._query_count,
            self.synapse_class,
            self.subnet_id,
            list(inputs.keys()),
        )

        timeout_secs = deadline if deadline is not None else self.default_timeout
        async with asyncio.timeout(timeout_secs):
            result = await self.forward_fn(synapse)
        self._results.append(dict(result.fields))
        return result

    async def run_test_suite(
        self,
        test_cases: list[dict[str, Any]],
        deadline: float | None = None,
    ) -> list[MockSynapse]:
        """Run a batch of test inputs sequentially through the miner.

        Args:
            test_cases: List of input field dicts to send as separate queries.
            deadline: Per-query timeout override in seconds.

        Returns:
            List of MockSynapse results for every successful test case.
        """
        results: list[MockSynapse] = []
        for idx, case in enumerate(test_cases):
            logger.info("[MockValidator] Test case %d/%d", idx + 1, len(test_cases))
            try:
                results.append(await self.query(case, deadline=deadline))
            except Exception:
                logger.exception("[MockValidator] Test case %d failed.", idx + 1)
        return results

    @property
    def query_count(self) -> int:
        """Total number of queries dispatched."""
        return self._query_count

    @property
    def results(self) -> list[dict[str, Any]]:
        """Snapshot of all output field dicts returned by the miner."""
        return list(self._results)
