"""Tests for the MockValidator local testing harness."""

from __future__ import annotations

import asyncio

import pytest

from torch2bt.testing import MockSynapse, MockValidator


async def _echo_forward(synapse: MockSynapse) -> MockSynapse:
    """Simple forward that echoes input messages back as completion."""
    messages = synapse.messages or []
    synapse.completion = " ".join(messages)
    return synapse


async def _slow_forward(synapse: MockSynapse) -> MockSynapse:
    await asyncio.sleep(5)
    return synapse


def test_mock_synapse_field_proxy() -> None:
    """MockSynapse proxies attribute reads/writes through its fields dict."""
    s = MockSynapse()
    s.completion = "hello"
    assert s.completion == "hello"
    assert s.fields["completion"] == "hello"


def test_mock_synapse_missing_attr_returns_none() -> None:
    s = MockSynapse()
    assert s.nonexistent is None


@pytest.mark.asyncio
async def test_query_returns_populated_synapse() -> None:
    validator = MockValidator("Prompting", subnet_id=1, forward_fn=_echo_forward)
    result = await validator.query({"roles": ["user"], "messages": ["hello", "world"]})
    assert result.completion == "hello world"
    assert validator.query_count == 1


@pytest.mark.asyncio
async def test_results_accumulate() -> None:
    validator = MockValidator("Prompting", subnet_id=1, forward_fn=_echo_forward)
    await validator.query({"messages": ["a"]})
    await validator.query({"messages": ["b"]})
    assert len(validator.results) == 2
    assert validator.results[0]["completion"] == "a"
    assert validator.results[1]["completion"] == "b"


@pytest.mark.asyncio
async def test_query_raises_without_forward_fn() -> None:
    validator = MockValidator("Prompting", subnet_id=1)
    with pytest.raises(RuntimeError, match="No forward_fn"):
        await validator.query({"messages": ["test"]})


@pytest.mark.asyncio
async def test_query_timeout_raises() -> None:
    validator = MockValidator("Prompting", subnet_id=1, forward_fn=_slow_forward)
    with pytest.raises(asyncio.TimeoutError):
        await validator.query({"messages": ["test"]}, deadline=0.01)


@pytest.mark.asyncio
async def test_run_test_suite_skips_failed_cases() -> None:
    call_count = 0

    async def _failing_forward(synapse: MockSynapse) -> MockSynapse:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            msg = "simulated failure"
            raise RuntimeError(msg)
        synapse.completion = "ok"
        return synapse

    validator = MockValidator("Prompting", subnet_id=1, forward_fn=_failing_forward)
    results = await validator.run_test_suite(
        [
            {"messages": ["a"]},
            {"messages": ["b"]},
            {"messages": ["c"]},
        ]
    )
    assert len(results) == 2
