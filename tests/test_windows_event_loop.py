from __future__ import annotations

import asyncio
import os
import sys

import pytest

from agents import Agent, Runner
from agents.run_internal.agent_runner_helpers import (
    _WINDOWS_EVENT_LOOP_POLICY_ENV_VAR,
    get_or_create_default_event_loop,
)
from agents.testing import (
    ScriptedModel,
    assistant_message,
)

_WINDOWS_REASON = "Windows event loop policy override applies on win32"


@pytest.fixture(autouse=True)
def restore_event_loop_policy():
    """Ensure each test runs with a fresh policy and restores the ambient policy afterwards."""
    original_policy = asyncio.get_event_loop_policy()
    original_env = os.environ.get(_WINDOWS_EVENT_LOOP_POLICY_ENV_VAR)
    yield
    if original_env is not None:
        os.environ[_WINDOWS_EVENT_LOOP_POLICY_ENV_VAR] = original_env
    else:
        os.environ.pop(_WINDOWS_EVENT_LOOP_POLICY_ENV_VAR, None)
    asyncio.set_event_loop_policy(original_policy)


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_windows_selector_policy_override_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting the environment variable switches to WindowsSelectorEventLoopPolicy."""
    monkeypatch.setenv(_WINDOWS_EVENT_LOOP_POLICY_ENV_VAR, "selector")
    loop = get_or_create_default_event_loop()
    try:
        current_policy = asyncio.get_event_loop_policy()
        assert isinstance(current_policy, asyncio.WindowsSelectorEventLoopPolicy)
        expected_loop_type = getattr(
            asyncio, "_WindowsSelectorEventLoop", asyncio.SelectorEventLoop
        )
        assert isinstance(loop, expected_loop_type)
    finally:
        if not loop.is_closed():
            loop.close()


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_windows_selector_policy_override_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the environment variable is not set, the ambient event loop policy is preserved."""
    monkeypatch.delenv(_WINDOWS_EVENT_LOOP_POLICY_ENV_VAR, raising=False)
    initial_policy = asyncio.WindowsProactorEventLoopPolicy()
    asyncio.set_event_loop_policy(initial_policy)

    loop = get_or_create_default_event_loop()
    try:
        current_policy = asyncio.get_event_loop_policy()
        assert isinstance(current_policy, asyncio.WindowsProactorEventLoopPolicy)
    finally:
        if not loop.is_closed():
            loop.close()


@pytest.mark.skipif(sys.platform != "win32", reason=_WINDOWS_REASON)
def test_runner_run_sync_with_selector_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner.run_sync completes successfully under the WindowsSelectorEventLoopPolicy."""
    monkeypatch.setenv(_WINDOWS_EVENT_LOOP_POLICY_ENV_VAR, "selector")

    agent = Agent(
        name="test_agent",
        instructions="Say hello.",
        model=ScriptedModel([[assistant_message("Hello from scripted model!")]]),
    )

    result = Runner.run_sync(agent, "Hi")
    assert result.final_output == "Hello from scripted model!"
