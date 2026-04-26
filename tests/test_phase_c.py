"""Phase C unit tests — per-room mode switching and CLI transcript forwarding.

All tests are pure unit tests: no network, no real LiveKit room, no real
AgentSession. Helpers from livekit_agent are extracted functions (not closures)
so they can be called directly here.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_agent.livekit_agent import (
    handle_set_mode,
    handle_user_input_transcribed,
    post_to_cli,
)


# ---------------------------------------------------------------------------
# Test 1: set_mode data message updates nonlocal_mode
# ---------------------------------------------------------------------------

def test_set_mode_agent_to_cli():
    nonlocal_mode = {"mode": "agent"}
    payload = {"type": "set_mode", "mode": "cli"}
    result = handle_set_mode(payload, nonlocal_mode)
    assert result is True
    assert nonlocal_mode["mode"] == "cli"


def test_set_mode_cli_to_agent():
    nonlocal_mode = {"mode": "cli"}
    payload = {"type": "set_mode", "mode": "agent"}
    result = handle_set_mode(payload, nonlocal_mode)
    assert result is True
    assert nonlocal_mode["mode"] == "agent"


def test_set_mode_ignores_other_types():
    nonlocal_mode = {"mode": "agent"}
    payload = {"type": "external_say", "text": "hello"}
    result = handle_set_mode(payload, nonlocal_mode)
    assert result is False
    assert nonlocal_mode["mode"] == "agent"  # unchanged


def test_set_mode_defaults_to_agent_on_missing_mode():
    nonlocal_mode = {"mode": "cli"}
    payload = {"type": "set_mode"}  # no "mode" key
    handle_set_mode(payload, nonlocal_mode)
    assert nonlocal_mode["mode"] == "agent"


# ---------------------------------------------------------------------------
# Test 2: CLI mode — POST sent on final transcript (LLM is silenced via llm_node)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cli_mode_triggers_post():
    nonlocal_mode = {"mode": "cli"}
    session = MagicMock()
    ev = SimpleNamespace(transcript="привет", is_final=True)

    tasks = []

    def fake_create_task(coro):
        loop = asyncio.get_event_loop()
        t = loop.create_task(coro)
        tasks.append(t)
        return t

    with patch("voice_agent.livekit_agent.asyncio.create_task", side_effect=fake_create_task):
        with patch("voice_agent.livekit_agent.post_to_cli", new_callable=AsyncMock) as mock_post:
            handle_user_input_transcribed(ev, nonlocal_mode, session)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            mock_post.assert_awaited_once_with("привет")

    # Worker LLM is silenced by _Assistant.llm_node — interrupt is no longer called.
    # Only post_to_cli runs as a task.
    assert len(tasks) == 1
    session.interrupt.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: agent mode — hook does nothing
# ---------------------------------------------------------------------------

def test_agent_mode_does_nothing():
    nonlocal_mode = {"mode": "agent"}
    session = MagicMock()
    ev = SimpleNamespace(transcript="hello", is_final=True)

    with patch("voice_agent.livekit_agent.asyncio.create_task") as mock_task:
        handle_user_input_transcribed(ev, nonlocal_mode, session)
        mock_task.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: is_final=False — POST does not happen
# ---------------------------------------------------------------------------

def test_cli_mode_non_final_transcript_ignored():
    nonlocal_mode = {"mode": "cli"}
    session = MagicMock()
    ev = SimpleNamespace(transcript="привет", is_final=False)

    with patch("voice_agent.livekit_agent.asyncio.create_task") as mock_task:
        handle_user_input_transcribed(ev, nonlocal_mode, session)
        mock_task.assert_not_called()


def test_cli_mode_empty_transcript_ignored():
    nonlocal_mode = {"mode": "cli"}
    session = MagicMock()
    ev = SimpleNamespace(transcript="   ", is_final=True)

    with patch("voice_agent.livekit_agent.asyncio.create_task") as mock_task:
        handle_user_input_transcribed(ev, nonlocal_mode, session)
        mock_task.assert_not_called()


# ---------------------------------------------------------------------------
# Test 5: post_to_cli — errors are logged, not raised
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_to_cli_handles_network_error():
    """A connection error must be caught and logged, not propagated."""
    import httpx

    with patch("voice_agent.livekit_agent.httpx.AsyncClient") as MockClient:
        instance = AsyncMock()
        instance.post.side_effect = httpx.ConnectError("refused")
        MockClient.return_value.__aenter__ = AsyncMock(return_value=instance)
        MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
        # Should not raise
        await post_to_cli("test text")
