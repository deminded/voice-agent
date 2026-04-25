"""Unit tests for SessionRecorder — transcript persistence and synthesis dispatch.

Synthesis itself uses a mocked anthropic client; we don't burn API credit in tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from voice_agent import session_recorder as sr
from voice_agent.session_recorder import SessionRecorder


@pytest.fixture
def tmp_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(sr, "SESSIONS_ROOT", tmp_path)
    monkeypatch.setattr(sr, "REFLECTIONS_DIR", tmp_path / "reflections")
    return tmp_path


def _msg(role: str, text: str) -> SimpleNamespace:
    """Approximate a livekit ChatMessage: role + content list of str-or-text-objects."""
    return SimpleNamespace(role=role, content=[text])


def _msg_obj(role: str, text: str) -> SimpleNamespace:
    """ChatMessage variant where content items are wrapping objects with .text attr."""
    return SimpleNamespace(role=role, content=[SimpleNamespace(text=text)])


def test_extract_text_handles_str_content() -> None:
    text = SessionRecorder._extract_text(["привет", "Евгений"])
    assert text == "привет Евгений"


def test_extract_text_handles_text_objects() -> None:
    items = [SimpleNamespace(text="да"), SimpleNamespace(text="нет")]
    assert SessionRecorder._extract_text(items) == "да нет"


def test_extract_text_returns_empty_for_no_content() -> None:
    assert SessionRecorder._extract_text(None) == ""
    assert SessionRecorder._extract_text([]) == ""


def test_record_item_writes_jsonl(tmp_sessions: Path) -> None:
    rec = SessionRecorder(room_name="room")
    rec._record_item(_msg("user", "привет"), 1700000000.0)
    rec._record_item(_msg_obj("assistant", "слушаю"), 1700000001.5)

    lines = rec.transcript_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    a, b = (json.loads(l) for l in lines)
    assert a == {"role": "user", "text": "привет", "ts": 1700000000.0}
    assert b == {"role": "assistant", "text": "слушаю", "ts": 1700000001.5}


def test_record_item_skips_empty_text(tmp_sessions: Path) -> None:
    rec = SessionRecorder()
    rec._record_item(_msg("user", "   "), 1.0)
    assert rec._items == []
    assert not rec.transcript_file.exists()


def test_record_item_skips_items_without_role(tmp_sessions: Path) -> None:
    rec = SessionRecorder()
    handoff = SimpleNamespace(target="other_agent")  # no role/content
    rec._record_item(handoff, 1.0)
    assert rec._items == []


def test_split_title_body_strips_markdown_heading() -> None:
    title, body = SessionRecorder._split_title_body("# О собаках\n\nВот тело рефлексии.")
    assert title == "О собаках"
    assert body == "Вот тело рефлексии."


def test_split_title_body_strips_quotes() -> None:
    title, _ = SessionRecorder._split_title_body('«О собаках»\n\nтело')
    assert title == "О собаках"


def test_split_title_body_falls_back_when_empty() -> None:
    title, body = SessionRecorder._split_title_body("")
    assert title == "Голосовая сессия"
    assert body == ""


@pytest.mark.asyncio
async def test_finalize_skips_session_with_too_few_items(tmp_sessions: Path) -> None:
    rec = SessionRecorder()
    rec._record_item(_msg("user", "привет"), 1.0)
    rec._record_item(_msg("assistant", "и тебе"), 2.0)  # only 2 items, threshold is 4

    with patch("voice_agent.session_recorder.AsyncAnthropic") as mock_client:
        await rec.finalize(reason="test")
        mock_client.assert_not_called()
    # nothing in reflections dir
    assert not (tmp_sessions / "reflections").exists() or not list((tmp_sessions / "reflections").iterdir())


@pytest.mark.asyncio
async def test_finalize_runs_synthesis_and_saves_reflection(
    tmp_sessions: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    rec = SessionRecorder(room_name="testroom")
    for i in range(2):
        rec._record_item(_msg("user", f"вопрос {i}"), float(i * 2))
        rec._record_item(_msg("assistant", f"ответ {i}"), float(i * 2 + 1))

    fake_response = SimpleNamespace(
        content=[SimpleNamespace(text="# О собаках\n\nЛюди и собаки разделяют ритм дня.")]
    )
    fake_messages = SimpleNamespace(create=AsyncMock(return_value=fake_response))
    fake_client = SimpleNamespace(messages=fake_messages)

    with patch("voice_agent.session_recorder.AsyncAnthropic", return_value=fake_client):
        await rec.finalize(reason="test")

    # reflection file written with extracted title
    ref_dir = tmp_sessions / "reflections"
    files = list(ref_dir.glob("*.md"))
    assert len(files) == 1, f"expected one reflection, got {[f.name for f in files]}"
    body = files[0].read_text(encoding="utf-8")
    assert "title: О собаках" in body
    assert f"session: {rec.session_id}" in body
    assert "Люди и собаки разделяют ритм дня." in body

    # synthesis was called with the formatted transcript
    call_args = fake_messages.create.call_args
    assert call_args.kwargs["model"] == sr.SYNTHESIS_MODEL
    user_content = call_args.kwargs["messages"][0]["content"]
    assert "вопрос 0" in user_content
    assert "ответ 1" in user_content


@pytest.mark.asyncio
async def test_finalize_is_idempotent(tmp_sessions: Path) -> None:
    rec = SessionRecorder()
    # leave items list empty, so synthesis short-circuits anyway
    await rec.finalize(reason="first")
    assert rec._closed is True
    # second call must not raise and not retrigger anything
    await rec.finalize(reason="second")
