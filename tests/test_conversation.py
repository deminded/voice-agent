"""Pipeline-level tests using mock providers. No network."""
from __future__ import annotations

import json
import pytest

from voice_agent.conversation import Conversation
from voice_agent.providers.echo import EchoLLM


class MockSTT:
    def __init__(self, text: str) -> None:
        self.text = text
        self.last_format: str | None = None

    async def transcribe(self, audio: bytes, *, format: str = "opus") -> str:
        self.last_format = format
        return self.text


class MockTTS:
    def __init__(self, audio: bytes) -> None:
        self.audio = audio
        self.last_text: str | None = None
        self.last_voice: str | None = None
        self.last_format: str | None = None

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "Tur_24000",
        format: str = "opus",
        ssml: bool = False,
    ) -> bytes:
        self.last_text = text
        self.last_voice = voice
        self.last_format = format
        return self.audio


@pytest.mark.asyncio
async def test_full_turn_flows_format_and_text() -> None:
    stt = MockSTT("привет")
    tts = MockTTS(b"FAKE_OPUS")
    convo = Conversation(stt, EchoLLM(), tts)

    result = await convo.turn(b"raw-pcm-bytes", in_format="pcm16")

    assert stt.last_format == "pcm16"
    assert result.transcript == "привет"
    assert result.response_text == "Я услышал: привет"
    assert tts.last_text == "Я услышал: привет"
    assert tts.last_voice == "Tur_24000"
    assert tts.last_format == "opus"
    assert result.response_audio == b"FAKE_OPUS"


@pytest.mark.asyncio
async def test_empty_transcript_falls_back_to_repeat_prompt() -> None:
    convo = Conversation(MockSTT(""), EchoLLM(), MockTTS(b""))
    result = await convo.turn(b"silence", in_format="pcm16")
    assert "Не услышал" in result.response_text


def test_websocket_round_trip_with_mocks() -> None:
    """Wire mocks into the FastAPI app and prove the WS protocol contract."""
    from fastapi.testclient import TestClient
    from voice_agent.server import app

    stt = MockSTT("тест-реплика")
    tts = MockTTS(b"FAKE_OPUS_REPLY")

    with TestClient(app) as client:
        # swap providers after lifespan ran
        app.state.stt = stt
        app.state.tts = tts
        app.state.llm = EchoLLM()

        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "audio_start", "format": "pcm16"}))
            ws.send_bytes(b"audio-payload")
            ws.send_text(json.dumps({"type": "audio_end"}))

            events: list[dict | bytes] = []
            while True:
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    evt = json.loads(msg["text"])
                    events.append(evt)
                    if evt["type"] in ("audio_end", "error"):
                        break
                elif "bytes" in msg and msg["bytes"]:
                    events.append(msg["bytes"])

    types_in_order = [e["type"] if isinstance(e, dict) else "audio_bytes" for e in events]
    assert types_in_order == [
        "transcript",
        "response_text",
        "audio_start",
        "audio_bytes",
        "audio_end",
    ]
    assert events[0]["text"] == "тест-реплика"
    assert events[1]["text"] == "Я услышал: тест-реплика"
    assert events[2]["format"] == "opus"
    assert events[3] == b"FAKE_OPUS_REPLY"
    assert stt.last_format == "pcm16"
