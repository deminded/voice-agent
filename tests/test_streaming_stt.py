"""Integration smoke test for streaming STT adapter.

We mock the gRPC channel so no real network call is made.
The test verifies:
1. stream() returns a RecognizeStream without raising
2. Pushing audio frames + flush causes events to be emitted
3. At least START_OF_SPEECH, INTERIM_TRANSCRIPT, FINAL_TRANSCRIPT, END_OF_SPEECH are seen
"""
from __future__ import annotations

import asyncio
import math
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from livekit import rtc
from livekit.agents import stt

from voice_agent.grpc_stubs import recognition_pb2


def _make_sine_frame(*, sample_rate: int = 16000, duration_ms: int = 320, freq: float = 440.0) -> rtc.AudioFrame:
    """Generate a PCM16 sine-wave frame — Salute will see legit audio, not silence."""
    n_samples = int(sample_rate * duration_ms / 1000)
    samples = [
        int(32767 * math.sin(2 * math.pi * freq * i / sample_rate))
        for i in range(n_samples)
    ]
    data = struct.pack(f"<{n_samples}h", *samples)
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=1,
        samples_per_channel=n_samples,
    )


def _make_fake_response(text: str, eou: bool) -> recognition_pb2.RecognitionResponse:
    hyp = recognition_pb2.Hypothesis(text=text, normalized_text=text)
    return recognition_pb2.RecognitionResponse(results=[hyp], eou=eou)


class _FakeCall:
    """Async iterator that yields canned responses when consumed."""

    def __init__(self, responses):
        self._responses = iter(responses)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._responses)
        except StopIteration:
            raise StopAsyncIteration


@pytest.mark.asyncio
async def test_streaming_stt_emits_events():
    from voice_agent.livekit_adapters.salute_stt import SaluteSTTAdapter

    adapter = SaluteSTTAdapter(language="ru-RU")
    assert adapter.capabilities.streaming is True
    assert adapter.capabilities.interim_results is True

    fake_responses = [
        _make_fake_response("привет", eou=False),
        _make_fake_response("привет мир", eou=True),
    ]

    # Patch out the gRPC channel so nothing touches the network.
    fake_call = _FakeCall(fake_responses)
    stub_mock = MagicMock()
    stub_mock.Recognize.return_value = fake_call

    with (
        patch("voice_agent.livekit_adapters.salute_stt.SaluteAuth.get_token", new=AsyncMock(return_value="fake-token")),
        patch("voice_agent.livekit_adapters.salute_stt.grpc.aio.secure_channel") as mock_channel_ctx,
    ):
        # secure_channel is used as an async context manager
        mock_channel = AsyncMock()
        mock_channel_ctx.return_value.__aenter__ = AsyncMock(return_value=mock_channel)
        mock_channel_ctx.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("voice_agent.livekit_adapters.salute_stt.recognition_pb2_grpc.SmartSpeechStub", return_value=stub_mock):
            stream = adapter.stream()

            # Push 3 frames then flush to signal end of utterance.
            frame = _make_sine_frame()
            stream.push_frame(frame)
            stream.push_frame(frame)
            stream.push_frame(frame)
            stream.end_input()

            events: list[stt.SpeechEvent] = []
            async for event in stream:
                events.append(event)

    event_types = [e.type for e in events]
    assert stt.SpeechEventType.START_OF_SPEECH in event_types, f"missing START_OF_SPEECH in {event_types}"
    assert stt.SpeechEventType.INTERIM_TRANSCRIPT in event_types, f"missing INTERIM_TRANSCRIPT in {event_types}"
    assert stt.SpeechEventType.FINAL_TRANSCRIPT in event_types, f"missing FINAL_TRANSCRIPT in {event_types}"
    assert stt.SpeechEventType.END_OF_SPEECH in event_types, f"missing END_OF_SPEECH in {event_types}"

    final_events = [e for e in events if e.type == stt.SpeechEventType.FINAL_TRANSCRIPT]
    assert final_events[0].alternatives[0].text == "привет мир"
