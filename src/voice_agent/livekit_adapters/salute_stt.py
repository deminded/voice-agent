"""LiveKit STT adapter that wraps our SaluteSTT provider.

LiveKit gives us an AudioBuffer (one or more rtc.AudioFrame instances at
some negotiated sample rate, typically 48kHz for WebRTC). We merge it into
a single PCM16 chunk and ship to Salute's `speech:recognize`. The agent
runtime owns the VAD, segmentation, and turn detection — we don't.

This is non-streaming MVP: every utterance is one full Salute call. Salute
also has a websocket streaming endpoint we can swap in later for partial
transcripts and lower latency.
"""
from __future__ import annotations

import uuid

from livekit.agents import APIConnectOptions, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from voice_agent.providers.salute import SaluteSTT as _SaluteSTT


class SaluteSTTAdapter(stt.STT):
    def __init__(self, *, language: str = "ru-RU") -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False),
        )
        self._inner = _SaluteSTT()
        self._language = language

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        # merge_frames concatenates rtc.AudioFrame[] into one frame; .data is
        # raw PCM16 little-endian at the frame's sample_rate.
        merged = utils.merge_frames(buffer)
        pcm_bytes = bytes(merged.data)
        sample_rate = merged.sample_rate

        if sample_rate == 16000:
            # Salute's pcm16 MIME pin is rate=16000 — feed the bytes as-is.
            text = await self._inner.transcribe(pcm_bytes, format="pcm16", language=self._language)
        else:
            # Resample to 16k for Salute's PCM endpoint, or send as opus container.
            # Simpler path: encode as wav16 (Salute accepts arbitrary sample rate
            # in WAV header form). But our SaluteSTT MIME map doesn't list wav16,
            # so we resample to 16k via livekit utils.
            resampled = _resample_to_16k(pcm_bytes, sample_rate)
            text = await self._inner.transcribe(resampled, format="pcm16", language=self._language)

        actual_lang = language if language is not NOT_GIVEN else self._language
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[stt.SpeechData(language=actual_lang, text=text)],
        )


def _resample_to_16k(pcm_bytes: bytes, src_rate: int) -> bytes:
    """Linear resampler; not perfect but Salute STT tolerates plenty of distortion.

    Replace with audioop or numpy when we care about quality."""
    import audioop
    out, _ = audioop.ratecv(pcm_bytes, 2, 1, src_rate, 16000, None)
    return out
