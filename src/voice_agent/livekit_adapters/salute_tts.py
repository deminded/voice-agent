"""LiveKit TTS adapter that wraps our SaluteTTS provider.

We request pcm16 from Salute (raw 16-bit LE PCM at 24kHz, mono for Tur_24000)
and stream the response body into LiveKit's AudioEmitter chunk-by-chunk, so
playback starts as soon as the first bytes arrive (TTFB latency, not full
synthesis latency).
"""
from __future__ import annotations

import uuid

from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import NotGivenOr

from voice_agent.providers.salute import SaluteTTS as _SaluteTTS


SAMPLE_RATE = 24000  # Tur_24000 native rate; other voices are also _24000 named


class SaluteTTSAdapter(tts.TTS):
    def __init__(self, *, voice: str = "Tur_24000") -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._inner = _SaluteTTS()
        self._voice = voice

    def synthesize(
        self,
        text: str,
        *,
        conn_options: NotGivenOr[APIConnectOptions] = ...,
    ) -> tts.ChunkedStream:
        return _SaluteChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options if conn_options is not ... else APIConnectOptions(),
        )


class _SaluteChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        salute: _SaluteTTS = self._tts._inner  # type: ignore[attr-defined]
        voice: str = self._tts._voice  # type: ignore[attr-defined]

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        # Push each HTTP response chunk immediately so playback starts at TTFB.
        async for chunk in salute.synthesize_stream(self._input_text, voice=voice, format="pcm16"):
            output_emitter.push(chunk)
        output_emitter.flush()
