"""LiveKit TTS adapter that wraps our SaluteTTS provider.

We request pcm16 from Salute (raw 16-bit LE PCM at 24kHz, mono for Tur_24000)
and push it as a single segment into LiveKit's AudioEmitter. Streaming
synthesis (chunked output as Claude generates) is a follow-up — for first
working live diалог it's fine to deliver the whole utterance at once.
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
        # Request raw PCM directly so we don't decode opus on the way back.
        salute: _SaluteTTS = self._tts._inner  # type: ignore[attr-defined]
        voice: str = self._tts._voice  # type: ignore[attr-defined]
        audio = await salute.synthesize(self._input_text, voice=voice, format="pcm16")

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )
        output_emitter.push(audio)
        output_emitter.flush()
