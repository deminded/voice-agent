"""Single-turn voice pipeline:

    audio bytes -> STT -> LLM.reply -> TTS -> audio bytes

Built around protocols, so providers swap freely. One Conversation per
WebSocket connection — instances hold no per-turn state today, but this
is where memory/system prompt will land once we plug Claude.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from voice_agent.providers.base import STTProvider, TTSProvider


class LLMProvider(Protocol):
    async def reply(self, user_text: str) -> str: ...


@dataclass
class TurnResult:
    transcript: str
    response_text: str
    response_audio: bytes


class Conversation:
    def __init__(
        self,
        stt: STTProvider,
        llm: LLMProvider,
        tts: TTSProvider,
        *,
        voice: str = "Tur_24000",
        out_format: str = "opus",
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._voice = voice
        self._out_format = out_format

    @property
    def out_format(self) -> str:
        return self._out_format

    async def turn(self, audio: bytes, *, in_format: str = "opus") -> TurnResult:
        transcript = await self._stt.transcribe(audio, format=in_format)
        response_text = await self._llm.reply(transcript)
        response_audio = await self._tts.synthesize(
            response_text, voice=self._voice, format=self._out_format
        )
        return TurnResult(
            transcript=transcript,
            response_text=response_text,
            response_audio=response_audio,
        )
