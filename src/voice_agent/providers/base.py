"""Protocols for swappable STT/TTS providers."""
from typing import Protocol


class TTSProvider(Protocol):
    async def synthesize(
        self,
        text: str,
        *,
        voice: str,
        format: str = "opus",
        ssml: bool = False,
    ) -> bytes:
        """Return audio bytes for the given text. Format: opus|wav|mp3."""
        ...


class STTProvider(Protocol):
    async def transcribe(self, audio: bytes, *, language: str = "ru-RU") -> str:
        """Return transcript text from audio bytes."""
        ...
