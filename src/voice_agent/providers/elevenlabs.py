"""ElevenLabs TTS provider.

ElevenLabs blocks RU-attributed IPs at the policy level. Routing requests
through a SOCKS5 proxy (e.g. an SSH dynamic forward to a non-blocked VPS)
restores access. Set ELEVENLABS_PROXY in .env to enable.

Voice settings stay configurable per provider instance — `alive` profile
(stability=0.3, style=0.4) gives the "thinking-out-loud" prosody we picked
on Voice Design preview 3.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import httpx

from voice_agent.config import settings


@dataclass(frozen=True)
class VoiceSettings:
    stability: float = 0.3
    similarity_boost: float = 0.75
    style: float = 0.4
    use_speaker_boost: bool = True

    def as_payload(self) -> dict:
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }


# Output formats Salute and ElevenLabs both speak — names match our protocol.
_EL_FORMAT = {
    "opus": "opus_48000_96",
    "mp3": "mp3_44100_128",
    "pcm16": "pcm_16000",
}


class ElevenLabsTTS:
    """text -> audio bytes via ElevenLabs Multilingual v2."""

    def __init__(
        self,
        *,
        voice_id: str | None = None,
        voice_settings: VoiceSettings | None = None,
        api_key: str | None = None,
        proxy: str | None = None,
        model_id: str = "eleven_multilingual_v2",
    ) -> None:
        self._voice_id = voice_id or settings.elevenlabs_voice_id
        self._voice_settings = voice_settings or VoiceSettings()
        self._api_key = api_key or os.environ["ELEVENLABS_API_KEY"]
        self._proxy = proxy or settings.elevenlabs_proxy
        self._model_id = model_id

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        format: str = "opus",
        ssml: bool = False,
    ) -> bytes:
        if ssml:
            raise NotImplementedError("ElevenLabs has no native SSML; pass plain text")
        out_format = _EL_FORMAT[format]
        async with httpx.AsyncClient(proxy=self._proxy, timeout=60.0) as c:
            r = await c.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice or self._voice_id}",
                params={"output_format": out_format},
                headers={
                    "xi-api-key": self._api_key,
                    "Content-Type": "application/json",
                    "Accept": f"audio/{format}",
                },
                json={
                    "text": text,
                    "model_id": self._model_id,
                    "voice_settings": self._voice_settings.as_payload(),
                },
            )
            r.raise_for_status()
        return r.content
