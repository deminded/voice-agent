"""SaluteSpeech (Sber) provider — TTS + STT.

Auth: Basic key from `.env` -> oauth -> bearer access_token, refreshed
30s before expiry. One client instance per process; thread-safe via
asyncio.Lock around the refresh.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

from voice_agent.config import settings


@dataclass
class _Token:
    value: str
    expires_at: float  # unix seconds


class SaluteAuth:
    """OAuth bearer-token holder. Refreshes lazily."""

    REFRESH_MARGIN_S = 30.0

    def __init__(self, *, auth_key: str | None = None, scope: str | None = None) -> None:
        self._auth_key = auth_key or settings.salute_auth_key
        self._scope = scope or settings.salute_scope
        self._token: _Token | None = None
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        async with self._lock:
            now = time.time()
            if self._token and self._token.expires_at - self.REFRESH_MARGIN_S > now:
                return self._token.value
            self._token = await self._fetch()
            return self._token.value

    async def _fetch(self) -> _Token:
        async with httpx.AsyncClient(verify=False, timeout=10.0) as c:
            r = await c.post(
                settings.salute_oauth_url,
                headers={
                    "Authorization": f"Basic {self._auth_key}",
                    "RqUID": str(uuid.uuid4()),
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
                data={"scope": self._scope},
            )
            r.raise_for_status()
            payload = r.json()
        return _Token(
            value=payload["access_token"],
            expires_at=payload["expires_at"] / 1000.0,  # ms -> s
        )


class SaluteTTS:
    """text -> audio bytes via SmartSpeech synth."""

    def __init__(self, auth: SaluteAuth | None = None) -> None:
        self._auth = auth or SaluteAuth()

    async def synthesize(
        self,
        text: str,
        *,
        voice: str = "Tur_24000",
        format: str = "opus",
        ssml: bool = False,
    ) -> bytes:
        token = await self._auth.get_token()
        content_type = "application/ssml" if ssml else "application/text"
        async with httpx.AsyncClient(verify=False, timeout=30.0) as c:
            r = await c.post(
                f"{settings.salute_api_url}/text:synthesize",
                params={"format": format, "voice": voice},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": content_type,
                },
                content=text.encode("utf-8"),
            )
            r.raise_for_status()
        return r.content

    async def synthesize_stream(
        self,
        text: str,
        *,
        voice: str = "Tur_24000",
        format: str = "pcm16",
        ssml: bool = False,
        chunk_size: int = 4096,
    ) -> AsyncIterator[bytes]:
        """Yield raw PCM bytes as they arrive from Salute's HTTP response.

        Uses httpx streaming so the caller can push each chunk to LiveKit's
        AudioEmitter immediately — latency drops to TTFB instead of full
        synthesis time.  Errors (4xx/5xx) propagate via raise_for_status
        before iteration begins.
        """
        token = await self._auth.get_token()
        content_type = "application/ssml" if ssml else "application/text"
        async with httpx.AsyncClient(verify=False, timeout=30.0) as c:
            async with c.stream(
                "POST",
                f"{settings.salute_api_url}/text:synthesize",
                params={"format": format, "voice": voice},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": content_type,
                },
                content=text.encode("utf-8"),
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                    if chunk:
                        yield chunk


# Сопоставление наших имён формата с MIME-типами Salute speech:recognize.
# OPUS они принимают как audio/ogg;codecs=opus, PCM 16-bit как audio/x-pcm.
_SALUTE_MIME = {
    "opus": "audio/ogg;codecs=opus",
    "pcm16": "audio/x-pcm;bit=16;rate=16000",
    "mp3": "audio/mp3",
    "flac": "audio/x-flac",
}


class SaluteSTT:
    """audio bytes -> transcript text via SmartSpeech speech:recognize.

    Suitable for short utterances (one push-to-talk segment). For continuous
    streaming Salute exposes a websocket endpoint — out of MVP scope.
    """

    def __init__(self, auth: SaluteAuth | None = None) -> None:
        self._auth = auth or SaluteAuth()

    async def transcribe(
        self,
        audio: bytes,
        *,
        format: str = "opus",
        language: str = "ru-RU",
    ) -> str:
        token = await self._auth.get_token()
        mime = _SALUTE_MIME[format]
        async with httpx.AsyncClient(verify=False, timeout=60.0) as c:
            r = await c.post(
                f"{settings.salute_api_url}/speech:recognize",
                params={"language": language, "model": "general"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": mime,
                },
                content=audio,
            )
            r.raise_for_status()
            payload = r.json()
        # Salute returns: {"status": <int>, "result": ["text1", "text2"], "request_id": "..."}
        results = payload.get("result") or []
        return " ".join(results).strip()
