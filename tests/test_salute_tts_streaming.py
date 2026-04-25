"""Unit tests for Salute streaming TTS — provider-side iteration and adapter wiring.

Httpx is mocked so we don't talk to Salute. Auth is bypassed.
"""
from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from voice_agent.providers.salute import SaluteTTS


class _FakeAsyncStream:
    """Stand-in for httpx.AsyncClient.stream() return value."""

    def __init__(self, chunks: list[bytes], status: int = 200) -> None:
        self._chunks = chunks
        self._status = status

    async def __aenter__(self) -> "_FakeAsyncStream":
        return self

    async def __aexit__(self, *exc) -> None:  # noqa: ANN001
        return None

    def raise_for_status(self) -> None:
        if self._status >= 400:
            raise httpx.HTTPStatusError(
                f"status {self._status}",
                request=httpx.Request("POST", "https://x"),
                response=httpx.Response(self._status),
            )

    async def aiter_bytes(self, chunk_size: int = 4096):  # noqa: ANN201
        for c in self._chunks:
            yield c


class _FakeAsyncClient:
    """Stand-in for httpx.AsyncClient — only the stream() method matters here."""

    def __init__(self, chunks: list[bytes], status: int = 200, **_kw) -> None:
        self._chunks = chunks
        self._status = status
        self.last_request: dict | None = None

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc) -> None:  # noqa: ANN001
        return None

    def stream(self, method: str, url: str, **kwargs) -> _FakeAsyncStream:  # noqa: ANN201
        self.last_request = {"method": method, "url": url, **kwargs}
        return _FakeAsyncStream(self._chunks, self._status)


def _patch_salute_httpx(chunks: list[bytes], status: int = 200):
    """Patch httpx.AsyncClient inside salute provider, return the captured client."""
    captured: dict = {}

    def factory(**kwargs):
        c = _FakeAsyncClient(chunks, status, **kwargs)
        captured["client"] = c
        return c

    return patch("voice_agent.providers.salute.httpx.AsyncClient", side_effect=factory), captured


@pytest.fixture
def fake_auth() -> AsyncMock:
    auth = AsyncMock()
    auth.get_token = AsyncMock(return_value="fake-bearer-token")
    return auth


@pytest.mark.asyncio
async def test_synthesize_stream_yields_each_chunk(fake_auth: AsyncMock) -> None:
    chunks = [b"AAAA", b"BBBB", b"CCCC"]
    patcher, captured = _patch_salute_httpx(chunks)
    with patcher:
        tts = SaluteTTS(auth=fake_auth)
        out: list[bytes] = []
        async for c in tts.synthesize_stream("привет", voice="Tur_24000", format="pcm16"):
            out.append(c)
    assert out == chunks


@pytest.mark.asyncio
async def test_synthesize_stream_filters_empty_chunks(fake_auth: AsyncMock) -> None:
    chunks = [b"data1", b"", b"data2", b""]
    patcher, _ = _patch_salute_httpx(chunks)
    with patcher:
        tts = SaluteTTS(auth=fake_auth)
        out = [c async for c in tts.synthesize_stream("text")]
    assert out == [b"data1", b"data2"]


@pytest.mark.asyncio
async def test_synthesize_stream_propagates_4xx(fake_auth: AsyncMock) -> None:
    patcher, _ = _patch_salute_httpx([b"unused"], status=401)
    with patcher:
        tts = SaluteTTS(auth=fake_auth)
        with pytest.raises(httpx.HTTPStatusError):
            async for _ in tts.synthesize_stream("text"):
                pass


@pytest.mark.asyncio
async def test_synthesize_stream_passes_voice_and_format_params(fake_auth: AsyncMock) -> None:
    patcher, captured = _patch_salute_httpx([b"x"])
    with patcher:
        tts = SaluteTTS(auth=fake_auth)
        async for _ in tts.synthesize_stream("text", voice="Nec_24000", format="pcm16"):
            pass

    req = captured["client"].last_request
    assert req["params"] == {"format": "pcm16", "voice": "Nec_24000"}
    assert req["headers"]["Authorization"] == "Bearer fake-bearer-token"
    assert req["headers"]["Content-Type"] == "application/text"
    assert req["content"] == "text".encode("utf-8")


@pytest.mark.asyncio
async def test_synthesize_stream_supports_ssml(fake_auth: AsyncMock) -> None:
    patcher, captured = _patch_salute_httpx([b"x"])
    with patcher:
        tts = SaluteTTS(auth=fake_auth)
        async for _ in tts.synthesize_stream("<speak>x</speak>", ssml=True):
            pass
    assert captured["client"].last_request["headers"]["Content-Type"] == "application/ssml"


# --- adapter-level wiring tests ---


@pytest.mark.asyncio
async def test_adapter_pushes_each_chunk_to_emitter() -> None:
    """_SaluteChunkedStream._run iterates the stream and pushes per chunk."""
    from voice_agent.livekit_adapters.salute_tts import SaluteTTSAdapter, _SaluteChunkedStream

    chunks_in = [b"PCM1", b"PCM2", b"PCM3"]

    # mock the inner provider's synthesize_stream
    async def fake_stream(text: str, *, voice: str, format: str):
        for c in chunks_in:
            yield c

    adapter = SaluteTTSAdapter()
    adapter._inner = MagicMock()
    adapter._inner.synthesize_stream = fake_stream

    emitter = MagicMock()

    # Stand up a stream object without running through ChunkedStream's init
    # (it expects real LiveKit conn-options); we only exercise _run.
    stream_obj = _SaluteChunkedStream.__new__(_SaluteChunkedStream)
    stream_obj._tts = adapter
    stream_obj._input_text = "тест"

    await stream_obj._run(emitter)

    # initialize called once with our PCM mime + sample rate
    init_call = emitter.initialize.call_args
    assert init_call.kwargs["sample_rate"] == 24000
    assert init_call.kwargs["mime_type"] == "audio/pcm"
    # one push per chunk, in order
    assert [c.args[0] for c in emitter.push.call_args_list] == chunks_in
    # flush at the end
    emitter.flush.assert_called_once()
