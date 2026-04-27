"""LiveKit TTS adapter — Yandex SpeechKit v3 via gRPC unary-stream.

Mirrors SaluteTTSAdapter contract exactly so registry.build_tts() can swap them.

Stubs path: src/voice_agent/yandex_grpc_stubs/yandex/cloud/ai/tts/v3/
RPC used: speechkit.tts.v3.Synthesizer/UtteranceSynthesis (unary -> server stream).

Yandex returns raw PCM16 at the requested sample rate.  We ask for 22050 Hz (the
Yandex default for "general" voices) to avoid WAV header overhead.

Auth: Api-Key passed as gRPC call metadata header 'authorization: Api-Key <key>'.
Key read from YANDEX_API_KEY env var. Empty key -> RuntimeError on first use.
"""
from __future__ import annotations

import os
import uuid

import grpc
import grpc.aio

from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import NotGivenOr

# Trigger sys.path injection in the stubs package before importing stubs.
import voice_agent.yandex_grpc_stubs  # noqa: F401
from yandex.cloud.ai.tts.v3 import tts_pb2, tts_service_pb2_grpc

_TTS_HOST = "tts.api.cloud.yandex.net:443"

# Yandex standard voices output 22050 Hz; anton is a male premium voice.
SAMPLE_RATE = 22050
_DEFAULT_VOICE = "anton"


class YandexTTSAdapter(tts.TTS):
    def __init__(self, *, voice: str = _DEFAULT_VOICE) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        self._voice = voice
        # Defer key validation to first use so registry import never crashes.
        self._api_key: str | None = None

    def _get_api_key(self) -> str:
        if self._api_key is None:
            key = os.environ.get("YANDEX_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "YANDEX_API_KEY is not set — cannot use Yandex TTS. "
                    "Add it to your .env file."
                )
            self._api_key = key
        return self._api_key

    def synthesize(
        self,
        text: str,
        *,
        conn_options: NotGivenOr[APIConnectOptions] = ...,
    ) -> tts.ChunkedStream:
        return _YandexChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options if conn_options is not ... else APIConnectOptions(),
        )


class _YandexChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        adapter: YandexTTSAdapter = self._tts  # type: ignore[assignment]
        api_key = adapter._get_api_key()
        voice = adapter._voice

        ssl_creds = grpc.ssl_channel_credentials()
        # See yandex_stt.py for why metadata_call_credentials is used here:
        # access_token_call_credentials prepends "Bearer " which breaks
        # Yandex's `Api-Key <key>` auth scheme.
        def _auth_plugin(context, callback):
            callback((("authorization", f"Api-Key {api_key}"),), None)
        call_creds = grpc.metadata_call_credentials(_auth_plugin)
        combined = grpc.composite_channel_credentials(ssl_creds, call_creds)

        output_emitter.initialize(
            request_id=str(uuid.uuid4()),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
        )

        async with grpc.aio.secure_channel(_TTS_HOST, combined) as channel:
            stub = tts_service_pb2_grpc.SynthesizerStub(channel)
            request = tts_pb2.UtteranceSynthesisRequest(
                text=self._input_text,
                hints=[tts_pb2.Hints(voice=voice)],
                output_audio_spec=tts_pb2.AudioFormatOptions(
                    raw_audio=tts_pb2.RawAudio(
                        audio_encoding=tts_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=SAMPLE_RATE,
                    )
                ),
                loudness_normalization_type=tts_pb2.UtteranceSynthesisRequest.LUFS,
            )
            # UtteranceSynthesis is unary->stream: one request, response is a stream of chunks.
            async for resp in stub.UtteranceSynthesis(request):
                if resp.audio_chunk.data:
                    output_emitter.push(resp.audio_chunk.data)

        output_emitter.flush()
