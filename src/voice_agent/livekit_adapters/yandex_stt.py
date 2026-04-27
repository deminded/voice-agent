"""LiveKit STT adapter — Yandex SpeechKit v3 streaming via gRPC.

Mirrors SaluteSTTAdapter contract exactly so registry.build_stt() can swap them.

Stubs path: src/voice_agent/yandex_grpc_stubs/yandex/cloud/ai/stt/v3/
RPC used: speechkit.stt.v3.Recognizer/RecognizeStreaming (bidirectional stream).

Auth: Api-Key passed as gRPC call metadata header 'authorization: Api-Key <key>'.
Key read from YANDEX_API_KEY env var. Empty key -> RuntimeError on first use.
"""
from __future__ import annotations

import os
import uuid
from typing import AsyncGenerator

import grpc
import grpc.aio

from livekit.agents import APIConnectOptions, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

# Trigger sys.path injection in the stubs package before importing stubs.
import voice_agent.yandex_grpc_stubs  # noqa: F401
from yandex.cloud.ai.stt.v3 import stt_pb2, stt_service_pb2_grpc

_STT_HOST = "stt.api.cloud.yandex.net:443"
_CHUNK_BYTES = 10_240  # same chunk size as Salute adapter for consistency


class YandexSTTAdapter(stt.STT):
    def __init__(self, *, language: str = "ru-RU") -> None:
        super().__init__(
            # streaming=True tells LiveKit to call stream() instead of recognize().
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True),
        )
        self._language = language
        # Defer key validation to first use so registry import never crashes.
        self._api_key: str | None = None

    def _get_api_key(self) -> str:
        if self._api_key is None:
            key = os.environ.get("YANDEX_API_KEY", "").strip()
            if not key:
                raise RuntimeError(
                    "YANDEX_API_KEY is not set — cannot use Yandex STT. "
                    "Add it to your .env file."
                )
            self._api_key = key
        return self._api_key

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        # Non-streaming fallback — not primary path but required by STT base.
        raise NotImplementedError(
            "YandexSTTAdapter only supports streaming; use stream() instead."
        )

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> stt.RecognizeStream:
        actual_lang = language if language is not NOT_GIVEN else self._language
        return _YandexRecognizeStream(
            stt=self,
            conn_options=conn_options,
            language=actual_lang,
            api_key_fn=self._get_api_key,
        )


class _YandexRecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: YandexSTTAdapter,
        conn_options: APIConnectOptions,
        language: str,
        api_key_fn,
    ) -> None:
        # sample_rate=8000 accepted by Yandex; we request 16k in StreamingOptions.
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)
        self._language = language
        self._api_key_fn = api_key_fn

    async def _run(self) -> None:
        api_key = self._api_key_fn()
        # Yandex uses standard TLS; no custom root CA needed.
        ssl_creds = grpc.ssl_channel_credentials()
        # Yandex expects header `authorization: Api-Key <key>`.
        # access_token_call_credentials() prepends "Bearer " to its value,
        # producing `Bearer Api-Key <key>` which Yandex rejects with
        # "Unknown api key". Use metadata_call_credentials with an explicit
        # callback to set the header verbatim.
        def _auth_plugin(context, callback):
            callback((("authorization", f"Api-Key {api_key}"),), None)
        call_creds = grpc.metadata_call_credentials(_auth_plugin)
        combined = grpc.composite_channel_credentials(ssl_creds, call_creds)

        async with grpc.aio.secure_channel(_STT_HOST, combined) as channel:
            stub = stt_service_pb2_grpc.RecognizerStub(channel)
            call = stub.RecognizeStreaming(
                _audio_request_generator(self._input_ch, self._language)
            )

            cur_req_id = str(uuid.uuid4())
            started = False

            async for resp in call:
                # Each StreamingResponse carries one of: partial, final, eou_update, etc.
                if resp.HasField("partial"):
                    alts = resp.partial.alternatives
                    if not alts:
                        continue
                    text = alts[0].text
                    if not text:
                        continue
                    if not started:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.START_OF_SPEECH,
                                request_id=cur_req_id,
                            )
                        )
                        started = True
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=cur_req_id,
                            alternatives=[stt.SpeechData(language=self._language, text=text)],
                        )
                    )

                elif resp.HasField("final"):
                    alts = resp.final.alternatives
                    text = alts[0].text if alts else ""
                    if not started and text:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.START_OF_SPEECH,
                                request_id=cur_req_id,
                            )
                        )
                    if text:
                        self._event_ch.send_nowait(
                            stt.SpeechEvent(
                                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                                request_id=cur_req_id,
                                alternatives=[stt.SpeechData(language=self._language, text=text)],
                            )
                        )
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.END_OF_SPEECH,
                            request_id=cur_req_id,
                        )
                    )
                    # Reset for next utterance in the stream.
                    started = False
                    cur_req_id = str(uuid.uuid4())


async def _audio_request_generator(
    input_ch: utils.aio.Chan,
    language: str,
) -> AsyncGenerator[stt_pb2.StreamingRequest, None]:
    """Yield the session options header then audio chunks from the input channel."""
    # First message must be session_options with RecognitionModelOptions.
    yield stt_pb2.StreamingRequest(
        session_options=stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                model="general",
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=16000,
                        audio_channel_count=1,
                    )
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=[language],
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
            ),
        )
    )

    async for item in input_ch:
        if isinstance(item, stt.RecognizeStream._FlushSentinel):
            # Yandex detects utterance boundaries from VAD internally.
            continue
        pcm = bytes(item.data)
        for i in range(0, len(pcm), _CHUNK_BYTES):
            yield stt_pb2.StreamingRequest(
                chunk=stt_pb2.AudioChunk(data=pcm[i : i + _CHUNK_BYTES])
            )
