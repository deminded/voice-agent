"""LiveKit STT adapter — SaluteSpeech streaming via gRPC.

Non-streaming path (_recognize_impl) remains for backwards compat.
Streaming path (stream()) opens a gRPC bidirectional stream to
smartspeech.sber.ru:443, feeds audio frames as they arrive, and emits
INTERIM_TRANSCRIPT / FINAL_TRANSCRIPT events as Salute responds.

Protocol: gRPC over TLS (grpc.aio).  Auth: Bearer token in per-call
metadata, refreshed via SaluteAuth.get_token() before each stream.
"""
from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import AsyncGenerator

import grpc
import grpc.aio

from livekit import rtc
from livekit.agents import APIConnectOptions, stt, utils
from livekit.agents.types import NOT_GIVEN, NotGivenOr

from voice_agent.grpc_stubs import recognition_pb2, recognition_pb2_grpc
from voice_agent.providers.salute import SaluteAuth, SaluteSTT as _SaluteSTT

# Salute gRPC endpoint — TLS on 443, same host as REST.
_GRPC_HOST = "smartspeech.sber.ru:443"

# Sber's TLS chain is signed by the Russian Trusted Root CA (Минцифры),
# which isn't in the certifi bundle. We ship the root with the project
# so grpc can verify the handshake without disabling TLS verification.
_RUSSIAN_ROOT_CA_PATH = Path(__file__).resolve().parents[3] / "certs" / "russian_trusted_root_ca.pem"

# Send 320 ms chunks at 16 kHz mono PCM16 = 10 240 bytes per chunk.
# Small enough for low latency, large enough to not spam gRPC.
_CHUNK_BYTES = 10_240


class SaluteSTTAdapter(stt.STT):
    def __init__(self, *, language: str = "ru-RU") -> None:
        super().__init__(
            # streaming=True tells LiveKit to call stream() instead of recognize().
            capabilities=stt.STTCapabilities(streaming=True, interim_results=True),
        )
        self._inner = _SaluteSTT()
        self._auth = SaluteAuth()
        self._language = language

    # ------------------------------------------------------------------
    # Non-streaming fallback — kept intact, used when caller picks
    # recognize() explicitly or capabilities.streaming gets set False.
    # ------------------------------------------------------------------

    async def _recognize_impl(
        self,
        buffer: utils.AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        merged = utils.merge_frames(buffer)
        pcm_bytes = bytes(merged.data)
        sample_rate = merged.sample_rate

        if sample_rate != 16000:
            pcm_bytes = _resample_to_16k(pcm_bytes, sample_rate)

        actual_lang = language if language is not NOT_GIVEN else self._language
        text = await self._inner.transcribe(pcm_bytes, format="pcm16", language=actual_lang)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            request_id=str(uuid.uuid4()),
            alternatives=[stt.SpeechData(language=actual_lang, text=text)],
        )

    # ------------------------------------------------------------------
    # Streaming path
    # ------------------------------------------------------------------

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = APIConnectOptions(),
    ) -> stt.RecognizeStream:
        actual_lang = language if language is not NOT_GIVEN else self._language
        return _SaluteRecognizeStream(
            stt=self,
            conn_options=conn_options,
            language=actual_lang,
            auth=self._auth,
        )


class _SaluteRecognizeStream(stt.RecognizeStream):
    def __init__(
        self,
        *,
        stt: SaluteSTTAdapter,
        conn_options: APIConnectOptions,
        language: str,
        auth: SaluteAuth,
    ) -> None:
        # sample_rate=16000 tells RecognizeStream base to auto-resample input.
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)
        self._language = language
        self._auth = auth

    async def _run(self) -> None:
        token = await self._auth.get_token()
        req_id = str(uuid.uuid4())

        root_certs = _RUSSIAN_ROOT_CA_PATH.read_bytes()
        ssl_creds = grpc.ssl_channel_credentials(root_certificates=root_certs)
        token_creds = grpc.access_token_call_credentials(token)
        combined = grpc.composite_channel_credentials(ssl_creds, token_creds)

        async with grpc.aio.secure_channel(_GRPC_HOST, combined) as channel:
            stub = recognition_pb2_grpc.SmartSpeechStub(channel)
            call = stub.Recognize(_audio_request_generator(self._input_ch, self._language))

            # In multi-utterance mode Salute keeps the gRPC stream alive across
            # utterances and emits eou=True at each boundary. We track per-utterance
            # state with new request_ids so transcripts don't collide.
            cur_req_id = str(uuid.uuid4())
            started = False
            async for resp in call:
                if not resp.results:
                    if not resp.eou:
                        continue
                    # eou with no text — utterance ended silently; reset for next.
                    started = False
                    cur_req_id = str(uuid.uuid4())
                    continue

                text = resp.results[0].normalized_text or resp.results[0].text

                if not started and text:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.START_OF_SPEECH,
                            request_id=cur_req_id,
                        )
                    )
                    started = True

                if resp.eou:
                    # Final for this utterance — emit FINAL, then reset for next one.
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
                    started = False
                    cur_req_id = str(uuid.uuid4())
                else:
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(
                            type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                            request_id=cur_req_id,
                            alternatives=[stt.SpeechData(language=self._language, text=text)],
                        )
                    )


async def _audio_request_generator(
    input_ch: utils.aio.Chan,
    language: str,
) -> AsyncGenerator[recognition_pb2.RecognitionRequest, None]:
    """Yield the options header then audio chunks from the input channel."""
    yield recognition_pb2.RecognitionRequest(
        options=recognition_pb2.RecognitionOptions(
            audio_encoding=recognition_pb2.RecognitionOptions.PCM_S16LE,
            sample_rate=16000,
            language=language,
            model="general",
            enable_partial_results=True,
            enable_multi_utterance=True,  # keep stream alive across multiple turns
            channels_count=1,
        )
    )

    async for item in input_ch:
        if isinstance(item, stt.RecognizeStream._FlushSentinel):
            # Multi-utterance: Salute detects boundaries from audio VAD itself.
            # LiveKit's flush is just a hint — skip it and keep streaming.
            continue
        # item is an rtc.AudioFrame; .data is a memoryview of PCM16 LE bytes.
        pcm = bytes(item.data)
        # Split into fixed-size chunks so gRPC message size stays small.
        for i in range(0, len(pcm), _CHUNK_BYTES):
            yield recognition_pb2.RecognitionRequest(audio_chunk=pcm[i : i + _CHUNK_BYTES])


def _resample_to_16k(pcm_bytes: bytes, src_rate: int) -> bytes:
    """Linear resampler — adequate for STT, not for music.

    audioop ships with CPython stdlib; no extra dep needed."""
    import audioop
    out, _ = audioop.ratecv(pcm_bytes, 2, 1, src_rate, 16000, None)
    return out
