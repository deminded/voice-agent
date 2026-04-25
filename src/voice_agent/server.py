"""Voice-agent WebSocket server.

Protocol per turn (push-to-talk client model):

    client -> server  text  {"type": "audio_start", "format": "opus"}
    client -> server  bin   <audio bytes>
    client -> server  text  {"type": "audio_end"}
    server -> client  text  {"type": "transcript", "text": "..."}
    server -> client  text  {"type": "response_text", "text": "..."}
    server -> client  text  {"type": "audio_start", "format": "opus"}
    server -> client  bin   <audio bytes>
    server -> client  text  {"type": "audio_end"}

One Conversation per connection. Errors come back as
{"type": "error", "message": "..."}, then the connection stays open for
the next turn.
"""
from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from voice_agent.config import settings
from voice_agent.conversation import Conversation
from voice_agent.providers.anthropic_llm import ClaudeLLM
from voice_agent.providers.elevenlabs import ElevenLabsTTS
from voice_agent.providers.salute import SaluteAuth, SaluteSTT, SaluteTTS

log = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()  # picks up voice-agent/.env when started from project root
    auth = SaluteAuth()
    app.state.stt = SaluteSTT(auth)
    if settings.tts_provider == "elevenlabs":
        app.state.tts = ElevenLabsTTS()
        tts_label = f"ElevenLabs(voice={settings.elevenlabs_voice_id}, proxy={settings.elevenlabs_proxy or 'direct'})"
    else:
        app.state.tts = SaluteTTS(auth)
        tts_label = "SaluteSpeech"
    # llm_factory: one Claude per connection so history stays per-client.
    # Tests override with `app.state.llm_factory = lambda: MockLLM()`.
    app.state.llm_factory = lambda: ClaudeLLM()
    log.info("voice-agent ready: STT=SaluteSpeech, TTS=%s, LLM=Claude", tts_label)
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    llm = ws.app.state.llm_factory()
    convo = Conversation(ws.app.state.stt, llm, ws.app.state.tts)
    log.info("client connected")
    try:
        while True:
            await _handle_turn(ws, convo)
    except WebSocketDisconnect:
        log.info("client disconnected")


async def _handle_turn(ws: WebSocket, convo: Conversation) -> None:
    # 1) wait for audio_start (text)
    msg = await ws.receive()
    if msg["type"] == "websocket.disconnect":
        raise WebSocketDisconnect()
    start = json.loads(msg["text"])
    if start.get("type") != "audio_start":
        await ws.send_json({"type": "error", "message": f"expected audio_start, got {start.get('type')}"})
        return
    audio_format = start.get("format", "opus")

    # 2) collect binary chunks until audio_end (text)
    chunks: list[bytes] = []
    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            raise WebSocketDisconnect()
        if "bytes" in msg and msg["bytes"] is not None:
            chunks.append(msg["bytes"])
            continue
        if "text" in msg and msg["text"] is not None:
            evt = json.loads(msg["text"])
            if evt.get("type") == "audio_end":
                break
            await ws.send_json({"type": "error", "message": f"expected audio_end, got {evt.get('type')}"})
            return

    audio = b"".join(chunks)
    log.info("turn: %d bytes %s in", len(audio), audio_format)

    # 3) run pipeline, emit results
    try:
        result = await convo.turn(audio, in_format=audio_format)
    except Exception as e:
        log.exception("turn failed")
        await ws.send_json({"type": "error", "message": str(e)})
        return

    log.info("turn: transcript=%r reply=%r %d bytes out", result.transcript, result.response_text, len(result.response_audio))
    await ws.send_json({"type": "transcript", "text": result.transcript})
    await ws.send_json({"type": "response_text", "text": result.response_text})
    await ws.send_json({"type": "audio_start", "format": convo.out_format})
    await ws.send_bytes(result.response_audio)
    await ws.send_json({"type": "audio_end"})
