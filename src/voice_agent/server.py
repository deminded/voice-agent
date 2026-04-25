"""Voice-agent server.

WebSocket protocol per turn (push-to-talk):

    client -> server  text  {"type": "audio_start", "format": "pcm16",
                              "tts": "salute" | "elevenlabs"}   # tts is optional
    client -> server  bin   <audio bytes>
    client -> server  text  {"type": "audio_end"}
    server -> client  text  {"type": "transcript", "text": "..."}
    server -> client  text  {"type": "response_text", "text": "..."}
    server -> client  text  {"type": "audio_start", "format": "opus", "tts": "..."}
    server -> client  bin   <audio bytes>
    server -> client  text  {"type": "audio_end"}

One Conversation per connection. Errors come back as
{"type": "error", "message": "..."}, then the connection stays open
for the next turn.

GET / serves the bundled HTML+JS push-to-talk client (client/web).
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from voice_agent.config import settings
from voice_agent.conversation import Conversation
from voice_agent.providers.anthropic_llm import ClaudeLLM
from voice_agent.providers.base import TTSProvider
from voice_agent.providers.elevenlabs import ElevenLabsTTS
from voice_agent.providers.salute import SaluteAuth, SaluteSTT, SaluteTTS

log = logging.getLogger("voice-agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

WEB_DIR = Path(__file__).parent.parent.parent / "client" / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()  # picks up voice-agent/.env when started from project root
    auth = SaluteAuth()
    app.state.stt = SaluteSTT(auth)

    # tts_pool: every available provider keyed by short name. The client
    # picks per turn via {"tts": "..."} in audio_start; absent key falls
    # back to settings.tts_provider as the default.
    pool: dict[str, TTSProvider] = {"salute": SaluteTTS(auth)}
    if os.getenv("ELEVENLABS_API_KEY"):
        pool["elevenlabs"] = ElevenLabsTTS()
    app.state.tts_pool = pool
    app.state.default_tts_name = settings.tts_provider if settings.tts_provider in pool else "salute"
    app.state.tts = pool[app.state.default_tts_name]

    # llm_factory: one Claude per connection so history stays per-client.
    # Tests override with `app.state.llm_factory = lambda: MockLLM()`.
    app.state.llm_factory = lambda: ClaudeLLM()
    log.info(
        "voice-agent ready: STT=SaluteSpeech, TTS pool=%s, default=%s, LLM=Claude",
        list(pool.keys()), app.state.default_tts_name,
    )
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.get("/providers")
async def list_providers() -> dict:
    """Used by the web client to populate the TTS toggle."""
    return {
        "tts": list(app.state.tts_pool.keys()),
        "default": app.state.default_tts_name,
    }


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/lk")
async def lk_index() -> FileResponse:
    """LiveKit-based client (real live dialog with VAD, turn-taking, interruptions)."""
    return FileResponse(WEB_DIR / "lk.html")


@app.get("/livekit/token")
async def livekit_token(identity: str = "guest") -> dict:
    """Mint a short-lived JWT for the browser to join a LiveKit room.

    Each visit gets a per-identity room so multiple testers don't collide.
    The agent worker auto-dispatches into any room it sees.
    """
    from livekit import api as lk_api

    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    url = os.environ.get("LIVEKIT_URL")
    if not (api_key and api_secret and url):
        return {"error": "LIVEKIT_* env vars missing"}

    room_name = f"voice-agent-{identity}"
    token = (
        lk_api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(lk_api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True))
    )
    return {"url": url, "token": token.to_jwt(), "room": room_name}


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

    pool = ws.app.state.tts_pool
    default_name = ws.app.state.default_tts_name
    requested_tts = start.get("tts")
    if requested_tts and requested_tts not in pool:
        await ws.send_json({"type": "error", "message": f"unknown tts: {requested_tts}, available: {list(pool)}"})
        return
    tts_name = requested_tts or default_name
    chosen_tts = pool[tts_name]

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
    log.info("turn: %d bytes %s in, tts=%s", len(audio), audio_format, tts_name)

    # 3) run pipeline, emit results
    try:
        result = await convo.turn(audio, in_format=audio_format, tts=chosen_tts)
    except Exception as e:
        log.exception("turn failed")
        await ws.send_json({"type": "error", "message": str(e)})
        return

    log.info("turn: transcript=%r reply=%r %d bytes out", result.transcript, result.response_text, len(result.response_audio))
    await ws.send_json({"type": "transcript", "text": result.transcript})
    await ws.send_json({"type": "response_text", "text": result.response_text})
    await ws.send_json({"type": "audio_start", "format": convo.out_format, "tts": tts_name})
    await ws.send_bytes(result.response_audio)
    await ws.send_json({"type": "audio_end"})
