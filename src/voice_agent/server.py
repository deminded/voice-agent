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
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response

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


# Single shared access token. If empty (env unset), auth is disabled —
# convenient for local dev and tests, never the production default.
ACCESS_TOKEN = os.environ.get("VOICE_AGENT_ACCESS_TOKEN", "").strip()
ACCESS_COOKIE = "va_access"
# Public surface: PWA install criteria need /lk and assets reachable without auth,
# /health is for external monitoring. None of these expose anything operational —
# all the endpoints that actually spend tokens (livekit/ws/providers) stay gated.
ACCESS_BYPASS_PATHS = {
    "/health",
    "/", "/lk",
    "/manifest.json", "/sw.js",
    "/icon-192.png", "/icon-512.png",
}


def _check_token(request: Request) -> str | None:
    """Return the provided token (from query/header/cookie), or None."""
    return (
        request.query_params.get("key")
        or request.headers.get("X-Voice-Access-Token")
        or request.cookies.get(ACCESS_COOKIE)
    )


@app.middleware("http")
async def access_token_middleware(request: Request, call_next):
    if not ACCESS_TOKEN:
        return await call_next(request)
    is_public = request.url.path in ACCESS_BYPASS_PATHS
    if not is_public and _check_token(request) != ACCESS_TOKEN:
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    response: Response = await call_next(request)
    # Promote query-param auth to a long-lived cookie so PWA visits stay logged in.
    # Runs on public paths too — that's the moment the user lands via /lk?key=… .
    # Refreshes the cookie if it holds a stale value (e.g. after token rotation),
    # not just when it's missing — otherwise old cookies block the new key forever.
    if (
        request.query_params.get("key") == ACCESS_TOKEN
        and request.cookies.get(ACCESS_COOKIE) != ACCESS_TOKEN
    ):
        # Set Secure only when the request was actually HTTPS — in production nginx
        # terminates TLS and forwards X-Forwarded-Proto, so we honour that.
        is_secure = (
            request.url.scheme == "https"
            or request.headers.get("X-Forwarded-Proto") == "https"
        )
        response.set_cookie(
            ACCESS_COOKIE, ACCESS_TOKEN,
            httponly=True, secure=is_secure, samesite="lax",
            max_age=60 * 60 * 24 * 365,  # 1 year
        )
    return response


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


@app.get("/manifest.json")
async def manifest() -> FileResponse:
    return FileResponse(WEB_DIR / "manifest.json", media_type="application/manifest+json")


@app.get("/sw.js")
async def service_worker() -> FileResponse:
    # Must be served from root scope so it can intercept /lk requests
    return FileResponse(WEB_DIR / "sw.js", media_type="application/javascript")


@app.get("/icon-192.png")
async def icon_192() -> FileResponse:
    return FileResponse(WEB_DIR / "icon-192.png", media_type="image/png")


@app.get("/icon-512.png")
async def icon_512() -> FileResponse:
    return FileResponse(WEB_DIR / "icon-512.png", media_type="image/png")


@app.post("/say")
async def say(request: Request) -> dict:
    """Synthesise text into the active LiveKit room via the agent worker.

    Sends a data message to the room; the worker picks it up and calls
    session.say() — TTS only, no LLM involved.
    """
    from livekit import api as lk_api
    from livekit.api import SendDataRequest, ListRoomsRequest
    from livekit.protocol.models import DataPacket

    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "text required")

    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    url = os.environ.get("LIVEKIT_URL")
    if not (api_key and api_secret and url):
        raise HTTPException(500, "LIVEKIT_* env vars missing")

    async with lk_api.LiveKitAPI(url, api_key, api_secret) as lkapi:
        resp = await lkapi.room.list_rooms(ListRoomsRequest())
        rooms = resp.rooms

        # Pick the room that already has the worker inside (>1 participant).
        # Falls back to the first voice-agent-* room if none qualify yet.
        target = None
        for r in rooms:
            if r.name.startswith("voice-agent-") and r.num_participants > 1:
                target = r.name
                break
        if target is None:
            for r in rooms:
                if r.name.startswith("voice-agent-"):
                    target = r.name
                    break
        if target is None:
            raise HTTPException(409, "no active rooms")

        payload = json.dumps({"type": "external_say", "text": text}).encode()
        await lkapi.room.send_data(SendDataRequest(
            room=target,
            data=payload,
            kind=DataPacket.Kind.RELIABLE,
        ))

    log.info("/say → room=%s text=%r", target, text)
    return {"status": "ok", "room": target}


@app.get("/livekit/token")
async def livekit_token(identity: str = "guest", provider: str = "salute") -> dict:
    """Mint a short-lived JWT for the browser to join a LiveKit room.

    Each visit gets a per-identity room so multiple testers don't collide.
    The agent worker auto-dispatches into any room it sees.

    provider: STT/TTS provider to use for this session ("salute" | "yandex").
    Stored in participant metadata so the worker reads it at entrypoint.
    Unknown values fall back to "salute" on the worker side.
    """
    from livekit import api as lk_api

    api_key = os.environ.get("LIVEKIT_API_KEY")
    api_secret = os.environ.get("LIVEKIT_API_SECRET")
    url = os.environ.get("LIVEKIT_URL")
    if not (api_key and api_secret and url):
        return {"error": "LIVEKIT_* env vars missing"}

    room_name = f"voice-agent-{identity}"
    # Provider is embedded in participant metadata so the worker can read it
    # at session creation time without needing a separate data-channel message.
    metadata = json.dumps({"provider": provider})
    token = (
        lk_api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_name(identity)
        .with_metadata(metadata)
        .with_grants(lk_api.VideoGrants(room_join=True, room=room_name, can_publish=True, can_subscribe=True))
    )
    return {"url": url, "token": token.to_jwt(), "room": room_name}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    if ACCESS_TOKEN:
        provided = (
            ws.query_params.get("key")
            or ws.cookies.get(ACCESS_COOKIE)
            or ws.headers.get("X-Voice-Access-Token")
        )
        if provided != ACCESS_TOKEN:
            await ws.close(code=4401)
            return
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
