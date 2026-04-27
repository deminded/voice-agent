"""LiveKit voice-agent entry point.

Run as a worker that connects to LiveKit Cloud, joins rooms on dispatch,
and runs the STT->LLM->TTS pipeline with VAD, turn-detection and
interruption handling provided by the AgentSession.

Identity: CLAUDE.md + core memories baked into the system prompt at boot
(see memory_loader). Topical recall via vault-memory MCP server attached
to the Agent — Claude pulls past reflections on demand.

Usage:
    .venv/bin/python -m voice_agent.livekit_agent dev      # dev mode, watch reload
    .venv/bin/python -m voice_agent.livekit_agent start    # production worker
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import httpx
from dotenv import load_dotenv
load_dotenv()  # picks up LIVEKIT_*, ANTHROPIC_API_KEY, SALUTE_*

from livekit import agents
from livekit.agents import Agent, AgentSession, mcp
from livekit.plugins import anthropic, silero

from voice_agent.livekit_adapters.registry import STT_PROVIDERS, TTS_PROVIDERS, build_stt, build_tts
from voice_agent.memory_loader import load_identity_prompt
from voice_agent.session_recorder import SessionRecorder


log = logging.getLogger("voice-agent.livekit")


# Optional MCP servers — set the env vars to enable. Empty/unset means
# the worker runs without that capability rather than crashing on missing
# scripts, so the agent stays usable on a fresh install.
VAULT_MEMORY_SCRIPT = os.environ.get("VAULT_MEMORY_MCP_SCRIPT", "").strip()

# URL of voice-channel-plugin intake — CLI mode forwards transcripts here.
# Override via VOICE_CLI_INTAKE_URL if the plugin runs on a non-default port.
# Must point at a loopback address: transcripts contain user utterances and
# the worker has access to identity material; sending them off-host through
# this env var would be silent exfiltration.
def _validate_loopback_url(raw: str) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            f"VOICE_CLI_INTAKE_URL must point at loopback (got host={host!r}). "
            "Transcripts are sensitive — refusing to forward off-host."
        )
    return raw


_CLI_INTAKE_URL = _validate_loopback_url(
    os.environ.get("VOICE_CLI_INTAKE_URL", "http://127.0.0.1:8910/utterance")
)


# ---------------------------------------------------------------------------
# Phase C helpers — extracted for testability (not buried in closures)
# ---------------------------------------------------------------------------

def handle_set_mode(payload: dict, nonlocal_mode: dict) -> bool:
    """Process a set_mode data message. Returns True if handled, else False.

    Why a separate function: keeps the closure in entrypoint thin and lets
    unit tests call this directly without a real LiveKit room.
    """
    if payload.get("type") != "set_mode":
        return False
    nonlocal_mode["mode"] = payload.get("mode") or "agent"
    log.info("set_mode -> %s", nonlocal_mode["mode"])
    return True


async def post_to_cli(text: str) -> None:
    """Forward a final user transcript to the voice-channel-plugin intake.

    Why httpx: it's already a dependency of voice-agent (pyproject.toml).
    Errors are logged but never raised — a failed POST must not crash the worker.

    The intake authenticates via the shared VOICE_INTAKE_SECRET. The header is
    only sent when the secret is configured — otherwise we POST without it and
    rely on the plugin's "unauthenticated mode" warning, preserving the legacy
    open-loopback behaviour for setups that haven't rotated to a secret yet.
    """
    utterance_id = f"lk-{int(time.time() * 1000)}"
    headers = {}
    secret = os.environ.get("VOICE_INTAKE_SECRET", "").strip()
    if secret:
        headers["X-Voice-Intake-Secret"] = secret
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.post(
                _CLI_INTAKE_URL,
                json={"text": text, "user": "voice", "utterance_id": utterance_id},
                headers=headers,
            )
        log.info("post_to_cli: status=%s utterance_id=%s", r.status_code, utterance_id)
    except Exception as exc:
        log.warning("post_to_cli failed: %s: %s", type(exc).__name__, exc)


def handle_user_input_transcribed(ev, nonlocal_mode: dict, session) -> None:
    """React to a user_input_transcribed event.

    In 'cli' mode: suppress the default LLM generation and forward the
    transcript to the voice-channel-plugin. In 'agent' mode: do nothing
    (the AgentSession handles the turn as normal).

    Event fields confirmed via livekit-agents 1.5.5 source:
      livekit/agents/voice/events.py :: UserInputTranscribedEvent
        .transcript  — the STT text
        .is_final    — True when the utterance is complete
    """
    if nonlocal_mode["mode"] != "cli":
        return
    text = getattr(ev, "transcript", "") or getattr(ev, "text", "")
    is_final = getattr(ev, "is_final", True)
    if not is_final or not text.strip():
        return
    # No interrupt needed — _Assistant.llm_node short-circuits in CLI mode,
    # so the worker LLM never generates and TTS stays silent. Just forward.
    asyncio.create_task(post_to_cli(text.strip()))


def _build_mcp_servers() -> list:
    servers: list = []
    # vault-memory MCP — operator-supplied identity/recall server.
    # Only attach if the script path is configured AND exists; missing
    # script must not crash the worker.
    if VAULT_MEMORY_SCRIPT and os.path.exists(VAULT_MEMORY_SCRIPT):
        servers.append(mcp.MCPServerStdio(
            command="/usr/bin/python3",
            args=[VAULT_MEMORY_SCRIPT],
            client_session_timeout_seconds=10,
        ))
    elif VAULT_MEMORY_SCRIPT:
        log.warning("VAULT_MEMORY_MCP_SCRIPT set but file not found: %s", VAULT_MEMORY_SCRIPT)
    # Web search via Exa — only attach if key is present so missing key
    # degrades to "no search tool" instead of crashing the worker.
    exa_key = os.environ.get("EXA_API_KEY", "").strip()
    if exa_key:
        servers.append(mcp.MCPServerStdio(
            command="/usr/bin/npx",
            args=["-y", "exa-mcp-server"],
            env={"EXA_API_KEY": exa_key, "PATH": os.environ.get("PATH", "")},
            client_session_timeout_seconds=15,  # npx cold-starts can be slow
        ))
    return servers


class _Assistant(Agent):
    def __init__(self, mode_state: dict) -> None:
        super().__init__(
            instructions=load_identity_prompt(),
            mcp_servers=_build_mcp_servers(),
        )
        self._mode_state = mode_state

    async def llm_node(self, chat_ctx, tools, model_settings):
        # CLI mode: main Claude Code session generates the reply via the
        # voice-channel-plugin path; the worker LLM must stay silent.
        # Returning before the first yield gives the framework an empty stream,
        # so no ChatChunks reach TTS. session.interrupt() can't cover this race
        # because generation may have already begun before the transcript hook
        # fires — suppressing at the node level is the only race-free fix.
        if self._mode_state["mode"] == "cli":
            return
            yield  # unreachable, marks function as async generator
        async for chunk in Agent.default.llm_node(self, chat_ctx, tools, model_settings):
            yield chunk


async def entrypoint(ctx: agents.JobContext) -> None:
    log.info("agent connecting to room %s", ctx.room.name if ctx.room else "<none>")

    # Phase C: per-room mode state (mutable dict so the Agent's llm_node and
    # the data/transcript hooks all share one reference). Default 'agent'.
    nonlocal_mode: dict = {"mode": "agent"}
    # session ref via mutable holder — _on_data is registered before
    # AgentSession exists (we need the data hook live before connect to
    # avoid losing the client's first set_mode), so it accesses the
    # session through this dict at fire time.
    session_ref: dict = {"session": None}

    @ctx.room.on("data_received")
    def _on_data(packet) -> None:
        try:
            payload = json.loads(packet.data.decode("utf-8"))
        except Exception:
            return
        if handle_set_mode(payload, nonlocal_mode):
            return
        if payload.get("type") != "external_say":
            return
        text = (payload.get("text") or "").strip()
        if not text:
            return
        log.info("external_say: %r", text)
        sess = session_ref["session"]
        if sess is None:
            log.warning("external_say dropped: session not yet started")
            return
        # session.say() in livekit-agents 1.5.5 is sync — returns SpeechHandle,
        # not a coroutine. Same gotcha as session.interrupt(). Don't wrap in
        # create_task; the call itself schedules the TTS in background.
        sess.say(text, allow_interruptions=True)

    # Connect first so remote participants and their metadata are visible.
    # The previous order read remote_participants BEFORE connect — at that
    # moment the worker hasn't joined yet, the dict is empty, and provider
    # always falls back to salute regardless of what the client sent.
    await ctx.connect()
    participant = await ctx.wait_for_participant()
    provider = "salute"
    try:
        meta = json.loads(participant.metadata or "{}")
        candidate = meta.get("provider", "salute")
        if candidate in STT_PROVIDERS:
            provider = candidate
    except Exception:
        pass
    log.info("provider selected: %s", provider)

    session = AgentSession(
        vad=silero.VAD.load(),
        stt=build_stt(provider),
        llm=anthropic.LLM(model="claude-sonnet-4-6"),
        tts=build_tts(provider),
    )
    session_ref["session"] = session
    recorder = SessionRecorder(room_name=ctx.room.name if ctx.room else None)
    recorder.attach(session)
    ctx.add_shutdown_callback(lambda: recorder.finalize(reason="job_shutdown"))

    @session.on("user_input_transcribed")
    def _on_user_text(ev) -> None:
        handle_user_input_transcribed(ev, nonlocal_mode, session)

    await session.start(room=ctx.room, agent=_Assistant(nonlocal_mode))
    log.info("agent connected, session live (recording to %s)", recorder.session_dir)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
