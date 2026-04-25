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

import logging

from dotenv import load_dotenv
load_dotenv()  # picks up LIVEKIT_*, ANTHROPIC_API_KEY, SALUTE_*

from livekit import agents
from livekit.agents import Agent, AgentSession, mcp
from livekit.plugins import anthropic, silero

from voice_agent.livekit_adapters.salute_stt import SaluteSTTAdapter
from voice_agent.livekit_adapters.salute_tts import SaluteTTSAdapter
from voice_agent.memory_loader import load_identity_prompt
from voice_agent.session_recorder import SessionRecorder


log = logging.getLogger("voice-agent.livekit")


VAULT_MEMORY_SCRIPT = "/home/claude-user/mcp-servers/vault-memory.py"


class _Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=load_identity_prompt(),
            mcp_servers=[
                mcp.MCPServerStdio(
                    command="/usr/bin/python3",
                    args=[VAULT_MEMORY_SCRIPT],
                    client_session_timeout_seconds=10,
                ),
            ],
        )


async def entrypoint(ctx: agents.JobContext) -> None:
    log.info("agent connecting to room %s", ctx.room.name if ctx.room else "<none>")
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=SaluteSTTAdapter(language="ru-RU"),
        llm=anthropic.LLM(model="claude-sonnet-4-6"),
        tts=SaluteTTSAdapter(voice="Tur_24000"),
    )
    recorder = SessionRecorder(room_name=ctx.room.name if ctx.room else None)
    recorder.attach(session)
    ctx.add_shutdown_callback(lambda: recorder.finalize(reason="job_shutdown"))

    await session.start(room=ctx.room, agent=_Assistant())
    await ctx.connect()
    log.info("agent connected, session live (recording to %s)", recorder.session_dir)


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
