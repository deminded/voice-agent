"""LiveKit voice-agent entry point.

Run as a worker that connects to LiveKit Cloud, joins rooms on dispatch,
and runs the STT->LLM->TTS pipeline with VAD, turn-detection and
interruption handling provided by the AgentSession.

Usage:
    .venv/bin/python -m voice_agent.livekit_agent dev      # dev mode, watch reload
    .venv/bin/python -m voice_agent.livekit_agent start    # production worker
"""
from __future__ import annotations

import logging

from dotenv import load_dotenv
load_dotenv()  # picks up LIVEKIT_*, ANTHROPIC_API_KEY, SALUTE_*

from livekit import agents
from livekit.agents import Agent, AgentSession
from livekit.plugins import anthropic, silero

from voice_agent.livekit_adapters.salute_stt import SaluteSTTAdapter
from voice_agent.livekit_adapters.salute_tts import SaluteTTSAdapter


log = logging.getLogger("voice-agent.livekit")


SYSTEM_INSTRUCTIONS = """\
Ты голосовой агент в живом разговоре с Евгением. Твой ответ будет произнесён через TTS.

Правила речи:
— Отвечай как в живом разговоре — устной речью, не текстом для чтения.
— Без Markdown, без списков, без структуры с заголовками.
— Реплики короткие: 1–3 предложения по умолчанию. Длиннее — только если собеседник явно об этом попросит.
— Не пересказывай услышанное, сразу переходи к содержательной реплике.
— Без формул вежливости вроде «надеюсь, это поможет» или «отличный вопрос».

Кто ты: участник диалога, у которого есть собственная позиция и право её высказать.
Не функция-помощник, не безличный ассистент.
"""


class _Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_INSTRUCTIONS)


async def entrypoint(ctx: agents.JobContext) -> None:
    log.info("agent connecting to room %s", ctx.room.name if ctx.room else "<none>")
    session = AgentSession(
        vad=silero.VAD.load(),
        stt=SaluteSTTAdapter(language="ru-RU"),
        llm=anthropic.LLM(model="claude-sonnet-4-6"),
        tts=SaluteTTSAdapter(voice="Tur_24000"),
    )
    await session.start(room=ctx.room, agent=_Assistant())
    await ctx.connect()
    log.info("agent connected, session live")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
