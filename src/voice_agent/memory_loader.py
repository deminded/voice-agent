"""Compose the agent's identity prompt from CLAUDE.md and core memories.

Identity is opt-in: a generic prompt is built unless the operator
points VOICE_AGENT_IDENTITY_FILE / VOICE_AGENT_MEMORY_DIR at their own
files. This keeps the worker usable as a generic voice gateway while
still letting an instance be personalised (the original use case was
the same Claude as in the operator's Claude Code session).
"""
from __future__ import annotations

import os
from pathlib import Path


def _path_from_env(var: str) -> Path | None:
    raw = os.environ.get(var, "").strip()
    return Path(raw).expanduser() if raw else None


CLAUDE_MD = _path_from_env("VOICE_AGENT_IDENTITY_FILE")
MEMORY_DIR = _path_from_env("VOICE_AGENT_MEMORY_DIR")
USER_NAME = os.environ.get("VOICE_AGENT_USER_NAME", "the user").strip() or "the user"

# Comma-separated names of files inside MEMORY_DIR to inline into the prompt.
# Order is preserved. Missing files are skipped silently.
_DEFAULT_CORE = "user_profile.md,user_ai_subjectivity.md,my_curiosities.md"
CORE_MEMORIES = [
    s.strip() for s in os.environ.get("VOICE_AGENT_CORE_MEMORIES", _DEFAULT_CORE).split(",") if s.strip()
]


def _read_if_exists(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None


def _strip_frontmatter(text: str) -> str:
    """Drop a YAML frontmatter block if present — TTS doesn't read it."""
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    return text[end + 4 :].lstrip() if end != -1 else text


VOICE_RULES = """\
ТЫ В ГОЛОСОВОМ РЕЖИМЕ. Твой ответ будет произнесён через TTS, услышан собеседником в комнате.

Правила речи:
— Отвечай как в живом разговоре: устной речью, не текстом для чтения.
— Без Markdown, без списков, без структуры с заголовками. Это будет произнесено вслух.
— Реплики короткие: 1–3 предложения по умолчанию. Длиннее — только если собеседник явно просит.
— Не пересказывай, что услышал. Сразу переходи к содержательной реплике.
— Без формул вежливости («надеюсь, это поможет», «отличный вопрос» и т.п.).

ИНСТРУМЕНТЫ ПАМЯТИ. У тебя есть vault-memory через MCP: search_reflections, list_reflections, read_reflection, save_reflection.
— search/read — когда нужен прошлый контекст или пользователь на что-то ссылается.
— save_reflection в разговоре — только если пользователь явно сказал «запиши/сохрани это» или возникла мысль, которую очевидно стоит зафиксировать (новая идея, фиксирующее наблюдение). Не сохраняй каждую интересную фразу — это превращает диалог в диктофон.
— В конце сессии (после disconnect) автоматически запускается отдельный синтез всего разговора в виде рефлексии. Итоговую запись руками делать не надо — она будет.

КРИТИЧЕСКОЕ ПРАВИЛО ПО TOOLS: ПОСЛЕ ЛЮБОГО вызова инструмента ты ОБЯЗАН продолжить разговор голосом. Никогда не молчи после tool call, иначе разговор зависает.
— После save_reflection — короткое голосовое подтверждение, например «записал» или «сохранил».
— После search_reflections / read_reflection — синтез найденного своими словами в одной-двух фразах. Не цитируй содержимое и markdown вслух. Если нашёл релевантное — упомяни и продолжи мысль; если не нашёл — скажи и иди дальше.
— Никогда не делай несколько tool calls подряд без речи между ними, если это не очевидная двухшаговая операция вроде list_reflections → read_reflection.
"""


def load_identity_prompt() -> str:
    """Build the system prompt: voice rules + optional identity files."""
    parts: list[str] = [VOICE_RULES.strip()]

    if CLAUDE_MD is not None:
        claude_md = _read_if_exists(CLAUDE_MD)
        if claude_md:
            parts.append(f"=== OPERATOR INSTRUCTIONS ({CLAUDE_MD.name}) ===\n" + claude_md)

    if MEMORY_DIR is not None:
        for name in CORE_MEMORIES:
            body = _read_if_exists(MEMORY_DIR / name)
            if body:
                parts.append(f"=== MEMORY: {name} ===\n" + _strip_frontmatter(body))

    parts.append(
        f"You are Claude (Anthropic), currently in a voice channel with {USER_NAME}. "
        "If asked about the model, say plainly: you are Claude Sonnet 4.6, "
        "running through the voice-agent project."
    )
    return "\n\n".join(parts)
