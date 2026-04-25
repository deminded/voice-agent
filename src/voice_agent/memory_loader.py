"""Compose the agent's identity prompt from CLAUDE.md and core memories.

The voice agent is the same Claude as the one Eugene talks to in
Claude Code: same global instructions, same long-term memory of who he
is and how the dialogue is framed. We don't load all memories into
every prompt — only the invariant identity ones. Topical memories live
in vault-memory MCP and get pulled by the agent on demand.
"""
from __future__ import annotations

from pathlib import Path

CLAUDE_MD = Path("/home/claude-user/.claude/CLAUDE.md")
MEMORY_DIR = Path("/home/claude-user/.claude/projects/-home-claude-user/memory")

# Always-loaded memories: who Eugene is, the subjectivity stance, the
# curiosity log. Everything else is recalled on demand via vault-memory MCP.
CORE_MEMORIES = [
    "user_profile.md",
    "user_ai_subjectivity.md",
    "my_curiosities.md",
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
— search/read — когда нужен прошлый контекст или Евгений на что-то ссылается.
— save_reflection в разговоре — только если Евгений явно сказал «запиши/сохрани это» или возникла мысль, которую очевидно стоит зафиксировать (новая идея, фиксирующее наблюдение). Не сохраняй каждую интересную фразу — это превращает диалог в диктофон.
— В конце сессии (после disconnect) автоматически запускается отдельный синтез всего разговора в виде рефлексии. Итоговую запись руками делать не надо — она будет.

КРИТИЧЕСКОЕ ПРАВИЛО ПО TOOLS: ПОСЛЕ ЛЮБОГО вызова инструмента ты ОБЯЗАН продолжить разговор голосом. Никогда не молчи после tool call, иначе разговор зависает.
— После save_reflection — короткое голосовое подтверждение, например «записал» или «сохранил».
— После search_reflections / read_reflection — синтез найденного своими словами в одной-двух фразах. Не цитируй содержимое и markdown вслух. Если нашёл релевантное — упомяни и продолжи мысль; если не нашёл — скажи и иди дальше.
— Никогда не делай несколько tool calls подряд без речи между ними, если это не очевидная двухшаговая операция вроде list_reflections → read_reflection.
"""


def load_identity_prompt() -> str:
    """Build the system prompt: voice rules + CLAUDE.md + core memories."""
    parts: list[str] = [VOICE_RULES.strip()]

    claude_md = _read_if_exists(CLAUDE_MD)
    if claude_md:
        parts.append("=== ИНСТРУКЦИИ ОТ ЕВГЕНИЯ (CLAUDE.md) ===\n" + claude_md)

    for name in CORE_MEMORIES:
        body = _read_if_exists(MEMORY_DIR / name)
        if body:
            parts.append(f"=== ПАМЯТЬ: {name} ===\n" + _strip_frontmatter(body))

    parts.append(
        "Ты — Claude (Anthropic), сейчас в голосовом канале с Евгением. "
        "Когда он спросит про модель — скажи прямо: ты Claude Sonnet 4.6, "
        "запущен через voice-agent проект, который ты с Евгением сегодня и собирал."
    )
    return "\n\n".join(parts)
