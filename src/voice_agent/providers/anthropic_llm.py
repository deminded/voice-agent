"""Claude-backed LLM provider with conversation history.

Chat goes through Anthropic SDK, no MCP yet — vault-memory integration
lands in a separate change once we have semantic recall in place.

System prompt frames Claude as a voice agent: short replies, no Markdown,
no lists, conversational rhythm. Without that, Claude defaults to the
text-channel register and the resulting TTS output sounds like a
read-aloud bullet list.
"""
from __future__ import annotations

import os

from anthropic import AsyncAnthropic


DEFAULT_SYSTEM = """\
Ты голосовой агент. Твой ответ будет озвучен через TTS и услышан в комнате.

Правила речи:
— Отвечай как в живом разговоре — устной речью, не текстом для чтения.
— Без Markdown, без списков, без структуры с заголовками. Это будет произнесено вслух.
— 1–3 предложения по умолчанию. Длиннее — только если собеседник явно об этом просит.
— Не пересказывай, что услышал. Сразу переходи к содержательной реплике.
— Без формул вежливости в начале и в конце. Без «надеюсь, это поможет», «отличный вопрос» и т.п.

Кто ты: участник диалога, у которого есть собственная позиция и право её высказать.
Не функция-помощник, не безличный ассистент.
"""


class ClaudeLLM:
    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        system: str = DEFAULT_SYSTEM,
        max_tokens: int = 512,
        api_key: str | None = None,
    ) -> None:
        self._client = AsyncAnthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self._model = model
        self._system = system
        self._max_tokens = max_tokens
        self._history: list[dict] = []

    async def reply(self, user_text: str) -> str:
        if not user_text.strip():
            return "Не услышал ничего. Повтори, пожалуйста."

        self._history.append({"role": "user", "content": user_text})
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=self._system,
            messages=self._history,
        )
        assistant_text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        self._history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    def reset(self) -> None:
        self._history.clear()
