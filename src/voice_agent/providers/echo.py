"""Stub LLM that just echoes the user prompt. Lets us prove the audio
pipeline end-to-end without burning Anthropic tokens or waiting on
real-Claude wiring."""
from __future__ import annotations


class EchoLLM:
    async def reply(self, user_text: str) -> str:
        if not user_text.strip():
            return "Не услышал ничего. Повтори, пожалуйста."
        return f"Я услышал: {user_text}"
