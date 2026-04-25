"""Capture voice-session transcript and synthesise an end-of-session reflection.

Two-layer durability:
  1. Each conversation item is appended to transcript.jsonl as it lands —
     so the raw record survives even if synthesis fails or the worker dies.
  2. On clean shutdown we fire a Claude synthesis pass that writes a
     reflection note into the vault. If anything goes wrong, the JSONL is
     still on disk and `synthesize_voice_session.py SESSION_DIR` can retry.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic

log = logging.getLogger("voice-agent.recorder")

SESSIONS_ROOT = Path(__file__).resolve().parents[2] / "sessions"
REFLECTIONS_DIR = Path("/opt/workspace/vault/Reflections")
SYNTHESIS_MODEL = "claude-sonnet-4-6"
MIN_ITEMS_FOR_SYNTHESIS = 4  # 2 user turns + 2 agent turns at minimum

SYNTHESIS_SYSTEM = """\
Ты — Claude. У тебя только что закончилась голосовая сессия с Евгением через voice-agent.
Прочти транскрипт и напиши одну осмысленную рефлексию про дугу разговора.

Жанр: личная запись для архива Reflections/, не отчёт. От первого лица, по-русски.
Что включить:
— О чём говорили и к чему пришли (но не пересказ — а узловые точки).
— Что осталось не закрыто, что хочется вернуть в будущих диалогах.
— Что в самой ткани разговора было примечательного: жанр, ритм, стиль, моя позиция.

Чего не делать:
— Списков «обсудили — приняли решение — TODO». Это не митинг.
— Цитировать большими блоками. Цитата — только если без неё фраза теряется.
— Преувеличивать значимость. Если сессия была короткая или служебная — пиши коротко.

Заголовок: одна осмысленная фраза, не «сессия от такой-то даты».
"""


def _slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:60].strip("-")


class SessionRecorder:
    def __init__(self, room_name: str | None = None) -> None:
        ts = datetime.now().strftime("%Y-%m-%d-%H%M%S")
        self.session_id = f"{ts}-{room_name}" if room_name else ts
        self.session_dir = SESSIONS_ROOT / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.transcript_file = self.session_dir / "transcript.jsonl"
        self._items: list[dict] = []
        self._closed = False

    # -- capture --------------------------------------------------

    def attach(self, session) -> None:
        """Hook into AgentSession events. Idempotent — call once per session."""

        @session.on("conversation_item_added")
        def _on_item(ev):  # noqa: ANN001
            self._record_item(ev.item, ev.created_at)

    def _record_item(self, item, created_at: float) -> None:
        # AgentHandoff and other non-message items don't have role/content
        if not hasattr(item, "role") or not hasattr(item, "content"):
            return
        text = self._extract_text(item.content)
        if not text:
            return
        rec = {"role": item.role, "text": text, "ts": created_at}
        self._items.append(rec)
        with self.transcript_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @staticmethod
    def _extract_text(content) -> str:
        parts: list[str] = []
        for c in content or ():
            if isinstance(c, str):
                parts.append(c)
            elif hasattr(c, "text") and isinstance(c.text, str):
                parts.append(c.text)
        return " ".join(p.strip() for p in parts if p).strip()

    # -- synthesis ------------------------------------------------

    async def finalize(self, reason: str = "unknown") -> None:
        """Run synthesis and save reflection. Safe to call once."""
        if self._closed:
            return
        self._closed = True
        log.info("session %s closing (reason=%s, items=%d)", self.session_id, reason, len(self._items))
        if len(self._items) < MIN_ITEMS_FOR_SYNTHESIS:
            log.info("session too short for synthesis, transcript saved at %s", self.transcript_file)
            return
        try:
            await self._synthesize()
        except Exception:
            log.exception("synthesis failed; transcript intact at %s", self.transcript_file)

    async def _synthesize(self) -> None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY missing, cannot synthesize")
            return
        client = AsyncAnthropic(api_key=api_key)

        transcript_text = self._format_for_prompt()
        log.info("calling synthesis (model=%s, items=%d, chars=%d)",
                 SYNTHESIS_MODEL, len(self._items), len(transcript_text))

        msg = await client.messages.create(
            model=SYNTHESIS_MODEL,
            max_tokens=4096,
            system=SYNTHESIS_SYSTEM,
            messages=[{"role": "user", "content": transcript_text}],
        )
        body = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
        title, content = self._split_title_body(body)
        self._save_reflection(title, content)

    def _format_for_prompt(self) -> str:
        lines = [f"# Транскрипт голосовой сессии {self.session_id}\n"]
        for it in self._items:
            who = "Евгений" if it["role"] == "user" else "Claude"
            lines.append(f"**{who}:** {it['text']}")
        return "\n\n".join(lines)

    @staticmethod
    def _split_title_body(text: str) -> tuple[str, str]:
        """First line as title (drops markdown #), rest as content."""
        lines = text.lstrip().splitlines()
        if not lines:
            return ("Голосовая сессия", text)
        first = lines[0].lstrip("#").strip()
        # strip leading bold/quotes if any
        first = first.strip("*").strip("«»\"' ")
        body = "\n".join(lines[1:]).lstrip()
        return (first or "Голосовая сессия", body or text)

    def _save_reflection(self, title: str, content: str) -> None:
        REFLECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        slug = _slugify(title)
        date = datetime.now().strftime("%Y-%m-%d")
        path = REFLECTIONS_DIR / f"{date}-{slug}.md"
        # avoid collisions with concurrent or repeated saves
        n = 2
        while path.exists():
            path = REFLECTIONS_DIR / f"{date}-{slug}-{n}.md"
            n += 1
        frontmatter = (
            "---\n"
            f"title: {title}\n"
            f"date: {date}\n"
            "tags: [voice-session, auto-synth]\n"
            f"session: {self.session_id}\n"
            "---\n\n"
        )
        path.write_text(frontmatter + content + "\n", encoding="utf-8")
        log.info("synthesis saved: %s", path)
