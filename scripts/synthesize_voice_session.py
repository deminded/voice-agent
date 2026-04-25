#!/usr/bin/env python3
"""Re-synthesise a voice session reflection from a saved transcript.jsonl.

Used when the on-close synthesis failed (network blip, API quota), or to
regenerate after tweaking the synthesis prompt.

Usage:
    .venv/bin/python scripts/synthesize_voice_session.py sessions/2026-04-25-180530-livekit-room
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")

from voice_agent.session_recorder import SessionRecorder


async def main(session_dir: Path) -> None:
    transcript = session_dir / "transcript.jsonl"
    if not transcript.exists():
        sys.exit(f"no transcript at {transcript}")

    rec = SessionRecorder.__new__(SessionRecorder)
    rec.session_id = session_dir.name
    rec.session_dir = session_dir
    rec.transcript_file = transcript
    rec._items = [json.loads(l) for l in transcript.read_text(encoding="utf-8").splitlines() if l.strip()]
    rec._closed = False

    if not rec._items:
        sys.exit("transcript empty")
    print(f"loaded {len(rec._items)} items, running synthesis…")
    await rec._synthesize()
    print("done")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: synthesize_voice_session.py <session-dir>")
    asyncio.run(main(Path(sys.argv[1]).resolve()))
