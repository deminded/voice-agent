"""CLI: voice-tts "текст" --voice Tur_24000 --out file.opus

Sanity check that auth + TTS round-trip works without spinning up a server.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from voice_agent.providers.salute import SaluteTTS


async def _main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", help="text to synthesize (or stdin if omitted)")
    p.add_argument("--voice", default="Tur_24000")
    p.add_argument("--format", default="opus", choices=["opus", "wav", "mp3"])
    p.add_argument("--ssml", action="store_true")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        p.error("empty text")

    tts = SaluteTTS()
    audio = await tts.synthesize(
        text, voice=args.voice, format=args.format, ssml=args.ssml
    )
    args.out.write_bytes(audio)
    print(f"{len(audio)} bytes -> {args.out}", file=sys.stderr)
    return 0


def main() -> None:
    sys.exit(asyncio.run(_main()))


if __name__ == "__main__":
    main()
