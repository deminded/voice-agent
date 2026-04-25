"""Verify that SaluteSpeech /text:synthesize actually streams chunks incrementally.

Prints TTFB, chunk count, total bytes, and total duration so we can judge
whether server-side streaming is real or the body arrives all at once.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from voice_agent.providers.salute import SaluteTTS  # noqa: E402 (after load_dotenv)

TEST_TEXT = (
    "Привет, Евгений. "
    "Это тест потокового синтеза речи. "
    "Если всё работает — слышно сразу."
)


async def main() -> None:
    tts = SaluteTTS()
    chunks: list[tuple[float, int]] = []  # (relative_ts, len)
    start = time.perf_counter()

    async for chunk in tts.synthesize_stream(TEST_TEXT, voice="Tur_24000", format="pcm16"):
        ts = time.perf_counter() - start
        chunks.append((ts, len(chunk)))

    end = time.perf_counter() - start

    if not chunks:
        print("ERROR: no chunks received")
        return

    ttfb = chunks[0][0]
    total_bytes = sum(n for _, n in chunks)
    # Tur_24000: 24000 samples/sec * 2 bytes = 48000 bytes/sec
    expected_speech_sec = total_bytes / 48000

    print(f"Chunks received : {len(chunks)}")
    print(f"Total bytes     : {total_bytes:,}  (~{expected_speech_sec:.1f}s of speech at 24kHz pcm16)")
    print(f"TTFB            : {ttfb*1000:.0f} ms")
    print(f"Total duration  : {end*1000:.0f} ms  (request → last byte)")
    print()
    if len(chunks) > 1:
        print("Chunk timeline (relative ms : bytes):")
        for ts, n in chunks:
            print(f"  {ts*1000:6.0f} ms : {n} bytes")
    else:
        print("Single chunk — Salute is NOT streaming server-side.")

    print()
    if ttfb < 0.5 and len(chunks) > 3:
        print("VERDICT: real streaming — TTFB well under 500 ms, multiple chunks.")
    elif len(chunks) == 1:
        print("VERDICT: no streaming — entire response arrived as one chunk.")
    else:
        print("VERDICT: partial streaming or large chunks — check timeline above.")


if __name__ == "__main__":
    asyncio.run(main())
