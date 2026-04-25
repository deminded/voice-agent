#!/usr/bin/env python
"""Timing verification for streaming STT.

Pushes pre-generated PCM16 audio (sine wave at speech-like 200 Hz) to
the SaluteSTTAdapter stream in real-time pacing and measures:
  - time to first interim transcript
  - time to final transcript after end_input()
  - count of interim updates

Usage (requires real .env with SALUTE_AUTH_KEY):
  cd /home/claude-user/voice-agent
  .venv/bin/python scripts/verify_streaming_stt.py [--duration 5] [--wav /tmp/test.wav]

If a WAV file is provided it must be 16kHz mono PCM16.
Without --wav the script synthesises a 200 Hz sine wave for --duration seconds.
"""
from __future__ import annotations

import argparse
import asyncio
import math
import struct
import time
import wave
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from livekit import rtc
from livekit.agents import stt

from voice_agent.livekit_adapters.salute_stt import SaluteSTTAdapter


SAMPLE_RATE = 16000
CHUNK_MS = 320   # 320 ms per push — matches gRPC _CHUNK_BYTES boundary
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000  # 5120 samples


def _make_sine_pcm(duration_s: float, freq: float = 200.0) -> bytes:
    n = int(SAMPLE_RATE * duration_s)
    samples = [int(32767 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n)]
    return struct.pack(f"<{n}h", *samples)


def _load_wav(path: str) -> bytes:
    with wave.open(path, "rb") as f:
        assert f.getsampwidth() == 2, "WAV must be 16-bit"
        assert f.getnchannels() == 1, "WAV must be mono"
        assert f.getframerate() == SAMPLE_RATE, f"WAV must be {SAMPLE_RATE} Hz"
        return f.readframes(f.getnframes())


async def main(duration: float, wav_path: str | None) -> None:
    pcm = _load_wav(wav_path) if wav_path else _make_sine_pcm(duration)
    print(f"Audio: {len(pcm)/2/SAMPLE_RATE:.2f}s of PCM16 ({len(pcm)} bytes)")

    adapter = SaluteSTTAdapter(language="ru-RU")
    stream = adapter.stream()

    # Producer: push frames at real-time pace then signal end_input.
    t0 = time.perf_counter()

    async def _push() -> None:
        for offset in range(0, len(pcm), CHUNK_SAMPLES * 2):
            chunk = pcm[offset : offset + CHUNK_SAMPLES * 2]
            if not chunk:
                break
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=len(chunk) // 2,
            )
            stream.push_frame(frame)
            await asyncio.sleep(CHUNK_MS / 1000)
        t_eoi = time.perf_counter() - t0
        print(f"  end_input() at t={t_eoi:.3f}s")
        stream.end_input()

    push_task = asyncio.create_task(_push())

    # Consumer: collect events and timestamp them.
    t_first_interim: float | None = None
    t_final: float | None = None
    interim_count = 0
    t_eoi_ref: float | None = None

    async for event in stream:
        t_now = time.perf_counter() - t0
        if event.type == stt.SpeechEventType.START_OF_SPEECH:
            print(f"  START_OF_SPEECH  t={t_now:.3f}s")
        elif event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
            interim_count += 1
            if t_first_interim is None:
                t_first_interim = t_now
                print(f"  FIRST INTERIM    t={t_first_interim:.3f}s  text={event.alternatives[0].text!r}")
            else:
                print(f"  INTERIM #{interim_count}      t={t_now:.3f}s  text={event.alternatives[0].text!r}")
        elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
            t_final = t_now
            print(f"  FINAL            t={t_final:.3f}s  text={event.alternatives[0].text!r}")
        elif event.type == stt.SpeechEventType.END_OF_SPEECH:
            print(f"  END_OF_SPEECH    t={t_now:.3f}s")

    await push_task

    total = time.perf_counter() - t0
    print()
    print("=== Summary ===")
    print(f"  Total wall time:         {total:.3f}s")
    print(f"  Time to first interim:   {t_first_interim:.3f}s" if t_first_interim else "  No interim transcript received")
    if t_final is not None:
        # Approximate EOI time from total audio length + small overhead.
        eoi_approx = len(pcm) / 2 / SAMPLE_RATE
        lag = t_final - eoi_approx
        print(f"  Time to final (after EOI): {lag:.3f}s  (final at {t_final:.3f}s, audio ends ~{eoi_approx:.3f}s)")
    else:
        print("  No final transcript received (silence audio returns empty text — expected)")
    print(f"  Interim updates:         {interim_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0, help="Sine wave duration in seconds (default 5)")
    parser.add_argument("--wav", type=str, default=None, help="Path to 16kHz mono PCM16 WAV file")
    args = parser.parse_args()
    asyncio.run(main(args.duration, args.wav))
