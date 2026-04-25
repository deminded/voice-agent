"""Push-to-talk WebSocket client for voice-agent.

Press Enter to start recording, press Enter again to stop and send.
Server replies with transcript, response text, and audio that we play back.

Run on the desktop with mic/speakers:
    pip install sounddevice numpy websockets
    python client/audio_client.py --server ws://localhost:8000/ws
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from queue import Queue, Empty

import numpy as np
import sounddevice as sd
import websockets


SAMPLE_RATE = 16000  # Salute STT works best at 16k mono
CHANNELS = 1
DTYPE = "int16"


def _wait_enter(prompt: str) -> None:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    sys.stdin.readline()


def record_until_enter() -> bytes:
    """Block-record until the user presses Enter, return raw PCM int16 bytes."""
    q: Queue[np.ndarray] = Queue()
    stop = threading.Event()

    def cb(indata, frames, time_info, status) -> None:
        if status:
            print(f"[mic] {status}", file=sys.stderr)
        q.put(indata.copy())

    print("recording... press Enter to stop")
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=cb
    )
    stream.start()

    def _stop_on_enter() -> None:
        sys.stdin.readline()
        stop.set()

    threading.Thread(target=_stop_on_enter, daemon=True).start()

    chunks: list[np.ndarray] = []
    while not stop.is_set():
        try:
            chunks.append(q.get(timeout=0.1))
        except Empty:
            continue
    stream.stop(); stream.close()
    while True:
        try:
            chunks.append(q.get_nowait())
        except Empty:
            break

    if not chunks:
        return b""

    audio = np.concatenate(chunks, axis=0)
    return audio.tobytes()


def play_opus(audio_bytes: bytes) -> None:
    """Decode opus through ffmpeg subprocess and play as PCM.

    Avoids pulling pyogg/opuslib as deps — ffmpeg is one binary install
    on every desktop OS.
    """
    import subprocess

    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le",
         "-ar", "24000", "-ac", "1", "pipe:1"],
        input=audio_bytes,
        capture_output=True,
        check=True,
    )
    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
    sd.play(pcm, samplerate=24000, blocking=True)


async def one_turn(ws_url: str) -> bool:
    """Run one push-to-talk turn. Returns False if the user wants to quit."""
    line = input("Press Enter to record, q+Enter to quit: ")
    if line.strip().lower() == "q":
        return False

    pcm = record_until_enter()
    if not pcm:
        print("(no audio captured)")
        return True

    print(f"sending {len(pcm)} bytes pcm16 ({len(pcm)/SAMPLE_RATE/2:.1f}s)...")
    async with websockets.connect(ws_url, max_size=20 * 1024 * 1024) as ws:
        await ws.send(json.dumps({"type": "audio_start", "format": "pcm16"}))
        await ws.send(pcm)
        await ws.send(json.dumps({"type": "audio_end"}))

        audio_chunks: list[bytes] = []
        audio_format = "opus"
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                evt = json.loads(msg)
                t = evt.get("type")
                if t == "transcript":
                    print(f"  YOU: {evt['text']}")
                elif t == "response_text":
                    print(f"  CLAUDE: {evt['text']}")
                elif t == "audio_start":
                    audio_format = evt.get("format", "opus")
                elif t == "audio_end":
                    break
                elif t == "error":
                    print(f"  ERROR: {evt['message']}")
                    return True
            else:
                audio_chunks.append(msg)

        audio = b"".join(audio_chunks)
        if audio:
            print(f"  ({len(audio)} bytes {audio_format} reply)")
            play_opus(audio)
    return True


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="ws://localhost:8000/ws")
    args = p.parse_args()
    print(f"voice-agent client -> {args.server}")
    while True:
        try:
            cont = await one_turn(args.server)
        except (KeyboardInterrupt, EOFError):
            break
        if not cont:
            break
    print("bye")


if __name__ == "__main__":
    asyncio.run(main())
