# voice-agent

Голосовой интерфейс к Claude instance с доступом к памяти.

**Стек:** SaluteSpeech (STT/TTS) → EchoLLM (заглушка вместо Claude) → WebSocket-сервер → push-to-talk клиент с микрофоном на десктопе.

## Что готово

- `SaluteSpeech` provider: OAuth с автообновлением, TTS (`Tur_24000` голос по умолчанию), STT через `speech:recognize`.
- `Conversation` — однооборотный pipeline `audio → STT → LLM → TTS → audio` с pluggable провайдерами.
- WebSocket-сервер (`/ws`): принимает `audio_start` + binary chunks + `audio_end`, отдаёт transcript, response_text, audio.
- Push-to-talk клиент (`client/audio_client.py`) с записью с микрофона и воспроизведением ответа.
- Mock-провайдеры + pytest (`tests/`) — pipeline и WebSocket контракт проверяются in-process без сети.
- CLI `python -m voice_agent.tools.tts ...` для разовых синтезов.

## Запуск сервера

На VPS (где `.env` с `SALUTE_AUTH_KEY`):

```bash
cd voice-agent
uv venv --python 3.12
uv pip install -e .
.venv/bin/python -m uvicorn voice_agent.server:app --host 127.0.0.1 --port 8765
```

Health-check: `curl http://127.0.0.1:8765/health`.

## Запуск клиента (десктоп)

Через SSH-туннель к серверу:

```bash
ssh -L 8765:127.0.0.1:8765 user@vps
```

Потом локально:

```bash
pip install sounddevice numpy websockets
# нужен ffmpeg в PATH для воспроизведения opus-ответов

python client/audio_client.py --server ws://localhost:8765/ws
```

Push-to-talk: `Enter` → запись → `Enter` → отправка. `q + Enter` — выход.

## Тесты

```bash
.venv/bin/python -m pytest tests/ -v
```

Тесты используют mock-провайдеры — никаких реальных API-вызовов, никаких ключей.

## Дальше

- LLM: подключить Anthropic SDK с MCP `vault-memory` вместо `EchoLLM`.
- VAD: silero-vad для open-mic режима (без push-to-talk).
- Moderator-режим: PTT + явная передача слова через клиент фасилитатора.
- SSML на лету: автоматическое расставление пауз и акцентов в ответе.
- Дополнительные транспорты: WebRTC, Telegram voice, Recall.ai/Zoom SDK для созвонов.
