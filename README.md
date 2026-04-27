# voice-agent

A LiveKit-backed voice gateway for Claude. Talk to Claude from a browser
with real STT/LLM/TTS streaming, VAD, turn-taking and barge-in. Pluggable
STT/TTS providers (SaluteSpeech, Yandex SpeechKit, ElevenLabs).

Pairs with [`claude-cli-voice-channel`](https://github.com/deminded/claude-cli-voice-channel)
to forward voice into a running Claude Code session — the worker silences
its own LLM in CLI mode and Claude Code answers via the plugin's `reply`
tool, which loops back through this agent's `/say` endpoint for TTS.

> This is the operator's reference implementation, published as-is. It
> runs in production at `va.noomarxism.ru` and is the same Claude that
> the operator talks to in Claude Code. Fork and customise — paths,
> identity files and providers are env-configurable.

## What it does

```
                ┌──────────────────────────────────────┐
                │           browser /lk PWA            │
                │  (mic, speaker, mode + provider UI)  │
                └───────────────┬──────────────────────┘
                                │ WebRTC (LiveKit Cloud)
                                ▼
        ┌─────────────────────────────────────────────────┐
        │            voice-agent worker                   │
        │   STT  →  Claude (Anthropic) + MCP tools  →  TTS│
        │            ▲                              │     │
        │            │ user_input_transcribed       │     │
        │            ▼                              ▼     │
        │   ┌─────────────────┐          ┌──────────────┐ │
        │   │  CLI mode:      │          │ Agent mode:  │ │
        │   │  POST /utterance│          │ session.say  │ │
        │   │  → plugin       │          │ via TTS      │ │
        │   └────────┬────────┘          └──────────────┘ │
        └────────────┼────────────────────────────────────┘
                     │ HTTP
                     ▼
            voice-channel-plugin (separate repo)
                     │
                     │ inject <channel> into Claude Code
                     ▼
              Claude Code session
                     │ reply tool → POST /say
                     ▼
              voice-agent worker → TTS → browser
```

Two roles in one stack:

1. **Standalone voice agent.** Open `/lk`, speak, get a streamed response
   from Claude with vault-memory recall.
2. **Voice I/O for Claude Code.** Toggle the `/lk` UI to *CLI* mode —
   transcripts forward to the plugin, Claude Code answers, and the answer
   is spoken back through this agent.

## Status

- **Phase A** — incoming voice into Claude Code (via the plugin's
  `/utterance` intake): working.
- **Phase B** — outgoing TTS bridge (plugin's `reply` tool POSTs to
  `/say`, the worker speaks via `session.say()`): working.
- **Phase C** — `/lk` UI toggle between *Voice Agent* (low-latency direct
  reply from the worker LLM) and *CLI* (transcript routed into a Claude
  Code main session): working.
- **Streaming STT + TTS**, **vault-memory MCP tool**, **session
  recording + Opus end-of-session reflection synthesis**, **PWA install
  on Android Chrome**: all implemented.

## Quick start

```bash
git clone <this-repo> voice-agent
cd voice-agent
uv venv --python 3.12
uv pip install -e .
cp .env.example .env
# edit .env — at minimum: ANTHROPIC_API_KEY, LIVEKIT_*, SALUTE_AUTH_KEY
.venv/bin/python -m voice_agent.livekit_agent dev
```

In another terminal:

```bash
.venv/bin/python -m uvicorn voice_agent.server:app --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/lk`, click *connect*, talk.

## Configuration

All config is via env (loaded from `.env`). See `.env.example` for the
full list with comments. The minimum to get a voice session is:

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | LLM (Claude) |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Transport (LiveKit Cloud) |
| `SALUTE_AUTH_KEY`, `SALUTE_SCOPE` | Default STT/TTS provider |

Optional providers and integrations:

| Variable | Effect |
| --- | --- |
| `YANDEX_API_KEY` | Enable Yandex SpeechKit as a switchable provider |
| `ELEVENLABS_API_KEY` | Enable ElevenLabs TTS in the pool |
| `EXA_API_KEY` | Adds an Exa-backed web search MCP tool |
| `VAULT_MEMORY_MCP_SCRIPT` | Path to a vault-memory MCP server (search/save_reflection tools) |
| `VOICE_AGENT_IDENTITY_FILE`, `VOICE_AGENT_MEMORY_DIR` | Personalise the system prompt (CLAUDE.md + memory files) |
| `VOICE_AGENT_USER_NAME` | Name woven into the synthesis prompt |
| `VOICE_AGENT_REFLECTIONS_DIR` | Where end-of-session reflections are written |
| `VOICE_AGENT_ACCESS_TOKEN` | Gate the web UI behind a single shared token |

## Deployment

`deploy/` contains systemd units and an nginx site template. They assume
a dedicated `voice-agent` user and `/opt/voice-agent` checkout — edit
the `.service` files for your layout.

```bash
sudo bash deploy/install.sh                    # install systemd units
sudo VOICE_DOMAIN=voice.example.com \
     VOICE_EMAIL=you@example.com \
     bash deploy/install_nginx.sh              # nginx + Let's Encrypt
```

The PWA shell (`/lk`, manifest, service worker, icons, `/health`) is
public so Chrome can validate it for install. All operational endpoints
(`/livekit/token`, `/say`, `/ws`) are gated by `VOICE_AGENT_ACCESS_TOKEN`
when set.

## Pairing with voice-channel-plugin

Install the plugin into Claude Code (see its
[README](https://github.com/deminded/claude-cli-voice-channel)). The
plugin's `reply` tool reads `VOICE_AGENT_ACCESS_TOKEN` from this repo's
`.env` (or its own) and POSTs to `http://127.0.0.1:8765/say`. CLI mode
in `/lk` forwards transcripts to the plugin's intake on
`http://127.0.0.1:8910/utterance` (override via `VOICE_CLI_INTAKE_URL`).

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

67 tests, mock-only — no API keys, no network.

## License

Apache License 2.0 — see [LICENSE](LICENSE).

Generated Yandex SpeechKit gRPC stubs in `src/voice_agent/yandex_grpc_stubs/`
are derived from [`yandex-cloud/cloudapi`](https://github.com/yandex-cloud/cloudapi)
(also Apache-2.0).
