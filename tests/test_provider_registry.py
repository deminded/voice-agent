"""Unit tests for provider registry and Yandex adapter instantiation.

All tests are pure unit tests: no network, no real gRPC calls.
Yandex adapters are instantiated with mocked YANDEX_API_KEY env var.

Unknown provider name -> ValueError (no silent fallback).
This is intentional: misconfig should fail loudly at startup.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from voice_agent.livekit_adapters.registry import (
    STT_PROVIDERS,
    TTS_PROVIDERS,
    build_stt,
    build_tts,
)
from voice_agent.livekit_adapters.salute_stt import SaluteSTTAdapter
from voice_agent.livekit_adapters.salute_tts import SaluteTTSAdapter
from voice_agent.livekit_adapters.yandex_stt import YandexSTTAdapter
from voice_agent.livekit_adapters.yandex_tts import YandexTTSAdapter


# ---------------------------------------------------------------------------
# Registry contents
# ---------------------------------------------------------------------------

def test_stt_providers_contains_salute_and_yandex():
    assert "salute" in STT_PROVIDERS
    assert "yandex" in STT_PROVIDERS


def test_tts_providers_contains_salute_and_yandex():
    assert "salute" in TTS_PROVIDERS
    assert "yandex" in TTS_PROVIDERS


# ---------------------------------------------------------------------------
# build_stt
# ---------------------------------------------------------------------------

def test_build_stt_salute_returns_correct_type():
    adapter = build_stt("salute")
    assert isinstance(adapter, SaluteSTTAdapter)


def test_build_stt_yandex_returns_correct_type():
    # Yandex adapter constructor never reads the key — defer to first use.
    adapter = build_stt("yandex")
    assert isinstance(adapter, YandexSTTAdapter)


def test_build_stt_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown STT provider"):
        build_stt("unknown_provider")


def test_build_stt_unknown_does_not_fallback():
    # Explicit design: unknown -> ValueError, not silent salute fallback.
    with pytest.raises(ValueError):
        build_stt("openai_whisper")


# ---------------------------------------------------------------------------
# build_tts
# ---------------------------------------------------------------------------

def test_build_tts_salute_returns_correct_type():
    adapter = build_tts("salute")
    assert isinstance(adapter, SaluteTTSAdapter)


def test_build_tts_yandex_returns_correct_type():
    adapter = build_tts("yandex")
    assert isinstance(adapter, YandexTTSAdapter)


def test_build_tts_unknown_raises_value_error():
    with pytest.raises(ValueError, match="Unknown TTS provider"):
        build_tts("elevenlabs_v2")


# ---------------------------------------------------------------------------
# Yandex adapter instantiation (no key in env — still instantiates)
# ---------------------------------------------------------------------------

def test_yandex_stt_instantiates_without_api_key():
    """Constructor must not read key — only _get_api_key() raises."""
    with patch.dict(os.environ, {}, clear=False):
        env = dict(os.environ)
        env.pop("YANDEX_API_KEY", None)
        with patch.dict(os.environ, env, clear=True):
            adapter = YandexSTTAdapter(language="ru-RU")
            assert adapter._language == "ru-RU"
            assert adapter._api_key is None


def test_yandex_stt_get_api_key_raises_when_missing():
    env = dict(os.environ)
    env.pop("YANDEX_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        adapter = YandexSTTAdapter()
        with pytest.raises(RuntimeError, match="YANDEX_API_KEY"):
            adapter._get_api_key()


def test_yandex_tts_instantiates_without_api_key():
    """Constructor must not read key — only _get_api_key() raises."""
    env = dict(os.environ)
    env.pop("YANDEX_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        adapter = YandexTTSAdapter(voice="oksana")
        assert adapter._voice == "oksana"
        assert adapter._api_key is None


def test_yandex_tts_get_api_key_raises_when_missing():
    env = dict(os.environ)
    env.pop("YANDEX_API_KEY", None)
    with patch.dict(os.environ, env, clear=True):
        adapter = YandexTTSAdapter()
        with pytest.raises(RuntimeError, match="YANDEX_API_KEY"):
            adapter._get_api_key()


def test_yandex_stt_reads_api_key_from_env():
    with patch.dict(os.environ, {"YANDEX_API_KEY": "test-key-123"}):
        adapter = YandexSTTAdapter()
        assert adapter._get_api_key() == "test-key-123"
        # Cached after first read
        assert adapter._api_key == "test-key-123"


def test_yandex_tts_reads_api_key_from_env():
    with patch.dict(os.environ, {"YANDEX_API_KEY": "test-key-456"}):
        adapter = YandexTTSAdapter()
        assert adapter._get_api_key() == "test-key-456"


# ---------------------------------------------------------------------------
# Per-room provider state simulation (mirrors entrypoint logic)
# ---------------------------------------------------------------------------

def _simulate_provider_from_metadata(metadata_str: str) -> str:
    """Replicate the entrypoint metadata-reading logic for testing."""
    import json
    try:
        meta = json.loads(metadata_str or "{}")
        candidate = meta.get("provider", "salute")
        return candidate if candidate in STT_PROVIDERS else "salute"
    except Exception:
        return "salute"


def test_provider_state_salute_from_metadata():
    assert _simulate_provider_from_metadata('{"provider": "salute"}') == "salute"


def test_provider_state_yandex_from_metadata():
    assert _simulate_provider_from_metadata('{"provider": "yandex"}') == "yandex"


def test_provider_state_unknown_falls_back_to_salute():
    assert _simulate_provider_from_metadata('{"provider": "openai"}') == "salute"


def test_provider_state_empty_metadata_defaults_to_salute():
    assert _simulate_provider_from_metadata("") == "salute"
    assert _simulate_provider_from_metadata("{}") == "salute"
    assert _simulate_provider_from_metadata(None) == "salute"


def test_provider_state_invalid_json_falls_back_to_salute():
    assert _simulate_provider_from_metadata("not-json") == "salute"
