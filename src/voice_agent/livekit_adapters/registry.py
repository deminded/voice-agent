"""Provider registry for STT/TTS adapters.

Single source of truth for all pluggable speech providers.
build_stt / build_tts instantiate with provider-appropriate defaults.

Behaviour on unknown name: raise ValueError (no silent fallback — caller must
validate before calling; entrypoint uses .get() with default="salute").
"""
from __future__ import annotations

from livekit.agents import stt, tts

from voice_agent.livekit_adapters.salute_stt import SaluteSTTAdapter
from voice_agent.livekit_adapters.salute_tts import SaluteTTSAdapter
from voice_agent.livekit_adapters.yandex_stt import YandexSTTAdapter
from voice_agent.livekit_adapters.yandex_tts import YandexTTSAdapter

STT_PROVIDERS: dict[str, type] = {
    "salute": SaluteSTTAdapter,
    "yandex": YandexSTTAdapter,
}

TTS_PROVIDERS: dict[str, type] = {
    "salute": SaluteTTSAdapter,
    "yandex": YandexTTSAdapter,
}

# Sensible defaults per provider so callers don't need to know internals.
_STT_DEFAULTS: dict[str, dict] = {
    "salute": {"language": "ru-RU"},
    "yandex": {"language": "ru-RU"},
}

_TTS_DEFAULTS: dict[str, dict] = {
    "salute": {"voice": "Tur_24000"},
    "yandex": {"voice": "anton"},
}


def build_stt(name: str, **kwargs) -> stt.STT:
    """Instantiate an STT adapter by provider name.

    Raises ValueError for unknown names so misconfigurations surface early.
    kwargs override per-provider defaults.
    """
    if name not in STT_PROVIDERS:
        raise ValueError(
            f"Unknown STT provider {name!r}. Available: {list(STT_PROVIDERS)}"
        )
    defaults = dict(_STT_DEFAULTS.get(name, {}))
    defaults.update(kwargs)
    return STT_PROVIDERS[name](**defaults)


def build_tts(name: str, **kwargs) -> tts.TTS:
    """Instantiate a TTS adapter by provider name.

    Raises ValueError for unknown names so misconfigurations surface early.
    kwargs override per-provider defaults.
    """
    if name not in TTS_PROVIDERS:
        raise ValueError(
            f"Unknown TTS provider {name!r}. Available: {list(TTS_PROVIDERS)}"
        )
    defaults = dict(_TTS_DEFAULTS.get(name, {}))
    defaults.update(kwargs)
    return TTS_PROVIDERS[name](**defaults)
