"""Token-gate middleware: blocks unauthorized HTTP and WS, lets healthcheck through.

We patch the ACCESS_TOKEN module global per test so we don't depend on env state.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from voice_agent import server as srv


@pytest.fixture
def secret(monkeypatch: pytest.MonkeyPatch) -> str:
    token = "test-secret-token-12345"
    monkeypatch.setattr(srv, "ACCESS_TOKEN", token)
    return token


def test_health_bypasses_token(secret: str) -> None:
    with TestClient(srv.app) as client:
        r = client.get("/health")
        assert r.status_code == 200


def test_lk_is_public(secret: str) -> None:
    """PWA shell must be reachable without auth so Chrome validates start_url."""
    with TestClient(srv.app) as client:
        r = client.get("/lk")
        assert r.status_code == 200


def test_pwa_assets_are_public(secret: str) -> None:
    with TestClient(srv.app) as client:
        for path in ("/manifest.json", "/sw.js", "/icon-192.png"):
            assert client.get(path).status_code == 200, path


def test_protected_endpoint_blocked_without_token(secret: str) -> None:
    with TestClient(srv.app) as client:
        r = client.get("/providers")
        assert r.status_code == 401


def test_protected_endpoint_allowed_with_key(secret: str) -> None:
    with TestClient(srv.app) as client:
        r = client.get(f"/providers?key={secret}")
        assert r.status_code == 200


def test_lk_query_key_promotes_cookie(secret: str) -> None:
    with TestClient(srv.app) as client:
        r = client.get(f"/lk?key={secret}")
        assert r.status_code == 200
        assert "va_access" in r.cookies


def test_cookie_unlocks_protected_endpoint(secret: str) -> None:
    with TestClient(srv.app) as client:
        client.get(f"/lk?key={secret}")  # sets cookie via public path
        r = client.get("/providers")     # protected, but cookie is now sent
        assert r.status_code == 200


def test_wrong_token_rejected(secret: str) -> None:
    with TestClient(srv.app) as client:
        r = client.get("/providers?key=wrong")
        assert r.status_code == 401


def test_ws_rejects_without_token(secret: str) -> None:
    with TestClient(srv.app) as client:
        # token gate fires before app accepts the upgrade — handshake fails
        with pytest.raises(Exception):
            with client.websocket_connect("/ws"):
                pass


def test_ws_allowed_with_query_key(secret: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Smoke check the WS path admits a valid token before the protocol kicks in."""
    from voice_agent.providers.echo import EchoLLM

    class _MockSTT:
        async def transcribe(self, audio: bytes, *, format: str = "opus") -> str:
            return "x"

    class _MockTTS:
        async def synthesize(self, text: str, *, voice: str = "Tur_24000", format: str = "opus", ssml: bool = False) -> bytes:
            return b""

    with TestClient(srv.app) as client:
        srv.app.state.stt = _MockSTT()
        srv.app.state.tts = _MockTTS()
        srv.app.state.tts_pool = {"mock": _MockTTS()}
        srv.app.state.default_tts_name = "mock"
        srv.app.state.llm_factory = lambda: EchoLLM()

        with client.websocket_connect(f"/ws?key={secret}") as ws:
            ws.send_text(json.dumps({"type": "audio_start", "format": "pcm16"}))
            ws.send_bytes(b"x")
            ws.send_text(json.dumps({"type": "audio_end"}))
            evt = json.loads(ws.receive_text())
            assert evt["type"] == "transcript"


def test_token_disabled_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(srv, "ACCESS_TOKEN", "")
    with TestClient(srv.app) as client:
        r = client.get("/lk")
        assert r.status_code == 200  # no auth required


def test_stale_cookie_refreshes_on_query_key(secret: str) -> None:
    """After token rotation the operator hits /lk?key=<new> with an old
    cookie still in the browser; the middleware must overwrite it instead
    of leaving the stale cookie to block subsequent /livekit/token calls."""
    with TestClient(srv.app) as client:
        client.cookies.set("va_access", "stale-old-token")
        r = client.get(f"/lk?key={secret}")
        assert r.status_code == 200
        assert r.cookies.get("va_access") == secret


def test_token_compare_is_constant_time() -> None:
    """_token_eq must use secrets.compare_digest. We don't measure timing
    here, just confirm it tolerates None and string types without crashing
    and rejects non-equal values without raising."""
    assert srv._token_eq(None, "x") is False
    assert srv._token_eq("", "x") is False
    assert srv._token_eq("x", "x") is True
    assert srv._token_eq("x", "y") is False


def test_livekit_token_rejects_invalid_identity(secret: str) -> None:
    with TestClient(srv.app) as client:
        r = client.get(
            f"/livekit/token?key={secret}&identity=../../etc/passwd&provider=salute"
        )
        assert r.status_code == 400
        assert "identity" in r.json()["detail"].lower()


def test_livekit_token_rejects_unknown_provider(secret: str) -> None:
    with TestClient(srv.app) as client:
        r = client.get(
            f"/livekit/token?key={secret}&identity=alice&provider=acme"
        )
        assert r.status_code == 400
        assert "provider" in r.json()["detail"].lower()


def test_livekit_token_accepts_valid_inputs(secret: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Identity in [A-Za-z0-9_-]{1,64} and provider in {salute, yandex} pass.
    LIVEKIT_* env may be missing in CI — accept both 200 and 500-ish env-error."""
    with TestClient(srv.app) as client:
        r = client.get(
            f"/livekit/token?key={secret}&identity=alice_bob-1&provider=yandex"
        )
        # Validation passes (no 400). LiveKit env may be missing in test env,
        # in which case the handler returns 200 with {"error": "..."}.
        assert r.status_code == 200
