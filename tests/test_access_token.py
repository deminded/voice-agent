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
