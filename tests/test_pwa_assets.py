"""PWA asset tests — manifest, service worker, icons, lk.html integration."""
from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from voice_agent.server import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_manifest_json(client):
    r = client.get("/manifest.json")
    assert r.status_code == 200
    data = json.loads(r.text)
    assert data["name"] == "voice-agent"
    assert data["start_url"] == "/lk"


def test_sw_js(client):
    r = client.get("/sw.js")
    assert r.status_code == 200
    ct = r.headers.get("content-type", "")
    assert "javascript" in ct


def test_icon_192(client):
    r = client.get("/icon-192.png")
    assert r.status_code == 200
    # PNG magic bytes: \x89 P N G
    assert r.content[0] == 0x89
    assert r.content[1:4] == b"PNG"


def test_icon_512(client):
    r = client.get("/icon-512.png")
    assert r.status_code == 200
    assert r.content[0] == 0x89
    assert r.content[1:4] == b"PNG"


def test_lk_has_manifest_and_sw(client):
    r = client.get("/lk")
    assert r.status_code == 200
    body = r.text
    assert "manifest.json" in body
    assert "serviceWorker.register" in body
