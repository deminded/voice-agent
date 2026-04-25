"""pytest plumbing — async by default."""
import pytest


@pytest.fixture(autouse=True)
def _no_real_secrets(monkeypatch):
    """Tests must not need real creds. Stub the env to be safe."""
    monkeypatch.setenv("SALUTE_AUTH_KEY", "test-only")
    monkeypatch.setenv("SALUTE_SCOPE", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-only")
