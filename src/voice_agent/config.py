"""Settings loaded from voice-agent/.env. Real env wins."""
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    salute_auth_key: str = Field(...)
    salute_scope: str = "SALUTE_SPEECH_PERS"
    salute_oauth_url: str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    salute_api_url: str = "https://smartspeech.sber.ru/rest/v1"

    elevenlabs_api_key: str | None = None


settings = Settings()
