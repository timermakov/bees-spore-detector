from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Bee BioData Platform"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False
    cors_origins: str = "http://localhost:5173"

    database_url: str = Field(
        default="postgresql+psycopg://bees:bees@localhost:5432/bees_db"
    )
    upload_dir: str = "uploads"
    analysis_mode: str = "yolo"

    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[2] / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @property
    def cors_origins_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
