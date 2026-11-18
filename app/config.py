"""Application settings loader combining environment variables and YAML configs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PIPELINE_CONFIG_PATH = PROJECT_ROOT / "configs" / "pipeline.yaml"
DEFAULT_FLAKE8_CONFIG_PATH = PROJECT_ROOT / "configs" / ".flake8"


def load_pipeline_yaml(
    config_path: Path = DEFAULT_PIPELINE_CONFIG_PATH,
) -> dict[str, Any]:
    """Load pipeline YAML configuration if present."""
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


class Settings(BaseSettings):
    """Typed runtime configuration loaded from `.env` and environment variables."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Secrets & external API credentials
    github_token: str = Field(default="", alias="GITHUB_TOKEN")
    github_webhook_secret: str = Field(default="", alias="GITHUB_WEBHOOK_SECRET")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # LLM parameters
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    llm_temperature: float = Field(default=0.1, alias="LLM_TEMPERATURE")

    # Persistence checkpointer
    checkpointer_type: Literal["sqlite", "memory"] = Field(
        default="sqlite", alias="CHECKPOINTER_TYPE"
    )
    sqlite_db_path: str = Field(
        default="data/review_checkpoints.sqlite", alias="SQLITE_DB_PATH"
    )

    # Sandbox linter configuration
    linter_timeout_seconds: int = Field(default=15, alias="LINTER_TIMEOUT_SECONDS")
    enable_flake8_check: bool = Field(default=True, alias="ENABLE_FLAKE8_CHECK")
    enable_black_check: bool = Field(default=True, alias="ENABLE_BLACK_CHECK")

    # Runtime flags
    dry_run_mode: bool = Field(default=True, alias="DRY_RUN_MODE")
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")

    # GitHub API settings
    github_api_base_url: str = Field(
        default="https://api.github.com", alias="GITHUB_API_BASE_URL"
    )
    github_timeout_seconds: int = Field(default=20, alias="GITHUB_TIMEOUT_SECONDS")

    @property
    def resolved_sqlite_path(self) -> Path:
        """Resolve SQLite checkpoint DB path relative to project root."""
        path = Path(self.sqlite_db_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    @property
    def flake8_config_path(self) -> Path:
        """Path to the flake8 configuration file."""
        return DEFAULT_FLAKE8_CONFIG_PATH

    @property
    def has_github_credentials(self) -> bool:
        """Return True when a non-placeholder GitHub token is configured."""
        return bool(
            self.github_token
            and not self.github_token.startswith("ghp_your_")
            and self.github_token != "your_github_token"
        )

    @property
    def has_openai_credentials(self) -> bool:
        """Return True when a non-placeholder OpenAI API key is configured."""
        return bool(
            self.openai_api_key
            and not self.openai_api_key.startswith("sk-your_")
            and self.openai_api_key != "your_openai_api_key"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()
