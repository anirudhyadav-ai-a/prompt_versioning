"""Configuration management via environment variables.

Loads settings from a .env file using python-dotenv and exposes them
through a validated Pydantic model.
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=1024)

    model_config = {"env_prefix": "", "populate_by_name": True}


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
