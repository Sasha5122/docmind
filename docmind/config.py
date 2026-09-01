"""Application settings, loaded from environment variables / a `.env` file.

Every module that needs a secret or a switch (LLM backend, DB URL, ...) reads it
from here. Nothing else in the code base touches `os.environ` directly.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LlmBackend = Literal["azure", "anthropic", "ollama"]
EmbeddingBackend = Literal["local", "azure"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- database ---
    database_url: str = Field(
        default="postgresql+psycopg://docmind:docmind@localhost:5432/docmind",
        description="SQLAlchemy URL; must match docker-compose.yml",
    )

    # --- LLM backend (the data-residency switch) ---
    llm_backend: LlmBackend = "azure"
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_deployment: str = "gpt-4o-mini"
    azure_openai_api_version: str = "2024-10-21"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # --- embeddings ---
    embedding_backend: EmbeddingBackend = "local"
    local_embedding_model: str = "BAAI/bge-m3"
    azure_embedding_deployment: str = "text-embedding-3-large"

    # --- retrieval ---
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- observability ---
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings object (parsed once, then cached)."""
    return Settings()
