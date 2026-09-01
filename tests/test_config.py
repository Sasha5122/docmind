"""Settings must have safe defaults and reject unknown backends."""

import pytest
from pydantic import ValidationError

from docmind.config import Settings


def test_defaults_point_at_local_docker_db(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("LLM_BACKEND", raising=False)  # CI exports it (=ollama)
    settings = Settings(_env_file=None)
    assert settings.llm_backend == "azure"
    assert settings.embedding_backend == "local"
    assert settings.database_url == "postgresql+psycopg://docmind:docmind@localhost:5432/docmind"


def test_env_var_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    settings = Settings(_env_file=None)
    assert settings.llm_backend == "ollama"


def test_unknown_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BACKEND", "openai")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)
