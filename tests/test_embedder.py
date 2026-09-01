"""Embedder contract tests. The real bge-m3 test only runs with DOCMIND_RUN_SLOW=1."""

import math
import os

import pytest

from docmind.config import Settings
from docmind.ingest.embedder import AzureEmbedder, FakeEmbedder, LocalEmbedder, get_embedder
from docmind.models import EMBEDDING_DIM


def length(vector: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vector))


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_fake_embedder_is_deterministic_and_unit_length() -> None:
    embedder = FakeEmbedder()
    a, b = embedder.embed(["Versicherung", "Versicherung"])
    assert a == b
    assert len(a) == EMBEDDING_DIM == embedder.dim
    assert length(a) == pytest.approx(1.0)


def test_fake_embedder_separates_different_texts() -> None:
    a, b = FakeEmbedder().embed(["Praemie", "Franchise"])
    assert a != b


def test_fake_embedder_handles_empty_input() -> None:
    assert FakeEmbedder().embed([]) == []


def test_get_embedder_local_is_lazy() -> None:
    """Choosing the local backend must not download the model yet."""
    settings = Settings(_env_file=None, embedding_backend="local")
    embedder = get_embedder(settings)
    assert isinstance(embedder, LocalEmbedder)
    assert embedder._model is None
    assert embedder.dim == EMBEDDING_DIM


def test_get_embedder_azure_requires_credentials() -> None:
    settings = Settings(_env_file=None, embedding_backend="azure", azure_openai_api_key="")
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        get_embedder(settings)


def test_get_embedder_azure_builds_client() -> None:
    settings = Settings(
        _env_file=None,
        embedding_backend="azure",
        azure_openai_api_key="test-key",
        azure_openai_endpoint="https://example.openai.azure.com",
    )
    embedder = get_embedder(settings)
    assert isinstance(embedder, AzureEmbedder)
    assert embedder.deployment == "text-embedding-3-large"


@pytest.mark.skipif(os.environ.get("DOCMIND_RUN_SLOW") != "1", reason="downloads ~2 GB model")
def test_local_bge_m3_puts_same_meaning_closer_across_languages() -> None:
    embedder = LocalEmbedder()
    de, fr, unrelated = embedder.embed(
        [
            "Die Versicherung deckt Schaeden durch Feuer.",
            "L'assurance couvre les dommages causes par le feu.",
            "The train to Zurich leaves at nine.",
        ]
    )
    assert len(de) == EMBEDDING_DIM
    assert length(de) == pytest.approx(1.0, abs=1e-3)
    assert dot(de, fr) > dot(de, unrelated)  # cross-language meaning beats unrelated text
