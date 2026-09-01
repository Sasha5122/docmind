"""Turn text into vectors (lists of numbers) so the database can search by meaning.

One `Embedder` contract, three implementations:
- `LocalEmbedder`  : BAAI/bge-m3 through sentence-transformers, runs on this machine,
                     no API key, no data leaves the laptop (the data-residency option).
- `AzureEmbedder`  : Azure OpenAI text-embedding-3-large, asked for 1024 dimensions so
                     it fits the same `Vector(1024)` column as bge-m3.
- `FakeEmbedder`   : deterministic hash-based vectors for tests; no model, instant.

All vectors are unit length (normalised), so cosine similarity == dot product and
pgvector's `<=>` (cosine distance) operator can be used for every backend.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

from docmind.config import Settings, get_settings
from docmind.models import EMBEDDING_DIM

_AZURE_BATCH = 100  # inputs per Azure request; well under the 2048 hard limit


class Embedder(Protocol):
    """Anything with `.dim` and `.embed(texts) -> vectors` can be plugged into ingestion."""

    dim: int

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text, each `dim` floats long."""
        ...


class FakeEmbedder:
    """Deterministic stand-in: same text -> same vector, different text -> different vector.

    Vectors carry no meaning; they only let tests exercise the pipeline offline.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dim:
            digest = hashlib.sha256(f"{counter}:{text}".encode()).digest()
            values.extend(b / 255.0 - 0.5 for b in digest)
            counter += 1
        return _normalise(values[: self.dim])


class LocalEmbedder:
    """bge-m3 on the local CPU/GPU. The model (~2.2 GB) is downloaded on first use."""

    dim = EMBEDDING_DIM

    def __init__(self, model_name: str = "BAAI/bge-m3", batch_size: int = 16) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None  # loaded lazily so importing this module stays cheap

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()


class AzureEmbedder:
    """Azure OpenAI embeddings, forced to `dim` so both backends share one table."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        deployment: str,
        api_version: str,
        dim: int = EMBEDDING_DIM,
    ) -> None:
        if not api_key or not endpoint:
            raise ValueError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set for "
                "EMBEDDING_BACKEND=azure"
            )
        from openai import AzureOpenAI

        self.dim = dim
        self.deployment = deployment
        self._client = AzureOpenAI(
            api_key=api_key, azure_endpoint=endpoint, api_version=api_version
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _AZURE_BATCH):
            batch = list(texts[start : start + _AZURE_BATCH])
            response = self._client.embeddings.create(
                model=self.deployment, input=batch, dimensions=self.dim
            )
            # The API may return items out of order; `index` restores the input order.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(_normalise(item.embedding) for item in ordered)
        return vectors


def get_embedder(settings: Settings | None = None) -> Embedder:
    """Pick the backend named by `EMBEDDING_BACKEND` (local | azure)."""
    settings = settings or get_settings()
    if settings.embedding_backend == "azure":
        return AzureEmbedder(
            api_key=settings.azure_openai_api_key,
            endpoint=settings.azure_openai_endpoint,
            deployment=settings.azure_embedding_deployment,
            api_version=settings.azure_openai_api_version,
        )
    return LocalEmbedder(model_name=settings.local_embedding_model)


def _normalise(values: Sequence[float]) -> list[float]:
    """Scale a vector to length 1 (leaves an all-zero vector unchanged)."""
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return list(values)
    return [v / norm for v in values]
