"""Prompt building, citation parsing, cost estimate, backend factory — all offline."""

import pytest

from docmind.config import Settings
from docmind.llm.backends import AnthropicLLM, AzureOpenAILLM, FakeLLM, OllamaLLM, get_llm
from docmind.llm.base import estimate_cost
from docmind.llm.prompt import (
    build_prompt,
    extract_citations,
    format_sources,
    sentences_without_citation,
)
from docmind.retrieval.search import RetrievedChunk


def chunk(i: int, filename: str = "avb.pdf", page: int = 3) -> RetrievedChunk:
    return RetrievedChunk(i, 1, filename, None, page, "de", f"Text of chunk {i}.", 0.5)


def test_format_sources_numbers_from_one() -> None:
    out = format_sources([chunk(10), chunk(11, "finma.pdf", 7)])
    assert out.startswith("[1] (avb.pdf, page 3)\nText of chunk 10.")
    assert "[2] (finma.pdf, page 7)" in out


def test_build_prompt_sets_language_and_question() -> None:
    system, user = build_prompt("Was ist versichert?", [chunk(1)], "de")
    assert "Answer in German" in system
    assert user.endswith("Question: Was ist versichert?")


def test_extract_citations_maps_numbers_and_ignores_bad_ones() -> None:
    chunks = [chunk(10), chunk(11, "finma.pdf", 7)]
    cites = extract_citations("Fact one [2]. Fact two [1][2]. Nonsense [9].", chunks)
    assert [(c.n, c.chunk_id, c.filename, c.page) for c in cites] == [
        (2, 11, "finma.pdf", 7),
        (1, 10, "avb.pdf", 3),
    ]
    assert cites[0].label == "[finma.pdf, p. 7]"


def test_sentences_without_citation() -> None:
    text = "Die Franchise betraegt 200 Franken [1]. Der Vertrag endet nach fuenf Jahren. Ja."
    assert sentences_without_citation(text) == ["Der Vertrag endet nach fuenf Jahren."]


def test_estimate_cost() -> None:
    assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)
    assert estimate_cost("gpt-4o-mini-2024-07-18", 0, 1_000_000) == pytest.approx(0.60)
    assert estimate_cost("qwen2.5:7b", 5000, 5000) == 0.0


def test_fake_llm_records_calls() -> None:
    llm = FakeLLM("hello [1]")
    out = llm.complete("sys", "usr")
    assert out.text == "hello [1]" and llm.calls == [("sys", "usr")]


def test_get_llm_factory() -> None:
    assert isinstance(get_llm(Settings(_env_file=None, llm_backend="ollama")), OllamaLLM)
    with pytest.raises(ValueError):
        get_llm(Settings(_env_file=None, llm_backend="azure", azure_openai_api_key=""))
    with pytest.raises(ValueError):
        get_llm(Settings(_env_file=None, llm_backend="anthropic", anthropic_api_key=""))
    azure = get_llm(
        Settings(
            _env_file=None,
            llm_backend="azure",
            azure_openai_api_key="k",
            azure_openai_endpoint="https://x.openai.azure.com",
        )
    )
    assert isinstance(azure, AzureOpenAILLM) and azure.model == "gpt-4o-mini"
    anthropic = get_llm(Settings(_env_file=None, llm_backend="anthropic", anthropic_api_key="k"))
    assert isinstance(anthropic, AnthropicLLM)
