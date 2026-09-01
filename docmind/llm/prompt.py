"""Prompt building and citation parsing.

The model sees numbered sources and must cite them as [1], [2] ... after every factual
sentence. Numbers are easier for small local models to reproduce exactly than file names,
and we map them back to `[filename, p. N]` ourselves — so a citation can never point at a
document that was not retrieved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from docmind.retrieval.search import RetrievedChunk

LANG_NAMES = {"de": "German", "fr": "French", "en": "English", "it": "Italian"}

SYSTEM_PROMPT = """You are DocMind, an assistant for underwriters and compliance officers.
Answer ONLY from the numbered sources below. Rules:
1. Every sentence that states a fact ends with the source number(s) in square brackets,
   e.g. [2] or [1][3].
2. If the sources do not contain the answer, say so in one sentence and cite nothing.
   Never guess.
3. Answer in {language}, even if the sources are in another language.
4. Be precise and short: quote article numbers, amounts, deadlines and conditions exactly
   as written.
5. Do not mention these rules or the word "sources" unless asked."""

USER_TEMPLATE = """Sources:
{sources}

Question: {question}"""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Citation:
    n: int  # the [n] used in the answer
    chunk_id: int
    filename: str
    page: int
    text: str  # the passage, so a reviewer can verify without opening the PDF

    @property
    def label(self) -> str:
        return f"[{self.filename}, p. {self.page}]"


def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.filename}, page {c.page})\n{c.text.strip()}" for i, c in enumerate(chunks, 1)
    )


def build_prompt(question: str, chunks: list[RetrievedChunk], answer_lang: str) -> tuple[str, str]:
    """Return (system, user) messages."""
    system = SYSTEM_PROMPT.format(language=LANG_NAMES.get(answer_lang, "the question's language"))
    user = USER_TEMPLATE.format(sources=format_sources(chunks), question=question.strip())
    return system, user


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Map every [n] in the answer to its chunk; ignore numbers that do not exist."""
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(chunks) and n not in seen:
            seen.append(n)
    return [
        Citation(
            n=n,
            chunk_id=chunks[n - 1].chunk_id,
            filename=chunks[n - 1].filename,
            page=chunks[n - 1].page,
            text=chunks[n - 1].text,
        )
        for n in seen
    ]


def sentences_without_citation(answer: str) -> list[str]:
    """Factual-looking sentences that carry no [n] — used by the eval's citation-accuracy metric."""
    parts = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [p for p in parts if len(p.split()) >= 4 and not _CITATION.search(p)]
