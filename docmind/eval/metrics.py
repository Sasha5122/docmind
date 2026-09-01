"""Evaluation metrics. The deterministic ones need no model; the judged ones use an LLM.

Deterministic (retrieval + citations):
- recall_at_k        : did the right page show up in the top-k retrieved chunks? (0/1 per
                       question, averaged). THE retrieval metric — if the page is not
                       retrieved, no LLM can answer correctly.
- mrr                : 1 / rank of the first correct chunk (how high, not just whether).
- context_hit        : recall_at_k applied to the k chunks handed to the LLM AFTER reranking
                       (recall_at_k / mrr score the fused candidates BEFORE reranking, so
                       only this one can show what the reranker does). Computed in runner.py.
- citation_precision : share of the answer's citations that point at an expected page.
- citation_coverage  : share of factual sentences that carry at least one citation.

Judged (LLM-as-judge, RAGAS-style, one call per question each):
- faithfulness       : share of answer statements supported by the retrieved contexts.
- answer_correctness : 0-1 agreement between the answer and the reference answer.

Page matching allows a tolerance of ±1 page because a chunk is cited by the page it
STARTS on and may run onto the next page.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from docmind.eval.golden import SourceRef
from docmind.llm.base import LLM
from docmind.llm.prompt import Citation, sentences_without_citation
from docmind.retrieval.search import RetrievedChunk

PAGE_TOLERANCE = 1

# "I can't answer this from the documents" in the four corpus languages. The prompt asks for one
# fixed sentence, but a 7B model paraphrases ("wird in den Quellen nicht erwähnt", "do not
# contain information about"), so the check is a set of patterns, not an exact string.
_ABSTAIN_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # de
        r"keine (spezifische[nr]? |konkrete[nr]? |weitere[nr]? )?"
        r"(angaben|informationen|information|hinweise|aussage)",
        r"nicht (erwähnt|enthalten|angegeben|genannt|aufgeführt|ersichtlich|zu finden|möglich"
        r"|hervor)",
        r"beziehen sich nicht auf",
        r"finde ich .{0,40}(keine|nicht)",
        # fr
        r"aucune information",
        r"ne (contiennent|contient|mentionnent|mentionne|figurent?|précisent?|permettent|permet)"
        r" pas",
        r"pas d[’']information",
        # en
        r"could not find",
        r"do(es)? not (contain|mention|provide|specify|include|state|indicate|address)",
        r"no (specific )?information (about|on|regarding)",
        r"not (mentioned|specified|provided|available|found|stated|addressed) in",
        r"none of the (provided|given|retrieved)",
        # it
        r"non trovo",
        r"non (contengono|contiene|menzionano|menziona|riportano|riporta)",
        r"nessuna informazione",
    )
]


def is_abstention(answer: str) -> bool:
    """True when the answer says the documents do not cover the question."""
    return any(pattern.search(answer) for pattern in _ABSTAIN_PATTERNS)


def _matches(chunk_file: str, chunk_page: int, expected: Sequence[SourceRef]) -> bool:
    return any(
        chunk_file == src.file and abs(chunk_page - src.page) <= PAGE_TOLERANCE for src in expected
    )


def recall_at_k(
    retrieved: Sequence[RetrievedChunk], expected: Sequence[SourceRef], k: int
) -> float:
    """1.0 if any of the first k chunks lands on an expected (file, page±1), else 0.0."""
    if not expected:
        return 0.0
    return float(any(_matches(c.filename, c.page, expected) for c in retrieved[:k]))


def mrr(retrieved: Sequence[RetrievedChunk], expected: Sequence[SourceRef]) -> float:
    for rank, chunk in enumerate(retrieved, start=1):
        if _matches(chunk.filename, chunk.page, expected):
            return 1.0 / rank
    return 0.0


def citation_precision(
    citations: Sequence[Citation], expected: Sequence[SourceRef]
) -> float | None:
    """Share of citations that point at an expected page; None when there are no citations."""
    if not citations:
        return None
    hits = sum(_matches(c.filename, c.page, expected) for c in citations)
    return hits / len(citations)


def citation_coverage(answer: str) -> float:
    """1 - (uncited factual sentences / all factual sentences)."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if len(s.split()) >= 4]
    if not sentences:
        return 1.0
    uncited = sentences_without_citation(answer)
    return 1.0 - len(uncited) / len(sentences)


# ------------------------------------------------------------------ LLM-as-judge

FAITHFULNESS_SYSTEM = """You are a strict auditor. You receive CONTEXT passages and an ANSWER.
Split the ANSWER into its individual factual statements. For each statement decide whether it is
fully supported by the CONTEXT (paraphrases count, inferences do not).
Reply with JSON only: {"statements": [{"text": "...", "supported": true|false}, ...]}"""

CORRECTNESS_SYSTEM = """You compare a candidate ANSWER with a REFERENCE answer to the same question.
Score 1.0 if the candidate states the same facts as the reference (wording may differ),
0.5 if it is partially correct or incomplete, 0.0 if it is wrong, contradicts the reference,
or says the information is unavailable when the reference gives it.
Reply with JSON only: {"score": <number>, "reason": "<one sentence>"}"""


@dataclass(frozen=True)
class Judgement:
    score: float
    detail: dict


def _json_from(text: str) -> dict:
    """Small local models sometimes wrap JSON in prose or code fences; dig it out."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"judge returned no JSON: {text[:200]!r}")
    return json.loads(match.group(0))


def judge_faithfulness(judge: LLM, answer: str, contexts: Sequence[str]) -> Judgement:
    user = "CONTEXT:\n" + "\n---\n".join(contexts) + f"\n\nANSWER:\n{answer}"
    reply = judge.complete(FAITHFULNESS_SYSTEM, user, max_tokens=800)
    data = _json_from(reply.text)
    statements = data.get("statements", [])
    if not statements:
        return Judgement(1.0, {"statements": [], "note": "no factual statements"})
    supported = sum(1 for s in statements if bool(s.get("supported")))
    return Judgement(supported / len(statements), {"statements": statements})


def judge_correctness(judge: LLM, question: str, answer: str, reference: str) -> Judgement:
    user = f"QUESTION: {question}\n\nREFERENCE: {reference}\n\nANSWER: {answer}"
    reply = judge.complete(CORRECTNESS_SYSTEM, user, max_tokens=200)
    data = _json_from(reply.text)
    score = min(1.0, max(0.0, float(data.get("score", 0.0))))
    return Judgement(score, {"reason": data.get("reason", "")})


def mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
