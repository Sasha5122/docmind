"""The golden set: questions with known answers and known source pages.

One JSON object per line in `data/eval/golden.jsonl`:
{
  "id": "avb-zurich-001",
  "lang": "de",
  "question": "Wie hoch ist der Selbstbehalt bei Glasbruch?",
  "reference_answer": "Es gilt kein Selbstbehalt bei Glasbruch.",
  "sources": [{"file": "zurich-avb-haushalt-de.pdf", "page": 12}],
  "category": "fact",            # fact | cross-lingual | multi-doc | unanswerable
  "origin": "manual"             # manual | generated-reviewed
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GOLDEN = Path("data/eval/golden.jsonl")
CATEGORIES = ("fact", "cross-lingual", "multi-doc", "unanswerable")


@dataclass(frozen=True)
class SourceRef:
    file: str
    page: int


@dataclass(frozen=True)
class GoldenItem:
    id: str
    lang: str
    question: str
    reference_answer: str
    sources: tuple[SourceRef, ...]
    category: str = "fact"
    origin: str = "manual"

    @property
    def answerable(self) -> bool:
        return self.category != "unanswerable"


def load_golden(path: Path = DEFAULT_GOLDEN, limit: int | None = None) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            raw = json.loads(line)
            item = GoldenItem(
                id=raw["id"],
                lang=raw["lang"],
                question=raw["question"],
                reference_answer=raw.get("reference_answer", ""),
                sources=tuple(SourceRef(s["file"], int(s["page"])) for s in raw.get("sources", [])),
                category=raw.get("category", "fact"),
                origin=raw.get("origin", "manual"),
            )
            if item.category not in CATEGORIES:
                raise ValueError(f"{path}:{line_no}: unknown category {item.category!r}")
            if item.id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {item.id!r}")
            if item.answerable and not item.sources:
                raise ValueError(f"{path}:{line_no}: {item.id} has no sources")
            seen.add(item.id)
            items.append(item)
    return items[:limit] if limit else items
