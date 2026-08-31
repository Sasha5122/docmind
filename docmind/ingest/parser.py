"""PDF -> pages of text.

First stage of ingestion. Output keeps the page number with every page's text so
that later chunks (and therefore citations) can point back to `[document, page]`.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # make language detection deterministic
SUPPORTED_LANGS = frozenset({"de", "fr", "en", "it"})
_LANG_SAMPLE_CHARS = 5000  # enough text to guess the language reliably


@dataclass(frozen=True)
class Page:
    number: int  # 1-based, as printed in a citation
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    filename: str
    sha256: str
    lang: str | None
    pages: list[Page]

    @property
    def page_count(self) -> int:
        return len(self.pages)


def sha256_of(path: Path) -> str:
    """Fingerprint of the file bytes; identical files give identical hashes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def detect_language(text: str) -> str | None:
    """Return an ISO 639-1 code from SUPPORTED_LANGS, or None if unsure."""
    sample = text.strip()[:_LANG_SAMPLE_CHARS]
    if not sample:
        return None
    try:
        code = detect(sample)
    except LangDetectException:
        return None
    return code if code in SUPPORTED_LANGS else None


def parse_pdf(path: Path) -> ParsedDocument:
    """Read every page of a PDF; raises FileNotFoundError / pymupdf errors on bad input."""
    with pymupdf.open(path) as doc:
        pages = [Page(number=i + 1, text=page.get_text("text")) for i, page in enumerate(doc)]
    full_text = "\n".join(p.text for p in pages)
    return ParsedDocument(
        filename=path.name,
        sha256=sha256_of(path),
        lang=detect_language(full_text),
        pages=pages,
    )
