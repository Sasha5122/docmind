"""Replace personal data with placeholders BEFORE text is chunked or stored.

Two kinds of detection, both run by Presidio's AnalyzerEngine:
- pattern rules (deterministic): IBAN, e-mail, phone number
- a spaCy language model (statistical): person names -> <PERSON>

Name detection uses one small spaCy model per language (de/fr/en/it `*_sm`, ~15 MB each).
The first version used the single multilingual `xx_ent_wiki_sm`; on this corpus it tagged
63 % of all chunks as containing a person ("Number of shares", "Schadenfalls", "Cash").
On top of the better models, `_looks_like_person_name` keeps only spans that look like a
real name (2-4 Title-Case tokens), trading a little recall (single surnames) for a lot of
precision — the right trade for regulatory documents, where names are rare and mangled
domain words would poison retrieval.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
    SpacyRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

LANGS = ("de", "fr", "en", "it")
_FALLBACK_LANG = "en"
SPACY_MODELS: dict[str, str] = {
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "en": "en_core_web_sm",
    "it": "it_core_news_sm",
}

# Presidio entity name -> placeholder that ends up in the stored text.
PLACEHOLDERS: dict[str, str] = {
    "PERSON": "<PERSON>",
    "IBAN_CODE": "<IBAN>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
}
_MIN_SCORE = 0.4  # Presidio's phone rule scores plausible-but-unverified numbers at 0.4
_PHONE_REGIONS = ("CH", "DE", "FR", "IT", "AT", "LI")

# A name token: Capital letter followed by lower-case letters (accents allowed), optional
# apostrophe/hyphen parts, e.g. "Müller", "O'Brien", "Jean-Pierre". Lower-case particles
# ("de", "von", "van", "du", "le", "da") may sit between tokens.
_NAME_TOKEN = re.compile(r"^[A-ZÄÖÜÀ-ÖØ-Þ][a-zäöüßà-öø-ÿ]+(?:[-'’][A-ZÄÖÜÀ-Þ]?[a-zäöüßà-ÿ]+)*$")
_PARTICLES = frozenset({"de", "von", "van", "du", "le", "la", "da", "der", "den", "di", "del"})

_ANONYMIZER = AnonymizerEngine()
_OPERATORS = {
    entity: OperatorConfig("replace", {"new_value": placeholder})
    for entity, placeholder in PLACEHOLDERS.items()
}


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)  # e.g. {"PERSON": 2, "IBAN": 1}

    @property
    def total(self) -> int:
        return sum(self.counts.values())


@lru_cache
def get_analyzer() -> AnalyzerEngine:
    """Build the engine once (loading four spaCy models takes a few seconds)."""
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": lang, "model_name": SPACY_MODELS[lang]} for lang in LANGS],
            "ner_model_configuration": {
                "model_to_presidio_entity_mapping": {"PER": "PERSON", "PERSON": "PERSON"},
                "labels_to_ignore": ["ORG", "LOC", "GPE", "MISC", "NORP", "FAC", "DATE", "MONEY"],
            },
        }
    ).create_engine()
    registry = RecognizerRegistry(supported_languages=list(LANGS))
    for lang in LANGS:
        registry.add_recognizer(IbanRecognizer(supported_language=lang))
        registry.add_recognizer(EmailRecognizer(supported_language=lang))
        registry.add_recognizer(
            PhoneRecognizer(supported_language=lang, supported_regions=_PHONE_REGIONS)
        )
        registry.add_recognizer(SpacyRecognizer(supported_language=lang))
    return AnalyzerEngine(
        nlp_engine=nlp_engine,
        registry=registry,
        supported_languages=list(LANGS),
        default_score_threshold=_MIN_SCORE,
    )


def _language(lang: str | None) -> str:
    return lang if lang in LANGS else _FALLBACK_LANG


def _looks_like_person_name(span: str) -> bool:
    """2-4 tokens, Title Case (lower-case particles allowed in the middle), no digits."""
    tokens = span.replace("\n", " ").split()
    if not 2 <= len(tokens) <= 4 or any(ch.isdigit() for ch in span):
        return False
    if not (_NAME_TOKEN.match(tokens[0]) and _NAME_TOKEN.match(tokens[-1])):
        return False
    return all(_NAME_TOKEN.match(t) or t.lower() in _PARTICLES for t in tokens[1:-1])


def find_pii(text: str, lang: str | None = None) -> list[RecognizerResult]:
    """Locate PII spans (start, end, entity_type, score) without changing the text."""
    if not text.strip():
        return []
    results = get_analyzer().analyze(
        text=text, language=_language(lang), entities=list(PLACEHOLDERS)
    )
    return [
        r
        for r in results
        if r.entity_type != "PERSON" or _looks_like_person_name(text[r.start : r.end])
    ]


def redact(text: str, lang: str | None = None) -> RedactionResult:
    """Return the text with PII replaced by placeholders, plus how many of each kind."""
    results = find_pii(text, lang)
    if not results:
        return RedactionResult(text=text)
    anonymized = _ANONYMIZER.anonymize(text=text, analyzer_results=results, operators=_OPERATORS)
    counts: dict[str, int] = {}
    for item in anonymized.items:
        label = PLACEHOLDERS[item.entity_type].strip("<>")
        counts[label] = counts.get(label, 0) + 1
    return RedactionResult(text=anonymized.text, counts=counts)
