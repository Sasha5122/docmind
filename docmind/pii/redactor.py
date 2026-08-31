"""Replace personal data with placeholders BEFORE text is chunked or stored.

Two kinds of detection, both run by Presidio's AnalyzerEngine:
- pattern rules (deterministic): IBAN, e-mail, phone number
- a spaCy language model (statistical): person names -> <PERSON>

The model is the small multilingual `xx_ent_wiki_sm`, registered once per
supported language, so the same engine serves German, French, English, Italian.
"""

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
_FALLBACK_LANG = "en"  # the multilingual model does not care; Presidio just needs a valid code
SPACY_MODEL = "xx_ent_wiki_sm"

# Presidio entity name -> placeholder that ends up in the stored text.
PLACEHOLDERS: dict[str, str] = {
    "PERSON": "<PERSON>",
    "IBAN_CODE": "<IBAN>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
}
_MIN_SCORE = 0.4  # Presidio's phone rule scores plausible-but-unverified numbers at 0.4
_PHONE_REGIONS = ("CH", "DE", "FR", "IT", "AT", "LI")

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
    """Build the engine once (loading the spaCy model takes ~1 s)."""
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": lang, "model_name": SPACY_MODEL} for lang in LANGS],
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


def find_pii(text: str, lang: str | None = None) -> list[RecognizerResult]:
    """Locate PII spans (start, end, entity_type, score) without changing the text."""
    if not text.strip():
        return []
    return get_analyzer().analyze(text=text, language=_language(lang), entities=list(PLACEHOLDERS))


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
