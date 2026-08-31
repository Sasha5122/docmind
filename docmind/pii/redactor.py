"""Replace personal data with placeholders BEFORE text is chunked or stored.

Stage 1 (this file): deterministic pattern rules — IBAN, e-mail, phone number.
Names (which need a language model) are added in a later step; the public
`redact()` signature will not change.
"""

from dataclasses import dataclass, field

from presidio_analyzer import RecognizerResult
from presidio_analyzer.predefined_recognizers import (
    EmailRecognizer,
    IbanRecognizer,
    PhoneRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Presidio entity name -> placeholder that ends up in the stored text.
PLACEHOLDERS: dict[str, str] = {
    "IBAN_CODE": "<IBAN>",
    "EMAIL_ADDRESS": "<EMAIL>",
    "PHONE_NUMBER": "<PHONE>",
}
_MIN_SCORE = 0.4  # Presidio's phone rule scores plausible-but-unverified numbers at 0.4
_PHONE_REGIONS = ("CH", "DE", "FR", "IT", "AT", "LI")

_RECOGNIZERS = (
    IbanRecognizer(),
    EmailRecognizer(),
    PhoneRecognizer(supported_regions=_PHONE_REGIONS),
)
_ANONYMIZER = AnonymizerEngine()
_OPERATORS = {
    entity: OperatorConfig("replace", {"new_value": placeholder})
    for entity, placeholder in PLACEHOLDERS.items()
}


@dataclass(frozen=True)
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)  # e.g. {"IBAN": 2, "EMAIL": 1}

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def find_pii(text: str) -> list[RecognizerResult]:
    """Locate PII spans (start, end, entity_type, score) without changing the text."""
    found: list[RecognizerResult] = []
    for recognizer in _RECOGNIZERS:
        found += recognizer.analyze(
            text, entities=recognizer.supported_entities, nlp_artifacts=None
        )
    return [r for r in found if r.score >= _MIN_SCORE]


def redact(text: str, lang: str | None = None) -> RedactionResult:
    """Return the text with PII replaced by placeholders, plus how many of each kind.

    `lang` is accepted now so callers do not change when name detection arrives.
    """
    if not text:
        return RedactionResult(text=text)
    results = find_pii(text)
    if not results:
        return RedactionResult(text=text)
    anonymized = _ANONYMIZER.anonymize(text=text, analyzer_results=results, operators=_OPERATORS)
    counts: dict[str, int] = {}
    for item in anonymized.items:
        label = PLACEHOLDERS[item.entity_type].strip("<>")
        counts[label] = counts.get(label, 0) + 1
    return RedactionResult(text=anonymized.text, counts=counts)
