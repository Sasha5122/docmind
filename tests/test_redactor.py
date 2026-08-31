from docmind.pii.redactor import find_pii, redact

SAMPLE = (
    "Kontakt: hans.muster@example.ch, Tel. +41 79 123 45 67, "
    "IBAN CH93 0076 2011 6238 5295 7. Prämie CHF 1'250 pro Jahr."
)


def test_redacts_iban_email_and_phone() -> None:
    result = redact(SAMPLE, lang="de")
    assert "<IBAN>" in result.text
    assert "<EMAIL>" in result.text
    assert "<PHONE>" in result.text
    for secret in ("CH93", "hans.muster", "79 123 45 67"):
        assert secret not in result.text
    assert result.counts == {"IBAN": 1, "EMAIL": 1, "PHONE": 1}
    assert result.total == 3


def test_non_pii_text_is_untouched() -> None:
    text = "Die Versicherung deckt Schäden bis CHF 100'000 pro Ereignis (Art. 5 Abs. 2)."
    result = redact(text)
    assert result.text == text
    assert result.counts == {}


def test_empty_text() -> None:
    assert redact("").text == ""
    assert redact("").total == 0


def test_multiple_ibans_are_all_replaced() -> None:
    text = "Konto A: DE89 3704 0044 0532 0130 00; Konto B: FR14 2004 1010 0505 0001 3M02 606."
    result = redact(text)
    assert result.counts["IBAN"] == 2
    assert "3704" not in result.text and "2004" not in result.text


def test_find_pii_reports_positions() -> None:
    text = "Mail: a.b@firma.de"
    (hit,) = find_pii(text)
    assert hit.entity_type == "EMAIL_ADDRESS"
    assert text[hit.start : hit.end] == "a.b@firma.de"
