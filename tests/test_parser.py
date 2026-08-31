from pathlib import Path

import pymupdf
import pytest

from docmind.ingest.parser import detect_language, parse_pdf, sha256_of

GERMAN = (
    "Allgemeine Versicherungsbedingungen. Der Versicherer erbringt die Leistungen "
    "gemäss den nachstehenden Bestimmungen und dem Versicherungsvertragsgesetz."
)
FRENCH = (
    "Conditions générales d'assurance. L'assureur fournit les prestations conformément "
    "aux dispositions ci-après et à la loi sur le contrat d'assurance."
)


def make_pdf(path: Path, page_texts: list[str]) -> Path:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        # A text box wraps long lines; plain insert_text would run off the page edge.
        page.insert_textbox(pymupdf.Rect(72, 72, 520, 700), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def german_pdf(tmp_path: Path) -> Path:
    return make_pdf(tmp_path / "avb.pdf", [GERMAN, GERMAN + " Zweite Seite."])


def test_parse_pdf_keeps_page_numbers_and_text(german_pdf: Path) -> None:
    parsed = parse_pdf(german_pdf)
    assert parsed.filename == "avb.pdf"
    assert parsed.page_count == 2
    assert [p.number for p in parsed.pages] == [1, 2]
    assert "Versicherungsbedingungen" in parsed.pages[0].text
    assert "Zweite Seite" in parsed.pages[1].text


def test_parse_pdf_detects_language(german_pdf: Path, tmp_path: Path) -> None:
    assert parse_pdf(german_pdf).lang == "de"
    assert parse_pdf(make_pdf(tmp_path / "cga.pdf", [FRENCH])).lang == "fr"


def test_sha256_is_stable_and_content_based(german_pdf: Path, tmp_path: Path) -> None:
    digest = sha256_of(german_pdf)
    assert len(digest) == 64
    assert digest == parse_pdf(german_pdf).sha256
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(german_pdf.read_bytes())
    assert sha256_of(copy) == digest  # same bytes, different name -> same fingerprint


def test_detect_language_handles_empty_and_unsupported() -> None:
    assert detect_language("") is None
    assert detect_language("   \n ") is None
    assert detect_language("これは日本語のテキストです。保険契約について説明します。") is None


def test_parse_pdf_missing_file(tmp_path: Path) -> None:
    with pytest.raises(pymupdf.FileNotFoundError):
        parse_pdf(tmp_path / "nope.pdf")
