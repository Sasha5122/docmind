"""Download the public demo corpus into data/raw/ (never committed).

Sources: FINMA circulars (DE/FR/EN), Swiss insurers' general conditions (AVB), and
annual reports. Every file is public. Re-running skips files that already exist.

Run: uv run python scripts/download_corpus.py
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pymupdf
import truststore

# Python's own certificate bundle does not know some corporate/regional CAs; use the
# Windows/macOS system store instead (this is what browsers do).
truststore.inject_into_ssl()

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

FINMA = (
    "https://www.finma.ch/{lang}/~/media/finma/dokumente/dokumentencenter/myfinma/rundschreiben/"
)

# (target filename, url)
CORPUS: list[tuple[str, str]] = [
    # --- FINMA circulars: same circular in three languages -> great for cross-lingual tests
    (
        "finma-rs-2023-01-oprisk-de.pdf",
        FINMA.format(lang="de") + "finma-rs-2023-01-20221207.pdf?sc_lang=de",
    ),
    (
        "finma-rs-2023-01-oprisk-fr.pdf",
        FINMA.format(lang="fr") + "finma-rs-2023-01-20221207.pdf?sc_lang=fr",
    ),
    (
        "finma-rs-2023-01-oprisk-en.pdf",
        FINMA.format(lang="en") + "finma-rs-2023-01-20221207.pdf?sc_lang=en",
    ),
    (
        "finma-rs-2026-01-nature-risks-de.pdf",
        FINMA.format(lang="de") + "finma-rs-2026-01.pdf?sc_lang=de",
    ),
    (
        "finma-rs-2026-01-nature-risks-fr.pdf",
        FINMA.format(lang="fr") + "finma-rs-2026-01.pdf?sc_lang=fr",
    ),
    (
        "finma-rs-2026-01-nature-risks-en.pdf",
        FINMA.format(lang="en") + "finma-rs-2026-01.pdf?sc_lang=en",
    ),
    # --- insurers' general conditions (AVB / CGA)
    (
        "mobiliar-avb-hausrat-junge-de.pdf",
        "https://www.mobiliar.ch/sites/default/files/2024-04/avb-hausrat-privathaftpflicht.junge_.pdf",
    ),
    (
        "mobiliar-avb-hausrat-minima-de.pdf",
        "https://www.mobiliar.ch/sites/default/files/2024-04/avb-hausrat-minima.pdf",
    ),
    (
        "zurich-avb-haushalt-de.pdf",
        "https://www.zurich.ch/-/media/zurich-site/content/privatkunden/haftung-recht/dokumente/avb-haushaltversicherung/avb-haushalt-versicherung.pdf?sc_lang=de",
    ),
    (
        "zurich-cga-menage-fr.pdf",
        "https://www.zurich.ch/-/media/zurich-site/content/privatkunden/haftung-recht/dokumente/avb-haushaltversicherung/avb-haushalt-versicherung.pdf?sc_lang=fr",
    ),
    (
        "axa-avb-haushalt-de.pdf",
        "https://www.axa.ch/servlets/external/docstoredocument?accesscode=af4fm",
    ),
    # --- annual reports (long, tables, English)
    (
        "swissre-annual-report-2024-financial-statements-en.pdf",
        "https://www.swissre.com/dam/jcr:2a7c343c-57fb-458d-b7a9-34455c8a30f6/2024-annual-report-financial-statements.pdf",
    ),
    (
        "swisslife-annual-report-2024-en.pdf",
        "https://www.swisslife.com/content/dam/com_rel/dokumente/fy_results/fy_2024_publish_16_april/Swiss_Life_Full-year_results_2024_Annual_Report_2024_EN.pdf",
    ),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (docmind portfolio project; public documents only)"}


def download(name: str, url: str) -> str:
    target = RAW / name
    if target.exists():
        return "exists"
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    if not data.startswith(b"%PDF"):
        return f"not a PDF ({len(data)} bytes)"
    target.write_bytes(data)
    with pymupdf.open(target) as doc:
        pages = doc.page_count
    return f"ok, {pages} pages, {len(data) // 1024} KB"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    failures = 0
    for name, url in CORPUS:
        try:
            status = download(name, url)
        except Exception as exc:  # noqa: BLE001
            status = f"FAILED: {exc}"
            failures += 1
        print(f"{name:55} {status}")
    print(f"\n{len(CORPUS) - failures}/{len(CORPUS)} available in {RAW}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
