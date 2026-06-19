"""Download the public PDF corpus for the visual document RAG demo.

The corpus is intentionally small and curated. Each source has a tenant, a
stable slug, source metadata, and a limited page selection so the demo can be
indexed quickly while still containing diagrams, schematics, screenshots, and
technical figures that reward visual retrieval.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SOURCES = [
    {
        "client": "embedded-lab",
        "slug": "raspberry-pi-pico-datasheet",
        "title": "Raspberry Pi Pico Datasheet",
        "publisher": "Raspberry Pi Ltd",
        "license": "CC BY-ND 4.0",
        "url": "https://datasheets.raspberrypi.com/pico/pico-datasheet.pdf",
        "pages": [4, 5, 6, 7, 8, 9],
    },
    {
        "client": "embedded-lab",
        "slug": "arduino-uno-r3-datasheet",
        "title": "Arduino UNO R3 Datasheet",
        "publisher": "Arduino",
        "license": "Arduino documentation / open hardware terms",
        "url": "https://docs.arduino.cc/resources/datasheets/A000066-datasheet.pdf",
        "pages": [5, 6, 7, 8, 9, 10, 11],
    },
    {
        "client": "embedded-lab",
        "slug": "arduino-uno-r3-schematic",
        "title": "Arduino UNO R3 Schematic",
        "publisher": "Arduino",
        "license": "CC BY-SA 4.0 hardware reference design",
        "url": "https://docs.arduino.cc/resources/schematics/A000066-schematics.pdf",
        "pages": [1, 2],
    },
    {
        "client": "ops-eng",
        "slug": "postgresql-18-manual",
        "title": "PostgreSQL 18 Documentation",
        "publisher": "PostgreSQL Global Development Group",
        "license": "PostgreSQL License",
        "url": "https://www.postgresql.org/files/documentation/pdf/18/postgresql-18-A4.pdf",
        "pages": [673, 674, 675, 676],
    },
    {
        "client": "ops-eng",
        "slug": "kubernetes-infrastructure-abstraction",
        "title": "Kubernetes as Infrastructure Abstraction",
        "publisher": "Cloud Native Computing Foundation",
        "license": "CNCF public presentation material",
        "url": "https://www.cncf.io/wp-content/uploads/2020/08/2019-09-Kubernetes-as-Infrastructure-Abstraction.pdf",
        "pages": [6, 7, 8, 9, 10, 11],
    },
    {
        "client": "ops-eng",
        "slug": "cloud-native-ai-whitepaper",
        "title": "Cloud Native Artificial Intelligence Whitepaper",
        "publisher": "Cloud Native Computing Foundation",
        "license": "CNCF documentation / report terms",
        "url": "https://www.cncf.io/wp-content/uploads/2024/03/cloud_native_ai24_031424a-2.pdf",
        "pages": [11, 12, 13, 14, 15, 16],
    },
    {
        "client": "aerospace",
        "slug": "solid-rocket-motor-nozzles",
        "title": "Solid Rocket Motor Nozzles (NASA SP-8115)",
        "publisher": "NASA Technical Reports Server",
        "license": "NASA STI public release",
        # NTRS citation-download API returns the actual scanned report (with the
        # engineering drawings); the /archive/ path returns the HTML landing page.
        "url": "https://ntrs.nasa.gov/api/citations/19760013126/downloads/19760013126.pdf",
        "pages": [14, 15, 20, 22, 49, 50],
    },
    {
        "client": "aerospace",
        "slug": "liquid-rocket-engine-nozzles",
        "title": "Liquid Rocket Engine Nozzles (NASA SP-8120)",
        "publisher": "NASA Technical Reports Server",
        "license": "NASA STI public release",
        "url": "https://ntrs.nasa.gov/api/citations/19770009165/downloads/19770009165.pdf",
        "pages": [23, 27, 42, 43, 46, 49],
    },
]


def _download(url: str, out: Path) -> bool:
    """Download url to out atomically. Return True when a new file was written."""
    if out.exists() and out.stat().st_size > 0:
        return False

    out.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        url,
        headers={
            "User-Agent": "sie-vision-doc-rag-demo/1.0",
            "Accept": "application/pdf,*/*",
        },
    )
    with tempfile.NamedTemporaryFile(delete=False, dir=out.parent, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        try:
            with urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, tmp)
        except (HTTPError, URLError, TimeoutError):
            tmp_path.unlink(missing_ok=True)
            raise

    tmp_path.replace(out)
    return True


def main() -> None:
    here = Path(__file__).resolve().parent
    pdf_root = here / "pdfs"
    manifest = []

    for source in SOURCES:
        pdf_path = pdf_root / source["client"] / f"{source['slug']}.pdf"
        try:
            downloaded = _download(source["url"], pdf_path)
        except Exception as exc:
            print(f"Failed to download {source['url']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise

        row = dict(source)
        row["pdf_path"] = str(pdf_path.relative_to(here))
        row["source_pdf"] = pdf_path.name
        manifest.append(row)

        status = "downloaded" if downloaded else "cached"
        print(f"  {status:10s} {source['client']:12s} {source['slug']} -> {row['pdf_path']}")

    out = here / "pdfs_manifest.json"
    out.write_text(json.dumps({"sources": manifest}, indent=2) + "\n")
    print(f"\nWrote {len(manifest)} PDF sources to {out}")


if __name__ == "__main__":
    main()
