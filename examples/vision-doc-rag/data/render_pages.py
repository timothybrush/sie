"""Rasterize the curated PDF corpus to page PNGs.

The script tries pdf2image first because it produces excellent page images
when Poppler is installed. If Poppler or pdf2image is unavailable, it falls
back to PyMuPDF so the demo still works with only Python package dependencies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


def _selected_pages(source: dict, total_pages: int) -> list[int]:
    pages = source.get("pages")
    if pages:
        selected = [int(p) for p in pages if 1 <= int(p) <= total_pages]
    else:
        start = int(source.get("start_page", 1))
        max_pages = int(source.get("max_pages", 6))
        selected = list(range(start, min(total_pages, start + max_pages - 1) + 1))

    if not selected:
        raise ValueError(f"No valid pages selected for {source['slug']} ({total_pages} pages)")
    return selected


def _pdf_page_count_with_pymupdf(pdf_path: Path) -> int:
    import fitz

    with fitz.open(pdf_path) as doc:
        return doc.page_count


def _render_with_pdf2image(pdf_path: Path, page_number: int, out_path: Path, dpi: int) -> None:
    from pdf2image import convert_from_path

    images = convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=page_number,
        last_page=page_number,
        fmt="png",
        single_file=True,
    )
    if not images:
        raise RuntimeError(f"pdf2image returned no image for {pdf_path} page {page_number}")
    images[0].save(out_path)


def _render_with_pymupdf(pdf_path: Path, page_number: int, out_path: Path, dpi: int) -> None:
    import fitz

    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_number - 1)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        pixmap.save(out_path)


def _render_page(pdf_path: Path, page_number: int, out_path: Path, dpi: int, backend: str) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if backend in {"auto", "pdf2image"}:
        try:
            _render_with_pdf2image(pdf_path, page_number, out_path, dpi)
            return "pdf2image"
        except Exception as exc:
            if backend == "pdf2image":
                raise
            print(
                f"  pdf2image unavailable for {pdf_path.name} p.{page_number} "
                f"({type(exc).__name__}); falling back to PyMuPDF",
                file=sys.stderr,
            )

    _render_with_pymupdf(pdf_path, page_number, out_path, dpi)
    return "pymupdf"


def main() -> None:
    here = Path(__file__).resolve().parent
    root = here.parent
    manifest_path = here / "pdfs_manifest.json"
    if not manifest_path.exists():
        print("pdfs_manifest.json not found; run `python data/fetch_pdfs.py` first", file=sys.stderr)
        sys.exit(1)

    config = yaml.safe_load((root / "config.yaml").read_text())
    render_config = config.get("render", {})
    dpi = int(render_config.get("dpi", 160))
    backend = render_config.get("backend", "auto")
    active_backend = backend
    out_dir = here / "pages"

    pdf_manifest = json.loads(manifest_path.read_text())
    page_manifest: list[dict] = []
    backend_counts: dict[str, int] = {}

    for source in pdf_manifest["sources"]:
        pdf_path = here / source["pdf_path"]
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}. Run data/fetch_pdfs.py.")

        total_pages = _pdf_page_count_with_pymupdf(pdf_path)
        for page_number in _selected_pages(source, total_pages):
            page_id = f"{source['client']}__{source['slug']}__p{page_number:03d}"
            image_path = out_dir / source["client"] / f"{source['slug']}_p{page_number:03d}.png"
            used_backend = _render_page(pdf_path, page_number, image_path, dpi, active_backend)
            if backend == "auto" and used_backend == "pymupdf":
                active_backend = "pymupdf"
            backend_counts[used_backend] = backend_counts.get(used_backend, 0) + 1

            rel_image_path = image_path.relative_to(here)
            page_manifest.append(
                {
                    "page_id": page_id,
                    "client": source["client"],
                    "title": source["title"],
                    "publisher": source["publisher"],
                    "license": source["license"],
                    "source_url": source["url"],
                    "source_pdf": source["source_pdf"],
                    "source_pdf_path": source["pdf_path"],
                    "page_number": page_number,
                    "image_path": str(rel_image_path),
                }
            )
            print(
                f"  {source['client']:12s} {source['slug']:38s} "
                f"p.{page_number:<4d} -> data/{rel_image_path}"
            )

    out = here / "pages_manifest.json"
    out.write_text(json.dumps(page_manifest, indent=2) + "\n")

    print(f"\nRendered {len(page_manifest)} pages to {out_dir}")
    print(f"Wrote page manifest to {out}")
    for name, count in sorted(backend_counts.items()):
        print(f"  {name}: {count} pages")


if __name__ == "__main__":
    main()
