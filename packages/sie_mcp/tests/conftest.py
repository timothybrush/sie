import io
from collections.abc import Callable

import pypdfium2 as pdfium
import pytest
from PIL import Image


@pytest.fixture
def make_pdf() -> Callable[[int], bytes]:
    """Factory for an in-memory blank PDF with ``n_pages`` pages."""

    def _make(n_pages: int) -> bytes:
        pdf = pdfium.PdfDocument.new()
        for _ in range(n_pages):
            pdf.new_page(200, 300)
        buf = io.BytesIO()
        pdf.save(buf)
        pdf.close()
        return buf.getvalue()

    return _make


@pytest.fixture
def png_bytes() -> bytes:
    """A minimal valid PNG image as bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def tiff_bytes() -> bytes:
    """A minimal valid TIFF image as bytes (a common scanned-document container)."""
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(buf, format="TIFF")
    return buf.getvalue()
