"""Render contract text to a page image.

Used to produce the "scanned page" the OCR and vision models read. (For a true
scan, point ``--scan`` at a real PDF/PNG instead — see the README.)
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int) -> ImageFont.ImageFont:
    """A scalable font with no external file dependency (Pillow >= 10.1)."""
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow without sized default
        return ImageFont.load_default()


def render_text_page(
    text: str,
    out_path: Path,
    *,
    title: str | None = None,
    width: int = 1000,
    line_height: int = 30,
    font_size: int = 20,
    margin: int = 60,
    max_lines: int = 46,
    wrap: int = 92,
) -> None:
    lines: list[str] = []
    for raw in text.splitlines():
        lines.extend(textwrap.wrap(raw, width=wrap) or [""])
    lines = lines[:max_lines]

    n = len(lines) + (2 if title else 0)
    height = max(margin * 2 + line_height * n, 400)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    body_font = load_font(font_size)

    y = margin
    if title:
        draw.text((margin, y), title[:80], fill="black", font=load_font(font_size + 4))
        y += line_height * 2
    for line in lines:
        draw.text((margin, y), line, fill="black", font=body_font)
        y += line_height
    img.save(out_path)
