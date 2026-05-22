"""Render the synthetic pages to PNG screenshots.

Each entry in pages.json becomes one image in data/pages/<page_id>.png. The
layout is intentionally plain — a title, a metadata line, and a body block —
so ColQwen2.5 sees the same kind of visual structure it would in real wikis,
docs, or PDFs. Replace this script with `pdf2image` (or screenshots) when
pointing at real content.
"""

import json
import sys
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    """Try the platform Helvetica, fall back to PIL's default bitmap font."""
    for path in [
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _wrap(text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Greedy word wrap so body paragraphs fit the page width."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if font.getlength(candidate) <= max_width:
                current = candidate
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def render_page(page: dict, width: int, height: int, body_size: int, title_size: int) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    title_font = _font(title_size)
    meta_font = _font(int(body_size * 0.9))
    body_font = _font(body_size)

    margin = 48
    cursor_y = margin
    draw.text((margin, cursor_y), page["title"], fill="black", font=title_font)
    cursor_y += int(title_size * 1.6)
    meta = f"{page['space']}  ·  {page['author']}  ·  {page['page_id']}"
    draw.text((margin, cursor_y), meta, fill=(96, 96, 96), font=meta_font)
    cursor_y += int(title_size * 1.2)
    draw.line([(margin, cursor_y), (width - margin, cursor_y)], fill=(200, 200, 200), width=2)
    cursor_y += int(body_size * 1.2)

    max_text_width = width - 2 * margin
    line_gap = int(body_size * 1.5)
    for bullet in page["body"]:
        # Render each body line as a wrapped paragraph block.
        lines = _wrap(bullet, body_font, max_text_width)
        for line in lines:
            draw.text((margin, cursor_y), line, fill="black", font=body_font)
            cursor_y += line_gap
        cursor_y += int(line_gap * 0.4)  # paragraph spacing

    return img


def main():
    here = Path(__file__).resolve().parent
    pages_path = here / "pages.json"
    if not pages_path.exists():
        print("pages.json not found; run fetch_dataset.py first", file=sys.stderr)
        sys.exit(1)
    config = yaml.safe_load((here.parent / "config.yaml").read_text())
    render = config["render"]
    out_dir = here / "pages"
    out_dir.mkdir(exist_ok=True)

    pages = json.loads(pages_path.read_text())
    for p in pages:
        img = render_page(
            p,
            width=render["width"],
            height=render["height"],
            body_size=render["body_font_size"],
            title_size=render["title_font_size"],
        )
        out = out_dir / f"{p['page_id']}.png"
        img.save(out)
        print(f"  {p['client']:10s}  {p['page_id']:10s}  ->  {out.relative_to(here.parent)}")
    print(f"Rendered {len(pages)} pages to {out_dir}")


if __name__ == "__main__":
    main()
