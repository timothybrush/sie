"""Generate the bundled sample document images.

Synthetic, public-domain content: no real receipts/invoices/customer data.
Run once to regenerate `data/samples/*.png`. The output is committed; this
script is here so the fixtures are reproducible.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates_regular = [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    candidates_bold = [
        "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ]
    options = candidates_bold if bold else candidates_regular
    for p in options:
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render_lines(lines: list[tuple[str, int, bool]], width: int = 480, padding: int = 24, line_height: int | None = None) -> Image.Image:
    height_estimate = padding * 2 + sum((line_height or sz + 6) for (_, sz, _) in lines)
    img = Image.new("RGB", (width, height_estimate), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    y = padding
    for text, size, bold in lines:
        f = font(size, bold=bold)
        draw.text((padding, y), text, fill=(0, 0, 0), font=f)
        y += (line_height or size + 6)
    return img


def make_receipt() -> Image.Image:
    lines = [
        ("ACME GROCERS", 22, True),
        ("123 Market Lane", 12, False),
        ("Springfield, IL 62701", 12, False),
        ("Tel: (555) 010-2034", 12, False),
        ("", 8, False),
        ("Date: 2026-04-12   Time: 14:32", 12, False),
        ("Cashier: Aki                    ", 12, False),
        ("--------------------------------", 12, False),
        ("Whole Wheat Bread        4.50", 14, False),
        ("Organic Milk 1L          3.20", 14, False),
        ("Bananas (1.2 kg)         2.40", 14, False),
        ("Free Range Eggs x12      6.99", 14, False),
        ("Coffee Beans 250g        9.75", 14, False),
        ("--------------------------------", 12, False),
        ("Subtotal                26.84", 14, False),
        ("Sales Tax (6.25%)        1.68", 14, False),
        ("TOTAL                   28.52", 16, True),
        ("", 8, False),
        ("Paid: VISA ****1234", 12, False),
        ("Thank you, please come again!", 12, False),
    ]
    return render_lines(lines, width=460)


def make_invoice() -> Image.Image:
    lines = [
        ("BLUERIDGE SUPPLIES, LLC", 20, True),
        ("Invoice No: INV-2026-0381", 14, True),
        ("Date Issued: 2026-03-22", 12, False),
        ("Due Date: 2026-04-21", 12, False),
        ("", 8, False),
        ("Bill To:", 13, True),
        ("Northwind Coffee Co.", 12, False),
        ("412 Pine Street, Suite 5", 12, False),
        ("Portland, OR 97204", 12, False),
        ("", 8, False),
        ("--------------------------------------------", 11, False),
        ("Description                Qty  Price   Amount", 12, True),
        ("--------------------------------------------", 11, False),
        ("Espresso machine filter     4   45.00   180.00", 12, False),
        ("Replacement gasket          8    6.50    52.00", 12, False),
        ("Shop towels (case)          2   24.00    48.00", 12, False),
        ("Calibration service         1   95.00    95.00", 12, False),
        ("--------------------------------------------", 11, False),
        ("                              Subtotal  375.00", 12, False),
        ("                              Tax (8%)   30.00", 12, False),
        ("                              TOTAL    405.00", 14, True),
        ("", 8, False),
        ("Payment terms: Net 30. Wire to acct 4421-0119.", 11, False),
    ]
    return render_lines(lines, width=600)


def make_business_card() -> Image.Image:
    img = Image.new("RGB", (520, 260), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (519, 22)], fill=(124, 93, 255))
    draw.text((20, 4), "STARGAZER ROBOTICS", fill=(255, 255, 255), font=font(13, bold=True))

    draw.text((30, 50), "Dr. Mira Okafor", fill=(0, 0, 0), font=font(22, bold=True))
    draw.text((30, 84), "Principal Research Engineer", fill=(60, 60, 80), font=font(14, bold=False))

    draw.text((30, 130), "Email: m.okafor@stargazer-robotics.com", fill=(0, 0, 0), font=font(13, bold=False))
    draw.text((30, 155), "Phone: +1 (415) 555-0188", fill=(0, 0, 0), font=font(13, bold=False))
    draw.text((30, 180), "Office: 1900 Embarcadero, Oakland CA 94606", fill=(0, 0, 0), font=font(13, bold=False))
    draw.text((30, 210), "stargazer-robotics.com / @stargazer_rob", fill=(60, 60, 90), font=font(12, bold=False))

    return img


def make_table() -> Image.Image:
    lines = [
        ("Q1 2026 Department Spend (in USD)", 18, True),
        ("Source: Internal finance, ledger close 2026-04-05", 11, False),
        ("", 6, False),
        ("Department      Headcount   Salary    Cloud   Travel   Total", 13, True),
        ("------------------------------------------------------------", 11, False),
        ("Engineering            42  581,200   91,400   12,800  685,400", 13, False),
        ("Sales                  18  240,300    4,100   38,900  283,300", 13, False),
        ("Marketing              11  168,750    7,900   15,200  191,850", 13, False),
        ("Operations              7   88,400    1,200    3,400   93,000", 13, False),
        ("Customer Success       12  148,900    2,400    9,500  160,800", 13, False),
        ("Legal & Compliance      4   72,600      900      750   74,250", 13, False),
        ("------------------------------------------------------------", 11, False),
        ("TOTAL                  94  1,300,150 107,900   80,550 1,488,600", 14, True),
        ("", 6, False),
        ("Notes: Cloud excludes prepaid commits ($240k YTD).", 11, False),
        ("       Travel includes T&E, but not relocation expenses.", 11, False),
    ]
    return render_lines(lines, width=720)


def make_handwritten_note() -> Image.Image:
    # Not actually handwriting (we lack a reliable cursive font), so this is
    # a "casual jotting" style: tighter line spacing, a few intentional jitters,
    # to give the OCR-VLM something less template-like to read.
    img = Image.new("RGB", (540, 360), color=(252, 252, 240))
    draw = ImageDraw.Draw(img)
    rng = random.Random(42)

    def jitter_text(xy, text, size):
        x, y = xy
        for ch in text:
            dx = rng.randint(-1, 1)
            dy = rng.randint(-1, 1)
            draw.text((x + dx, y + dy), ch, fill=(20, 20, 30), font=font(size, bold=False))
            x += int(size * 0.55) + rng.randint(-1, 2)

    jitter_text((30, 24), "Reminders for Friday:", 16)
    jitter_text((40, 60), "1. Call Tom about the warranty claim", 14)
    jitter_text((40, 88), "   - they need photo by 5pm", 13)
    jitter_text((40, 120), "2. Pick up book from library", 14)
    jitter_text((40, 148), "   (Hawking, A Brief History of Time)", 12)
    jitter_text((40, 184), "3. Sushi for dinner - Mira likes Aoba", 14)
    jitter_text((40, 218), "4. Order replacement HDMI cable", 14)
    jitter_text((40, 246), "   2m or 3m, USB-C end", 12)
    jitter_text((30, 296), "Total errands today: 4", 14)
    jitter_text((30, 322), "Budget: $80 cash, $200 card", 13)
    return img


def make_multi_column() -> Image.Image:
    # Two columns of "newspaper-like" text to test reading-order handling.
    img = Image.new("RGB", (740, 460), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((30, 20), "THE MORNING WIRE", fill=(0, 0, 0), font=font(22, bold=True))
    draw.text((30, 50), "Vol. XII, No. 47       Tuesday, March 17, 2026       Price: 1.50", fill=(60, 60, 60), font=font(11, bold=False))
    draw.line([(30, 80), (710, 80)], fill=(0, 0, 0), width=1)

    col_a = [
        "City Council Approves",
        "Riverside Park Expansion",
        "",
        "After two years of public debate, the",
        "city council voted 7-2 yesterday to",
        "approve the long-discussed Riverside",
        "Park expansion. The 14-acre extension",
        "will include a wetland boardwalk, two",
        "playgrounds, and a community amphi-",
        "theater. Funding comes from a 2024",
        "bond measure plus a state matching",
        "grant.",
        "",
        "Construction is expected to begin in",
        "May with phased openings through",
        "late 2027.",
    ]
    col_b = [
        "Local Bakery Closes",
        "After Forty Years",
        "",
        "Sundance Bakery, a fixture on Elm",
        "Street since 1986, will close at the",
        "end of next month. Owner Lila",
        "Carnesale cited rising commercial",
        "rents and the difficulty of finding",
        "overnight bakers as the deciding",
        "factors.",
        "",
        "A farewell event is planned for the",
        "final Saturday, with proceeds going",
        "to the regional food bank.",
        "",
        "Loyal customers are already pre-",
        "ordering their favorites for the last",
        "week of operation.",
    ]

    y = 100
    for line in col_a:
        size = 14 if not line.startswith(line[:2].upper()) or line == "" else 14
        draw.text((30, y), line, fill=(0, 0, 0), font=font(13, bold=line in {"City Council Approves", "Riverside Park Expansion"}))
        y += 18

    y = 100
    for line in col_b:
        draw.text((400, y), line, fill=(0, 0, 0), font=font(13, bold=line in {"Local Bakery Closes", "After Forty Years"}))
        y += 18

    return img


def make_event_poster() -> Image.Image:
    # Poster-style. Florence-2 OCR shines on this: large text, simple layout,
    # high contrast, plenty of whitespace.
    img = Image.new("RGB", (780, 1040), color=(252, 246, 235))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (779, 90)], fill=(124, 93, 255))
    draw.text((40, 18), "SPRING JAZZ FESTIVAL", fill=(255, 255, 255), font=font(34, bold=True))
    draw.text((40, 58), "Riverside Park Amphitheater", fill=(240, 230, 255), font=font(18, bold=False))

    draw.text((40, 150), "Saturday, May 24, 2026", fill=(20, 20, 30), font=font(28, bold=True))
    draw.text((40, 200), "Gates open 6:00 PM   Music 7:00 PM - 10:00 PM", fill=(40, 40, 60), font=font(18, bold=False))
    draw.line([(40, 250), (740, 250)], fill=(124, 93, 255), width=2)

    draw.text((40, 280), "FEATURED ARTISTS", fill=(124, 93, 255), font=font(20, bold=True))
    draw.text((40, 320), "Maya Brennan Quintet", fill=(20, 20, 30), font=font(22, bold=False))
    draw.text((40, 358), "Otis Park & The Resolve", fill=(20, 20, 30), font=font(22, bold=False))
    draw.text((40, 396), "Jeong-min Trio (special guest)", fill=(20, 20, 30), font=font(22, bold=False))

    draw.line([(40, 460), (740, 460)], fill=(124, 93, 255), width=2)
    draw.text((40, 490), "TICKETS", fill=(124, 93, 255), font=font(20, bold=True))
    draw.text((40, 530), "Advance: $18   Door: $24   Under 12: free", fill=(20, 20, 30), font=font(20, bold=False))
    draw.text((40, 570), "Visit springjazzfest.org or call 555-0142", fill=(20, 20, 30), font=font(18, bold=False))

    draw.text((40, 660), "Presented by Riverside Arts Council", fill=(60, 60, 90), font=font(16, bold=False))
    draw.text((40, 690), "with support from Bluebird Coffee Roasters", fill=(60, 60, 90), font=font(16, bold=False))
    return img


def make_slide() -> Image.Image:
    # Presentation slide. Clean, lots of whitespace, big text. Florence-2 OCR
    # tends to read this well: it's the kind of "single image with prominent
    # text" the model was trained on.
    img = Image.new("RGB", (960, 540), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (959, 70)], fill=(20, 20, 40))
    draw.text((40, 18), "Q2 2026 Product Roadmap", fill=(255, 255, 255), font=font(28, bold=True))

    draw.text((50, 130), "1.  Ship the unified search experience", fill=(20, 20, 30), font=font(24, bold=False))
    draw.text((80, 168), "across web and mobile clients", fill=(80, 80, 100), font=font(18, bold=False))

    draw.text((50, 230), "2.  Move billing onto the new ledger service", fill=(20, 20, 30), font=font(24, bold=False))
    draw.text((80, 268), "complete the dual-write cutover by June 15", fill=(80, 80, 100), font=font(18, bold=False))

    draw.text((50, 330), "3.  Launch the developer documentation site", fill=(20, 20, 30), font=font(24, bold=False))
    draw.text((80, 368), "public preview at WWDC keynote", fill=(80, 80, 100), font=font(18, bold=False))

    draw.text((50, 470), "Owners: P. Lin (search), M. Okafor (billing), R. Tate (docs)", fill=(60, 60, 90), font=font(14, bold=False))
    return img


def make_letter() -> Image.Image:
    # Short business letter. Clean printed paragraphs.
    lines = [
        ("Westmoor Acoustics, Inc.", 16, True),
        ("418 Crown Street, Sausalito, CA 94965", 12, False),
        ("", 8, False),
        ("April 8, 2026", 13, False),
        ("", 8, False),
        ("Dear Ms. Hernandez,", 14, False),
        ("", 6, False),
        ("Thank you for your interest in the model 218 studio monitor.", 13, False),
        ("We have reserved a pair under your name; they will ship from", 13, False),
        ("our Oakland warehouse on April 15 and should arrive by", 13, False),
        ("April 20 via FedEx Ground.", 13, False),
        ("", 6, False),
        ("Your order total of $1,940.00 includes the 5-year transferable", 13, False),
        ("warranty and one calibration visit. Please retain the enclosed", 13, False),
        ("serial number card for any future service request.", 13, False),
        ("", 6, False),
        ("If you have questions, our service line is (415) 555-0173,", 13, False),
        ("open weekdays 9 AM to 5 PM Pacific time.", 13, False),
        ("", 6, False),
        ("Sincerely,", 13, False),
        ("", 8, False),
        ("Jane Salisbury", 14, True),
        ("Customer Account Manager", 12, False),
    ]
    return render_lines(lines, width=640, padding=32)


SAMPLES = [
    ("receipt.png", "Grocery receipt", "Printed receipt with line items, subtotal, tax, total. Donut on CORD reads this end-to-end.", make_receipt, ["merchant", "date", "line_item", "subtotal", "tax", "total", "payment_method"]),
    ("invoice.png", "Vendor invoice", "Multi-column invoice with billing party, line items, subtotal, tax, total.", make_invoice, ["vendor", "invoice_number", "date", "due_date", "billing_party", "line_item", "total"]),
    ("business-card.png", "Business card", "Tight layout, mixed text sizes, multiple contact fields. Good NER showcase.", make_business_card, ["company", "person", "role", "email", "phone", "address", "website"]),
    ("event-poster.png", "Event poster", "Large-text poster with title, date, artists, ticket info. Florence-2 OCR's home turf.", make_event_poster, ["event", "date", "venue", "artist", "price", "organization"]),
    ("slide.png", "Presentation slide", "Roadmap slide with title and three numbered items. Clean printed text on a single background.", make_slide, ["initiative", "person", "date", "quarter"]),
    ("letter.png", "Business letter", "Short printed business letter with sender, date, recipient, body, and signature.", make_letter, ["company", "person", "address", "date", "amount", "phone"]),
]


def main() -> None:
    index = []
    for filename, label, description, factory, labels in SAMPLES:
        img = factory()
        out_path = OUT_DIR / filename
        img.save(out_path, "PNG", optimize=True)
        index.append({
            "id": filename.replace(".png", ""),
            "filename": filename,
            "label": label,
            "description": description,
            "labels": labels,
        })
        print(f"wrote {out_path} ({img.size[0]}x{img.size[1]})")

    (OUT_DIR / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {OUT_DIR / 'index.json'} ({len(index)} samples)")


if __name__ == "__main__":
    main()
