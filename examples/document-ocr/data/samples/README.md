# Bundled sample documents

Six synthetic, public-domain images covering the document shapes most
real-world OCR pipelines hit:

- `receipt.png`: printed grocery receipt, line items + totals
- `invoice.png`: vendor invoice, multi-column form layout
- `business-card.png`: tight contact card, mixed text sizes
- `table.png`: dense numerical table with totals row
- `handwritten.png`: jittered text that simulates informal handwriting
- `multi-column.png`: two-column newspaper-style layout where reading order
  matters

`index.json` carries metadata for each: the GLiNER labels we ask for, plus a
short description shown in the UI.

Regenerate with `python scripts/generate_samples.py`. Pillow is the only
dep; no real customer data is involved.
