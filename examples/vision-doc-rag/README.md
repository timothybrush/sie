# Vision-first document RAG

Retrieve by image, answer by image. ColQwen2.5 reads each PDF page as a
picture and ranks pages via late interaction; Florence-2-DocVQA reads the
winning page and produces the textual answer. OCR never enters the score path,
so schematics, pinout diagrams, architecture slides, charts, and other layout
cues still drive ranking. Everything runs on one SIE endpoint.

Each page also carries a `client` tag, so the same corpus serves multiple
tenants from one index. Queries scoped to `embedded-lab` cannot retrieve
`ops-eng` or `aerospace` pages.

## Corpus

The demo fetches a small public PDF batch on demand and renders selected pages
to PNGs. The page selections are deliberately capped so local ingest stays
fast while still indexing visually rich pages.

| Tenant | Sources | Visual signal |
|---|---|---|
| `embedded-lab` | Raspberry Pi Pico datasheet, Arduino UNO R3 datasheet, Arduino UNO R3 schematic | Pinout diagrams, board diagrams, circuit schematics |
| `ops-eng` | PostgreSQL manual, CNCF Kubernetes / cloud-native architecture material | Architecture diagrams, operational flows, dense technical tables |
| `aerospace` | NASA NTRS nozzle and booster reports | Engineering drawings, cross-sections, charts, mission technical figures |

Generated files are ignored:

```text
data/pdfs/                # downloaded PDFs
data/pdfs_manifest.json   # source manifest from fetch_pdfs.py
data/pages/               # rendered PNG pages
data/pages_manifest.json  # page-level metadata from render_pages.py
data/metadata.json        # index metadata from ingest.py
data/multivectors.npz     # page multivectors from ingest.py
```

## SIE features used

- `encode` - `vidore/colqwen2.5-v0.2` on page images at ingest and on query
  text at search time. Output is a `[tokens, 128]` multivector. Late
  interaction (`sie_sdk.scoring.maxsim`) is the first-stage ranking signal.
- `extract` - `mynkchaudhry/Florence-2-FT-DocVQA`. Called with
  `instruction=<your question>` to get a textual answer for the top page, and
  without `instruction` to OCR the same page for a display snippet. The OCR
  snippet is UX-only; it never enters ranking.
- `score` optional - `Qwen/Qwen3-VL-Reranker-2B` second-stage rerank over
  `(query text, page image)`. Off by default while we wait for an upstream
  adapter fix; flip `search.visual_rerank: true` in `config.yaml` to enable it
  on a cluster that's ready.

## Run it

You need Python 3.12 and a reachable SIE cluster.

```bash
# 1. SIE locally, or point SIE_CLUSTER_URL / SIE_API_KEY at a managed cluster.
docker run -p 8080:8080 ghcr.io/superlinked/sie-server:latest-cpu-default

# 2. Fetch public PDFs and render selected pages to PNG.
cd examples/vision-doc-rag
pip install -r python/requirements.txt
python data/fetch_pdfs.py
python data/render_pages.py

# 3. Encode every rendered page with ColQwen2.5 and save the multivectors.
python python/ingest.py

# 4a. CLI demo.
python python/search.py

# 4b. Or start the UI.
uvicorn --app-dir python server:app --port 8888
open http://localhost:8888
```

`render_pages.py` uses `pdf2image` when Poppler is available. If Poppler is
not installed, it falls back to PyMuPDF, which is installed from
`python/requirements.txt`.

First run on a cold cluster pays a one-time model load. ColQwen2.5 and
Florence-2 are both several GB, so expect roughly a minute on CPU and a few
seconds on GPU before the warm path kicks in.

### Managed cluster

```bash
export SIE_CLUSTER_URL="https://your-cluster-host:8080"
export SIE_API_KEY="SL-..."
```

The defaults in `config.yaml` point at `http://localhost:8080`. Set
`cluster.gpu` to a profile name like `l4-spot` if the cluster needs an
explicit GPU class.

## Try these queries

Queries are grouped by what they exercise. Each row names the expected target
page so you can spot regressions at a glance.

### Visual signal — the ranking comes from the page image, not OCR

| Tenant | Query | Expected target | Why it's interesting |
|---|---|---|---|
| `embedded-lab` | Raspberry Pi Pico pinout GP21 | Pi Pico datasheet pinout (pp 4-5) | Abbreviated visual label still drives retrieval. |
| `embedded-lab` | where is the ATmega16U2 on the schematic? | Arduino UNO R3 schematic (pp 1-2) | Circuit schematic retrieval, not prose. |
| `ops-eng` | cloud native architecture diagram | CNCF AI whitepaper or Kubernetes slides | Visual architecture page instead of OCR text. |
| `aerospace` | solid rocket motor nozzle design figure | Solid rocket motor nozzles report | Engineering drawing in a figure-heavy report. |

### Table / value lookups — the DocVQA answer is the point

| Tenant | Query | Expected target | Expected answer |
|---|---|---|---|
| `embedded-lab` | What is the operating voltage range of the Raspberry Pi Pico? | Pi Pico datasheet electrical characteristics (pp 6-8) | A voltage range, e.g. 1.8-5.5 V |
| `embedded-lab` | Which Arduino UNO pin is the built-in LED on? | UNO R3 datasheet pinout (pp 5-11) | D13 / PB5 |
| `ops-eng` | PostgreSQL default listening port | PG 18 manual config section (pp 19-24) | 5432 |
| `ops-eng` | What is the default value of max_connections in PostgreSQL? | PG 18 manual parameter table (pp 19-24) | 100 |
| `aerospace` | What is the throat diameter shown in the nozzle drawing? | Nozzle design figure | A labeled dimension off the drawing |

### Disambiguation — two PDFs in one tenant, the right one must win

| Tenant | Query | Should pick | Should beat |
|---|---|---|---|
| `aerospace` | solid propellant rocket nozzle cross-section | `solid-rocket-motor-nozzles.pdf` | `liquid-rocket-engine-nozzles.pdf` |
| `aerospace` | regeneratively cooled nozzle | `liquid-rocket-engine-nozzles.pdf` (regen cooling is liquid-specific) | `solid-rocket-motor-nozzles.pdf` |
| `embedded-lab` | USB-to-serial interface chip on the schematic | `arduino-uno-r3-schematic.pdf` (ATmega16U2) | `raspberry-pi-pico-datasheet.pdf` |
| `embedded-lab` | RP2040 GPIO function table | `raspberry-pi-pico-datasheet.pdf` | `arduino-uno-r3-datasheet.pdf` |

### Tenant-leak negatives — the matching content lives in a different tenant

| Scoped to | Query | Pass condition |
|---|---|---|
| `ops-eng` | Raspberry Pi Pico pinout GP21 | No embedded-lab pages return. |
| `ops-eng` | regeneratively cooled nozzle | No aerospace pages return. |
| `aerospace` | cloud native architecture diagram | No ops-eng pages return. |
| `embedded-lab` | PostgreSQL connection pool | No ops-eng pages return. |

## API

### `GET /api/search`

| Parameter | Required | Description |
|---|---|---|
| `q` | yes | Search query |
| `client` | no | Tenant filter, for example `embedded-lab`. Omitted means search all tenants. |

```bash
curl "http://localhost:8888/api/search?q=Raspberry+Pi+Pico+pinout+GP21&client=embedded-lab"
```

```json
{
  "query": "Raspberry Pi Pico pinout GP21",
  "client": "embedded-lab",
  "answer": "GP21 can be used for ...",
  "results": [
    {
      "page_id": "embedded-lab__raspberry-pi-pico-datasheet__p005",
      "client": "embedded-lab",
      "title": "Raspberry Pi Pico Datasheet",
      "publisher": "Raspberry Pi Ltd",
      "source_pdf": "raspberry-pi-pico-datasheet.pdf",
      "page_number": 5,
      "citation": "raspberry-pi-pico-datasheet.pdf · p.5",
      "page_image": "/pages/embedded-lab/raspberry-pi-pico-datasheet_p005.png",
      "scores": { "maxsim": 14.44, "rerank": null }
    }
  ]
}
```

### `GET /api/clients`, `GET /api/stats`

Tenant list and runtime config: active models, rerank on/off, and page count.

## How it works

```text
        ingest.py  (once per corpus)
        fetch_pdfs.py -> data/pdfs/{tenant}/*.pdf
             -> render_pages.py -> data/pages/{tenant}/*.png
             -> data/pages_manifest.json
             -> SIE.encode(ColQwen2.5, images, multivector)
             -> data/multivectors.npz + data/metadata.json

        search.py / server.py  (per query)
        q -> SIE.encode(ColQwen2.5, text, is_query=True)
          -> filter metadata by tenant
          -> sie_sdk.scoring.maxsim -> top_k_candidates
          -> optional SIE.score(Qwen3-VL-Reranker, q, images)
          -> SIE.extract(Florence-2-DocVQA, instruction=q, images=[top_page])
          -> SIE.extract(Florence-2-DocVQA, images=[top_page]) for display OCR
```

OCR is never on the score path. The visual reranker, when enabled, ranks over
the same modality as retrieval, so layout cues survive both stages.

The corpus is small enough that MaxSim runs in Python. For thousands of pages,
hand the multivectors to LanceDB, Vespa, or another multivector store; the SIE
calls stay the same.

## Customize

`data/fetch_pdfs.py` owns the curated source list. Add a source with:

```python
{
    "client": "my-tenant",
    "slug": "my-manual",
    "title": "My Manual",
    "publisher": "Example Publisher",
    "license": "CC BY 4.0",
    "url": "https://example.com/my-manual.pdf",
    "pages": [1, 2, 7, 8],
}
```

Then rerun:

```bash
python data/fetch_pdfs.py
python data/render_pages.py
python python/ingest.py
```

`config.yaml` is the model and rendering tuning surface:

```yaml
models:
  retriever: "vidore/colqwen2.5-v0.2"
  docvqa: "mynkchaudhry/Florence-2-FT-DocVQA"
  reranker: "Qwen/Qwen3-VL-Reranker-2B"
render:
  backend: "auto"
  dpi: 160
search:
  top_k_candidates: 5
  top_k_results: 3
  visual_rerank: false
  answer: true
  ocr_snippet: true
```

## Project layout

```text
examples/vision-doc-rag/
├── config.yaml
├── data/
│   ├── fetch_pdfs.py          # curated public PDF source list + downloader
│   ├── render_pages.py        # PDFs -> PNG pages + pages_manifest.json
│   ├── pdfs/                  # generated
│   ├── pages/                 # generated PNGs
│   ├── metadata.json          # generated by ingest
│   └── multivectors.npz       # generated by ingest
├── python/
│   ├── ingest.py
│   ├── search.py
│   ├── server.py
│   └── requirements.txt
└── static/
    └── index.html
```
