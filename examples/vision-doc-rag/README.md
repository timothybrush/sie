# Vision-first document RAG

Retrieve by image, answer by image. ColQwen2.5 reads each page as a picture
and ranks them via late interaction; Florence-2-DocVQA reads the winning
page and produces the textual answer. OCR never enters the score path, so
charts, screenshots, tables, and any other layout cue that would die in a
text round-trip still drives ranking. Everything runs on one SIE endpoint.

Each page also carries a `client` tag, so the same corpus serves multiple
tenants from one index — queries scoped to `acme-corp` cannot retrieve a
`globex` page, no separate index per tenant required.

## SIE features used

- `encode` — `vidore/colqwen2.5-v0.2` on page images at ingest and on the
  query text at search time. Output is a `[tokens, 128]` multivector. Late
  interaction (`sie_sdk.scoring.maxsim`) is the only ranking signal.
- `extract` — `mynkchaudhry/Florence-2-FT-DocVQA`. Called twice, with two
  jobs: with `instruction=<your question>` to get a textual answer for the
  top page, and without `instruction` to OCR the same page for a display
  snippet. The OCR snippet is UX-only — it never enters the score path.
- `score` *(optional)* — `Qwen/Qwen3-VL-Reranker-2B` second-stage rerank
  over `(query text, page image)`. Off by default while we wait for an
  upstream adapter fix; flip `search.visual_rerank: true` in `config.yaml`
  to enable it on a cluster that's ready.

## Why vision end-to-end

OCR-then-text-rerank throws away the exact signal we pick ColQwen for —
charts, screenshots, tables, callouts, and the spatial layout that tells
a wiki page apart from a checklist. The rerank stays visual or doesn't
happen. The OCR step shows on-screen text next to the page image so the
user can copy/paste from the result, nothing more.

## Multi-tenant by construction

Every page carries a `client` field in `data/pages.json`. The metadata list
loaded by `python/search.py` is filtered by `client_name` before MaxSim
runs, so a query scoped to `acme-corp` cannot retrieve a `globex` page.
Real deployments would push `client` down into the multivector store's
filter expression; the demo keeps everything in memory because the corpus
is tiny.

## Run it

You need Python 3.12 and a reachable SIE cluster (or local `docker run`).

```bash
# 1. SIE locally (or point SIE_CLUSTER_URL / SIE_API_KEY at a managed cluster).
docker run -p 8080:8080 ghcr.io/superlinked/sie-server:latest-cpu-default

# 2. Generate the synthetic corpus and render each page to a PNG.
cd examples/vision-doc-rag
pip install -r python/requirements.txt
python data/fetch_dataset.py
python data/render_pages.py

# 3. Encode every page with ColQwen2.5 and save the multivectors.
python python/ingest.py

# 4a. CLI demo — runs four scoped queries and prints results.
python python/search.py

# 4b. Or start the UI.
uvicorn --app-dir python server:app --port 8888
open http://localhost:8888
```

First run on a cold cluster pays a one-time model load: ColQwen2.5 and
Florence-2 are both several GB, expect roughly a minute on CPU and a few
seconds on GPU before the warm path kicks in.

### Pointing at a managed cluster

```bash
export SIE_CLUSTER_URL="https://your-cluster-host:8080"
export SIE_API_KEY="SL-..."
```

The defaults in `config.yaml` point at `http://localhost:8080` so the env
vars only matter when you're hitting something remote. Set `cluster.gpu`
to a profile name like `l4-spot` if the cluster needs an explicit GPU
class.

## Try these queries

| Tenant | Query | Why it's interesting |
|---|---|---|
| `acme-corp` | how do I sign in to the VPN? | Visual layout match — the page is titled "VPN setup for new engineers" with a bulleted body, and ColQwen2.5 picks it without keyword overlap with "sign in". DocVQA reads the page and answers with the client name and the auth method. |
| `globex` | what is the parental leave policy? | Disambiguates from "time off" — the right page mentions parental leave only halfway down the body. The textual answer cites the week count. |
| `initech` | audit prep evidence and walkthroughs | All three Initech pages are compliance-flavored; the visual model breaks the tie by reading the checklist layout. |
| `globex` | how do I sign in to the VPN? | Tenant filter — even though the same query hit acme-corp earlier, scoping to globex returns the closest globex page (Wi-Fi guide) and never leaks acme content. |

## API

### `GET /api/search`

| Parameter | Required | Description |
|---|---|---|
| `q` | yes | Search query |
| `client` | no | Tenant filter (e.g. `acme-corp`). Omitted ⇒ search runs across all tenants. |

```bash
curl "http://localhost:8888/api/search?q=how+do+I+sign+in+to+the+VPN&client=acme-corp"
```

```json
{
  "query": "how do I sign in to the VPN",
  "client": "acme-corp",
  "answer": "Okta credentials with Duo Push for 2FA",
  "timings": {
    "encode_query_s": 0.12,
    "maxsim_s": 0.003,
    "docvqa_s": 0.91,
    "ocr_snippet_s": 0.84
  },
  "results": [
    {
      "page_id": "ACME-101",
      "client": "acme-corp",
      "title": "VPN setup for new engineers",
      "space": "Engineering",
      "author": "alice@acme",
      "web_url": "https://acme.atlassian.net/wiki/spaces/ENG/pages/101",
      "page_image": "/pages/ACME-101.png",
      "ocr_snippet": "VPN Setup for New Engineers · ...",
      "scores": { "maxsim": 14.44, "rerank": null }
    }
  ]
}
```

### `GET /api/clients`, `GET /api/stats`

Tenant list and runtime config (active models, rerank on/off, page count).

## How it works

```
        ┌──────────────────────────────────────────────────────────────┐
        │  ingest.py  (once per corpus)                                │
        │  pages.json ─▶ render_pages.py ─▶ data/pages/*.png           │
        │      ─▶ SIE.encode(ColQwen2.5, images, multivector)          │
        │      ─▶ data/multivectors.npz + data/metadata.json           │
        └──────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
        ┌──────────────────────────────────────────────────────────────┐
        │  search.py / server.py  (per query)                          │
        │  q ─▶ SIE.encode(ColQwen2.5, text, is_query=True)            │
        │    ─▶ filter metadata by tenant                              │
        │    ─▶ sie_sdk.scoring.maxsim → top_k_candidates              │
        │    ─▶ [optional] SIE.score(Qwen3-VL-Reranker, q, images)     │
        │    ─▶ SIE.extract(Florence-2-DocVQA, instruction=q,          │
        │                   images=[top_page])  ⇒  textual answer      │
        │    ─▶ SIE.extract(Florence-2-DocVQA, images=[top_page])      │
        │                                       ⇒  OCR snippet (UI)   │
        └──────────────────────────────────────────────────────────────┘
```

OCR is never on the score path. The visual reranker (when enabled) ranks
over the same modality as retrieval, so layout cues survive both stages.

The corpus is small enough that MaxSim runs in Python. For thousands of
pages, hand the multivectors to LanceDB or Vespa; only the SIE calls stay
the same.

## Customize

`config.yaml` is the single tuning surface:

```yaml
models:
  retriever: "vidore/colqwen2.5-v0.2"      # smaller: vidore/colpali-v1.3-hf
  docvqa: "mynkchaudhry/Florence-2-FT-DocVQA"
  reranker: "Qwen/Qwen3-VL-Reranker-2B"    # used only when search.visual_rerank: true
search:
  top_k_candidates: 5
  top_k_results: 3
  visual_rerank: false
  answer: true
  ocr_snippet: true
```

Swap any model for another from the
[SIE model catalog](https://superlinked.com/models) and the pipeline keeps
working.

## Project layout

```text
examples/vision-doc-rag/
├── config.yaml
├── data/
│   ├── fetch_dataset.py        # synthetic 3-tenant page corpus
│   ├── render_pages.py         # pages.json → PNG screenshots
│   ├── pages.json              # generated
│   ├── pages/                  # generated PNGs
│   ├── metadata.json           # generated by ingest
│   └── multivectors.npz        # generated by ingest
├── python/
│   ├── ingest.py
│   ├── search.py
│   ├── server.py
│   └── requirements.txt
└── static/
    └── index.html
```
