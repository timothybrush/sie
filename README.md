<div align="center">

<picture>
  <source srcset="https://cdn.prod.website-files.com/65dce6831bf9f730421e2915/66ef0317ed8616151ee1d451_superlinked_logo_white.png"
          media="(prefers-color-scheme: dark)">
  <img width="320"
       src="https://cdn.prod.website-files.com/65dce6831bf9f730421e2915/65dce6831bf9f730421e2929_superlinked_logo.svg"
       alt="Superlinked logo">
</picture>

<h1>SIE: Superlinked Inference Engine</h1>

<p><strong>Self-hosted inference for agents. Every model your agents call, served from one open-source cluster in your cloud.</strong></p>
<p>85+ models: Stella, SPLADE, Qwen3, GLiNER, SigLIP, and more. One API. From laptop to Kubernetes. All Apache 2.0.</p>

<p>
  <a href="https://superlinked.com/docs/">Docs</a> |
  <a href="https://superlinked.com/docs/quickstart/">Quickstart</a> |
  <a href="https://superlinked.com/docs/reference/api/">API Reference</a> |
  <a href="https://superlinked.com/models">Models</a>
</p>

[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/sie-sdk?style=flat-square)](https://pypi.org/project/sie-sdk/)
[![GitHub stars](https://img.shields.io/github/stars/superlinked/sie?style=flat-square)](https://github.com/superlinked/sie/stargazers)

</div>

## About

SIE is an open-source inference engine that runs the models behind every agent task through one API: search and retrieval, document-to-markdown conversion, structured output, content safety, and the agent loop itself. It replaces the patchwork of a separate model server per task with one system that serves 85+ models, loading each on demand.

- 85+ pre-configured models, hot-swappable, all quality-verified against MTEB in CI
- Serves multiple models simultaneously with on-demand loading and LRU eviction
- Ships the full production stack: load-balancing gateway, KEDA autoscaling, Grafana dashboards, Terraform for GKE/EKS
- Integrates with LangChain, LlamaIndex, Haystack, DSPy, CrewAI, Chroma, Qdrant, and Weaviate
- OpenAI-compatible `/v1/embeddings` endpoint for drop-in migration

## Tasks

One SIE cluster runs the inference behind a whole agent. Each task is a handful of swappable models; browse [`packages/sie_server/models/`](https://github.com/superlinked/sie/tree/main/packages/sie_server/models) for the full set.

| Task | What it does | Models |
|---|---|---|
| **Search** | Embed, match, and rerank to retrieve the right context. | `bge-m3`, `splade-v3`, `colbertv2`, `qwen3-reranker` |
| **Document to markdown** | PDFs, Office files, and scans become clean markdown. | `glm-ocr`, `mineru`, `paddleocr-vl`, `docling` |
| **Structured output** | Schema-valid JSON, extracted or generated. | `gliner2`, `nuner-zero`, `qwen3.6-27b` |
| **Guard content** | A safety verdict with a probability you threshold. | `granite-guardian-2b` |
| **Run the agent loop** | Plan steps and call tools with an open LLM, streaming included. | `qwen3.6-27b` |

## Quickstart

SIE runs as a server you call over HTTP — a Docker container, or a native macOS pip install for Apple Silicon. Start it, install the SDK, run the example.

**1. Run the engine**

```bash
# macOS (Apple Silicon) — native, served on Metal (requires Python 3.12).
# Embeddings + reranking (torch-MPS):
pip install "sie-server[local]" && sie-server serve
# Generation (Apple MLX), in its own env — keeps mlx-lm's transformers>=5 out of the
# embed/rerank lock:
#   uvx --with "mlx-lm>=0.30.7" --from sie-server sie-server serve -b sglang -p 8081
# Or run the Linux CPU image under emulation:
#   docker run --platform linux/amd64 -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cpu-default

# Linux, CPU
docker run -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cpu-default

# Linux, NVIDIA GPU
docker run --gpus all -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cuda12-default
```

Confirm it is up:

```bash
curl http://localhost:8080/readyz   # expect: ok
```

**2. Use SIE from Python or TypeScript**

```bash
pip install sie-sdk           # Python
pnpm add @superlinked/sie-sdk # TypeScript
```

The same `SIEClient` talks to every model. Four of them, one call each:

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient("http://localhost:8080")
# First call to each model downloads weights from Hugging Face (seconds for
# these tinies, longer for larger models). After that, calls are warm in ms.

# all-MiniLM-L6-v2: compact dense embeddings (~90 MB)
result = client.encode("sentence-transformers/all-MiniLM-L6-v2", Item(text="Hello world"))
print(result["dense"].shape)  # (384,)

# ms-marco-MiniLM: cross-encoder that reranks documents by relevance (~80 MB)
scores = client.score(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    Item(text="What is machine learning?"),
    [Item(text="ML learns from data."), Item(text="The weather is sunny.")]
)
print(scores["scores"])
# [{'item_id': 'item-0', 'score': -7.1,    'rank': 0},
#  {'item_id': 'item-1', 'score': -11.048, 'rank': 1}]
# (cross-encoder logits; relative order is what matters, not the absolute value)

# GLiNER: zero-shot entity extraction with any labels, no training data
result = client.extract(
    "urchade/gliner_multi-v2.1",
    Item(text="Tim Cook is the CEO of Apple."),
    labels=["person", "organization"]
)
print(result["entities"])
# [{'text': 'Tim Cook', 'label': 'person',       'score': 0.991},
#  {'text': 'Apple',    'label': 'organization', 'score': 0.978}]

# Qwen3-0.6B: open-weight text generation (~1.2 GB). Generation needs a GPU and
# the generation image (latest-cuda12-sglang below), not the latest-cpu-default
# server above; SGLang has no CPU path.
result = client.generate(
    "Qwen/Qwen3-0.6B",
    "Reply with a single word: the capital of France.",
    max_new_tokens=16,
)
print(result["text"])   # 'Paris'
print(result["usage"])  # {'prompt_tokens': 12, 'completion_tokens': 1, 'total_tokens': 13}
```

`encode`, `score`, and `extract` run on the `latest-cpu-default` server above.
Generation ships in a separate GPU image (the `sglang` bundle); start it with:

```bash
docker run --gpus all -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cuda12-sglang
```

For the equivalent TypeScript example, see the [TypeScript SDK docs](https://superlinked.com/docs/reference/typescript-sdk/). For more, see the [full quickstart guide](https://superlinked.com/docs/quickstart/) and [SDK reference](https://superlinked.com/docs/reference/sdk/).

---

### Production

The same code works against a production cluster. SIE ships a load-balancing gateway, KEDA autoscaling (scale to zero), Grafana dashboards, and Terraform modules for GKE and EKS. Not just the server, the whole stack. All Apache 2.0.

```bash
helm upgrade --install sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster \
  --namespace sie --create-namespace \
  --set hfToken.create=true \
  --set hfToken.value=YOUR_HF_TOKEN \
  -f deploy/helm/sie-cluster/values-{gke|aws}.yaml
```

See the [deployment guide](https://superlinked.com/docs/deployment/).

> **Telemetry**: SIE collects anonymous usage data (version, OS, architecture, GPU type) to understand adoption. No IP addresses, hostnames, or request data are collected. Disable with `SIE_TELEMETRY_DISABLED=1` or `DO_NOT_TRACK=1`.

---

### Explore

[**85+ models**](https://superlinked.com/models) across embedders, rerankers, extractors, and generators: dense, sparse, multi-vector, vision, cross-encoder, and generative architectures. All pre-configured, all quality-verified in CI.
Every model is a config in [`packages/sie_server/models/`](https://github.com/superlinked/sie/tree/main/packages/sie_server/models); pass its full Hugging Face ID to the SDK (e.g. `sentence-transformers/all-MiniLM-L6-v2`, `Qwen/Qwen3-4B-Instruct-2507`). Browse the rendered [catalog](https://superlinked.com/models) for the complete list.

[**Integrations**](https://superlinked.com/docs/integrations/): LangChain, LlamaIndex, Haystack, DSPy, CrewAI, Chroma, Qdrant, Weaviate.

[**Notebooks**](notebooks/): Quickstarts and walkthroughs

[**Examples**](examples/): End-to-end project gallery

[**Why we built SIE**](https://www.youtube.com/watch?v=qdh_x-uRs9g): The motivation, told at AI Engineer Europe 2026.

---

<p align="center">
  <a href="https://superlinked.com/docs"><strong>superlinked.com/docs</strong></a> | Apache 2.0
</p>
