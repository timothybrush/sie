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

- OpenAI-compatible API for drop-in migration: `/v1/embeddings`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`
- Pre-configured model catalog: Stella, SPLADE, Qwen3, GLiNER, SigLIP, and more, all quality-verified against MTEB
- Serves multiple models simultaneously with on-demand loading and LRU eviction
- Ships the full production stack: load-balancing gateway, KEDA autoscaling, Grafana dashboards, Terraform for GKE, EKS, and AKS
- Integrates with LangChain, LlamaIndex, Haystack, DSPy, CrewAI, Chroma, Qdrant, and Weaviate

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

**1. Start the engine**

```bash
# macOS (Apple Silicon) or Linux, native (requires Python 3.12)
pip install "sie-server[local]" && sie-server serve

# Linux, NVIDIA GPU
docker run --gpus all -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cuda12-default

# Linux, CPU
docker run -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cpu-default
```

```bash
curl http://localhost:8080/readyz   # expect: ok
```

**2. Install the SDK**

```bash
pip install sie-sdk           # Python
pnpm add @superlinked/sie-sdk # TypeScript
```

**3. Call models**

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient("http://localhost:8080")

# Dense embeddings
result = client.encode("sentence-transformers/all-MiniLM-L6-v2", Item(text="Hello world"))
print(result["dense"].shape)  # (384,)

# Rerank documents against a query
scores = client.score(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    Item(text="What is machine learning?"),
    [Item(text="ML learns from data."), Item(text="The weather is sunny.")],
)
print(scores["scores"][0])  # {'item_id': 'item-0', 'score': -7.1, 'rank': 0}

# Zero-shot entity extraction
result = client.extract(
    "urchade/gliner_multi-v2.1",
    Item(text="Tim Cook is the CEO of Apple."),
    labels=["person", "organization"],
)
print(result["entities"][0])  # {'text': 'Tim Cook', 'label': 'person', 'score': 0.991}
```

The first call to a model downloads its weights from Hugging Face; after that, calls are warm. Text generation runs on the GPU generation image:

```bash
docker run --gpus all -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cuda12-sglang
```

```python
result = client.generate("Qwen/Qwen3-0.6B", "Reply with a single word: the capital of France.", max_new_tokens=16)
print(result["text"])  # Paris
```

For generation on Apple Silicon (MLX), the TypeScript walkthrough, and every configuration in between, see the [quickstart guide](https://superlinked.com/docs/quickstart/), [TypeScript SDK docs](https://superlinked.com/docs/reference/typescript-sdk/), and [SDK reference](https://superlinked.com/docs/reference/sdk/).

---

### Production

The same code works against a production cluster. SIE ships a load-balancing gateway, KEDA autoscaling (scale to zero), Grafana dashboards, and Terraform modules for [GKE](https://github.com/superlinked/terraform-google-sie), [EKS](https://github.com/superlinked/terraform-aws-sie), and [AKS](https://github.com/superlinked/terraform-azure-sie). Not just the server, the whole stack. All Apache 2.0.

```bash
helm upgrade --install sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster \
  --namespace sie --create-namespace \
  --set hfToken.create=true \
  --set hfToken.value=YOUR_HF_TOKEN \
  -f deploy/helm/sie-cluster/values-{gke|aws|aks}.yaml
```

See the [deployment guide](https://superlinked.com/docs/deployment/).

> **Telemetry**: SIE collects anonymous usage data (version, OS, architecture, GPU type) to understand adoption. No IP addresses, hostnames, or request data are collected. Disable with `SIE_TELEMETRY_DISABLED=1` or `DO_NOT_TRACK=1`.

---

### Explore

[**85+ models**](https://superlinked.com/models): dense, sparse, multi-vector, vision, cross-encoder, and generative architectures. Every model is a config in [`packages/sie_server/models/`](https://github.com/superlinked/sie/tree/main/packages/sie_server/models); pass its full Hugging Face ID to the SDK (e.g. `sentence-transformers/all-MiniLM-L6-v2`, `Qwen/Qwen3-4B-Instruct-2507`).

[**Integrations**](https://superlinked.com/docs/integrations/): LangChain, LlamaIndex, Haystack, DSPy, CrewAI, Chroma, Qdrant, Weaviate.

[**Examples**](examples/): A quickstart notebook and an end-to-end project gallery.

[**Why we built SIE**](https://www.youtube.com/watch?v=qdh_x-uRs9g): The motivation, told at AI Engineer Europe 2026.

---

<p align="center">
  <a href="https://superlinked.com/docs"><strong>superlinked.com/docs</strong></a> | Apache 2.0
</p>
