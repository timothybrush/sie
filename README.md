<div align="center">

<picture>
  <source srcset="https://cdn.prod.website-files.com/65dce6831bf9f730421e2915/66ef0317ed8616151ee1d451_superlinked_logo_white.png"
          media="(prefers-color-scheme: dark)">
  <img width="320"
       src="https://cdn.prod.website-files.com/65dce6831bf9f730421e2915/65dce6831bf9f730421e2929_superlinked_logo.svg"
       alt="Superlinked logo">
</picture>

<h1>SIE: Superlinked Inference Engine</h1>

<p><strong>Open-source inference server and production cluster for embeddings, reranking, and extraction.</strong></p>
<p>85+ models. Three functions. From laptop to Kubernetes. All Apache 2.0.</p>

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

SIE is an open-source inference engine that serves embeddings, reranking, and entity extraction through a single unified API. It replaces the patchwork of separate model servers with one system that handles 85+ models across dense, sparse, multi-vector, vision, and cross-encoder architectures.

- Three functions (`encode`, `score`, `extract`) cover the entire embedding, reranking, and extraction pipeline
- 85+ pre-configured models, hot-swappable, all quality-verified against MTEB in CI
- Serves multiple models simultaneously with on-demand loading and LRU eviction
- Ships the full production stack: load-balancing gateway, KEDA autoscaling, Grafana dashboards, Terraform for GKE/EKS
- Integrates with LangChain, LlamaIndex, Haystack, DSPy, CrewAI, Chroma, Qdrant, and Weaviate
- OpenAI-compatible `/v1/embeddings` endpoint for drop-in migration

## Quickstart

SIE runs as a Docker container that your code calls over HTTP. Start the container first, then install the SDK in your app and run the example below against it. Docker is the recommended path for local work: it ships the model runtime, CUDA bits, and dependencies as one image, so you avoid Python and driver setup.

**1. Run the engine**

```bash
docker run -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cpu-default                # CPU
docker run --gpus all -p 8080:8080 -v sie-hf-cache:/app/.cache/huggingface ghcr.io/superlinked/sie-server:latest-cuda12-default  # GPU
```

The first image pull is around 1 GB (CPU) or 9 GB (GPU). After the pull, the container reaches `/readyz` in a few seconds. The `-v sie-hf-cache:/app/.cache/huggingface` flag persists model weights in a named Docker volume across runs; without it, every fresh container re-downloads them. (A named volume is preferred over a host bind-mount because the container runs as a non-root user; a bind-mount can hit UID permission issues on macOS and Windows.) The first call to a given model downloads its weights from Hugging Face on demand (typically tens of seconds to a couple of minutes depending on model size and connection); subsequent calls hit the cache and return in milliseconds. Apple Silicon users: prepend `--platform linux/amd64` (the image is `linux/amd64` only and runs under Rosetta).

See the [deployment guide](https://superlinked.com/docs/deployment/docker) for GPU pinning, read-only filesystems, and tuning.

**2. Use SIE from Python or TypeScript**

```bash
pip install sie-sdk           # Python
pnpm add @superlinked/sie-sdk # TypeScript
```

The SDKs are small HTTP clients; they do not run any models locally. The entire API is three functions: `encode`, `score`, `extract`.

```python
from sie_sdk import SIEClient
from sie_sdk.types import Item

client = SIEClient("http://localhost:8080")

# Encode: dense embeddings, 400M-parameter model
result = client.encode("NovaSearch/stella_en_400M_v5", Item(text="Hello world"))
print(result["dense"].shape)  # (1024,)

# Score: rerank documents by relevance
scores = client.score(
    "BAAI/bge-reranker-v2-m3",
    Item(text="What is machine learning?"),
    [Item(text="ML learns from data."), Item(text="The weather is sunny.")]
)
print(scores["scores"])
# [{'item_id': 'item-0', 'score': 0.998, 'rank': 0},
#  {'item_id': 'item-1', 'score': 0.012, 'rank': 1}]

# Extract: zero-shot named entity recognition, no training data
result = client.extract(
    "urchade/gliner_multi-v2.1",
    Item(text="Tim Cook is the CEO of Apple."),
    labels=["person", "organization"]
)
print(result["entities"])
# [{'text': 'Tim Cook', 'label': 'person', 'score': 0.96},
#  {'text': 'Apple', 'label': 'organization', 'score': 0.91}]
```

For the equivalent TypeScript example, see the [TypeScript SDK docs](https://superlinked.com/docs/reference/typescript-sdk/).

See the [full quickstart guide](https://superlinked.com/docs/quickstart/) and [SDK reference](https://superlinked.com/docs/reference/sdk/).

---

### Production

The same code works against a production cluster. SIE ships a load-balancing gateway, KEDA autoscaling (scale to zero), Grafana dashboards, and Terraform modules for GKE and EKS. Not just the server, the whole stack. All Apache 2.0.

```bash
helm upgrade --install sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster \
  --namespace sie --create-namespace \
  --set hfToken.create=true \
  --set hfToken.value=<TOKEN> \
  -f deploy/helm/sie-cluster/values-{gke|aws}.yaml
```

See the [deployment guide](https://superlinked.com/docs/deployment/).

> **Telemetry**: SIE collects anonymous usage data (version, OS, architecture, GPU type) to understand adoption. No IP addresses, hostnames, or request data are collected. Disable with `SIE_TELEMETRY_DISABLED=1` or `DO_NOT_TRACK=1`.

---

### Explore

[**85+ models**](https://superlinked.com/models): `Stella v5`, `BGE-M3`, `SPLADE v3`, `SigLIP`, `ColQwen2.5`, `BGE-reranker`, `GLiNER`, `Florence-2`, and [more](https://superlinked.com/models).
Dense, sparse, multi-vector, vision, rerankers, extractors. All pre-configured. All quality-verified against MTEB in CI.

[**Integrations**](https://superlinked.com/docs/integrations/): LangChain, LlamaIndex, Haystack, DSPy, CrewAI, Chroma, Qdrant, Weaviate.

[**Notebooks**](notebooks/): Quickstarts and walkthroughs

[**Examples**](examples/): End-to-end project gallery

---

<p align="center">
  <a href="https://superlinked.com/docs"><strong>superlinked.com/docs</strong></a> | Apache 2.0
</p>
