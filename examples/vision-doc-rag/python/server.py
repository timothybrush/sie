"""FastAPI backend for the multi-tenant visual-document search + QA demo."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sie_sdk import SIEClient

from search import load_index, search

config = None
multivectors = None
metadata = None
client = None
clients_index: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, multivectors, metadata, client, clients_index
    root = Path(__file__).resolve().parent.parent
    config = yaml.safe_load((root / "config.yaml").read_text())
    multivectors, metadata = load_index()
    cluster_url = os.environ.get("SIE_CLUSTER_URL", config["cluster"]["url"])
    api_key = os.environ.get("SIE_API_KEY", config["cluster"]["api_key"])
    client = SIEClient(cluster_url, api_key=api_key)
    clients_index = sorted({m["client"] for m in metadata})
    yield
    client.close()


app = FastAPI(title="SIE Vision-First Document RAG", lifespan=lifespan)

root = Path(__file__).resolve().parent.parent
static_dir = root / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
app.mount("/pages", StaticFiles(directory=str(root / "data" / "pages")), name="pages")


@app.get("/")
def index():
    return FileResponse(str(static_dir / "index.html"))


@app.get("/api/clients")
def api_clients():
    return clients_index


@app.get("/api/stats")
def api_stats():
    return {
        "total_pages": len(metadata),
        "clients": clients_index,
        "models": config["models"],
        "visual_rerank": config["search"].get("visual_rerank", False),
        "answer": config["search"].get("answer", True),
    }


@app.get("/api/search")
def api_search(
    q: str = Query(..., min_length=1),
    client_name: str | None = Query(None, alias="client"),
):
    out = search(client, config, multivectors, metadata, q, client_name)
    return {
        "query": q,
        "client": client_name,
        "answer": out["answer"],
        "timings": out["timings"],
        "results": [
            {
                "page_id": r["page_id"],
                "client": r["client"],
                "title": r["title"],
                "space": r["space"],
                "author": r["author"],
                "web_url": r["web_url"],
                "page_image": f"/pages/{r['page_id']}.png",
                "ocr_snippet": r.get("ocr_snippet", ""),
                "scores": {
                    "maxsim": round(r["_maxsim_score"], 4),
                    "rerank": round(r["_rerank_score"], 4) if r.get("_rerank_score") is not None else None,
                },
            }
            for r in out["results"]
        ],
    }
