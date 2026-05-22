"""Visual document search + question answering, vision end-to-end.

Pipeline per query:
  1. encode(ColQwen2.5, text)          — query multivector
  2. sie_sdk.scoring.maxsim             — late interaction against page images
  3. score(Qwen3-VL-Reranker, query, images)   — optional, off by default
  4. extract(Florence-2-FT-DocVQA, instruction=query, images=[top page])
                                        — textual answer + citation
  5. extract(Florence-2-FT-DocVQA, images=[top page])
                                        — OCR snippet for the UI (display only,
                                          NOT in the ranking path)

The ranking is decided by a vision model looking at the page image, so charts,
screenshots, tables, and any other visual signal that OCR would erase still
contributes. OCR runs only on the chosen page, only to provide on-screen text
the user can read or copy.

Multi-tenant isolation is a Python filter on metadata before MaxSim, so a
query scoped to one client never sees another client's pages.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

from sie_sdk import SIEClient
from sie_sdk.scoring import maxsim
from sie_sdk.types import Item


def load_config():
    return yaml.safe_load((Path(__file__).resolve().parent.parent / "config.yaml").read_text())


def load_index():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    if not (data_dir / "multivectors.npz").exists():
        raise FileNotFoundError("data/multivectors.npz missing. Run `python python/ingest.py` first.")
    npz = np.load(data_dir / "multivectors.npz")
    metadata = json.loads((data_dir / "metadata.json").read_text())
    multivectors = {m["page_id"]: npz[m["page_id"]] for m in metadata}
    return multivectors, metadata


def _ocr_snippet(entities: list[dict], max_chars: int = 400) -> str:
    """Concatenate OCR text regions into a single readable snippet."""
    pieces = []
    for e in entities or []:
        text = (e.get("text") or "").replace("</s>", "").strip()
        if text:
            pieces.append(text)
    joined = " · ".join(pieces)
    if len(joined) > max_chars:
        return joined[: max_chars - 1] + "…"
    return joined


def _docvqa_answer(entities: list[dict]) -> str:
    """Pick the answer string out of a Florence-2 DocVQA response.

    Florence-2 returns the answer as an entity (often the single one when the
    `<DocVQA>` task token is dispatched). We take the first non-empty text.
    """
    for e in entities or []:
        text = (e.get("text") or "").replace("</s>", "").strip()
        if text:
            return text
    return ""


def search(
    client: SIEClient,
    config: dict,
    multivectors: dict[str, np.ndarray],
    metadata: list[dict],
    query: str,
    client_filter: str | None = None,
) -> dict:
    gpu = config["cluster"]["gpu"]
    timeout = config["cluster"]["provision_timeout_s"]
    top_k_candidates = config["search"]["top_k_candidates"]
    top_k_results = config["search"]["top_k_results"]
    do_visual_rerank = config["search"].get("visual_rerank", False)
    do_answer = config["search"].get("answer", True)
    do_ocr_snippet = config["search"].get("ocr_snippet", True)

    corpus = [m for m in metadata if not client_filter or m["client"] == client_filter]
    if not corpus:
        return {"results": [], "answer": None, "timings": {}}

    timings: dict[str, float] = {}
    pages_root = Path(__file__).resolve().parent.parent / "data"

    # 1. Encode query (text side of ColQwen2.5).
    t0 = time.time()
    q_result = client.encode(
        config["models"]["retriever"],
        Item(text=query),
        output_types=["multivector"],
        is_query=True,
        gpu=gpu,
        wait_for_capacity=True,
        provision_timeout_s=timeout,
    )
    timings["encode_query_s"] = round(time.time() - t0, 3)
    query_mv = q_result["multivector"].astype(np.float32)

    # 2. MaxSim against in-memory multivectors.
    doc_mvs = [multivectors[m["page_id"]] for m in corpus]
    t0 = time.time()
    maxsim_scores = maxsim(query_mv, doc_mvs)
    timings["maxsim_s"] = round(time.time() - t0, 3)

    order = np.argsort(maxsim_scores)[::-1][:top_k_candidates]
    candidates: list[dict] = []
    for idx in order:
        c = dict(corpus[idx])
        c["_maxsim_score"] = float(maxsim_scores[idx])
        c["_rerank_score"] = None
        candidates.append(c)

    # 3. Optional visual rerank. Image-in cross-encoder so OCR never enters the
    #    ranking path. Disabled by default — see config.yaml for the cluster
    #    bug we're waiting on.
    if do_visual_rerank and candidates:
        try:
            t0 = time.time()
            rerank_items = [
                Item(id=c["page_id"], images=[str(pages_root / c["image_path"])])
                for c in candidates
            ]
            rerank = client.score(
                config["models"]["reranker"],
                Item(text=query),
                rerank_items,
                gpu=gpu,
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            timings["visual_rerank_s"] = round(time.time() - t0, 3)
            rerank_by_id = {s["item_id"]: s for s in rerank["scores"]}
            for c in candidates:
                s = rerank_by_id.get(c["page_id"])
                c["_rerank_score"] = float(s["score"]) if s else 0.0
            candidates.sort(key=lambda c: c["_rerank_score"] or 0.0, reverse=True)
        except Exception as exc:
            # Cluster adapter bug fallback: keep MaxSim ordering, surface the
            # failure to the caller. See sie-internal#1026.
            timings["visual_rerank_error"] = type(exc).__name__

    results = candidates[:top_k_results]

    # 4. DocVQA answer from the top page image. instruction= goes in as the
    #    plain question; the adapter prepends Florence-2's `<DocVQA>` task
    #    token. See superlinked.com/docs/extract/vision.
    answer = None
    if do_answer and results:
        top = results[0]
        try:
            t0 = time.time()
            qa = client.extract(
                config["models"]["docvqa"],
                Item(images=[str(pages_root / top["image_path"])]),
                instruction=query,
                gpu=gpu,
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            timings["docvqa_s"] = round(time.time() - t0, 3)
            answer = _docvqa_answer(qa[0].get("entities", []) if qa else [])
        except Exception as exc:
            timings["docvqa_error"] = type(exc).__name__

    # 5. OCR snippet for display — only on the top result so users see the
    #    text on the page they're being shown. Never used as a ranking signal.
    if do_ocr_snippet and results:
        top = results[0]
        try:
            t0 = time.time()
            ocr = client.extract(
                config["models"]["docvqa"],   # same model, no `instruction` ⇒ OCR mode
                Item(images=[str(pages_root / top["image_path"])]),
                gpu=gpu,
                wait_for_capacity=True,
                provision_timeout_s=timeout,
            )
            timings["ocr_snippet_s"] = round(time.time() - t0, 3)
            top["ocr_snippet"] = _ocr_snippet(ocr[0].get("entities", []) if ocr else [])
        except Exception as exc:
            timings["ocr_snippet_error"] = type(exc).__name__

    return {"results": results, "answer": answer, "timings": timings}


def print_run(out: dict, query: str, client_filter: str | None):
    scope = client_filter or "all clients"
    print(f'\n  Query: "{query}"  ({scope})')
    print(f"  Timings: {out['timings']}")
    if out["answer"]:
        print(f"\n  Answer: {out['answer']}")
    if not out["results"]:
        print("  No results.")
        return
    for i, r in enumerate(out["results"], 1):
        rerank = r.get("_rerank_score")
        rerank_str = f"rerank={rerank:.4f}" if rerank is not None else "rerank=—"
        print(f"\n  {i}. [{r['client']}] {r['title']}")
        print(f"     {r['page_id']}  ·  {r['space']}  ·  {r['author']}")
        print(f"     maxsim={r['_maxsim_score']:.3f}  {rerank_str}")
        if r.get("ocr_snippet"):
            print(f"     OCR snippet: {r['ocr_snippet'][:200]}")
        print(f"     url: {r['web_url']}")


def main():
    config = load_config()
    multivectors, metadata = load_index()
    print(f"Loaded index: {len(metadata)} pages")

    cluster_url = os.environ.get("SIE_CLUSTER_URL", config["cluster"]["url"])
    api_key = os.environ.get("SIE_API_KEY", config["cluster"]["api_key"])

    demo = [
        ("how do I sign in to the VPN?", "acme-corp"),
        ("what is the parental leave policy?", "globex"),
        ("audit prep evidence and walkthroughs", "initech"),
        # No tenant filter: shows the query routes across tenants.
        ("expense reports and per diem", None),
    ]
    with SIEClient(cluster_url, api_key=api_key) as client:
        for query, tenant in demo:
            out = search(client, config, multivectors, metadata, query, tenant)
            print_run(out, query, tenant)


if __name__ == "__main__":
    main()
