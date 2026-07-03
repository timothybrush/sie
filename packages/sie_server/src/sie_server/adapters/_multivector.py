"""Shared multivector (late-interaction) MaxSim scoring for ColBERT-style adapters.

ColBERT / ColPali / ColQwen-style adapters score a query against documents with
MaxSim: for each query token take the max dot product over the document's
tokens, then sum over query tokens. Historically every multivector adapter
reimplemented this. This module is the single deep implementation. See #1539.

Token embeddings are assumed L2-normalized by the caller (the adapters normalize
at encode time), so the dot product is cosine similarity. Both functions place
no device of their own — the result follows ``query.device``.
"""

from __future__ import annotations

import torch


def maxsim_scores(query: torch.Tensor, docs: list[torch.Tensor]) -> list[float]:
    """MaxSim score of each document against ``query`` (pairwise).

    For each document, sum over query tokens of the max over document tokens of
    the dot product.

    Args:
        query: Query token embeddings, shape ``[num_query_tokens, dim]``.
        docs: Per-document token embeddings, each ``[num_doc_tokens, dim]``,
            on the same device/dtype as ``query``.

    Returns:
        One score per document, in input order.
    """
    scores: list[float] = []
    for doc in docs:
        if doc.shape[0] == 0:
            # No document tokens: MaxSim is -inf (consistent with the batched path).
            scores.append(float("-inf"))
            continue
        sim = torch.matmul(query, doc.T)
        scores.append(sim.max(dim=-1).values.sum().item())
    return scores


def maxsim_scores_batched(query: torch.Tensor, docs: list[torch.Tensor]) -> list[float]:
    """MaxSim scores via a single padded, masked batched matmul.

    Equivalent to :func:`maxsim_scores` but pads the documents to a uniform
    length and runs one batched matmul — faster when reranking many documents.
    Padded positions are masked with ``-inf`` so they never win the per-query
    token max.

    Args:
        query: Query token embeddings, shape ``[num_query_tokens, dim]``.
        docs: Per-document token embeddings, each ``[num_doc_tokens, dim]``,
            on the same device/dtype as ``query``.

    Returns:
        One score per document, in input order.
    """
    if not docs:
        return []
    device = query.device
    doc_lengths = [d.shape[0] for d in docs]
    max_doc_tokens = max(doc_lengths)
    if max_doc_tokens == 0:
        # Every document is empty (e.g. punctuation-only docs after special-token
        # stripping). A zero-length doc dimension would make sim.max(dim=-1) raise;
        # MaxSim over zero doc tokens is -inf for each, matching the per-doc mask.
        return [float("-inf")] * len(docs)
    dim = query.shape[1]

    docs_padded = torch.zeros((len(docs), max_doc_tokens, dim), dtype=query.dtype, device=device)
    for i, doc in enumerate(docs):
        docs_padded[i, : doc.shape[0]] = doc

    # Batched sim: [Q, D] @ [N, D, T] -> [N, Q, T]
    sim = torch.matmul(query, docs_padded.transpose(1, 2))

    # Mask padded positions so they cannot win the per-query-token max.
    lengths_t = torch.tensor(doc_lengths, device=device)
    mask = torch.arange(max_doc_tokens, device=device).unsqueeze(0) < lengths_t.unsqueeze(1)
    sim.masked_fill_(~mask.unsqueeze(1), float("-inf"))

    # MaxSim: max over doc tokens, sum over query tokens -> [N].
    return sim.max(dim=-1).values.sum(dim=-1).tolist()
