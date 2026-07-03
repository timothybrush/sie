"""Tests for the shared multivector MaxSim scoring helpers (issue #1539).

Pins ``maxsim_scores`` (pairwise) and ``maxsim_scores_batched`` (padded) to a
naive reference and to each other, and checks that padding never inflates a
short document's score.
"""

from __future__ import annotations

import torch
from sie_server.adapters._multivector import maxsim_scores, maxsim_scores_batched


def _naive(query: torch.Tensor, docs: list[torch.Tensor]) -> list[float]:
    return [sum(max(float(q @ d_tok) for d_tok in doc) for q in query) for doc in docs]


def _fixture() -> tuple[torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(0)
    query = torch.randn(5, 16)
    docs = [torch.randn(n, 16) for n in (3, 7, 1, 4)]
    return query, docs


def test_pairwise_matches_naive() -> None:
    query, docs = _fixture()
    got, exp = maxsim_scores(query, docs), _naive(query, docs)
    assert all(abs(a - b) < 1e-4 for a, b in zip(got, exp, strict=True))


def test_batched_matches_pairwise() -> None:
    query, docs = _fixture()
    pair, batched = maxsim_scores(query, docs), maxsim_scores_batched(query, docs)
    assert all(abs(a - b) < 1e-4 for a, b in zip(pair, batched, strict=True))


def test_batched_padding_does_not_inflate_short_doc() -> None:
    # A 1-token doc batched with a 6-token doc must score identically to solo.
    torch.manual_seed(1)
    query = torch.randn(3, 8)
    docs = [torch.randn(1, 8), torch.randn(6, 8)]
    solo = maxsim_scores_batched(query, [docs[0]])[0]
    in_batch = maxsim_scores_batched(query, docs)[0]
    assert abs(solo - in_batch) < 1e-5


def test_single_document() -> None:
    query, docs = _fixture()
    exp = _naive(query, docs[:1])
    assert len(exp) == 1
    assert abs(maxsim_scores(query, docs[:1])[0] - exp[0]) < 1e-4
    assert abs(maxsim_scores_batched(query, docs[:1])[0] - exp[0]) < 1e-4


def test_empty_docs_list_returns_empty() -> None:
    query = torch.randn(3, 8)
    assert maxsim_scores(query, []) == []
    assert maxsim_scores_batched(query, []) == []


def test_all_empty_docs_return_neg_inf() -> None:
    # Punctuation-only docs can reach the helper with zero tokens after the
    # caller strips special tokens; this must not raise.
    query = torch.randn(3, 8)
    empties = [torch.zeros(0, 8), torch.zeros(0, 8)]
    assert maxsim_scores(query, empties) == [float("-inf"), float("-inf")]
    assert maxsim_scores_batched(query, empties) == [float("-inf"), float("-inf")]


def test_mixed_empty_and_nonempty_doc() -> None:
    torch.manual_seed(2)
    query = torch.randn(3, 8)
    docs = [torch.zeros(0, 8), torch.randn(5, 8)]
    pair, batched = maxsim_scores(query, docs), maxsim_scores_batched(query, docs)
    assert pair[0] == float("-inf")
    assert batched[0] == float("-inf")
    assert abs(pair[1] - batched[1]) < 1e-4
