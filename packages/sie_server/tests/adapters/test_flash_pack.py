"""Equivalence tests for the shared flash varlen packing helper (issue #1538).

``build_position_ids`` replaces per-adapter ``torch.arange`` loops. These tests
pin it to be bit-identical to those loops (zero-based and padding-offset) across
``int32``/``int64`` ``cu_seqlens`` and zero-length sequences.
"""

from __future__ import annotations

import torch
from sie_server.adapters._flash_pack import build_position_ids

# Includes a zero-length middle sequence ([0, 2, 2, 5]) and single-token seqs.
_CASES = [[0, 3, 7, 8], [0, 5], [0, 1, 4, 4, 9], [0, 2, 2, 5], [0, 1, 2, 3]]


def _loop(cu: list[int], *, offset: int) -> torch.Tensor:
    parts = []
    for i in range(len(cu) - 1):
        seq_len = cu[i + 1] - cu[i]
        parts.append(torch.arange(offset, offset + seq_len, dtype=torch.long))
    return torch.cat(parts)


def test_matches_zero_based_loop() -> None:
    for dtype in (torch.int32, torch.int64):
        for case in _CASES:
            cu = torch.tensor(case, dtype=dtype)
            assert torch.equal(build_position_ids(cu), _loop(case, offset=0))


def test_matches_padding_offset_loop() -> None:
    # XLM-RoBERTa style: positions start at padding_idx + 1 (here padding_idx=1).
    for dtype in (torch.int32, torch.int64):
        for case in _CASES:
            cu = torch.tensor(case, dtype=dtype)
            assert torch.equal(build_position_ids(cu, offset=2), _loop(case, offset=2))


def test_result_is_int64() -> None:
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    assert build_position_ids(cu).dtype == torch.int64


def test_single_token_sequences() -> None:
    cu = torch.tensor([0, 1, 2, 3])
    assert torch.equal(build_position_ids(cu), torch.tensor([0, 0, 0]))
