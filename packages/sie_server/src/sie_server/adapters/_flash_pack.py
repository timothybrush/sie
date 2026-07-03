"""Shared variable-length packing helpers for flash-attention adapters.

Flash-attention encoders pack a batch of sequences into one flat token stream
and drive attention with cumulative sequence lengths (``cu_seqlens``). Each
packed sequence needs position ids restarting at the model's base offset.

Historically every ``*_flash`` adapter reimplemented this with a per-sequence
``torch.arange`` loop (with a ``.item()`` CUDA sync per sequence). This module
is the single deep implementation those adapters delegate to. See issue #1538.
"""

from __future__ import annotations

import torch


def build_position_ids(cu_seqlens: torch.Tensor, *, offset: int = 0) -> torch.Tensor:
    """Position ids for a varlen-packed batch, restarting per sequence.

    Given ``cu_seqlens`` (cumulative token counts, shape ``[num_seqs + 1]``),
    return a flat ``int64`` tensor where packed sequence ``i`` carries
    ``[offset, offset + 1, ..., offset + len_i - 1]``.

    Fully vectorized — no per-sequence ``.item()`` CUDA sync — and bit-identical
    to the per-sequence ``torch.arange`` loops it replaces. The result is placed
    on ``cu_seqlens.device``.

    Args:
        cu_seqlens: Cumulative sequence lengths, shape ``[num_seqs + 1]``.
        offset: Base position of each sequence. ``0`` for most encoders;
            ``padding_idx + 1`` for XLM-RoBERTa-style position ids.

    Returns:
        A 1-D ``int64`` tensor of length ``cu_seqlens[-1]`` on ``cu_seqlens.device``.
    """
    total_tokens = int(cu_seqlens[-1].item())
    positions = torch.arange(total_tokens, device=cu_seqlens.device)
    seq_starts = torch.repeat_interleave(cu_seqlens[:-1], cu_seqlens[1:] - cu_seqlens[:-1])
    positions = positions - seq_starts
    return positions + offset if offset else positions
