"""Tokenization utilities."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def load_tokenizer(
    model_path: str | Path,
    *,
    trust_remote_code: bool = False,
) -> PreTrainedTokenizer | PreTrainedTokenizerFast:
    """Load a HuggingFace tokenizer from a local path or model id."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        trust_remote_code=trust_remote_code,
    )
    logger.debug(
        "Loaded tokenizer from %s (fast=%s)",
        model_path,
        getattr(tokenizer, "is_fast", False),
    )
    return tokenizer
