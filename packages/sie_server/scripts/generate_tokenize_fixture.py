#!/usr/bin/env python3
"""Generate a byte-identity fixture for the Rust tokeniser parity test.

Companion to ``packages/sie_server_sidecar/tests/byte_identity.rs``.

The worker-sidecar can pre-tokenise text on behalf of the Python adapter
and ship the ids over IPC. Python falls back to its own tokenisation
whenever the content-hash of the two tokenizers disagrees, but that
guard is only useful if the two sides produce the SAME output for
the SAME ``tokenizer.json``.

This script is the ground-truth generator for that check:

  1. Downloads the target model's ``tokenizer.json`` via
     ``huggingface_hub`` (uses the standard HF cache).
  2. Loads it with ``transformers.AutoTokenizer`` using the exact
     kwargs our Python adapters use today:
     ``padding=False, truncation=True, max_length=N``.
  3. Runs a curated set of test texts (short, long, multilingual,
     unicode edge cases) through the tokenizer per-text.
  4. Emits a JSON fixture consumed by the Rust test.

Usage:

    uv run --with transformers --with huggingface_hub \\
        packages/sie_server/scripts/generate_tokenize_fixture.py \\
        --model Alibaba-NLP/gte-multilingual-base \\
        --max-seq-len 512 \\
        --out /tmp/gte_fixture.json

Then:

    SIE_BYTE_IDENTITY_FIXTURE=/tmp/gte_fixture.json \\
        cargo test --manifest-path packages/sie_server_sidecar/Cargo.toml --test byte_identity -- --nocapture

Intentionally self-contained: the only runtime deps are
``transformers`` + ``huggingface_hub``; no sie_server imports, so
the fixture can be regenerated on any machine that can reach HF Hub
(or has the model cached).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Curated text cases. Designed to exercise the corners where
# tokenizer implementations most often diverge:
#   * plain ASCII (trivial)
#   * leading/trailing whitespace (pre-tokenizer stripping)
#   * Unicode normalisation (combining chars, CJK)
#   * punctuation clusters
#   * very long text that WILL trigger truncation to max_seq_len
#   * empty string (edge case; some tokenizers emit just special tokens)
DEFAULT_CASES: list[str] = [
    "hello world",
    "Hello, World!",
    "  leading and trailing whitespace   ",
    "The quick brown fox jumps over the lazy dog.",
    "A. B. C.  Multiple.   Spaces.",
    # Multilingual — the main reason we use gte-multilingual-base
    "Bonjour le monde",
    "你好，世界",
    "こんにちは世界",
    "مرحبا بالعالم",
    "Привет мир",
    # Unicode combining / normalisation edge cases
    "café vs cafe\u0301",  # NFC vs NFD form of "café"
    "é\u200bé",  # zero-width space between graphemes
    # Long text — tests truncation behaviour
    " ".join(["lorem ipsum"] * 300),
    # Empty string
    "",
]


def resolve_tokenizer_path(model_id: str) -> str:
    """Return the absolute filesystem path to ``tokenizer.json`` for
    ``model_id``, downloading it to the HF cache if necessary.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        print(
            "huggingface_hub is required; install with "
            "`pip install huggingface_hub` (or use the `--with` uv invocation "
            "shown in the module docstring).",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
    return str(Path(path).resolve())


def load_tokenizer(model_id: str) -> Any:
    """Load the HF ``AutoTokenizer`` for ``model_id``. We deliberately
    call ``AutoTokenizer`` (not ``Tokenizer.from_file``) because that
    is what every in-tree adapter uses — if the two wrappers differ
    on any config, the fixture will catch it.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        print(
            "transformers is required; install with "
            "`pip install transformers` (or use the `--with` uv invocation "
            "shown in the module docstring).",
            file=sys.stderr,
        )
        raise SystemExit(2) from e

    return AutoTokenizer.from_pretrained(model_id, use_fast=True)


def build_case(tokenizer: Any, text: str, max_seq_len: int) -> dict[str, Any]:
    """Tokenise ``text`` with the Python-adapter-equivalent kwargs and
    return a JSON-serialisable case dict.
    """
    enc = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=max_seq_len,
        return_attention_mask=True,
        return_token_type_ids=None,  # let the tokenizer decide
    )
    # ``transformers`` returns BatchEncoding-wrapped lists when given
    # a single string. Convert to plain Python lists for JSON.
    input_ids = list(enc["input_ids"])
    attention_mask = list(enc["attention_mask"])
    # Some tokenizers (XLM-R, most modern multilingual models) don't
    # emit token_type_ids at all. In that case the key is absent; we
    # encode that as an empty list in the fixture and the Rust test
    # treats "empty" as "the Python side didn't emit segment ids".
    token_type_ids: list[int] = list(enc["token_type_ids"]) if "token_type_ids" in enc else []
    return {
        "text": text,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def main() -> None:
    description = (__doc__ or "").splitlines()[0] if __doc__ else ""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace repo id, e.g. 'Alibaba-NLP/gte-multilingual-base'.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=512,
        help="Max sequence length for truncation (must match the Rust side's max_seq_len for the same model).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the fixture JSON.",
    )
    parser.add_argument(
        "--texts",
        type=Path,
        default=None,
        help="Optional path to a newline-delimited text file of additional "
        "custom cases. If absent, a curated default set is used.",
    )
    args = parser.parse_args()

    texts: list[str] = list(DEFAULT_CASES)
    if args.texts is not None:
        extra = args.texts.read_text(encoding="utf-8").splitlines()
        texts.extend(t for t in extra if t)

    tokenizer_path = resolve_tokenizer_path(args.model)
    tokenizer = load_tokenizer(args.model)

    cases = [build_case(tokenizer, text, args.max_seq_len) for text in texts]

    fixture = {
        "model_id": args.model,
        "tokenizer_path": tokenizer_path,
        "max_seq_len": args.max_seq_len,
        "cases": cases,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {len(cases)} cases for {args.model} to {args.out} (tokenizer.json = {tokenizer_path})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
