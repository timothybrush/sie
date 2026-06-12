"""Text preprocessors for tokenization and cost estimation.

This module contains preprocessors for text modality:
- TextPreprocessor: Full tokenization using HuggingFace tokenizers
- CharCountPreprocessor: Cost estimation for library-wrapped adapters
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sie_server.core.prepared import PreparedBatch, PreparedItem, TextPayload

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    from sie_server.config.model import ModelConfig
    from sie_server.ipc_types import PreparedTokens
    from sie_server.types.inputs import Item

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocessor for text tokenization.

    Wraps a HuggingFace tokenizer to produce TextPayload items.
    Thread-safe: tokenizers handle concurrent calls internally.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        model_name: str,
    ) -> None:
        """Initialize with a tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer instance.
            model_name: Model name for logging.
        """
        self._tokenizer = tokenizer
        self._model_name = model_name
        # Cached cross-language content hash of this tokenizer. Used
        # by `try_prepare_from_prepared_tokens` to decide whether
        # Rust-side pre-tokenised inputs can be trusted without
        # re-running Python tokenisation. Lazily computed — Rust-fast
        # models pay it once at adapter load; others never do.
        self._tokenizer_id: str | None = None

    @property
    def tokenizer_id(self) -> str | None:
        """Return the BLAKE3 content hash of this tokenizer, matching
        what the worker-sidecar emits in ``PreparedTokens.tokenizer_id``.

        Returns ``None`` when the tokenizer is not a HuggingFace fast
        tokenizer (slow tokenizers don't expose a canonical JSON
        serialisation and therefore cannot participate in the Rust
        fast-path).

        Hash is computed on first access and cached thereafter.
        """
        if self._tokenizer_id is not None:
            return self._tokenizer_id

        # Fast tokenizers expose `.backend_tokenizer` → the underlying
        # Rust `tokenizers::Tokenizer`. Slow tokenizers don't, and
        # will return None (which the caller interprets as "this
        # model cannot fast-path — always tokenise in Python").
        backend = getattr(self._tokenizer, "backend_tokenizer", None)
        if backend is None:
            return None

        try:
            import blake3

            canonical = backend.to_str(pretty=False).encode("utf-8")
            self._tokenizer_id = blake3.blake3(canonical).hexdigest()[:32]
        except Exception:  # noqa: BLE001
            logger.warning(
                "tokenizer_id: failed to compute hash for %s; fast-path disabled",
                self._model_name,
                exc_info=True,
            )
            return None

        return self._tokenizer_id

    def try_prepare_from_prepared_tokens(
        self,
        items: list[Item],
        prepared_tokens_per_item: list[PreparedTokens | None],
        *,
        config: ModelConfig,
    ) -> PreparedBatch[TextPayload] | None:
        """Fast-path builder: assemble a ``PreparedBatch`` directly from
        Rust-supplied ``PreparedTokens`` without re-tokenising.

        Returns ``None`` when the fast path is not viable for the
        batch — in which case the caller must call ``prepare()`` as
        usual.

        Per-item hybrid acceptance (v2 of this gate):

        * Items whose ``pt`` is present and passes all in-item checks
          (``tokenizer_id`` match, 1 sequence, correct shape, zero
          ``token_type_ids``, ``max_seq_len`` within model cap) build
          their ``TextPayload`` directly from the Rust bytes.
        * Items whose ``pt`` is ``None`` (Rust deliberately skipped
          them — empty text, multimodal fields, instruction/is_query
          subgroup, ...) are tokenised in Python via ``prepare()`` on
          the subset, then spliced back in the original order. Every
          item ends up with a valid ``TextPayload``; the adapter sees
          a homogeneous batch and does a single forward pass.

        Whole-batch fallback (return ``None``) is reserved for signals
        that correctness of *other* items in the batch is suspect:

        * Length mismatch between ``items`` and ``prepared_tokens_per_item``.
        * ``tokenizer_id`` disagrees with this preprocessor's hash —
          indicates a drift between what Rust loaded and what Python
          loaded (different revisions, different ``tokenizer_config``
          overrides, ...). Safer to re-tokenise everything under the
          Python policy.
        * A ``prepared_tokens`` whose sequence was truncated to a
          larger ``max_seq_len`` than this model's
          ``config.max_sequence_length`` — staying on the Python path
          lets the config-level truncation apply consistently.
        * Malformed wire shape (``len(input_ids) != 1`` or explicit
          ``attention_mask`` width does not match ``input_ids``).
        """
        if not items:
            return PreparedBatch(items=[], total_cost=0, modality="text")

        if len(prepared_tokens_per_item) != len(items):
            logger.debug(
                "rust-tokenize fast-path: prepared_tokens length mismatch (%d vs %d items); falling back",
                len(prepared_tokens_per_item),
                len(items),
            )
            return None

        my_id = self.tokenizer_id
        if my_id is None:
            return None

        model_max_seq = getattr(config, "max_sequence_length", None)

        # Classify each item into fast (Rust bytes usable) or slow
        # (Python tokeniser needed). Drift / malformed signals on any
        # item collapse the whole batch to slow via an early return.
        fast_payloads: dict[int, TextPayload] = {}
        slow_indices: list[int] = []
        for i, pt in enumerate(prepared_tokens_per_item):
            if pt is None:
                slow_indices.append(i)
                continue
            if pt.tokenizer_id != my_id:
                # Tokenizer drift on any item = batch-wide fallback.
                # Silent at debug to avoid flooding during rolling
                # deploys where Rust and Python legitimately disagree
                # for a few minutes.
                logger.debug(
                    "rust-tokenize fast-path: tokenizer_id mismatch for %s (python=%s, rust=%s); falling back",
                    self._model_name,
                    my_id,
                    pt.tokenizer_id,
                )
                return None
            if len(pt.input_ids) != 1:
                logger.debug(
                    "rust-tokenize fast-path: item %d has %d sequences, expected 1; falling back",
                    i,
                    len(pt.input_ids),
                )
                return None
            if model_max_seq is not None and pt.max_seq_len > model_max_seq:
                logger.debug(
                    "rust-tokenize fast-path: item %d truncated at %d > model cap %d; falling back",
                    i,
                    pt.max_seq_len,
                    model_max_seq,
                )
                return None
            # Non-zero segment ids are a per-item concern: the sidecar's
            # text payload can't ferry them, so this item falls
            # back to Python (which regenerates the correct segment
            # synthesis the adapter expects). Other items in the
            # batch can still use the fast path.
            if pt.token_type_ids and any(any(t != 0 for t in row) for row in pt.token_type_ids):
                logger.debug(
                    "rust-tokenize fast-path: non-zero token_type_ids on item %d; tokenising in Python",
                    i,
                )
                slow_indices.append(i)
                continue

            input_ids = list(pt.input_ids[0])
            # Rust elides `attention_mask` entirely when it would be
            # all-ones (the no-padding case); rebuild the default.
            if pt.attention_mask:
                mask_shape_mismatch = len(pt.attention_mask) != len(pt.input_ids) or len(pt.attention_mask[0]) != len(
                    input_ids
                )
                if mask_shape_mismatch:
                    logger.debug(
                        "rust-tokenize fast-path: item %d has malformed attention_mask shape; falling back",
                        i,
                    )
                    return None
                attention_mask = list(pt.attention_mask[0])
            else:
                attention_mask = [1] * len(input_ids)

            fast_payloads[i] = TextPayload(input_ids=input_ids, attention_mask=attention_mask)

        # If no item hit the fast path, let the caller's normal
        # ``prepare()`` run — spawning a second tokeniser call for
        # an all-slow batch would just be overhead.
        if not fast_payloads:
            return None

        # Hybrid path: tokenise the slow subset in Python and splice
        # back. A synchronous call is fine — we're already inside the
        # tokenisation phase of ``_prepare_batch``.
        slow_payload_by_original_index: dict[int, TextPayload] = {}
        if slow_indices:
            slow_items = [items[i] for i in slow_indices]
            try:
                slow_batch = self.prepare(slow_items, config=config)
            except Exception:  # noqa: BLE001
                # Bail to whole-batch fallback if the slow subset
                # blows up — the caller's ``prepare()`` will retry
                # with the full item list and a uniform error path.
                logger.warning(
                    "rust-tokenize fast-path: hybrid slow-subset tokenise failed for %s; falling back whole batch",
                    self._model_name,
                    exc_info=True,
                )
                return None
            for prep_item in slow_batch.items:
                original_index = slow_indices[prep_item.original_index]
                slow_payload_by_original_index[original_index] = prep_item.payload

        # Assemble in original ``items`` order so downstream
        # ``original_index`` invariants are preserved (used by the
        # encode pipeline to route outputs back to request items).
        prepared_items: list[PreparedItem[TextPayload]] = []
        total_cost = 0
        for i in range(len(items)):
            payload = fast_payloads.get(i) or slow_payload_by_original_index.get(i)
            if payload is None:
                # Defensive: every index must be covered by exactly
                # one of the two sources. If this branch ever fires
                # it means the classification logic above diverged
                # from the merge logic here.
                logger.warning(
                    "rust-tokenize fast-path: item %d has no payload after hybrid merge; falling back",
                    i,
                )
                return None
            cost = len(payload.input_ids)
            prepared_items.append(PreparedItem(payload=payload, cost=cost, original_index=i))
            total_cost += cost

        return PreparedBatch(items=prepared_items, total_cost=total_cost, modality="text")

    @property
    def modality(self) -> str:
        """Return 'text'."""
        return "text"

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[TextPayload]:
        """Tokenize text items.

        Args:
            items: Items with text field.
            config: Model config with max_sequence_length.
            is_query: Whether items are queries (True) or documents (False).
                Currently unused; reserved for query-specific tokenization
                such as ColBERT query expansion or instruction prefixes.
            instruction: Optional instruction (unused for text preprocessing).
            task: Optional task token (unused for text preprocessing).

        Returns:
            PreparedBatch with TextPayload items.
        """
        # Note: is_query is reserved for future query-specific tokenization.
        _ = is_query
        del instruction, task  # Unused - only needed for vision models

        # Extract texts
        texts = [item.text or "" for item in items]

        # Build tokenizer kwargs
        kwargs: dict[str, Any] = {
            "padding": False,  # Pad later when forming batches
            "truncation": True,
            "return_attention_mask": True,
        }
        if config.max_sequence_length:
            kwargs["max_length"] = config.max_sequence_length

        # Tokenize all at once (HuggingFace handles batches efficiently)
        encoded = self._tokenizer(texts, **kwargs)

        # Build prepared items
        prepared_items: list[PreparedItem[TextPayload]] = []
        total_cost = 0

        for i, (input_ids, attention_mask) in enumerate(
            zip(encoded["input_ids"], encoded["attention_mask"], strict=True)
        ):
            token_count = len(input_ids)
            payload = TextPayload(input_ids=input_ids, attention_mask=attention_mask)
            prepared_items.append(PreparedItem(payload=payload, cost=token_count, original_index=i))
            total_cost += token_count

        return PreparedBatch(
            items=prepared_items,
            total_cost=total_cost,
            modality="text",
        )

    def collate(
        self,
        prepared: list[PreparedItem[TextPayload]],
        *,
        device: str,
        pad_token_id: int = 0,
    ) -> dict[str, Any]:
        """Collate tokenized items into padded tensors.

        Args:
            prepared: List of prepared text items.
            device: Target device.
            pad_token_id: Token ID for padding.

        Returns:
            Dict with 'input_ids' and 'attention_mask' tensors.
        """
        import torch

        if not prepared:
            return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([])}

        # Find max length
        max_length = max(p.payload.token_count for p in prepared)

        input_ids_batch = []
        attention_mask_batch = []

        for p in prepared:
            payload = p.payload
            padding_length = max_length - payload.token_count

            padded_ids = payload.input_ids + [pad_token_id] * padding_length
            padded_mask = payload.attention_mask + [0] * padding_length

            input_ids_batch.append(padded_ids)
            attention_mask_batch.append(padded_mask)

        return {
            "input_ids": torch.tensor(input_ids_batch, device=device),
            "attention_mask": torch.tensor(attention_mask_batch, device=device),
        }


class CharCountPreprocessor:
    """Simple cost estimator for library-wrapped adapters.

    Uses character count as a cost proxy instead of actual tokenization.
    This avoids tokenization overhead for adapters that handle tokenization
    internally (e.g., BGE-M3, Qwen embeddings, rerankers).

    The cost multiplier converts characters to approximate token count:
    - English: ~4 chars/token on average
    - With a buffer for safety, we use ~3.5 chars/token (multiplier=0.3)

    Usage:
        # In adapter's get_preprocessor():
        return CharCountPreprocessor(model_name="my-model", chars_per_token=4.0)
    """

    is_trivial: bool = True

    def __init__(
        self,
        model_name: str,
        chars_per_token: float = 4.0,
    ) -> None:
        """Initialize with cost estimation parameters.

        Args:
            model_name: Model name for logging.
            chars_per_token: Average characters per token for cost estimation.
        """
        self._model_name = model_name
        self._chars_per_token = chars_per_token

    @property
    def modality(self) -> str:
        """Return 'text'."""
        return "text"

    def prepare(
        self,
        items: list[Item],
        *,
        config: ModelConfig,
        is_query: bool = False,
        instruction: str | None = None,
        task: str | None = None,
    ) -> PreparedBatch[TextPayload]:
        """Estimate cost from character count.

        Creates TextPayload items with None for tokenization data since
        the adapter handles tokenization internally.

        Args:
            items: Items with text field.
            config: Model config (unused).
            is_query: Whether items are queries (unused).
            instruction: Optional instruction (unused).
            task: Optional task token (unused).

        Returns:
            PreparedBatch with estimated token counts.
        """
        prepared_items: list[PreparedItem[TextPayload]] = []
        total_cost = 0

        for i, item in enumerate(items):
            text = self._get_text_safe(item)
            char_count = len(text)

            # Estimate token count from character count
            estimated_tokens = max(1, int(char_count / self._chars_per_token))
            total_cost += estimated_tokens

            # Create payload with empty tokenization data
            # The adapter will handle tokenization internally
            # The cost for batching is set in PreparedItem.cost, not here
            payload = TextPayload(
                input_ids=[],  # Empty - adapter tokenizes internally
                attention_mask=[],
            )

            prepared_items.append(
                PreparedItem(
                    cost=estimated_tokens,
                    original_index=i,
                    payload=payload,
                )
            )

        return PreparedBatch(items=prepared_items, total_cost=total_cost)

    def _get_text_safe(self, item: Item) -> str:
        raw = item.text
        if raw is None:
            text = ""
        elif not isinstance(raw, str):
            raise ValueError(f"text item must be a string, got: {raw}")
        else:
            text = raw
        return text

    def collate(
        self,
        prepared: list[PreparedItem[TextPayload]],
        *,
        device: str,
    ) -> dict[str, Any]:
        """Collate is a no-op for CharCountPreprocessor.

        The adapter handles tokenization and tensor creation internally.

        Args:
            prepared: List of prepared items (unused).
            device: Target device (unused).

        Returns:
            Empty dict - adapter handles its own input preparation.
        """
        return {}
