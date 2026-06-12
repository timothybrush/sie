"""Adapter call loop for the Rust-driven scheduler.

When a request lands on the worker-sidecar path, the worker-sidecar forms
the batch itself (``sie_server_sidecar::scheduler::Scheduler``) and
dispatches it over a single :data:`~sie_server.ipc_types.METHOD_RUN_BATCH`
RPC. This module is the Python receiver for that RPC.

The previous ``SIE_RUST_SCHEDULER_MODELS`` allowlist has been retired:
routing decisions now live in the gateway's model-config API and pool
selection. Any model that lands here is running through the worker-sidecar
scheduler path.

Design rationale — why this isn't just a method on ``QueueExecutor``:

* The existing ``process_encode_batch`` / ``process_score_batch`` /
  ``process_extract_batch`` methods on ``QueueExecutor`` are the
  *authoritative* Python path. They stay untouched so direct Python
  execution and the focused IPC compatibility tests behave
  exactly as today — zero diff in the hot path.
* ``adapter_call_loop`` is a **strict subset** of what those three
  methods do: it accepts an already-formed batch (no batching, no
  coalesce, no adaptive control — those live in Rust now) and fans
  each ``RunBatchItem`` into the same per-op handlers the executor
  already calls. All the adapter-side logic (``handlers.make_config_key``,
  ``EncodePipeline.run_encode``, ``ModelWorker.submit``, ``_group_by_inference_config``,
  ``_complete_requests``) is reused verbatim.
* Keeping the dispatch thin here lets the Rust side own the only
  behaviourally-novel piece (batch formation + FCFS across LoRAs + PI
  loop) without duplicating the Python handler vocabulary.

LoRA plumbing (single source of truth for the wire-boundary contract):

* The Rust scheduler hands us **one** ``RunBatchRequest.lora_key`` per
  batch — items inside the batch are guaranteed homogeneous by the
  upstream FCFS-across-LoRAs scheduler. We forward that key into each
  ``EncodeBatchItem.options["lora"]`` / ``ExtractBatchItem.options["lora"]``
  before fanning out to the per-op handler. From there the existing
  ``ModelWorker.submit`` path picks it up at lines like
  ``options.get("lora")`` (``model_worker.py``) and routes the items into
  the matching per-LoRA ``BatchFormer``, where ``_process_loop`` calls
  ``self._adapter.set_active_lora(active_lora)`` before the forward
  pass.
* Empty / missing ``lora_key`` → no injection → ``options.get("lora")``
  resolves to ``None`` → base-model batcher. The empty-string-to-``None``
  coercion is the wire-boundary contract; a future native adapter must
  apply the same normalisation.
* Score is base-only — Python's ``ModelWorker.submit_score`` documents
  this as "Score operations use base model batcher (no LoRA support
  for reranking)". When ``RunBatch`` arrives with ``op="score"`` and a
  non-empty ``lora_key`` we **log** and serve the base path rather
  than fail the batch.
* Pre-existing ``options["lora"]`` on individual items is **never**
  silently overwritten — if a caller has already set it (e.g. legacy
  client code that constructs ``EncodeBatchItem`` directly) and it
  conflicts with the batch-level ``lora_key``, we log and trust the
  per-item value. The Rust scheduler never produces this combination
  today, so the warning surfaces a real protocol bug rather than
  papering over one.

Fallback contract:

* Unknown ``op`` → typed error per item (``error_code =
  "run_batch_unknown_op"``). Forward-compat safety: a newer Rust
  build can ship an op this Python doesn't know, and the operator
  sees a clean message instead of a crash.
* Mixed-op batch → wholesale rejection with every outcome tagged
  ``run_batch_mixed_op``. The Rust scheduler MUST only form
  homogeneous-op batches; this is defence-in-depth, not an expected
  code path.
* Op tag / payload mismatch (e.g. ``op = "encode"`` but
  ``encode = None``) → per-item ``run_batch_invalid_item``. Keeps
  one bad item from sinking the whole batch.

Every error path produces a ``publish_error_and_ack`` disposition —
not a ``nak_retry`` — because the payload is ill-formed at the wire
level and retrying would deliver the same bad frame. Matches the
existing behaviour of the per-op handlers when ``msgspec.convert``
fails.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sie_server.ipc_types import (
    BatchOutcome,
    EncodeBatchItem,
    ExtractBatchItem,
    ItemOutcome,
    ProcessEncodeBatchRequest,
    ProcessExtractBatchRequest,
    ProcessScoreBatchRequest,
    RunBatchRequest,
)

if TYPE_CHECKING:
    from sie_server.queue_executor import QueueExecutor

logger = logging.getLogger(__name__)


# Error codes emitted by this module. Kept as module-level constants
# so callers (tests, log-greppers, dashboards) can import them without
# stringly-typed drift. Mirrors the pattern in
# ``queue_executor._RAW_OUTPUT_ERROR_CODE``.
RUN_BATCH_EMPTY = "run_batch_empty"
RUN_BATCH_MIXED_OP = "run_batch_mixed_op"
RUN_BATCH_UNKNOWN_OP = "run_batch_unknown_op"
RUN_BATCH_INVALID_ITEM = "run_batch_invalid_item"


def _merge_lora_into_options(
    existing: dict[str, Any] | None,
    lora_key: str,
    *,
    item_id: str,
    batch_id: int,
    op: str,
) -> dict[str, Any] | None:
    """Return an options dict with ``options["lora"] = lora_key`` set.

    Matches the legacy ``ModelWorker.submit`` contract where the
    batcher selection is sourced from ``options.get("lora")``. Returns
    ``existing`` unchanged when ``lora_key`` is empty (the base-model
    case — leaving ``options["lora"]`` unset rather than setting it to
    ``None`` keeps the two paths byte-identical).

    Pre-existing per-item ``options["lora"]`` is never silently
    overwritten: if it conflicts with the batch-level key we log and
    trust the per-item value. The Rust scheduler never produces this
    combination today (it partitions by lora_key before forming the
    batch), so any conflict is a real upstream bug worth a WARN.
    """
    if not lora_key:
        return existing

    if existing is None:
        return {"lora": lora_key}

    prior = existing.get("lora")
    if prior is not None and prior != lora_key:
        logger.warning(
            "run_batch (%s) item=%r batch_id=%d has options['lora']=%r but "
            "RunBatchRequest.lora_key=%r — keeping per-item value; the Rust "
            "scheduler should not produce mixed-LoRA batches",
            op,
            item_id,
            batch_id,
            prior,
            lora_key,
        )
        return existing

    if prior == lora_key:
        # Already set to the same value — no copy needed.
        return existing

    # Avoid mutating the caller's dict. The msgspec deserialisation
    # produces a fresh dict per call so this copy is cheap (a handful
    # of items per batch in practice) and keeps the function pure.
    merged = dict(existing)
    merged["lora"] = lora_key
    return merged


async def handle_run_batch(
    executor: QueueExecutor,
    req: RunBatchRequest,
) -> BatchOutcome:
    """Dispatch a Rust-formed batch into the existing per-op handlers.

    Contract:
        * ``req.items`` must be non-empty and homogeneous in ``op``.
        * ``req.model_id`` must be a model the executor can serve —
          the gateway is responsible for only routing models served
          by this worker image to the worker-sidecar; we don't
          re-validate here because a model reaching this RPC means
          the gateway has already decided we can serve it.

    Returns a :class:`BatchOutcome` matching the per-op RPCs' shape
    so the Rust publisher path is identical regardless of which RPC
    produced the result (``process_*_batch`` or ``RunBatch``).
    """
    if not req.items:
        logger.warning(
            "run_batch received empty items list (model=%s, batch_id=%d, lora=%r)",
            req.model_id,
            req.batch_id,
            req.lora_key,
        )
        # Empty batch is a protocol violation on the Rust side —
        # scheduler should never flush zero items. Return an empty
        # BatchOutcome; the Rust publisher treats this as a no-op.
        return BatchOutcome(outcomes=[])

    op = req.items[0].op
    if any(item.op != op for item in req.items):
        logger.error(
            "run_batch got mixed-op batch (model=%s, batch_id=%d): "
            "ops=%s — Rust scheduler MUST emit homogeneous-op batches",
            req.model_id,
            req.batch_id,
            sorted({i.op for i in req.items}),
        )
        return _reject_all(req, RUN_BATCH_MIXED_OP, "mixed-op run_batch is not supported")

    if op == "encode":
        return await _dispatch_encode(executor, req)
    if op == "score":
        return await _dispatch_score(executor, req)
    if op == "extract":
        return await _dispatch_extract(executor, req)

    logger.error(
        "run_batch got unknown op=%r (model=%s, batch_id=%d)",
        op,
        req.model_id,
        req.batch_id,
    )
    return _reject_all(req, RUN_BATCH_UNKNOWN_OP, f"unknown op {op!r}")


# --------------------------------------------------------------------
# Per-op dispatchers
# --------------------------------------------------------------------
#
# Each dispatcher validates that every RunBatchItem has its matching
# payload populated, then unwraps into the existing Process*Request
# type and calls the same method the per-op RPC already uses.
# Any validation miss on a single item becomes a typed error on that
# item only — the rest of the batch proceeds.


async def _dispatch_encode(
    executor: QueueExecutor,
    req: RunBatchRequest,
) -> BatchOutcome:
    items: list[EncodeBatchItem] = []
    invalid: list[ItemOutcome] = []
    for ri in req.items:
        if ri.encode is None:
            invalid.append(
                _invalid_item_outcome(
                    request_id=ri.request_id,
                    work_item_id=ri.work_item_id,
                    item_index=ri.item_index,
                    reason="op=encode but encode payload missing",
                )
            )
            continue
        # Plumb RunBatchRequest.lora_key into the per-item options so
        # ``ModelWorker.submit`` (line ~411 of model_worker.py:
        # ``options.get("lora")``) routes the item into the matching
        # per-LoRA batcher and ``_process_loop`` calls
        # ``set_active_lora(active_lora)`` before the forward pass.
        merged_options = _merge_lora_into_options(
            ri.encode.options,
            req.lora_key,
            item_id=ri.encode.work_item_id,
            batch_id=req.batch_id,
            op="encode",
        )
        if merged_options is ri.encode.options:
            items.append(ri.encode)
        else:
            # ``msgspec.Struct.replace``-equivalent: build a new struct
            # rather than mutating the deserialised one. We only enter
            # this branch when an injection actually happens, so the
            # base-LoRA hot path stays allocation-free.
            items.append(_with_options(ri.encode, merged_options))

    if not items:
        return BatchOutcome(outcomes=invalid)

    inner = ProcessEncodeBatchRequest(model_id=req.model_id, items=items)
    outcome = await executor.process_encode_batch(inner)
    return BatchOutcome(outcomes=[*invalid, *outcome.outcomes])


async def _dispatch_score(
    executor: QueueExecutor,
    req: RunBatchRequest,
) -> BatchOutcome:
    # Score is base-only: ``ModelWorker.submit_score`` always uses
    # ``self._batchers[None]``. We "downgrade to WARN, serve base" here
    # so a misrouted batch produces a visible log line rather than a
    # silent drop or a NAK that confuses retry logic.
    if req.lora_key:
        logger.warning(
            "run_batch op=score for model=%s arrived with lora_key=%r — "
            "score has no LoRA support (ModelWorker.submit_score), "
            "serving base weights for batch_id=%d",
            req.model_id,
            req.lora_key,
            req.batch_id,
        )

    items = []
    invalid: list[ItemOutcome] = []
    for ri in req.items:
        if ri.score is None:
            invalid.append(
                _invalid_item_outcome(
                    request_id=ri.request_id,
                    work_item_id=ri.work_item_id,
                    item_index=ri.item_index,
                    reason="op=score but score payload missing",
                )
            )
            continue
        items.append(ri.score)

    if not items:
        return BatchOutcome(outcomes=invalid)

    inner = ProcessScoreBatchRequest(model_id=req.model_id, items=items)
    outcome = await executor.process_score_batch(inner)
    return BatchOutcome(outcomes=[*invalid, *outcome.outcomes])


async def _dispatch_extract(
    executor: QueueExecutor,
    req: RunBatchRequest,
) -> BatchOutcome:
    items: list[ExtractBatchItem] = []
    invalid: list[ItemOutcome] = []
    for ri in req.items:
        if ri.extract is None:
            invalid.append(
                _invalid_item_outcome(
                    request_id=ri.request_id,
                    work_item_id=ri.work_item_id,
                    item_index=ri.item_index,
                    reason="op=extract but extract payload missing",
                )
            )
            continue
        # Same reasoning as ``_dispatch_encode``: plumb the batch-level
        # lora_key into ``options["lora"]`` so the legacy
        # ``ModelWorker.submit_extract`` path picks the right batcher
        # (model_worker.py line ~464).
        merged_options = _merge_lora_into_options(
            ri.extract.options,
            req.lora_key,
            item_id=ri.extract.work_item_id,
            batch_id=req.batch_id,
            op="extract",
        )
        if merged_options is ri.extract.options:
            items.append(ri.extract)
        else:
            items.append(_with_options(ri.extract, merged_options))

    if not items:
        return BatchOutcome(outcomes=invalid)

    inner = ProcessExtractBatchRequest(model_id=req.model_id, items=items)
    outcome = await executor.process_extract_batch(inner)
    return BatchOutcome(outcomes=[*invalid, *outcome.outcomes])


def _with_options[OptionsItemT: (EncodeBatchItem, ExtractBatchItem)](
    item: OptionsItemT,
    options: dict[str, Any] | None,
) -> OptionsItemT:
    """Return a copy of ``item`` with ``options`` replaced.

    Uses ``msgspec.structs.replace`` semantics — preserves every other
    field byte-for-byte. We avoid mutating the input struct because the
    deserialised request may be referenced by RPC framing/logging
    helpers downstream and silent mutation is a debugging hazard.

    Generic over the two item structs that actually carry an
    ``options`` field so callers preserve their narrowed type
    (``items.append(_with_options(ri.encode, ...))`` keeps
    ``list[EncodeBatchItem]``).
    """
    import msgspec.structs  # noqa: PLC0415  (local import keeps the cold path cold)

    return msgspec.structs.replace(item, options=options)


# --------------------------------------------------------------------
# Error helpers
# --------------------------------------------------------------------


def _reject_all(req: RunBatchRequest, error_code: str, error: str) -> BatchOutcome:
    """Produce a BatchOutcome that fails every item uniformly.

    Used for batch-level protocol errors (empty, mixed-op, unknown
    op). Every item's disposition is ``publish_error_and_ack``; the
    Rust publisher surfaces the error_code on the wire so the SDK
    sees a structured error rather than a timeout.
    """
    outcomes: list[ItemOutcome] = []
    for idx, ri in enumerate(req.items):
        work_item_id = ri.work_item_id
        request_id = ri.request_id
        item_index = getattr(ri, "item_index", idx)
        # Pull the id out of whichever payload is populated — we do not know
        # which, because the whole point of this path is the protocol error.
        # Prefer the wrapped payload when present so malformed frames still get
        # useful IDs.
        payload = ri.encode or ri.score or ri.extract
        if payload is not None:
            work_item_id = payload.work_item_id
            request_id = payload.request_id
            item_index = payload.item_index
        outcomes.append(
            ItemOutcome(
                work_item_id=work_item_id,
                request_id=request_id,
                item_index=item_index,
                disposition="publish_error_and_ack",
                error=error,
                error_code=error_code,
            )
        )
    return BatchOutcome(outcomes=outcomes)


def _invalid_item_outcome(
    *,
    request_id: str,
    work_item_id: str,
    item_index: int,
    reason: str,
) -> ItemOutcome:
    """Single-item error outcome for op-tag / payload mismatches."""
    return ItemOutcome(
        work_item_id=work_item_id,
        request_id=request_id,
        item_index=item_index,
        disposition="publish_error_and_ack",
        error=reason,
        error_code=RUN_BATCH_INVALID_ITEM,
    )
