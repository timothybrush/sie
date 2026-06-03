"""Fixture tests for the RunBatch IPC contract.

These tests load shared fixtures from ``<repo-root>/tests/parity/`` and
assert that the Python adapter call loop
(``sie_server.adapter_call_loop.handle_run_batch``) produces the
canonical outcome shape declared in each fixture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from sie_server.adapter_call_loop import handle_run_batch
from sie_server.ipc_types import (
    BatchOutcome,
    EncodeBatchItem,
    ExtractBatchItem,
    ItemOutcome,
    RunBatchItem,
    RunBatchRequest,
    ScoreBatchItem,
)

# Resolve the shared fixtures directory once. Walks up from this test
# file rather than hard-coding a path so the test still works inside
# the workspace's various build / install layouts (uv venv vs.
# editable install vs. CI tarball).
PARITY_FIXTURES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "parity"

# Explicit fixture list (rather than glob) so the test reports each
# fixture as a separate pytest case with a stable id.
PARITY_FIXTURES: tuple[str, ...] = (
    "run_batch_empty.json",
    "run_batch_encode_no_lora.json",
    "run_batch_encode_lora.json",
    "run_batch_extract_lora.json",
    "run_batch_mixed_op.json",
    "run_batch_score_basic.json",
    "run_batch_score_lora_warns.json",
    "run_batch_unknown_op.json",
)


@dataclass(frozen=True)
class CanonicalOutcome:
    """Subset of ItemOutcome fields that must be parity-stable.

    Backend-specific fields (timing, raw inference bytes) are NOT in
    this struct; the fixture ``notes.elided_fields`` lists them
    explicitly and the comparator below ignores them.
    """

    work_item_id: str
    request_id: str
    item_index: int
    disposition: str
    error_code: str | None

    @classmethod
    def from_item_outcome(cls, oc: ItemOutcome) -> CanonicalOutcome:
        return cls(
            work_item_id=oc.work_item_id,
            request_id=oc.request_id,
            item_index=oc.item_index,
            disposition=oc.disposition,
            error_code=oc.error_code,
        )

    @classmethod
    def from_fixture_dict(cls, d: dict[str, Any]) -> CanonicalOutcome:
        return cls(
            work_item_id=d["work_item_id"],
            request_id=d["request_id"],
            item_index=d["item_index"],
            disposition=d["disposition"],
            error_code=d.get("error_code"),
        )


def _build_encode(d: dict[str, Any]) -> EncodeBatchItem:
    return EncodeBatchItem(
        work_item_id=d["work_item_id"],
        request_id=d["request_id"],
        item_index=d["item_index"],
        total_items=d["total_items"],
        timestamp=d["timestamp"],
        item=d["item"],
        options=d.get("options"),
    )


def _build_score(d: dict[str, Any]) -> ScoreBatchItem:
    return ScoreBatchItem(
        work_item_id=d["work_item_id"],
        request_id=d["request_id"],
        item_index=d["item_index"],
        total_items=d["total_items"],
        timestamp=d["timestamp"],
        query_item=d["query_item"],
        score_items=d["score_items"],
        options=d.get("options"),
    )


def _build_extract(d: dict[str, Any]) -> ExtractBatchItem:
    return ExtractBatchItem(
        work_item_id=d["work_item_id"],
        request_id=d["request_id"],
        item_index=d["item_index"],
        total_items=d["total_items"],
        timestamp=d["timestamp"],
        item=d["item"],
        labels=d.get("labels"),
        output_schema=d.get("output_schema"),
        options=d.get("options"),
    )


def _build_request(input_spec: dict[str, Any]) -> RunBatchRequest:
    """Materialise an in-memory RunBatchRequest from a fixture's ``input``.

    The fixture uses the same wire shape that ``msgspec.convert`` would
    accept off the UDS, but the Python tests construct the typed
    structs directly to keep msgspec out of the parity equation (we
    want this test to fail on real divergence, not on
    msgspec-vs-fixture-format quirks).
    """
    items: list[RunBatchItem] = []
    for raw in input_spec["items"]:
        encode = _build_encode(raw["encode"]) if raw.get("encode") is not None else None
        score = _build_score(raw["score"]) if raw.get("score") is not None else None
        extract = _build_extract(raw["extract"]) if raw.get("extract") is not None else None
        items.append(
            RunBatchItem(
                op=raw["op"],
                work_item_id=raw.get("work_item_id", ""),
                request_id=raw.get("request_id", ""),
                item_index=raw.get("item_index", 0),
                encode=encode,
                score=score,
                extract=extract,
            )
        )
    return RunBatchRequest(
        model_id=input_spec["model_id"],
        batch_id=input_spec["batch_id"],
        lora_key=input_spec["lora_key"],
        total_cost=input_spec["total_cost"],
        items=items,
    )


def _assert_lora_invariant(
    op: str,
    items: list[Any],
    expected_lora_in_options: str | None,
) -> None:
    """Per-item assertion for the lora-key plumbing contract.

    * ``expected_lora_in_options is not None`` and op in {encode,
      extract} → every item must arrive with options['lora'] == expected.
    * ``expected_lora_in_options is None`` → no item may have options['lora']
      set at all (the base fast path is required to be allocation-free
      and indistinguishable from a non-LoRA deploy).
    * ``op == "score"`` → score is base-only on both backends; a
      non-empty wire-level lora_key is dropped with a WARN, so even
      when ``expected_lora_in_options`` is provided we expect items
      to arrive WITHOUT options['lora'].
    """
    if op == "score":
        # Score never propagates lora_key into options regardless of
        # the wire-level value. This pins the WARN-and-serve-base
        # contract.
        for it in items:
            if it.options is not None:
                assert "lora" not in it.options, (
                    f"parity violation: score item {it.work_item_id!r} arrived "
                    f"with options['lora']={it.options.get('lora')!r}; "
                    f"score is base-only — the dispatcher must NOT inject "
                    f"options['lora'] on the score path."
                )
        return

    if expected_lora_in_options is None:
        for it in items:
            if it.options is not None:
                assert "lora" not in it.options, (
                    f"parity violation: {op} item {it.work_item_id!r} arrived "
                    f"with options['lora']={it.options.get('lora')!r} but the "
                    f"fixture's expected_lora_in_options is null. Empty "
                    f"lora_key MUST resolve to no injection (base fast path)."
                )
        return

    for it in items:
        assert it.options is not None, (
            f"parity violation: {op} item {it.work_item_id!r} arrived without "
            f"options even though RunBatchRequest.lora_key was non-empty. The "
            f"_dispatch_{op} path must inject lora_key into options['lora']."
        )
        assert it.options.get("lora") == expected_lora_in_options, (
            f"parity violation on {op} item {it.work_item_id!r}: expected "
            f"options['lora']={expected_lora_in_options!r}, got "
            f"{it.options.get('lora')!r}."
        )


def _make_echo_executor(
    expected_lora_in_options: str | None,
) -> AsyncMock:
    """Mock executor whose ``process_*_batch`` echoes its inputs.

    Echoing per-item identifiers (rather than emitting opaque
    ``ok.0`` / ``ok.1`` placeholders the way the Rust ``MockAdapter``
    historically did) lets the parity comparator check that
    work_item_id / request_id / item_index round-trip through the
    dispatcher unchanged. That is part of the wire contract: the
    publisher uses these to correlate outcomes back to NATS work
    items.

    Asserts the per-op LoRA-injection invariant via
    :func:`_assert_lora_invariant` for whichever ``process_*_batch``
    method gets called.
    """

    def _echo(op: str) -> AsyncMock:
        async def _impl(req: Any) -> BatchOutcome:
            _assert_lora_invariant(op, req.items, expected_lora_in_options)
            return BatchOutcome(
                outcomes=[
                    ItemOutcome(
                        work_item_id=it.work_item_id,
                        request_id=it.request_id,
                        item_index=it.item_index,
                        disposition="publish_and_ack",
                        result_msgpack=b"\x80",
                    )
                    for it in req.items
                ]
            )

        return AsyncMock(side_effect=_impl)

    exe = AsyncMock()
    exe.process_encode_batch = _echo("encode")
    exe.process_score_batch = _echo("score")
    exe.process_extract_batch = _echo("extract")
    return exe


# --------------------------------------------------------------------
# Fixture-driven parity test
# --------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_name", PARITY_FIXTURES, ids=lambda n: n[:-5])
async def test_parity_run_batch(fixture_name: str) -> None:
    fixture_path = PARITY_FIXTURES_DIR / fixture_name
    assert fixture_path.exists(), (
        f"parity fixture missing: {fixture_path}; keep PARITY_FIXTURES in sync with tests/parity/."
    )
    fixture = json.loads(fixture_path.read_text())

    request = _build_request(fixture["input"])
    expected_lora = fixture.get("expected_lora_in_options")
    executor = _make_echo_executor(expected_lora)

    actual = await handle_run_batch(executor, request)

    expected_canonical = [CanonicalOutcome.from_fixture_dict(d) for d in fixture["expected_canonical_outcomes"]]
    actual_canonical = [CanonicalOutcome.from_item_outcome(oc) for oc in actual.outcomes]

    # Pinpoint diffs by element-by-element comparison rather than a
    # single list equality so the failure message names the specific
    # outcome index that drifted. (List equality lights up the whole
    # list in pytest's diff which is harder to read.)
    assert len(actual_canonical) == len(expected_canonical), (
        f"parity violation on {fixture_name}: outcome count drift "
        f"(expected {len(expected_canonical)}, got {len(actual_canonical)}); "
        f"actual outcomes: {actual_canonical}"
    )
    for i, (got, want) in enumerate(zip(actual_canonical, expected_canonical, strict=True)):
        assert got == want, (
            f"parity violation in {fixture_name} at outcome[{i}]:\n  expected: {want}\n  got:      {got}"
        )

    # Sanity-check that the right per-op handler was invoked (and not
    # short-circuited by an empty-batch / mixed-op early-return).
    # Empty / rejected batches don't reach any handler.
    if not request.items:
        executor.process_encode_batch.assert_not_awaited()
        executor.process_score_batch.assert_not_awaited()
        executor.process_extract_batch.assert_not_awaited()
        return

    ops = {it.op for it in request.items}
    if len(ops) > 1:  # mixed-op batch — wholesale rejected, no handler call
        executor.process_encode_batch.assert_not_awaited()
        executor.process_score_batch.assert_not_awaited()
        executor.process_extract_batch.assert_not_awaited()
        return

    op = next(iter(ops))
    if op == "encode":
        # Some encode fixtures (e.g. all-invalid encode payloads) might
        # short-circuit on the "no valid items" branch; gate the
        # assertion on whether any encode item actually has a payload.
        if any(it.encode is not None for it in request.items):
            executor.process_encode_batch.assert_awaited_once()
        executor.process_score_batch.assert_not_awaited()
        executor.process_extract_batch.assert_not_awaited()
    elif op == "score":
        if any(it.score is not None for it in request.items):
            executor.process_score_batch.assert_awaited_once()
        executor.process_encode_batch.assert_not_awaited()
        executor.process_extract_batch.assert_not_awaited()
    elif op == "extract":
        if any(it.extract is not None for it in request.items):
            executor.process_extract_batch.assert_awaited_once()
        executor.process_encode_batch.assert_not_awaited()
        executor.process_score_batch.assert_not_awaited()
    else:
        # Unknown op → wholesale rejected, no handler should fire.
        executor.process_encode_batch.assert_not_awaited()
        executor.process_score_batch.assert_not_awaited()
        executor.process_extract_batch.assert_not_awaited()
