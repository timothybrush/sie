"""W3C Trace Context propagation through the worker streaming processor.

These tests pin the contract that
:meth:`sie_server.processors.streaming.StreamingProcessor.process`
extracts a parent context from the work-envelope ``traceparent`` /
``tracestate`` fields, opens a ``worker.streaming_processor`` span as
its child, and attaches the standard ``sie.*`` attributes.

The propagation contract is what makes the cross-process trace tree
join: the gateway publishes the envelope with its own span context
serialised into ``traceparent``, the worker reads those bytes back
into a parent ``Context``, and the worker span becomes a child of
the gateway span. Without this round-trip the worker's spans would
form a disconnected root tree and the gateway-and-worker latency
breakdown would be impossible to correlate.

The tests use an :class:`InMemorySpanExporter` (no network) so they
are deterministic and run as part of the unit suite.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import msgpack
import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from sie_server.adapter_call_loop import handle_run_batch
from sie_server.ipc_types import BatchOutcome, EncodeBatchItem, RunBatchItem, RunBatchRequest
from sie_server.processors.streaming import StreamingProcessor

# A well-known W3C traceparent value, taken verbatim from the W3C
# spec example. Using a fixed value (rather than `uuid.uuid4()`-ish
# randomness) lets us assert the exact trace_id parses out of the
# resulting worker span — proof that the worker's parent extract
# really used the wire bytes and didn't accidentally root the trace.
EXAMPLE_TRACE_ID_HEX = "0af7651916cd43dd8448eb211c80319c"
EXAMPLE_PARENT_SPAN_HEX = "b7ad6b7169203331"
EXAMPLE_TRACEPARENT = f"00-{EXAMPLE_TRACE_ID_HEX}-{EXAMPLE_PARENT_SPAN_HEX}-01"

# A second, distinct trace context used to exercise the run_batch fan-in
# path (two items from two different client requests in one batch).
EXAMPLE_TRACE_ID_HEX_2 = "1bf0762027de54ee9559fc322d91420d"
EXAMPLE_PARENT_SPAN_HEX_2 = "c8be7c8270314442"
EXAMPLE_TRACEPARENT_2 = f"00-{EXAMPLE_TRACE_ID_HEX_2}-{EXAMPLE_PARENT_SPAN_HEX_2}-01"


_MODULE_EXPORTER: InMemorySpanExporter | None = None


def _ensure_module_provider() -> InMemorySpanExporter:
    """Install the in-memory tracer provider once per process.

    OpenTelemetry's :func:`opentelemetry.trace.set_tracer_provider`
    refuses to override a previously-set provider (logs a warning
    and silently keeps the original). The worker codepath under
    test calls :func:`opentelemetry.trace.get_tracer`, which
    binds to whatever provider was first set — so we must install
    *our* provider before the first call, and reuse it across
    every test in this module. The exporter's span buffer is
    reset per-test via the fixture below.
    """
    global _MODULE_EXPORTER
    if _MODULE_EXPORTER is None:
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        otel_trace.set_tracer_provider(provider)
        _MODULE_EXPORTER = exporter
    return _MODULE_EXPORTER


@pytest.fixture
def in_memory_exporter() -> InMemorySpanExporter:
    """Return the module-scoped exporter with its span buffer cleared.

    Per-test isolation: clear the buffer so assertions see only the
    spans emitted by this test. The exporter itself is shared at
    module scope because the global tracer provider can only be
    set once per process.
    """
    exporter = _ensure_module_provider()
    exporter.clear()
    return exporter


def _make_envelope(
    *,
    traceparent: str | None,
    tracestate: str | None = None,
    request_id: str = "req-test-123",
    model_id: str = "test/model",
) -> dict[str, Any]:
    """Build a minimal generate work envelope.

    Only the fields the trace-context boundary touches need to be
    present: ``reply_subject`` so :meth:`process` doesn't short-
    circuit on the empty-reply guard, the trace fields under test,
    and the request metadata that lands on span attributes.
    """
    envelope: dict[str, Any] = {
        "work_item_id": f"{request_id}.0",
        "request_id": request_id,
        "item_index": 0,
        "total_items": 1,
        "operation": "generate",
        "model_id": model_id,
        "profile_id": "default",
        "pool_name": "default",
        "machine_profile": "cpu",
        "reply_subject": f"_INBOX.r1.{request_id}",
        "timestamp": 1.0,
        "router_id": "r1",
        # We intentionally do NOT populate `generate` here — the test
        # short-circuits inside `_process_inner` on the unloaded
        # model branch before any generation params are read.
    }
    if traceparent is not None:
        envelope["traceparent"] = traceparent
    if tracestate is not None:
        envelope["tracestate"] = tracestate
    return envelope


def _make_msg(envelope: dict[str, Any]) -> MagicMock:
    """Wrap an envelope in a fake JetStream message handle.

    The streaming processor reads `msg.data` (raw msgpack bytes)
    and awaits `msg.ack()` / `msg.nak()` on the various exit paths.
    We stub both as async no-ops so the test focuses on the
    propagation behaviour rather than the JetStream side effects.
    """
    msg = MagicMock()
    msg.data = msgpack.packb(envelope, use_bin_type=True)
    msg.ack = AsyncMock()
    msg.nak = AsyncMock()
    msg.in_progress = AsyncMock()
    return msg


def _make_processor() -> StreamingProcessor:
    """Build a `StreamingProcessor` with everything stubbed out.

    The full constructor needs an `nc` (NATS client) and a model
    registry — both of which we mock. The model registry returns
    "model not found" so `_process_inner` exits early on the
    `_ensure_loaded` KeyError path; that's all we need for the
    propagation contract.
    """
    nc = MagicMock()
    nc.publish = AsyncMock()
    registry = MagicMock()
    registry.get_config = MagicMock(side_effect=KeyError("model not registered"))
    return StreamingProcessor(nc=nc, registry=registry, worker_id="worker-test")


@pytest.mark.asyncio
async def test_worker_span_is_child_of_envelope_traceparent(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """The worker span's trace_id must match the inbound traceparent.

    This is the load-bearing assertion of the trace propagation wire — if
    the worker did not extract the parent context the span would
    root a *new* trace tree and the assertion fails. The
    `parent_span_id` check additionally proves it's a child of the
    *specific* span the gateway encoded (not just any span in the
    same trace).
    """
    processor = _make_processor()
    envelope = _make_envelope(traceparent=EXAMPLE_TRACEPARENT)
    msg = _make_msg(envelope)

    await processor.process(msg, "test/model")

    spans = in_memory_exporter.get_finished_spans()
    # We expect at least one span — the `worker.streaming_processor`
    # span — but adapter sub-spans may also have been emitted on
    # other paths. Find the one we own by name.
    matching = [s for s in spans if s.name == "worker.streaming_processor"]
    assert matching, f"worker.streaming_processor span not emitted; saw {[s.name for s in spans]}"
    worker_span = matching[0]

    # The trace_id is the propagation invariant: it must equal the
    # one we put on the envelope, regardless of what span_id the
    # worker generated for itself.
    assert format(worker_span.context.trace_id, "032x") == EXAMPLE_TRACE_ID_HEX, (
        "worker span did not inherit the gateway's trace_id; propagation extraction is broken"
    )
    # And the parent_span_id must equal the gateway's span_id, not
    # be absent (which would mean a root span) and not be the
    # worker's own span_id (which would mean a self-cycle).
    assert worker_span.parent is not None, "worker span has no parent — propagation produced a root"
    assert format(worker_span.parent.span_id, "016x") == EXAMPLE_PARENT_SPAN_HEX, (
        "worker span's parent_span_id does not match the gateway span_id from the envelope"
    )


@pytest.mark.asyncio
async def test_worker_span_attributes_include_sie_request_metadata(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Span carries the per-request labels operators dashboard on.

    `sie.request_id`, `sie.attempt_id`, `sie.model` are the
    correlation keys that join logs, metrics, and traces. They
    must be on the span so the trace itself is searchable without
    cross-referencing log lines.
    """
    processor = _make_processor()
    envelope = _make_envelope(
        traceparent=EXAMPLE_TRACEPARENT,
        request_id="req-attrs-456",
        model_id="BAAI/bge-m3-generate",
    )
    msg = _make_msg(envelope)

    await processor.process(msg, "BAAI/bge-m3-generate")

    spans = in_memory_exporter.get_finished_spans()
    matching = [s for s in spans if s.name == "worker.streaming_processor"]
    assert matching
    attrs = dict(matching[0].attributes or {})
    assert attrs.get("sie.request_id") == "req-attrs-456"
    assert attrs.get("sie.model") == "BAAI/bge-m3-generate"
    # attempt_id is generated per pickup; we only assert it's
    # present and a non-empty string (32-char hex).
    attempt_id = attrs.get("sie.attempt_id")
    assert isinstance(attempt_id, str), "sie.attempt_id missing or not a string"
    assert len(attempt_id) == 32, f"sie.attempt_id should be 32-char hex, got {len(attempt_id)}"


@pytest.mark.asyncio
async def test_worker_span_is_root_when_envelope_omits_traceparent(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Backward-compat: envelopes without `traceparent` still emit a span.

    Gateways without trace propagation do not populate the trace fields, and we must
    not crash or skip the worker span just because the envelope
    omits them. The worker emits a root span instead — the trace
    just won't link back to a gateway parent.
    """
    processor = _make_processor()
    envelope = _make_envelope(traceparent=None)
    msg = _make_msg(envelope)

    await processor.process(msg, "test/model")

    spans = in_memory_exporter.get_finished_spans()
    matching = [s for s in spans if s.name == "worker.streaming_processor"]
    assert matching, "worker.streaming_processor span must still emit without traceparent"
    # No parent — the worker becomes the root of its own trace.
    assert matching[0].parent is None


# ---------------------------------------------------------------------------
# Non-streaming worker loop (``handle_run_batch``) trace propagation.
#
# Mirrors the streaming assertions above: the gateway serialises its span
# into each work item's ``traceparent`` / ``tracestate``, the Rust
# scheduler copies those onto the wire ``RunBatchItem`` s, and
# ``handle_run_batch`` re-extracts them so ``worker.run_batch`` attaches to
# the gateway span instead of rooting a fresh trace.
# ---------------------------------------------------------------------------


def _make_encode_run_batch_item(
    *,
    traceparent: str | None,
    tracestate: str | None = None,
    work_item_id: str = "r.0",
    request_id: str = "r",
    item_index: int = 0,
) -> RunBatchItem:
    """Build an encode ``RunBatchItem`` with a populated encode payload.

    The encode payload must be present (``_dispatch_encode`` reads
    ``ri.encode``) so dispatch reaches the stubbed executor instead of
    rejecting the item as invalid.
    """
    return RunBatchItem(
        op="encode",
        work_item_id=work_item_id,
        request_id=request_id,
        item_index=item_index,
        encode=EncodeBatchItem(
            work_item_id=work_item_id,
            request_id=request_id,
            item_index=item_index,
            total_items=1,
            timestamp=0.0,
            item={"text": "hello"},
        ),
        traceparent=traceparent,
        tracestate=tracestate,
    )


def _make_run_batch_executor() -> MagicMock:
    """A ``QueueExecutor`` stub whose per-op dispatch is a trivial no-op.

    ``process_encode_batch`` returns an empty :class:`BatchOutcome` so no
    model is loaded — the test only exercises the trace-context boundary
    in ``handle_run_batch``, not the encode pipeline.
    """
    executor = MagicMock()
    executor.process_encode_batch = AsyncMock(return_value=BatchOutcome(outcomes=[]))
    return executor


def _run_batch_span(in_memory_exporter: InMemorySpanExporter) -> Any:
    spans = in_memory_exporter.get_finished_spans()
    matching = [s for s in spans if s.name == "worker.run_batch"]
    assert matching, f"worker.run_batch span not emitted; saw {[s.name for s in spans]}"
    return matching[0]


@pytest.mark.asyncio
async def test_run_batch_span_is_child_of_single_context(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """A single-context batch's span attaches to the gateway span.

    This is the acceptance criterion: a ``/v1/encode`` request carrying a
    ``traceparent`` produces a ``worker.run_batch`` span whose ``trace_id``
    equals the gateway trace_id and whose parent is the gateway span — not
    a fresh root.
    """
    executor = _make_run_batch_executor()
    req = RunBatchRequest(
        model_id="test/model",
        batch_id=7,
        lora_key="",
        total_cost=1,
        items=[_make_encode_run_batch_item(traceparent=EXAMPLE_TRACEPARENT)],
    )

    await handle_run_batch(executor, req)

    span = _run_batch_span(in_memory_exporter)
    assert format(span.context.trace_id, "032x") == EXAMPLE_TRACE_ID_HEX, (
        "run_batch span did not inherit the gateway's trace_id; extraction is broken"
    )
    assert span.parent is not None, "run_batch span has no parent — propagation produced a root"
    assert format(span.parent.span_id, "016x") == EXAMPLE_PARENT_SPAN_HEX, (
        "run_batch span's parent_span_id does not match the gateway span_id from the wire item"
    )
    assert not span.links, "single-context batch must not emit fan-in links"


@pytest.mark.asyncio
async def test_run_batch_span_attributes_include_sie_batch_metadata(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """Span carries the batch metadata attributes operators build dashboards on."""
    executor = _make_run_batch_executor()
    req = RunBatchRequest(
        model_id="test/model",
        batch_id=7,
        lora_key="",
        total_cost=1,
        items=[_make_encode_run_batch_item(traceparent=EXAMPLE_TRACEPARENT)],
    )

    await handle_run_batch(executor, req)

    span = _run_batch_span(in_memory_exporter)
    attrs = dict(span.attributes or {})
    assert attrs.get("sie.model") == "test/model"
    assert attrs.get("sie.batch_id") == 7
    assert attrs.get("sie.op") == "encode"
    assert attrs.get("sie.adapter") == "run_batch"
    assert attrs.get("sie.batch_size") == 1


@pytest.mark.asyncio
async def test_run_batch_span_fan_in_links_distinct_contexts(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """A fan-in batch parents to the first context and links the rest.

    Two items carrying two DISTINCT traceparents land in one batch: the
    span is a child of the first and carries exactly one OTel link to the
    second so the fan-in is visible in the trace graph.
    """
    executor = _make_run_batch_executor()
    req = RunBatchRequest(
        model_id="test/model",
        batch_id=11,
        lora_key="",
        total_cost=2,
        items=[
            _make_encode_run_batch_item(
                traceparent=EXAMPLE_TRACEPARENT, work_item_id="a.0", request_id="a", item_index=0
            ),
            _make_encode_run_batch_item(
                traceparent=EXAMPLE_TRACEPARENT_2, work_item_id="b.0", request_id="b", item_index=1
            ),
        ],
    )

    await handle_run_batch(executor, req)

    span = _run_batch_span(in_memory_exporter)
    # Child of the FIRST distinct context.
    assert format(span.context.trace_id, "032x") == EXAMPLE_TRACE_ID_HEX
    assert span.parent is not None
    assert format(span.parent.span_id, "016x") == EXAMPLE_PARENT_SPAN_HEX
    # Exactly one link, pointing at the SECOND distinct context.
    assert len(span.links) == 1, f"expected one fan-in link, got {len(span.links)}"
    link_ctx = span.links[0].context
    assert format(link_ctx.trace_id, "032x") == EXAMPLE_TRACE_ID_HEX_2
    assert format(link_ctx.span_id, "016x") == EXAMPLE_PARENT_SPAN_HEX_2


@pytest.mark.asyncio
async def test_run_batch_span_is_root_without_traceparent(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """A batch with no trace context still emits a (root) span, no crash."""
    executor = _make_run_batch_executor()
    req = RunBatchRequest(
        model_id="test/model",
        batch_id=3,
        lora_key="",
        total_cost=1,
        items=[_make_encode_run_batch_item(traceparent=None)],
    )

    await handle_run_batch(executor, req)

    span = _run_batch_span(in_memory_exporter)
    assert span.parent is None, "run_batch span should be a root when no traceparent is present"
    assert not span.links


@pytest.mark.asyncio
async def test_run_batch_span_is_root_when_traceparent_malformed(
    in_memory_exporter: InMemorySpanExporter,
) -> None:
    """A malformed traceparent opens a root span and no fan-in links."""
    executor = _make_run_batch_executor()
    req = RunBatchRequest(
        model_id="test/model",
        batch_id=13,
        lora_key="",
        total_cost=1,
        items=[_make_encode_run_batch_item(traceparent="not-a-valid-traceparent")],
    )

    await handle_run_batch(executor, req)

    span = _run_batch_span(in_memory_exporter)
    assert span.parent is None, "run_batch span should be a root when traceparent is malformed"
    assert not span.links
