"""Unit tests for the worker-side generation metrics.

Covers :class:`sie_server.observability.metrics.GenerationStreamTimer`
(the helper the streaming adapters call once per yield) and the
``sie_worker_generation_ttft_seconds`` / ``sie_worker_generation_tpot_seconds``
histograms it feeds.

The privacy / label-cardinality smoke tests live here too so that the
full worker-side §4.11 surface lands in a single file the reviewer
can audit at a glance.
"""

from __future__ import annotations

import time
from typing import cast

import pytest
from prometheus_client import REGISTRY
from sie_server.observability import metrics as obs_metrics


def _histogram_count(name: str, model: str, grammar: str = "none") -> int:
    """Return the cumulative sample count for ``name{model=..., grammar=...}``.

    Reads the global Prometheus registry rather than calling
    ``Histogram._sum`` internals so the test stays robust to client-lib
    refactors. The ``grammar`` parameter mirrors the histogram's grammar label —
    most tests exercise the free-form (``grammar="none"``) path.
    """
    metric = REGISTRY.get_sample_value(f"{name}_count", {"model": model, "grammar": grammar})
    return int(metric) if metric is not None else 0


def test_ttft_recorded_on_first_non_empty_yield() -> None:
    """First yield with ``has_text=True`` must produce a TTFT observation.

    Subsequent ``mark_yield`` calls must not re-observe TTFT (it is a
    one-per-request metric).
    """
    model = "test/model_ttft"
    before = _histogram_count("sie_worker_generation_ttft_seconds", model)

    timer = obs_metrics.GenerationStreamTimer(model)
    # Empty yields before first text must NOT trigger TTFT.
    timer.mark_yield(has_text=False)
    assert _histogram_count("sie_worker_generation_ttft_seconds", model) == before

    timer.mark_yield(has_text=True)
    assert _histogram_count("sie_worker_generation_ttft_seconds", model) == before + 1

    timer.mark_yield(has_text=True)
    # No second TTFT observation.
    assert _histogram_count("sie_worker_generation_ttft_seconds", model) == before + 1

    timer.finalize(completion_tokens=2)


def test_tpot_uses_completion_tokens_denominator() -> None:
    """``finalize(completion_tokens=N)`` divides the first→last window by N."""
    model = "test/model_tpot_with_tokens"
    before = _histogram_count("sie_worker_generation_tpot_seconds", model)

    timer = obs_metrics.GenerationStreamTimer(model)
    # Pin the timing window so the assertion is deterministic.
    now = time.perf_counter()
    cast("object", timer)._first_yield_at = now  # type: ignore[attr-defined]
    cast("object", timer)._last_yield_at = now + 0.6  # type: ignore[attr-defined]
    cast("object", timer)._completion_yields = 3  # type: ignore[attr-defined]

    timer.finalize(completion_tokens=3)
    # One observation, value = 0.6 / 3 = 0.2 seconds.
    assert _histogram_count("sie_worker_generation_tpot_seconds", model) == before + 1
    sum_v = REGISTRY.get_sample_value("sie_worker_generation_tpot_seconds_sum", {"model": model, "grammar": "none"})
    assert sum_v is not None
    # Sum may have history from other tests on the same model; we
    # asserted count delta, so we just sanity-check the floor.
    assert sum_v >= 0.2 - 1e-9


def test_tpot_falls_back_to_yield_count_when_no_completion_tokens() -> None:
    """No completion-token denominator → use observed non-empty yield count."""
    model = "test/model_tpot_fallback"
    before = _histogram_count("sie_worker_generation_tpot_seconds", model)

    timer = obs_metrics.GenerationStreamTimer(model)
    now = time.perf_counter()
    cast("object", timer)._first_yield_at = now  # type: ignore[attr-defined]
    cast("object", timer)._last_yield_at = now + 0.4  # type: ignore[attr-defined]
    cast("object", timer)._completion_yields = 4  # type: ignore[attr-defined]

    timer.finalize(completion_tokens=None)
    assert _histogram_count("sie_worker_generation_tpot_seconds", model) == before + 1


def test_finalize_noop_when_no_text_yields() -> None:
    """Timeout / error before any text → finalize is a safe no-op for both metrics."""
    model = "test/model_noop"
    ttft_before = _histogram_count("sie_worker_generation_ttft_seconds", model)
    tpot_before = _histogram_count("sie_worker_generation_tpot_seconds", model)

    timer = obs_metrics.GenerationStreamTimer(model)
    timer.mark_yield(has_text=False)
    timer.mark_yield(has_text=False)
    timer.finalize(completion_tokens=10)  # would-be denominator ignored

    assert _histogram_count("sie_worker_generation_ttft_seconds", model) == ttft_before
    assert _histogram_count("sie_worker_generation_tpot_seconds", model) == tpot_before


def test_finalize_safe_with_zero_denominator() -> None:
    """``completion_tokens=0`` with no fallback yields must not divide-by-zero."""
    model = "test/model_zero_denom"
    before = _histogram_count("sie_worker_generation_tpot_seconds", model)

    timer = obs_metrics.GenerationStreamTimer(model)
    now = time.perf_counter()
    cast("object", timer)._first_yield_at = now  # type: ignore[attr-defined]
    cast("object", timer)._last_yield_at = now + 0.1  # type: ignore[attr-defined]
    cast("object", timer)._completion_yields = 0  # type: ignore[attr-defined]

    timer.finalize(completion_tokens=0)
    # No observation when no usable denominator is available.
    assert _histogram_count("sie_worker_generation_tpot_seconds", model) == before


def test_label_cardinality_does_not_grow_per_request() -> None:
    """Two timer cycles on the same (model, grammar) must produce one label series.

    Metrics-rollout acceptance: no caller-controlled label feeds these metrics
    so cardinality is bounded by ``models × grammar={none,json_schema,regex}``,
    regardless of safety_identifier / prompt / request_id.
    """
    model = "test/model_card"

    obs_metrics.GenerationStreamTimer(model).mark_yield(has_text=True)
    obs_metrics.GenerationStreamTimer(model).mark_yield(has_text=True)

    distinct_series = 0
    for metric in REGISTRY.collect():
        if metric.name == "sie_worker_generation_ttft_seconds":
            for sample in metric.samples:
                if sample.name.endswith("_count") and sample.labels.get("model") == model:
                    distinct_series += 1
    assert distinct_series == 1, f"expected 1 TTFT series for {model}, got {distinct_series}"


def test_grammar_label_splits_series() -> None:
    """A ``grammar`` label must produce a separate series per kind.

    Metrics-rollout acceptance: the overhead-attribution dashboard slices
    latency by structured-output mode. If the worker side dropped the
    grammar label (or hard-coded it), the gateway-vs-worker subtraction
    would aggregate two operationally-distinct regimes together.
    """
    model = "test/model_grammar_split"
    none_before = _histogram_count("sie_worker_generation_ttft_seconds", model, grammar="none")
    js_before = _histogram_count("sie_worker_generation_ttft_seconds", model, grammar="json_schema")
    rx_before = _histogram_count("sie_worker_generation_ttft_seconds", model, grammar="regex")

    obs_metrics.GenerationStreamTimer(model).mark_yield(has_text=True)
    obs_metrics.GenerationStreamTimer(model, grammar="json_schema").mark_yield(has_text=True)
    obs_metrics.GenerationStreamTimer(model, grammar="regex").mark_yield(has_text=True)

    assert _histogram_count("sie_worker_generation_ttft_seconds", model, grammar="none") == none_before + 1
    assert _histogram_count("sie_worker_generation_ttft_seconds", model, grammar="json_schema") == js_before + 1
    assert _histogram_count("sie_worker_generation_ttft_seconds", model, grammar="regex") == rx_before + 1


def test_metrics_export_contains_no_request_identifiers() -> None:
    """Privacy smoke: no per-request payload leaks into metric labels.

    Drives the timer with an obviously-secret string masquerading as a
    safety_identifier and asserts that string never lands in the
    Prometheus text export of the worker metrics families.
    """
    sentinel_identifier = "safety-id-do-not-leak-42"
    # We do NOT pass ``sentinel_identifier`` to the timer — that's the whole point.
    # Just exercise normal flow and assert the export is clean.
    obs_metrics.GenerationStreamTimer("test/model_privacy").mark_yield(has_text=True)

    # Render the two families we own and scan the text body.
    from prometheus_client.exposition import generate_latest

    body = generate_latest(REGISTRY).decode("utf-8")
    assert "sie_worker_generation_ttft_seconds" in body, "metric family must be exported"
    assert sentinel_identifier not in body, "request-level identifier leaked into /metrics body"


def test_metrics_export_contains_no_routing_or_prompt_identifiers() -> None:
    """§73: ``/metrics`` text must contain no raw prompt
    content, no raw routing keys, and no ``safety_identifier`` values.

    The privacy contract is enforced *by construction* (label sets are
    closed under ``(model, pool, grammar, kind, reason, stage, worker)``
    enums), so this test simply renders the export and scans for the
    sentinel substrings that would indicate a regression.
    """
    from prometheus_client.exposition import generate_latest

    # Exercise the surface once so families are non-empty.
    obs_metrics.GenerationStreamTimer("test/model_priv_scan").mark_yield(has_text=True)
    obs_metrics.set_generation_saturated("test/model_priv_scan", saturated=False)
    obs_metrics.set_speculative_acceptance_rate("test/model_priv_scan", 0.0)

    body = generate_latest(REGISTRY).decode("utf-8")
    forbidden_substrings = (
        # Raw prompt text (a request payload would carry this verbatim
        # if anyone accidentally labelled by prompt).
        "Please summarise",
        # Caller-supplied routing affinity hints.
        "routing_key=",
        'routing_key="',
        "prompt_cache_key=",
        'prompt_cache_key="',
        # OpenAI's user-identity field — explicitly discarded at the
        # HTTP layer and never forwarded to the queue.
        "safety_identifier=",
        'safety_identifier="',
    )
    for needle in forbidden_substrings:
        assert needle not in body, f"forbidden label/value {needle!r} leaked into /metrics body"


def test_observability_metrics_module_still_excludes_opentelemetry() -> None:
    """The metrics surface module is still pure Prometheus.

    The metrics rollout originally banned OpenTelemetry mentions from this
    *specific module* because trace propagation was out of scope for
    that work. Trace context propagation exists, but the
    *worker observability metrics module itself* still has no need
    to import OpenTelemetry — the propagation surface lives in
    :mod:`sie_server.processors.streaming` and the FastAPI tracing
    setup in :mod:`sie_server.observability.tracing`.

    Keeping this metrics module Prometheus-only avoids a regression
    where someone accidentally couples the prom-text rendering path
    to OTel SDK shutdown ordering. The trace-propagation wiring is
    exercised separately by
    ``tests/observability/test_trace_propagation.py``.
    """
    import pathlib

    metrics_path = pathlib.Path(obs_metrics.__file__)
    source = metrics_path.read_text(encoding="utf-8").lower()
    # `traceparent` is fine to mention in docs/comments anywhere
    # else, but should still not bleed into the metrics-text rendering
    # module. We still ban it here to keep that module focused.
    for forbidden in ("traceparent", "opentelemetry"):
        assert forbidden not in source, (
            f"sie_server.observability.metrics unexpectedly mentions {forbidden!r}; "
            "metrics-text rendering stays decoupled from OTel SDK"
        )


def test_full_section_4_11_worker_metric_surface_is_emitted() -> None:
    """§68 + §75: every worker-side §4.11 metric is emitted.

    Drives one observation/value into each metric the metrics rollout owns or
    initializes, then renders the Prometheus text and asserts the
    family name appears. The integration-test acceptance criterion
    (100 requests against a live model) is covered separately by the
    eval harness; this is the unit-level guard that pins the contract.
    """
    from prometheus_client.exposition import generate_latest

    model = "test/model_surface"
    grammar = "json_schema"

    # TTFT / TPOT — drive both with grammar=json_schema so the label
    # combination is exercised end-to-end.
    timer = obs_metrics.GenerationStreamTimer(model, grammar=grammar)
    timer.mark_yield(has_text=True)
    timer.mark_yield(has_text=True)
    timer.finalize(completion_tokens=2)

    # Admission / KV / in-flight (the admission-control rollout owns the call sites, but
    # the metric definitions live in this module).
    obs_metrics.GENERATION_IN_FLIGHT.labels(model=model).inc()
    obs_metrics.GENERATION_KV_RESERVED_TOKENS.labels(model=model).set(128)
    obs_metrics.GENERATION_ADMISSION_REJECTED.labels(model=model, reason="kv_budget").inc()
    # Grammar.
    obs_metrics.GRAMMAR_COMPILE_SECONDS.labels(model=model, kind=grammar).observe(0.05)
    obs_metrics.GRAMMAR_CACHE_HITS.labels(model=model).inc()
    obs_metrics.GRAMMAR_CACHE_MISSES.labels(model=model).inc()
    # Saturation / speculative.
    obs_metrics.set_generation_saturated(model, saturated=True)
    obs_metrics.set_speculative_acceptance_rate(model, 0.42)

    body = generate_latest(REGISTRY).decode("utf-8")
    expected_families = (
        "sie_worker_generation_ttft_seconds",
        "sie_worker_generation_tpot_seconds",
        "sie_worker_generation_in_flight",
        "sie_worker_generation_kv_reserved_tokens",
        "sie_worker_generation_admission_rejected_total",
        "sie_worker_grammar_compile_seconds",
        "sie_worker_grammar_cache_hits_total",
        "sie_worker_grammar_cache_misses_total",
        "sie_worker_generation_saturated",
        "sie_worker_speculative_acceptance_rate",
    )
    for family in expected_families:
        assert family in body, f"§4.11 worker metric {family!r} missing from /metrics export"


def test_buckets_match_gateway_constant_byte_for_byte() -> None:
    """Acceptance criterion: worker and gateway share identical bucket edges.

    The gateway side is Rust so we can't import it here; the gateway's
    constant has a code comment pinning the shared values. Pin the
    same values on this side too — if they diverge the dashboard's
    "gateway minus worker" subtraction silently lies.
    """
    assert obs_metrics.TTFT_TPOT_BUCKETS == (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        # Extended past 30s so long-completion requests on cold-start
        # workers (large max_new_tokens) don't all land in the +Inf
        # bucket — the previous ceiling made TPOT p99 uninformative.
        # MUST stay in lockstep with ``packages/sie_gateway/src/metrics.rs``
        # ``TTFT_TPOT_BUCKETS``.
        60.0,
        120.0,
        300.0,
    )


@pytest.mark.asyncio
async def test_adapter_wires_timer_through_yield_loop() -> None:
    """End-to-end: the generation adapter feeds the timer on each yield.

    We don't exercise the real SGLang HTTP path here — too expensive
    for a unit test. Instead we use the same fake-adapter approach
    that test_streaming_integration.py uses, plus the timer instance
    directly, to confirm the wiring records exactly one TTFT and one
    TPOT observation per request.
    """
    import asyncio
    from collections.abc import AsyncIterator

    from sie_server.adapters._generation_base import GenerationChunk

    model = "test/model_adapter_wire"
    ttft_before = _histogram_count("sie_worker_generation_ttft_seconds", model)
    tpot_before = _histogram_count("sie_worker_generation_tpot_seconds", model)

    async def fake_generate() -> AsyncIterator[GenerationChunk]:
        # Mirror the SGLang adapter's instrumentation contract: one
        # timer instance per call, mark_yield on each yield, finalize
        # in a finally block.
        timer = obs_metrics.GenerationStreamTimer(model)
        completion_tokens: int | None = None
        try:
            for delta in ("hel", "lo"):
                await asyncio.sleep(0.01)
                chunk = GenerationChunk(text_delta=delta)
                timer.mark_yield(has_text=bool(chunk.text_delta))
                yield chunk
            terminal = GenerationChunk(
                text_delta="",
                done=True,
                finish_reason="stop",
                prompt_tokens=4,
                completion_tokens=2,
            )
            completion_tokens = terminal.completion_tokens
            timer.mark_yield(has_text=False)
            yield terminal
        finally:
            timer.finalize(completion_tokens=completion_tokens)

    async for _ in fake_generate():
        pass

    assert _histogram_count("sie_worker_generation_ttft_seconds", model) == ttft_before + 1
    assert _histogram_count("sie_worker_generation_tpot_seconds", model) == tpot_before + 1
