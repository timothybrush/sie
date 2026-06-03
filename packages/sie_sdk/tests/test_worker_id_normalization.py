"""Cross-language fixture for worker_id NATS subject normalization.

The gateway (Rust) publishes direct-dispatch work items to
``sie.work.{model}.{pool}.{worker_id}`` and applies its
``normalize_model_id`` to the worker_id before publishing. The Python SDK
helpers (``work_worker_subject``, ``work_worker_stream_subjects``,
``work_worker_stream_name``, ``work_worker_consumer_name``) and the worker
init (``NatsPullLoop.__init__``) now route worker IDs through
:func:`sie_sdk.queue_types.normalize_worker_id`, which delegates to
``normalize_model_id`` for byte-identical output.

If this file diverges from the Rust ``test_worker_id_normalization_cross_language``
fixture in ``packages/sie_gateway/src/queue/publisher.rs``, direct-dispatch
will silently miss every worker whose raw id contains a newly-changed
character — the pool fallback will eventually catch the work item but the
HRW routing decision is wasted.

This locks the cross-language normalization contract.
"""

from __future__ import annotations

import pytest
from sie_sdk.queue_types import (
    normalize_worker_id,
    work_worker_consumer_name,
    work_worker_stream_name,
    work_worker_stream_subjects,
    work_worker_subject,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # unchanged: clean ascii with hyphens
        ("worker-1", "worker-1"),
        # case preserved (NATS subjects are case-sensitive; we deliberately
        # do NOT lowercase — see the rationale in publisher.rs).
        ("Worker-1", "Worker-1"),
        ("WORKER", "WORKER"),
        # dotted Kubernetes pod hostname → each dot maps to "_dot_"
        (
            "sie-worker-7d9f-default-0.sie-worker.default.svc",
            "sie-worker-7d9f-default-0_dot_sie-worker_dot_default_dot_svc",
        ),
        # whitespace → "_"
        ("my worker", "my_worker"),
        # wildcard tokens → "_". Each ``.`` → ``_dot_`` and ``*`` → ``_``,
        # so ``worker.*.foo`` = ``worker`` + ``_dot_`` + ``_`` + ``_dot_`` + ``foo``.
        ("worker.*.foo", "worker_dot___dot_foo"),
        # leading/trailing whitespace is preserved as `_` — we do NOT
        # trim. The empty / whitespace-only case is rejected separately
        # (see test_normalize_worker_id_rejects_empty).
        ("  worker-1  ", "__worker-1__"),
        # consecutive separators are NOT collapsed — keeps the contract
        # identical to Rust's normalize_model_id and avoids surprising
        # operators who already rely on the existing mapping.
        ("worker--1", "worker--1"),
    ],
)
def test_normalize_worker_id_matches_rust_fixture(raw: str, expected: str) -> None:
    """Every input here MUST produce the same output as the Rust
    ``test_worker_id_normalization_cross_language`` fixture.
    """
    assert normalize_worker_id(raw) == expected


@pytest.mark.parametrize("raw", ["", " ", "\t", "   \n  "])
def test_normalize_worker_id_rejects_empty(raw: str) -> None:
    """Empty / whitespace-only input must raise ValueError.

    The previous helper silently substituted the literal string
    ``"worker"`` for an empty input, which collided durable JetStream
    consumers across processes when env vars were missing. Empty input
    is now a hard error — :class:`NatsPullLoop.__init__` falls back to
    ``uuid4().hex`` and logs the failure explicitly rather than letting
    a silent substitution propagate.
    """
    with pytest.raises(ValueError, match="empty"):
        normalize_worker_id(raw)


def test_subject_helpers_route_through_normalize() -> None:
    """All four subject helpers must apply ``normalize_worker_id``.

    Sanity-check on a worker_id that contains a dot — if any helper
    interpolated the raw value the resulting subject would have an
    extra token and the gateway and worker would bind to different
    subjects.
    """
    raw = "pod-0.svc"
    normalized = normalize_worker_id(raw)
    assert normalized == "pod-0_dot_svc"

    # per-worker publish subject
    subj = work_worker_subject("BAAI/bge-m3", "default", raw)
    assert subj == f"sie.work.BAAI__bge-m3.default.{normalized}"
    assert subj.count(".") == 4, "exactly 4 dots → 5 tokens after split"

    # per-worker stream subjects (consumer-side filter)
    subjects = work_worker_stream_subjects("default", raw)
    assert subjects == [f"sie.work.*.default.{normalized}"]

    # per-worker stream name (no dots — uses an underscore prefix)
    assert work_worker_stream_name(raw) == f"WORK_WORKER_{normalized}"

    # durable consumer name
    assert work_worker_consumer_name(raw) == f"gen-{normalized}"


def test_subject_helpers_reject_empty_worker_id() -> None:
    """All four subject helpers must reject empty worker_id rather than
    silently substituting a default.
    """
    with pytest.raises(ValueError, match="empty"):
        work_worker_subject("m", "p", "")
    with pytest.raises(ValueError, match="empty"):
        work_worker_stream_subjects("p", "")
    with pytest.raises(ValueError, match="empty"):
        work_worker_stream_name("")
    with pytest.raises(ValueError, match="empty"):
        work_worker_consumer_name("")
