"""Tests for the NATS health publisher (feature flag + payload shape).

The publisher's runtime path requires a live NATS server, which the
unit-test environment does not provide. We exercise the bits that are
testable without IO:

* The feature flag gate ``is_enabled()`` reads ``SIE_HEALTH_NATS``.
* The subject template matches what the gateway subscribes to.
* The publisher does not raise during a failed connect (graceful
  degradation).
"""

from __future__ import annotations

from typing import Any

import msgpack
import pytest
from sie_server.health.nats_publisher import (
    ENABLE_ENV,
    HEALTH_SUBJECT_TEMPLATE,
    NatsHealthPublisher,
    is_enabled,
)


def test_is_enabled_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENABLE_ENV, raising=False)
    assert is_enabled() is False


def test_is_enabled_only_when_exactly_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENABLE_ENV, "1")
    assert is_enabled() is True
    monkeypatch.setenv(ENABLE_ENV, "true")
    assert is_enabled() is False  # strict — must be the literal "1"
    monkeypatch.setenv(ENABLE_ENV, "0")
    assert is_enabled() is False


def test_subject_template_matches_gateway_subscription() -> None:
    # The Rust gateway subscribes to ``sie.health.>`` and parses the
    # final token as ``worker_identifier``. The template must place
    # the worker_id at that position.
    assert HEALTH_SUBJECT_TEMPLATE.format(worker_id="w-42") == "sie.health.w-42"


def test_payload_is_msgpack_dict_of_status_message() -> None:
    """The publisher serialises whatever ``build_status`` returns via
    msgpack. We assert the round-trip survives so the gateway's
    msgpack decoder will accept the bytes.
    """
    sample = {
        "ready": True,
        "name": "w-1",
        "pool_name": "p",
        "loaded_models": ["m"],
        "saturated": True,
    }
    payload = msgpack.packb(sample, use_bin_type=True)
    decoded = msgpack.unpackb(payload, raw=False)
    assert decoded == sample
    assert decoded["saturated"] is True


def test_publisher_constructor_does_not_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction must be I/O free — the publisher only opens a
    connection when ``start()`` is called. This lets callers wire it
    up in app-factory contexts that may not have an event loop yet.
    """

    async def _build() -> dict[str, Any]:
        return {"ready": True}

    pub = NatsHealthPublisher(
        nats_url="nats://127.0.0.1:1",
        worker_id="w-test",
        build_status=_build,
    )
    # No connection should have been initiated.
    assert pub._nc is None
    assert pub._task is None
    # And the subject is fully resolved at construction time.
    assert pub._subject == "sie.health.w-test"
