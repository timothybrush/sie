from __future__ import annotations

import pytest
from sie_server.core import readiness


@pytest.fixture(autouse=True)
def _reset_readiness_state():
    readiness.register_liveness_probe(None)
    readiness.mark_not_ready()
    yield
    readiness.register_liveness_probe(None)
    readiness.mark_not_ready()


def test_default_state_is_not_ready() -> None:
    assert readiness.is_ready() is False


def test_mark_ready_without_probe() -> None:
    readiness.mark_ready()
    assert readiness.is_ready() is True


def test_mark_not_ready_is_sticky_until_mark_ready() -> None:
    readiness.mark_ready()
    readiness.mark_not_ready()
    assert readiness.is_ready() is False


def test_liveness_probe_false_blocks_ready() -> None:
    readiness.mark_ready()
    readiness.register_liveness_probe(lambda: False)
    assert readiness.is_ready() is False


def test_liveness_probe_true_allows_ready() -> None:
    readiness.mark_ready()
    readiness.register_liveness_probe(lambda: True)
    assert readiness.is_ready() is True


def test_liveness_probe_does_not_override_not_ready() -> None:
    readiness.register_liveness_probe(lambda: True)
    assert readiness.is_ready() is False


def test_clearing_probe_with_none_restores_plain_ready_semantics() -> None:
    readiness.mark_ready()
    readiness.register_liveness_probe(lambda: False)
    assert readiness.is_ready() is False
    readiness.register_liveness_probe(None)
    assert readiness.is_ready() is True
