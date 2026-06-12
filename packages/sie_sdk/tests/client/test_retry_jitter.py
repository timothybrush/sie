# ruff: noqa: S311 — seeded random.Random is intentional for deterministic
# jitter tests; nothing here is security-sensitive.
#
# Tests for retry backoff jitter (Fix 3).
#
# Fixed / pure-exponential backoff makes a fleet of clients that lost a
# worker at the same instant retry in lockstep (thundering herd). The
# SDK applies *downward-only* "equal jitter": the returned delay is drawn
# uniformly from ``[delay * (1 - RETRY_JITTER_FRACTION), delay]`` and
# clamped to ``>= 0``. Downward-only guarantees the jittered value never
# exceeds the caller's existing caps / provision-timeout budget — so all
# pre-existing delay-bound tests remain valid.

from __future__ import annotations

import random

import pytest
from sie_sdk.client._shared import (
    MODEL_LOADING_DEFAULT_DELAY_S,
    RESOURCE_EXHAUSTED_MAX_DELAY_S,
    RETRY_JITTER_FRACTION,
    apply_jitter,
    compute_oom_backoff,
    compute_retry_delay,
)


class TestApplyJitter:
    @pytest.mark.parametrize("delay", [0.5, 1.0, 5.0, 30.0])
    def test_jitter_stays_within_downward_band(self, delay: float) -> None:
        rng = random.Random(1234)
        lo = delay * (1.0 - RETRY_JITTER_FRACTION)
        for _ in range(1000):
            j = apply_jitter(delay, rng=rng)
            assert lo <= j <= delay, f"{j} out of band [{lo}, {delay}]"

    def test_jitter_never_negative(self) -> None:
        rng = random.Random(0)
        for _ in range(1000):
            assert apply_jitter(0.0001, rng=rng) >= 0.0

    @pytest.mark.parametrize("delay", [0.0, -1.0, -0.001])
    def test_non_positive_delay_returned_clamped(self, delay: float) -> None:
        # Nothing to jitter; result is clamped to >= 0 and never positive.
        assert apply_jitter(delay) == 0.0

    def test_deterministic_with_seeded_rng(self) -> None:
        a = apply_jitter(10.0, rng=random.Random(42))
        b = apply_jitter(10.0, rng=random.Random(42))
        assert a == b

    def test_actually_varies_across_draws(self) -> None:
        rng = random.Random(7)
        draws = {apply_jitter(10.0, rng=rng) for _ in range(50)}
        # Vanishingly unlikely to collapse to a single value if jitter works.
        assert len(draws) > 1


class TestComputeRetryDelayJitter:
    def test_within_cap_and_positive(self) -> None:
        rng = random.Random(99)
        # Large timeout so the cap is MODEL_LOADING_DEFAULT_DELAY_S.
        for _ in range(200):
            d = compute_retry_delay(
                start_time=0.0,
                timeout=1e9,  # effectively unbounded; monotonic() >> 0
                error_label="x",
                error=RuntimeError("boom"),
                rng=rng,
            )
            assert d is not None
            lo = MODEL_LOADING_DEFAULT_DELAY_S * (1.0 - RETRY_JITTER_FRACTION)
            assert lo <= d <= MODEL_LOADING_DEFAULT_DELAY_S

    def test_returns_none_when_budget_exhausted(self) -> None:
        # start_time far in the past relative to a tiny timeout -> elapsed >= timeout.
        import time

        d = compute_retry_delay(
            start_time=time.monotonic() - 100.0,
            timeout=1.0,
            error_label="x",
            error=RuntimeError("boom"),
        )
        assert d is None

    def test_never_exceeds_remaining_budget(self) -> None:
        # Tiny remaining budget caps the (pre-jitter) delay below the default;
        # jitter must keep it within that smaller cap.
        import time

        rng = random.Random(3)
        start = time.monotonic()
        timeout = 2.0  # elapsed is ~0, so remaining ~= 2.0 < default 5.0 cap
        for _ in range(200):
            d = compute_retry_delay(
                start_time=start, timeout=timeout, error_label="x", error=RuntimeError("b"), rng=rng
            )
            assert d is not None
            assert 0.0 <= d <= timeout


class TestComputeOomBackoffJitter:
    def test_first_attempt_retry_after_honoured_verbatim(self) -> None:
        # An explicit server "wait N seconds" hint is honoured exactly (no
        # jitter): the SDK only de-correlates its own exponential schedule.
        assert compute_oom_backoff(3.0, attempt=0, rng=random.Random(1)) == 3.0

    def test_first_attempt_retry_after_capped(self) -> None:
        assert compute_oom_backoff(1000.0, attempt=0) == RESOURCE_EXHAUSTED_MAX_DELAY_S

    def test_subsequent_attempts_jittered_within_cap(self) -> None:
        rng = random.Random(11)
        for attempt in (1, 2, 3, 4, 10):
            for _ in range(100):
                d = compute_oom_backoff(None, attempt=attempt, rng=rng)
                assert 0.0 <= d <= RESOURCE_EXHAUSTED_MAX_DELAY_S

    def test_negative_retry_after_never_negative(self) -> None:
        # Regression guard mirroring test_oom_retry.test_negative_retry_after:
        # a malformed negative header must never yield a negative sleep.
        rng = random.Random(5)
        for attempt in range(5):
            assert compute_oom_backoff(-5.0, attempt=attempt, rng=rng) >= 0.0

    def test_jitter_actually_applied_on_exponential_path(self) -> None:
        # attempt=1 with no Retry-After -> base 5 * 2 = 10, capped at 30.
        # Jitter should produce values strictly below 10 sometimes and never above.
        rng = random.Random(2024)
        draws = [compute_oom_backoff(None, attempt=1, rng=rng) for _ in range(100)]
        assert all(d <= 10.0 for d in draws)
        assert any(d < 10.0 for d in draws)
        assert len(set(draws)) > 1
