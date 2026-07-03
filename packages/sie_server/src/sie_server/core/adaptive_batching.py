from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

logger = logging.getLogger(__name__)


class AdaptiveBatchTarget(Protocol):
    """The mutable batch limits the controller drives.

    ``BatchConfig`` satisfies this structurally; declaring it here lets the
    controller own its write-back without importing the concrete config
    (which would couple this leaf module to ``batcher`` and risk a cycle).
    """

    max_batch_wait_ms: float
    max_batch_cost: int


@dataclass(slots=True)
class LatencyTracker:
    """Rolling percentile tracker for request latencies.

    Maintains a fixed-size window of recent latency samples and computes
    exact percentiles by sorting. A deque of 200 samples is trivial to
    sort (~1us) and gives exact results without approximation structures.
    """

    window_size: int = 200
    min_samples: int = 10
    _samples: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._samples = deque(maxlen=self.window_size)

    def record(self, total_ms: float) -> None:
        """Record a latency sample in milliseconds."""
        self._samples.append(total_ms)

    def percentile(self, p: float) -> float | None:
        """Compute the p-th percentile (0–100) of recent samples.

        Returns None if fewer than min_samples have been recorded.
        """
        n = len(self._samples)
        if n < self.min_samples:
            return None
        sorted_samples = sorted(self._samples)
        # Use nearest-rank method
        idx = int(p / 100.0 * (n - 1))
        return sorted_samples[idx]

    def p50(self) -> float | None:
        """Return the median (p50) of recent samples."""
        return self.percentile(50)

    def p90(self) -> float | None:
        return self.percentile(90)

    def p99(self) -> float | None:
        return self.percentile(99)

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def reset(self) -> None:
        """Clear all samples."""
        self._samples.clear()


@dataclass(slots=True)
class BatchEfficiencyTracker:
    """Tracks batch fill ratios to measure GPU saturation.

    Records (actual_batch_cost / max_batch_cost) for recent batches.
    A fill ratio near 1.0 means the GPU is fully saturated.
    A fill ratio near 0 means batches are flushing nearly empty.
    """

    window_size: int = 50
    _fill_ratios: deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        self._fill_ratios = deque(maxlen=self.window_size)

    def record(self, actual_cost: int, max_cost: int) -> None:
        """Record a batch fill ratio."""
        if max_cost > 0:
            self._fill_ratios.append(actual_cost / max_cost)

    def mean_fill_ratio(self) -> float | None:
        """Return the mean fill ratio, or None if no samples."""
        if not self._fill_ratios:
            return None
        return sum(self._fill_ratios) / len(self._fill_ratios)

    @property
    def sample_count(self) -> int:
        return len(self._fill_ratios)

    def reset(self) -> None:
        self._fill_ratios.clear()


@dataclass(frozen=True, slots=True)
class AdaptiveBatchState:
    """Immutable snapshot of adaptive controller state.

    Used by WebSocket status and gateway health to expose controller
    internals without coupling to private fields.
    """

    enabled: bool
    calibrated: bool
    target_p50_ms: float | None
    current_wait_ms: float
    current_batch_cost: int
    observed_p50_ms: float | None
    headroom_ms: float | None
    fill_ratio: float | None
    integral: float
    starvation_streak: int = 0
    starvation_resets: int = 0


@dataclass(slots=True)
class AdaptiveBatchController:
    """PI controller that adjusts batch wait and batch cost to maximize GPU
    saturation while respecting a latency SLO.

    Three knobs:
    1. **max_batch_wait_ms** — how long to wait for items before flushing.
       Adjusted by a PI (proportional-integral) controller based on the gap
       between target and observed p50 latency.
    2. **max_batch_cost** — token/cost limit per batch. Adjusted by a
       proportional-only controller gated on batch fill ratio.
    3. **target_p50_ms** — the latency SLO. Either set explicitly or
       auto-calibrated from observed inference latency.

    Auto-calibration:
        When ``target_p50_ms`` is None, the controller measures inference-only
        p50 (GPU forward pass, excluding queue/batch wait) during the first N
        requests and derives the target as ``inference_p50 × calibration_multiplier``.
        This avoids a feedback loop where conservative initial settings inflate
        early latency and poison the target.

    PI controller (wait knob):
        The integral term is time-normalized (``error × dt``) so Ki is stable
        across traffic rates. Saturation-aware anti-windup prevents integral
        accumulation when the output is clamped in the direction the error
        pushes. Idle decay prevents stale error from carrying into the next
        burst after a long idle period.

    Proportional-only controller (cost knob):
        Only adjusts when ``fill_ratio >= threshold`` (GPU is saturated at
        the current limit). This prevents wasteful increases when batches
        aren't filling. No integral term needed — the gating condition
        acts as a form of conditional integration.

    Starvation recovery (deadlock escape hatch):
        The PI loop has a pathological attractor when observed p50 stays
        above target for long enough to push both wait and cost to the
        floor. At the floor, every batch holds a single item; the single
        item has high p50 (dominated by one-item forward-pass + queue);
        headroom stays negative; the loop keeps both knobs pinned — it
        cannot raise wait (headroom < 0) and cannot raise cost
        (``headroom_frac × cost_gain < 0``). To escape, we track a streak
        of "batch produced at floor with size ≤ threshold". If the streak
        exceeds ``starvation_window`` AND both knobs are at their floors,
        we declare deadlock and hard-reset to a conservative recovery
        point half-way between floor and initial values. The integrator
        and ``last_step_time`` are also cleared so the next few samples
        come from a clean state.

        The counter only increments when ``step()`` is called with a
        non-None ``batch_size`` — i.e. real traffic is flowing. Idle
        worker loops don't trip it.
    """

    # Latency target — None means auto-calibrate from inference latency
    target_p50_ms: float | None = None

    # Auto-calibration parameters
    calibration_multiplier: float = 1.5
    min_target_p50_ms: float = 5.0
    max_target_p50_ms: float = 500.0

    # Wait time bounds. These are the PID-controller's internal default
    # clamps; production deployments construct ``AdaptiveBatchController``
    # explicitly from ``sie_server.config.engine.AdaptiveBatchingConfig``.
    min_wait_ms: float = 1.0
    max_wait_ms: float = 50.0

    # Batch cost bounds (token limit). These in-module defaults exist so
    # the controller is constructible without arguments for tests; real
    # deployments compute ``min_batch_cost`` from the model's
    # ``max_batch_tokens`` (see ``ModelWorker.__init__``) so the floor
    # scales with the model budget — a flat 256 is too low for anything
    # but the smallest models and lets the cost knob collapse to a
    # single-item batch under sustained negative-headroom load.
    min_batch_cost: int = 256
    max_batch_cost: int = 65536

    # Controller tuning
    gain: float = 0.3
    integral_gain: float = 0.05
    cost_gain: float = 0.15
    update_interval: int = 10
    fill_ratio_threshold: float = 0.7

    # Starvation detection / deadlock recovery
    starvation_recovery_enabled: bool = True
    starvation_window: int = 20
    starvation_batch_size: int = 1

    # Initial values used as recovery anchors (captured in __post_init__
    # from the _current_* fields so operator-provided startup values are
    # honoured rather than being replaced with hard-coded defaults).
    _initial_wait_ms: float = 0.0
    _initial_batch_cost: int = 0

    # Current state
    _current_wait_ms: float = 10.0
    _current_batch_cost: int = 16384
    _steps_since_update: int = 0

    # Calibration state
    _auto_calibrate: bool = False  # True if target_p50_ms was originally None
    _calibrated: bool = False
    _inference_tracker: LatencyTracker = field(default_factory=lambda: LatencyTracker(window_size=50, min_samples=10))

    # PI integral state
    _integral: float = 0.0
    _integral_max: float = 20.0
    _last_step_time: float | None = None

    # Starvation tracking (runtime)
    _starvation_streak: int = 0
    _starvation_resets: int = 0

    def __post_init__(self) -> None:
        if self.target_p50_ms is None:
            self._auto_calibrate = True
        else:
            self._calibrated = True
        # Snapshot the operator-provided start values so ``reset()`` and
        # starvation recovery can return to whatever the worker was
        # configured with (e.g. from helm values) rather than this
        # module's historic defaults.
        self._initial_wait_ms = self._current_wait_ms
        self._initial_batch_cost = self._current_batch_cost

    def record_inference_sample(self, inference_ms: float) -> None:
        """Record an inference-only latency sample for auto-calibration.

        Only collects samples before calibration completes. After calibration,
        this is a no-op.

        Args:
            inference_ms: GPU forward pass time from RequestTiming.inference_ms.
        """
        if not self._calibrated:
            self._inference_tracker.record(inference_ms)

    def step(
        self,
        observed_p50_ms: float | None,
        fill_ratio: float | None,
        batch_size: int | None = None,
    ) -> tuple[float, int]:
        """Advance the controller and return (new_wait_ms, new_batch_cost).

        Args:
            observed_p50_ms: Current rolling p50 latency (total_ms), or None
                if not enough samples yet.
            fill_ratio: Mean batch fill ratio (0.0–1.0), or None.
            batch_size: Size of the batch that triggered this step. Used by
                the starvation detector to count consecutive tiny batches at
                the floor. ``None`` (the default, for backward compatibility
                with callers that don't track per-batch size) disables the
                starvation counter — the loop still runs but can never
                trigger recovery.

        Returns:
            Tuple of (max_batch_wait_ms, max_batch_cost).
        """
        # Starvation tracking runs on every call (not every update_interval)
        # but only counts while both knobs are pinned at the floor. Tiny
        # batches before the floor are normal control-loop evidence, not a
        # deadlock signal.
        self._update_starvation_tracker(batch_size, pinned_at_floor=self._is_pinned_at_floor())

        self._steps_since_update += 1
        if self._steps_since_update < self.update_interval:
            return self._current_wait_ms, self._current_batch_cost

        self._steps_since_update = 0

        # --- Auto-calibration phase ---
        if not self._calibrated:
            inference_p50 = self._inference_tracker.p50()
            if inference_p50 is not None:
                self.target_p50_ms = _clamp(
                    inference_p50 * self.calibration_multiplier,
                    self.min_target_p50_ms,
                    self.max_target_p50_ms,
                )
                self._calibrated = True
                self._integral = 0.0
                self._last_step_time = None
                logger.info(
                    "Auto-calibrated: target_p50_ms=%.1fms (inference_p50=%.1fms x %.1f, clamped to [%.1f, %.1f])",
                    self.target_p50_ms,
                    inference_p50,
                    self.calibration_multiplier,
                    self.min_target_p50_ms,
                    self.max_target_p50_ms,
                )
            # Hold knobs at initial values until calibrated
            return self._current_wait_ms, self._current_batch_cost

        if observed_p50_ms is None or self.target_p50_ms is None:
            return self._current_wait_ms, self._current_batch_cost

        # --- Starvation recovery (before running the PI loop). ---
        # If we trip this, skip the PI update this step — the recovery
        # already set the knobs. The PI loop will resume next step with a
        # cleared integrator and fresh latency samples.
        if self._maybe_recover_from_starvation():
            return self._current_wait_ms, self._current_batch_cost

        target = self.target_p50_ms
        headroom_ms = target - observed_p50_ms
        headroom_frac = headroom_ms / target  # normalized (-inf, 1)

        # --- Compute dt for time-normalized integral ---
        now = time.monotonic()
        if self._last_step_time is not None:
            dt_s = min(now - self._last_step_time, 5.0)  # cap idle gap
        else:
            dt_s = 0.0  # first step after calibration, no integral contribution
        self._last_step_time = now

        # --- Idle decay: prevent stale error from carrying into next burst ---
        if dt_s > 2.0:
            decay_factor = 0.5 ** (dt_s - 2.0)
            self._integral *= decay_factor

        # --- Saturation-aware anti-windup ---
        # Only integrate when the output is NOT saturated in the direction
        # the error would push it. This prevents overshoot after recovery.
        output_at_max = self._current_wait_ms >= self.max_wait_ms
        output_at_min = self._current_wait_ms <= self.min_wait_ms

        can_integrate = True
        if output_at_max and headroom_ms > 0:
            can_integrate = False
        if output_at_min and headroom_ms < 0:
            can_integrate = False

        if can_integrate and self.integral_gain > 0:
            self._integral += headroom_ms * dt_s
            self._integral = _clamp(self._integral, -self._integral_max, self._integral_max)

        # --- Knob 1: batch wait (PI controller) ---
        wait_adjustment = headroom_ms * self.gain + self._integral * self.integral_gain
        new_wait = self._current_wait_ms + wait_adjustment
        self._current_wait_ms = _clamp(new_wait, self.min_wait_ms, self.max_wait_ms)

        # --- Knob 2: batch cost (proportional-only, gated on fill ratio) ---
        # Symmetric anti-windup: don't push the cost further into saturation
        # in the direction the error wants. Without this guard, once p50
        # exceeds target the loop shrinks cost from initial all the way to
        # ``min_batch_cost`` (since fill_ratio rises to ~1.0 as batches
        # collapse — tiny batches always fill their tiny budget). That's the
        # mechanism behind the queue-mode regression guarded by
        # packages/sie_server_sidecar/docs/architecture-guide.md.
        #
        # The anti-windup only stops further shrinking once at the floor,
        # it doesn't stop the collapse itself — that's why callers are
        # expected to set ``min_batch_cost`` to a value large enough to
        # keep GPU forwards amortized (see ModelWorker.__init__ which
        # anchors it at ``max_batch_tokens // 4``). A flat floor of 256
        # is too low for any non-trivial model.
        if fill_ratio is not None and fill_ratio >= self.fill_ratio_threshold:
            cost_at_max = self._current_batch_cost >= self.max_batch_cost
            cost_at_min = self._current_batch_cost <= self.min_batch_cost
            can_adjust_cost = True
            if cost_at_max and headroom_ms > 0:
                can_adjust_cost = False
            if cost_at_min and headroom_ms < 0:
                can_adjust_cost = False
            if can_adjust_cost:
                cost_adjustment = self._current_batch_cost * headroom_frac * self.cost_gain
                new_cost = self._current_batch_cost + cost_adjustment
                self._current_batch_cost = int(_clamp(new_cost, self.min_batch_cost, self.max_batch_cost))

        logger.debug(
            "Adaptive batch: p50=%.1fms (target=%.1f), headroom=%.1fms, integral=%.2f, "
            "fill=%.2f, wait=%.1fms, cost=%d, starvation_streak=%d",
            observed_p50_ms,
            target,
            headroom_ms,
            self._integral,
            fill_ratio or 0.0,
            self._current_wait_ms,
            self._current_batch_cost,
            self._starvation_streak,
        )

        return self._current_wait_ms, self._current_batch_cost

    def apply_step(
        self,
        target: AdaptiveBatchTarget,
        observed_p50_ms: float | None,
        fill_ratio: float | None,
        *,
        batch_size: int | None = None,
    ) -> tuple[float, int]:
        """Advance the controller and write the new limits into ``target``.

        Single owner of the controller-output → batch-config write-back that
        previously lived inline in ``ModelWorker._process_loop``. Returns
        ``(max_batch_wait_ms, max_batch_cost)`` for the caller's metrics.

        Synchronous (no ``await``): both assignments happen together on the
        event-loop thread, so the in-place mutation stays safe against
        ``BatchFormer``'s async lock, which cannot interleave with this block.
        """
        new_wait, new_cost = self.step(observed_p50_ms, fill_ratio, batch_size=batch_size)
        target.max_batch_wait_ms = new_wait
        target.max_batch_cost = new_cost
        return new_wait, new_cost

    # ------------------------------------------------------------------
    # Starvation detector helpers
    # ------------------------------------------------------------------

    def _is_pinned_at_floor(self) -> bool:
        return self._current_wait_ms <= self.min_wait_ms and self._current_batch_cost <= self.min_batch_cost

    def _update_starvation_tracker(self, batch_size: int | None, *, pinned_at_floor: bool) -> None:
        """Advance the consecutive-tiny-batches counter.

        The streak resets whenever we see a batch larger than the
        ``starvation_batch_size`` threshold — a single healthy batch is
        enough evidence that the loop is not deadlocked, regardless of
        what the PI math is about to do this step.

        ``batch_size is None`` means starvation detection is disabled
        for this step, so clear any stale streak before the recovery
        check runs.
        """
        if batch_size is None:
            self._starvation_streak = 0
            return
        if not pinned_at_floor:
            self._starvation_streak = 0
            return
        if batch_size > self.starvation_batch_size:
            self._starvation_streak = 0
        else:
            self._starvation_streak += 1

    def _maybe_recover_from_starvation(self) -> bool:
        """If deadlock is detected, reset knobs and return True.

        Deadlock = consecutive tiny batches + both knobs already pinned
        at their floors. Once tripped, we jump to a conservative midway
        point (half of initial) — aggressive enough to pull a few items
        per batch (which re-establishes a healthy p50 signal) but not so
        aggressive that we re-overshoot SLO immediately.

        Integrator, last-step-time and the streak counter are all
        cleared; the next iteration effectively starts the PI loop from
        scratch using fresh samples.
        """
        if not self.starvation_recovery_enabled:
            return False
        if self._starvation_streak < self.starvation_window:
            return False
        if not self._is_pinned_at_floor():
            return False

        recovery_wait = _clamp(
            (self.min_wait_ms + self._initial_wait_ms) / 2,
            self.min_wait_ms,
            self.max_wait_ms,
        )
        recovery_cost = int(
            _clamp(
                max(self._initial_batch_cost // 2, self.min_batch_cost * 4),
                self.min_batch_cost,
                self.max_batch_cost,
            )
        )

        logger.warning(
            "Adaptive batch: starvation recovery triggered after %d consecutive "
            "batches with size <= %d at the floor (wait=%.1fms, cost=%d). "
            "Resetting to wait=%.1fms, cost=%d.",
            self._starvation_streak,
            self.starvation_batch_size,
            self._current_wait_ms,
            self._current_batch_cost,
            recovery_wait,
            recovery_cost,
        )

        self._current_wait_ms = recovery_wait
        self._current_batch_cost = recovery_cost
        self._integral = 0.0
        self._last_step_time = None
        self._starvation_streak = 0
        self._starvation_resets += 1
        return True

    @property
    def current_wait_ms(self) -> float:
        return self._current_wait_ms

    @property
    def current_batch_cost(self) -> int:
        return self._current_batch_cost

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def starvation_streak(self) -> int:
        return self._starvation_streak

    @property
    def starvation_resets(self) -> int:
        return self._starvation_resets

    def snapshot(
        self,
        observed_p50_ms: float | None = None,
        fill_ratio: float | None = None,
    ) -> AdaptiveBatchState:
        """Return an immutable snapshot of the controller state."""
        target = self.target_p50_ms
        headroom = None
        if target is not None and observed_p50_ms is not None:
            headroom = target - observed_p50_ms
        return AdaptiveBatchState(
            enabled=True,
            calibrated=self._calibrated,
            target_p50_ms=target,
            current_wait_ms=self._current_wait_ms,
            current_batch_cost=self._current_batch_cost,
            observed_p50_ms=observed_p50_ms,
            headroom_ms=headroom,
            fill_ratio=fill_ratio,
            integral=self._integral,
            starvation_streak=self._starvation_streak,
            starvation_resets=self._starvation_resets,
        )

    def reset(self) -> None:
        """Reset the controller to the operator-provided initial state.

        ``_initial_wait_ms`` / ``_initial_batch_cost`` are captured in
        ``__post_init__`` so this restores whatever the worker was
        configured with at startup, not the module's legacy defaults.
        """
        self._current_wait_ms = _clamp(self._initial_wait_ms, self.min_wait_ms, self.max_wait_ms)
        self._current_batch_cost = int(_clamp(self._initial_batch_cost, self.min_batch_cost, self.max_batch_cost))
        self._steps_since_update = 0
        self._integral = 0.0
        self._last_step_time = None
        self._starvation_streak = 0
        if self._auto_calibrate:
            self._calibrated = False
            self.target_p50_ms = None
            self._inference_tracker.reset()
        # If target was explicit, _calibrated stays True.
        # ``_starvation_resets`` is a monotonic counter and deliberately not
        # cleared — it's an operational signal exposed via Prometheus.


# Keep backward compat alias
BatchWaitController = AdaptiveBatchController


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
