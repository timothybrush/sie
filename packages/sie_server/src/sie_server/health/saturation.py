"""Saturation hysteresis state machine.

The routing rollout surfaces a boolean ``saturated`` flag on the worker status
payload. The sidecar publishes that status over ``sie.health.{worker_id}`` in
queue-mode deployments, and the gateway uses it to exclude the worker from the
HRW direct-dispatch ring.

The flag is governed by a two-threshold hysteresis to avoid thrashing
under oscillation around a single setpoint:

* When ``in_flight / capacity >= HIGH`` (default 90%), the worker
  flips to ``saturated = True``.
* It stays saturated until ``in_flight / capacity <= LOW``
  (default 70%), at which point it flips back to ``False``.

The state machine is intentionally pure (no I/O, no clock) so it can be
unit-tested in isolation; status publishers drive it from the active saturation
signal for their pool. Non-generation pools use an in-flight/capacity ratio,
while generation pools use ``kv_reserved / kv_budget`` — the state machine
itself stays the same.
"""

from __future__ import annotations

from dataclasses import dataclass

# Defaults match the routing rollout's acceptance criteria. Both are inclusive
# (``>=`` / ``<=``) so the boundaries behave predictably under floating
# point rounding.
DEFAULT_HIGH_WATERMARK: float = 0.90
DEFAULT_LOW_WATERMARK: float = 0.70


@dataclass
class SaturationGate:
    """Hysteresis gate for the ``saturated`` flag.

    Attributes:
        high: Fraction at which to flip to saturated. Default 0.90.
        low: Fraction at which to flip back to not-saturated. Default 0.70.
        saturated: Current latched state. Starts at ``False``.

    Invariant: ``0 <= low <= high <= 1``. The dataclass validates this
    once in ``__post_init__`` so misconfiguration is loud.
    """

    high: float = DEFAULT_HIGH_WATERMARK
    low: float = DEFAULT_LOW_WATERMARK
    saturated: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.low <= self.high <= 1.0):
            raise ValueError(
                f"SaturationGate thresholds invalid: low={self.low}, high={self.high}; require 0 <= low <= high <= 1"
            )

    def update(self, in_flight: int, capacity: int) -> bool:
        """Feed the gate a fresh observation and return the latched state.

        Args:
            in_flight: Current number of admitted-but-not-completed requests.
            capacity: Configured upper bound (typically ``max_batch_requests``).

        Returns:
            The new latched ``saturated`` value.

        Notes:
            * ``capacity <= 0`` is treated as "no capacity configured" —
              in that case the gate cannot meaningfully decide, so it
              returns its current state unchanged. This matches the
              defensive behaviour upstream in ``WorkerState.memory_utilization``.
            * ``in_flight < 0`` is clamped to 0; callers should never
              pass negative values but we don't want to crash the
              status-update loop on a logic bug elsewhere.
        """
        if capacity <= 0:
            return self.saturated
        frac = max(0, in_flight) / capacity
        if self.saturated:
            # Already flipped — only un-flip if we drop at or below the low watermark.
            if frac <= self.low:
                self.saturated = False
        # Not yet flipped — flip if we reach the high watermark.
        elif frac >= self.high:
            self.saturated = True
        return self.saturated
