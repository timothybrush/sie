"""Tests for the saturation hysteresis gate."""

from __future__ import annotations

import pytest
from sie_server.health.saturation import (
    DEFAULT_HIGH_WATERMARK,
    DEFAULT_LOW_WATERMARK,
    SaturationGate,
)


def test_defaults_are_90_and_70() -> None:
    g = SaturationGate()
    assert g.high == DEFAULT_HIGH_WATERMARK == 0.90
    assert g.low == DEFAULT_LOW_WATERMARK == 0.70
    assert g.saturated is False


def test_invalid_thresholds_raise() -> None:
    with pytest.raises(ValueError, match="thresholds invalid"):
        SaturationGate(high=0.5, low=0.9)
    with pytest.raises(ValueError, match="thresholds invalid"):
        SaturationGate(high=1.5, low=0.7)
    with pytest.raises(ValueError, match="thresholds invalid"):
        SaturationGate(high=0.9, low=-0.1)


def test_below_high_does_not_flip() -> None:
    g = SaturationGate()
    # 80% utilisation — between low (70%) and high (90%): does not flip.
    assert g.update(in_flight=8, capacity=10) is False
    assert g.update(in_flight=8, capacity=10) is False


def test_at_high_flips_to_saturated() -> None:
    g = SaturationGate()
    assert g.update(in_flight=9, capacity=10) is True
    assert g.saturated is True


def test_drop_to_just_above_low_stays_saturated() -> None:
    g = SaturationGate()
    g.update(in_flight=10, capacity=10)  # latch
    # 75% — above the 70% low threshold: stay saturated.
    assert g.update(in_flight=75, capacity=100) is True


def test_drop_to_low_flips_back() -> None:
    g = SaturationGate()
    g.update(in_flight=10, capacity=10)
    assert g.update(in_flight=7, capacity=10) is False
    assert g.saturated is False


def test_oscillation_around_single_setpoint_does_not_thrash() -> None:
    """Oscillation between 75% and 85% must not flip the gate either way.

    This is the core property the hysteresis guarantees: with a single
    setpoint at, say, 80%, the gate would flap on every small jitter.
    With 90/70 hysteresis, only excursions outside that band flip.
    """
    g = SaturationGate()
    for _ in range(50):
        # 85%: between low and high, no flip in either direction.
        assert g.update(in_flight=85, capacity=100) is False
        assert g.update(in_flight=75, capacity=100) is False
    assert g.saturated is False

    # Push above high once, then oscillate inside the band.
    assert g.update(in_flight=90, capacity=100) is True
    for _ in range(50):
        assert g.update(in_flight=85, capacity=100) is True
        assert g.update(in_flight=75, capacity=100) is True
    assert g.saturated is True


def test_zero_capacity_returns_current_state_unchanged() -> None:
    g = SaturationGate()
    assert g.update(in_flight=10, capacity=0) is False
    g.saturated = True
    assert g.update(in_flight=10, capacity=0) is True


def test_negative_in_flight_clamps_to_zero() -> None:
    g = SaturationGate()
    g.saturated = True
    # Logic bug elsewhere should not crash the loop.
    assert g.update(in_flight=-5, capacity=10) is False


def test_custom_thresholds() -> None:
    g = SaturationGate(high=0.8, low=0.5)
    assert g.update(in_flight=7, capacity=10) is False
    assert g.update(in_flight=8, capacity=10) is True
    assert g.update(in_flight=6, capacity=10) is True
    assert g.update(in_flight=5, capacity=10) is False
