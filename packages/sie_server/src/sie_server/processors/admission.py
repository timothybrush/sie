"""KV-cache admission-control helpers.

The streaming processor owns the actual reserve/release state machine
(see :mod:`sie_server.processors.streaming`); this module is the
configuration resolver that decides whether admission is on or off for
a given worker at boot time.

The feature is gated by both a per-profile field
(:attr:`ProfileConfig.admission_enabled`) and an env-var override
(:envvar:`SIE_GENERATION_ADMISSION`). The env var wins when set to
``on`` or ``off``; ``auto`` (or any unset/unknown value) defers to the
profile. A profile that leaves ``admission_enabled`` unset defaults to
**off** for v1 — the calibration follow-up's ablation publishes the empirical default
once the calibration data lands.
"""

from __future__ import annotations

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

_ENV_VAR = "SIE_GENERATION_ADMISSION"

AdmissionMode = Literal["auto", "on", "off"]


def parse_admission_env(value: str | None) -> AdmissionMode:
    """Normalise the ``SIE_GENERATION_ADMISSION`` env value.

    Accepts ``auto`` / ``on`` / ``off`` case-insensitively. Anything
    else (including ``None`` / empty string / unknown values) maps to
    ``auto`` and emits a single debug log so misconfigured deployments
    can be diagnosed without crashing the worker.
    """
    if value is None or not value.strip():
        return "auto"
    normalised = value.strip().lower()
    if normalised == "auto":
        return "auto"
    if normalised == "on":
        return "on"
    if normalised == "off":
        return "off"
    logger.debug(
        "Unknown %s value %r; falling back to 'auto'",
        _ENV_VAR,
        value,
    )
    return "auto"


def resolve_admission_enabled(
    *,
    profile_admission: bool | None,
    env_value: str | None = None,
) -> bool:
    """Resolve the effective admission flag for a worker.

    Precedence (highest first):

    1. Env var ``SIE_GENERATION_ADMISSION=on`` / ``=off`` — operator
       override, regardless of profile.
    2. Profile ``admission_enabled: true|false`` — explicit
       per-profile choice.
    3. Default ``False`` — v1 ships admission off; the calibration
       follow-up's ablation flips this.
    """
    mode = parse_admission_env(env_value if env_value is not None else os.environ.get(_ENV_VAR))
    if mode == "on":
        return True
    if mode == "off":
        return False
    if profile_admission is not None:
        return profile_admission
    return False
