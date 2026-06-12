"""Cheap character-based token estimators.

Centralises the ``chars_per_token`` constant and ``len(text) / chars_per_token``
arithmetic so all consumers (admission control, preprocessor cost
estimation, calibration tooling) agree on the proxy. Real tokenization
still runs upstream where the exact count matters (the chat
context-length guard, the SGLang adapter); these helpers are for cost
proxies and pre-flight reservations.
"""

from __future__ import annotations

# Mirrors the ``CharCountPreprocessor`` default in
# :mod:`sie_server.core.preprocessor.text`. Keep these two values in
# lockstep when the KV calibration follow-up revises the proxy.
DEFAULT_CHARS_PER_TOKEN = 4.0


def estimate_tokens_from_chars(text: str, chars_per_token: float = DEFAULT_CHARS_PER_TOKEN) -> int:
    """Estimate token count from character count.

    Empty input returns ``0`` — admission control was previously
    charging 1 token per empty request, which silently leaked KV
    budget across high-volume callers that send sentinel pings. For
    non-empty inputs we still floor at 1 because rounding ``len/CPT``
    toward zero would under-count short inputs (e.g. a single
    BOM-only string).
    """
    if chars_per_token <= 0:
        msg = "chars_per_token must be positive"
        raise ValueError(msg)
    if not text:
        return 0
    return max(1, int(len(text) / chars_per_token))
