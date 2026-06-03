"""Observability module for SIE Server.

Provides Prometheus metrics and OpenTelemetry tracing.
"""

from sie_server.observability.metrics import (
    MODEL_LOADED,
    MODEL_MEMORY_BYTES,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
    record_request,
    set_model_loaded,
    set_model_memory,
)

__all__ = [
    "MODEL_LOADED",
    "MODEL_MEMORY_BYTES",
    "REQUESTS_TOTAL",
    "REQUEST_DURATION",
    "record_request",
    "set_model_loaded",
    "set_model_memory",
]
