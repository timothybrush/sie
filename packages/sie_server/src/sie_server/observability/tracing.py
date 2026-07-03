"""OpenTelemetry tracing setup for SIE Server.

Provides distributed tracing via OpenTelemetry with OTLP export.
Disabled by default; export requires SIE_TRACING_ENABLED=true (or --tracing)
and an explicit OTLP endpoint.

Configuration via standard OTel environment variables:
- OTEL_SERVICE_NAME: Service name (default: sie-server)
- OTEL_TRACES_EXPORTER: Exporter type (otlp, console, none)
- OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Trace-specific OTLP endpoint URL
- OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL
- OTEL_TRACES_SAMPLER: Sampling strategy
- OTEL_TRACES_SAMPLER_ARG: Sampling rate

See https://opentelemetry.io/docs/concepts/sdk-configuration/ for OTel config.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

# Module-level tracer for manual spans in endpoint handlers
# This is safe to import even when tracing is disabled (returns no-op tracer)
tracer = trace.get_tracer("sie_server")

# Bounded flush deadline (ms) so process exit can't stall on an unreachable collector.
TRACING_SHUTDOWN_TIMEOUT_MS = 3000

# Retained provider handle so shutdown_tracing() can flush deterministically.
_provider: TracerProvider | None = None


def is_tracing_enabled() -> bool:
    """Check if tracing is enabled via environment variable."""
    return os.environ.get("SIE_TRACING_ENABLED", "").strip().lower() in ("true", "1", "yes")


def setup_tracing(app: FastAPI) -> None:
    """Initialize OpenTelemetry tracing for the FastAPI app.

    This function:
    1. Creates a TracerProvider with service resource
    2. Configures OTLP exporter based on environment variables
    3. Auto-instruments FastAPI for HTTP span creation
    4. Handles W3C Trace Context propagation automatically

    Args:
        app: FastAPI application instance to instrument.

    Note:
        No-op unless SIE_TRACING_ENABLED is true and an OTLP endpoint is set.
        All configuration is via standard OTel environment variables.
    """
    if not is_tracing_enabled():
        logger.debug("Tracing disabled (SIE_TRACING_ENABLED not set)")
        return

    # Trim each var and drop whitespace-only values before the fallback so a
    # whitespace-only trace-specific endpoint can't shadow a valid generic one.
    endpoint = (
        os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "").strip()
        or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    )
    if not endpoint:
        logger.warning("SIE_TRACING_ENABLED is set but no OTLP endpoint configured; tracing disabled")
        return

    # Import here to avoid loading OTel machinery when tracing is disabled
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import (
        FastAPIInstrumentor,
    )

    # Get service name from env (OTel standard) or use default
    service_name = os.environ.get("OTEL_SERVICE_NAME", "sie-server")

    logger.info("Initializing OpenTelemetry tracing for service: %s", service_name)

    # Create resource identifying this service
    resource = Resource.create({SERVICE_NAME: service_name})

    # Create tracer provider. We own teardown (see shutdown_tracing), so disable
    # the SDK's unbounded atexit shutdown handler.
    provider = TracerProvider(resource=resource, shutdown_on_exit=False)

    # Configure OTLP exporter. The exporter's per-request timeout is the only
    # effective bound on shutdown flushing in the current SDK, so process exit
    # can't stall on an unreachable collector.
    try:
        exporter = OTLPSpanExporter(endpoint=endpoint, timeout=TRACING_SHUTDOWN_TIMEOUT_MS / 1000)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info("OTLP exporter configured for endpoint: %s", endpoint)
    except Exception:
        logger.exception("Failed to configure OTLP exporter")
        return

    # Set as global tracer provider
    trace.set_tracer_provider(provider)

    # Retain the provider so shutdown_tracing() can flush/shut down deterministically.
    global _provider
    _provider = provider

    # Update module-level tracer to use the new provider
    global tracer
    tracer = trace.get_tracer("sie_server")

    # Auto-instrument FastAPI
    # This creates HTTP spans for all requests and propagates W3C Trace Context
    FastAPIInstrumentor.instrument_app(app)

    logger.info("OpenTelemetry tracing initialized successfully")


def shutdown_tracing() -> None:
    """Bounded shutdown/flush of pending spans so exit can't stall on an unreachable collector.

    The bound comes from the exporter's per-request timeout set in
    ``setup_tracing`` — each export attempt inside ``provider.shutdown()`` is
    capped, so this returns promptly even when the collector is down.
    """
    global _provider
    provider = _provider
    if provider is None:
        return
    try:
        provider.shutdown()
    except Exception:  # best-effort; never block process exit on tracing
        logger.exception("Tracing shutdown failed")
    finally:
        _provider = None


def get_current_trace_id() -> str | None:
    """Get the current trace ID as a hex string.

    Returns:
        Trace ID as 32-character hex string, or None if no active span.
    """
    span = trace.get_current_span()
    if span is None:
        return None

    context = span.get_span_context()
    if context is None or not context.is_valid:
        return None

    return format(context.trace_id, "032x")
