//! OpenTelemetry tracer-provider setup for the gateway.
//!
//! All knobs are read from the standard OTel environment variables
//! (`OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_SERVICE_NAME`,
//! `OTEL_TRACES_SAMPLER`, …) — no SIE-specific env vars are
//! introduced. The W3C [`TraceContextPropagator`] is installed
//! globally even when no exporter is configured so that inbound
//! `traceparent` headers still propagate through to the worker via
//! the JetStream work envelope. Without an exporter the gateway
//! itself records no spans, but the IDs continue to flow.
//!
//! [`TraceContextPropagator`]: opentelemetry_sdk::propagation::TraceContextPropagator

use std::env;

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry::KeyValue;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::Tracer;
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Service name used when `OTEL_SERVICE_NAME` is not set.
const DEFAULT_SERVICE_NAME: &str = "sie-gateway";

/// Initialise OpenTelemetry + tracing-subscriber for the gateway.
///
/// Pipeline:
///   1. Install the global W3C [`TraceContextPropagator`] so the
///      `traceparent` / `tracestate` headers extract into a
///      `opentelemetry::Context`. **Always runs**, even without
///      an exporter — propagation is the load-bearing piece for
///      worker-side correlation.
///   2. If `OTEL_EXPORTER_OTLP_ENDPOINT` (or the trace-specific
///      `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is set, build a
///      [`opentelemetry_sdk::trace::TracerProvider`] with the OTLP
///      gRPC exporter, attach a
///      [`tracing_opentelemetry::OpenTelemetryLayer`] so existing
///      `tracing::*` spans become OTel spans, and set the provider
///      as global. If not set, only the tracing-subscriber fmt
///      layer is installed (legacy behaviour preserved).
pub fn init_tracing(level: &str, json: bool) {
    // Idempotency guard. The subscriber's ``.init()`` panics on a
    // second call ("a global default trace dispatcher has already
    // been set"). Tests that spin up the gateway in-process across
    // multiple cases would otherwise abort on the second case.
    use std::sync::atomic::{AtomicBool, Ordering};
    static INIT_GUARD: AtomicBool = AtomicBool::new(false);
    if INIT_GUARD.swap(true, Ordering::SeqCst) {
        tracing::debug!("init_tracing called more than once; skipping subsequent init");
        return;
    }

    // Step 1: always install the propagator. Even without an OTLP
    // exporter the gateway needs to *extract* inbound trace headers
    // and *inject* them into the work envelope so the worker side
    // (which runs the heavy adapter call) can continue the trace.
    global::set_text_map_propagator(TraceContextPropagator::new());

    let endpoint = env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .or_else(|_| env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
        .ok()
        .filter(|s| !s.is_empty());

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        let level_str = match level.to_lowercase().as_str() {
            "debug" => "debug",
            "warn" | "warning" => "warn",
            "error" => "error",
            _ => "info",
        };
        EnvFilter::new(level_str)
    });

    // Try to build the OTel tracer. On any failure the gateway
    // continues with the fmt-only subscriber (and propagator-only
    // OTel state) — operators get logs and worker-side trace
    // correlation, just no spans on the gateway itself.
    let tracer = endpoint.as_deref().and_then(|ep| match init_tracer(ep) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("warn: failed to init OTLP exporter ({e}); continuing without exporter");
            None
        }
    });

    // Place the (boxed) OTel layer FIRST, then the fmt and filter
    // layers. `OpenTelemetryLayer<S, T>: Layer<S>` so when we box it
    // against the inner `Registry` we get a `Box<dyn Layer<Registry>
    // + Send + Sync>` that `Option<L>: Layer<S> where L: Layer<S>`
    // composes cleanly. Doing it the other way round (OTel last) hits
    // a wall because the boxed `dyn Layer<Registry>` does not satisfy
    // `Layer<Layered<...>>`.
    let otel_layer_boxed: Option<Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync>> =
        tracer.map(|t| {
            let l: Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync> =
                Box::new(tracing_opentelemetry::layer().with_tracer(t));
            l
        });

    let base = tracing_subscriber::registry().with(otel_layer_boxed);
    if json {
        base.with(filter)
            .with(tracing_subscriber::fmt::layer().json())
            .init();
    } else {
        base.with(filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }

    if let Some(ep) = endpoint {
        tracing::info!(endpoint = %ep, "OpenTelemetry tracing initialized");
    } else {
        tracing::debug!(
            "OTEL_EXPORTER_OTLP_ENDPOINT not set; W3C propagator installed (no exporter)"
        );
    }
}

fn init_tracer(endpoint: &str) -> Result<Tracer, String> {
    let service_name =
        env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());

    let provider = opentelemetry_otlp::new_pipeline()
        .tracing()
        .with_trace_config(opentelemetry_sdk::trace::Config::default().with_resource(
            Resource::new(vec![KeyValue::new("service.name", service_name)]),
        ))
        .with_exporter(
            opentelemetry_otlp::new_exporter()
                .tonic()
                .with_endpoint(endpoint),
        )
        .install_batch(opentelemetry_sdk::runtime::Tokio)
        .map_err(|e| format!("install OTLP batch pipeline: {e}"))?;

    let tracer = provider.tracer("sie-gateway");
    // `install_batch` already sets the provider globally inside the
    // 0.17 pipeline; ensure the *exact* provider we built is the one
    // future `global::tracer(...)` calls see (no-op when identical).
    global::set_tracer_provider(provider);
    Ok(tracer)
}

/// Graceful shutdown — flush any pending spans.
///
/// Called from `main.rs` on the way out so the OTLP exporter has a
/// chance to drain its batch before the process exits.
pub fn shutdown_tracing() {
    global::shutdown_tracer_provider();
}
