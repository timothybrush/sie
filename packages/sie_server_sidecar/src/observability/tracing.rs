//! OpenTelemetry tracer-provider setup for the sidecar.
//!
//! Mirrors `packages/sie_gateway/src/observability/tracing.rs`. The
//! OTLP exporter is enabled only when `SIE_TRACING_ENABLED` is truthy
//! and an OTLP endpoint is configured via
//! `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` or `OTEL_EXPORTER_OTLP_ENDPOINT`.
//! The W3C [`TraceContextPropagator`] is installed globally regardless
//! of that exporter gate so that the inbound gateway
//! `traceparent` (carried on the work envelope) still propagates
//! through to the adapter worker. Without an exporter the sidecar
//! itself records no spans, but the IDs continue to flow — the
//! `into_run_batch_item_with_trace` fallback in the dispatcher copies
//! the gateway context onto the wire items unchanged.
//!
//! [`TraceContextPropagator`]: opentelemetry_sdk::propagation::TraceContextPropagator

use std::env;
use std::sync::OnceLock;
use std::time::Duration;

use opentelemetry::global;
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig;
use opentelemetry_sdk::propagation::TraceContextPropagator;
use opentelemetry_sdk::trace::{SdkTracerProvider, Tracer};
use opentelemetry_sdk::Resource;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};

/// Service name used when `OTEL_SERVICE_NAME` is not set.
const DEFAULT_SERVICE_NAME: &str = "sie-worker-sidecar";
static TRACER_PROVIDER: OnceLock<SdkTracerProvider> = OnceLock::new();

/// Bounded flush deadline (ms) so process exit can't stall on an unreachable collector.
const TRACING_SHUTDOWN_TIMEOUT_MS: u64 = 3_000;

/// Initialise OpenTelemetry + tracing-subscriber for the sidecar.
///
/// Pipeline:
///   1. Install the global W3C [`TraceContextPropagator`] so the
///      inbound `traceparent` / `tracestate` (read off the work
///      envelope) extract into an `opentelemetry::Context`. **Always
///      runs**, even without an exporter — propagation is the
///      load-bearing piece for worker-side correlation.
///   2. If `SIE_TRACING_ENABLED` is truthy and `OTEL_EXPORTER_OTLP_ENDPOINT`
///      (or the trace-specific `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT`) is
///      set, build a [`SdkTracerProvider`] with the OTLP gRPC exporter, attach a
///      [`tracing_opentelemetry::OpenTelemetryLayer`] so the
///      `sidecar.dispatch` `tracing::*` span becomes an OTel span,
///      and set the provider as global. Otherwise, only the JSON fmt layer is
///      installed while the propagator remains active.
///
/// Logs are always emitted as JSON, matching the sidecar's prior
/// behaviour. `RUST_LOG` is honoured via `EnvFilter`.
pub fn init_tracing() {
    // Idempotency guard. The subscriber's `.init()` panics on a second
    // call ("a global default trace dispatcher has already been set").
    // The integration tests spin the sidecar up in-process across
    // multiple cases and would otherwise abort on the second case.
    use std::sync::atomic::{AtomicBool, Ordering};
    static INIT_GUARD: AtomicBool = AtomicBool::new(false);
    if INIT_GUARD.swap(true, Ordering::SeqCst) {
        tracing::debug!("init_tracing called more than once; skipping subsequent init");
        return;
    }

    // Step 1: always install the propagator. Even without an OTLP
    // exporter the sidecar needs to *extract* the inbound gateway
    // context and *inject* it onto the wire items so the worker side
    // (which runs the heavy adapter call) can continue the trace.
    global::set_text_map_propagator(TraceContextPropagator::new());

    let endpoint = if sie_tracing_enabled() {
        cleaned_env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            .or_else(|| cleaned_env("OTEL_EXPORTER_OTLP_ENDPOINT"))
    } else {
        None
    };

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Try to build the OTel tracer. On any failure the sidecar
    // continues with the fmt-only subscriber (and propagator-only OTel
    // state) — operators get logs and worker-side trace correlation,
    // just no `sidecar.dispatch` spans.
    let tracer = endpoint.as_deref().and_then(|ep| match init_tracer(ep) {
        Ok(t) => Some(t),
        Err(e) => {
            eprintln!("warn: failed to init OTLP exporter ({e}); continuing without exporter");
            None
        }
    });

    // Place the (boxed) OTel layer FIRST, then the fmt and filter
    // layers — see the gateway's note: boxing against the inner
    // `Registry` is what makes `Option<L>: Layer<S>` compose; OTel-last
    // fails the `Layer<Layered<…>>` bound.
    let otel_layer_boxed: Option<Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync>> =
        tracer.map(|t| {
            let l: Box<dyn Layer<tracing_subscriber::Registry> + Send + Sync> =
                Box::new(tracing_opentelemetry::layer().with_tracer(t));
            l
        });

    tracing_subscriber::registry()
        .with(otel_layer_boxed)
        .with(filter)
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    if let Some(ep) = endpoint {
        tracing::info!(endpoint = %ep, "OpenTelemetry tracing initialized");
    } else {
        tracing::debug!(
            "SIE_TRACING_ENABLED not truthy or OTLP endpoint not set; W3C propagator installed (no exporter)"
        );
    }
}

fn tracing_flag_set(raw: Option<&str>) -> bool {
    raw.is_some_and(|value| {
        let value = value.trim();
        value.eq_ignore_ascii_case("true") || value == "1" || value.eq_ignore_ascii_case("yes")
    })
}

fn sie_tracing_enabled() -> bool {
    let raw = env::var("SIE_TRACING_ENABLED").ok();
    tracing_flag_set(raw.as_deref())
}

/// Read an env var, trimming surrounding whitespace and treating a
/// whitespace-only value as absent so it can't shadow a valid fallback.
fn cleaned_env(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn init_tracer(endpoint: &str) -> Result<Tracer, String> {
    let service_name =
        env::var("OTEL_SERVICE_NAME").unwrap_or_else(|_| DEFAULT_SERVICE_NAME.to_string());

    let exporter = opentelemetry_otlp::SpanExporter::builder()
        .with_tonic()
        .with_endpoint(endpoint)
        .build()
        .map_err(|e| format!("build OTLP span exporter: {e}"))?;

    let provider = SdkTracerProvider::builder()
        .with_resource(Resource::builder().with_service_name(service_name).build())
        .with_batch_exporter(exporter)
        .build();
    let tracer = provider.tracer("sie-worker-sidecar");
    global::set_tracer_provider(provider.clone());
    let _ = TRACER_PROVIDER.set(provider);
    Ok(tracer)
}

/// Graceful shutdown — flush any pending spans.
///
/// Called from `main.rs` on the way out so the OTLP exporter has a
/// chance to drain its batch before the process exits.
pub fn shutdown_tracing() {
    if let Some(provider) = TRACER_PROVIDER.get() {
        let _ = provider.shutdown_with_timeout(Duration::from_millis(TRACING_SHUTDOWN_TIMEOUT_MS));
    }
}

#[cfg(test)]
mod tests {
    use super::tracing_flag_set;

    #[test]
    fn tracing_flag_set_accepts_only_positive_truthy_values() {
        assert!(tracing_flag_set(Some("true")));
        assert!(tracing_flag_set(Some("1")));
        assert!(tracing_flag_set(Some("yes")));
        assert!(tracing_flag_set(Some("TRUE")));
        assert!(tracing_flag_set(Some(" true ")));

        assert!(!tracing_flag_set(Some("false")));
        assert!(!tracing_flag_set(Some("")));
        assert!(!tracing_flag_set(Some("   ")));
        assert!(!tracing_flag_set(None));
    }
}
