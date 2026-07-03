//! W3C Trace Context propagation helpers for the sidecar.
//!
//! The sidecar sits between two propagation boundaries on the queue
//! hop:
//!
//! 1. **Inbound envelope → in-process Context**. The gateway serialised
//!    its span into the work envelope's `traceparent` / `tracestate`
//!    strings ([`crate::work_types::WorkItem`]). We extract them via
//!    the globally-installed propagator and parent the
//!    `sidecar.dispatch` span on the result.
//!
//! 2. **In-process Context → outbound IPC item**. Before sending the
//!    [`crate::ipc_types::RunBatchItem`] to the Python worker we
//!    serialise the active `sidecar.dispatch` span back into the two
//!    header strings and write them onto the wire item, so the
//!    worker's `worker.run_batch` span nests under the sidecar span.
//!
//! Both directions go through the **same** propagator instance — the
//! global W3C propagator installed in [`super::tracing::init_tracing`]
//! — so the wire format is identical in both directions. Unlike the
//! gateway, the inbound carrier is a pair of W3C *strings* (not an
//! HTTP `HeaderMap`), so the extractor here is `HashMap`-backed.

use std::collections::HashMap;

use opentelemetry::global;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::trace::{SpanContext, TraceContextExt};
use opentelemetry::Context;

/// Adapter exposing a `HashMap<String, String>` as an OTel
/// [`Extractor`]. The inbound `traceparent` / `tracestate` strings off
/// the work envelope are loaded into this map and handed to the
/// propagator.
struct HashMapExtractor<'a>(&'a HashMap<String, String>);

impl Extractor for HashMapExtractor<'_> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).map(String::as_str)
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(String::as_str).collect()
    }
}

/// Adapter exposing a `HashMap<String, String>` as an OTel
/// [`Injector`]. The propagator writes the two W3C headers as
/// `String`s and we lift them out for the typed wire fields.
struct HashMapInjector<'a>(&'a mut HashMap<String, String>);

impl Injector for HashMapInjector<'_> {
    fn set(&mut self, key: &str, value: String) {
        self.0.insert(key.to_string(), value);
    }
}

/// Build the W3C carrier map from the optional envelope strings.
fn carrier_from_w3c(
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) -> HashMap<String, String> {
    let mut carrier: HashMap<String, String> = HashMap::with_capacity(2);
    if let Some(tp) = traceparent {
        carrier.insert("traceparent".to_string(), tp.to_string());
    }
    if let Some(ts) = tracestate {
        carrier.insert("tracestate".to_string(), ts.to_string());
    }
    carrier
}

/// Extract a parent [`Context`] from the inbound envelope's W3C
/// strings.
///
/// Returns the empty (root) context when no `traceparent` is present,
/// matching W3C semantics: callers should still open their own span;
/// it will simply not be a child of any external trace.
pub fn extract_context_from_w3c(traceparent: Option<&str>, tracestate: Option<&str>) -> Context {
    let carrier = carrier_from_w3c(traceparent, tracestate);
    global::get_text_map_propagator(|propagator| propagator.extract(&HashMapExtractor(&carrier)))
}

/// Extract the inbound [`SpanContext`] from the envelope's W3C
/// strings, for use as a span *link* (batch coalescing means a single
/// `sidecar.dispatch` span can parent items from several gateway
/// traces; the non-primary parents are recorded as links).
///
/// Returns `None` when the strings are absent or yield an invalid
/// context.
pub fn remote_span_context(
    traceparent: Option<&str>,
    tracestate: Option<&str>,
) -> Option<SpanContext> {
    let cx = extract_context_from_w3c(traceparent, tracestate);
    let span_cx = cx.span().span_context().clone();
    span_cx.is_valid().then_some(span_cx)
}

/// Serialise the current OTel [`Context`] (active span) back into the
/// two W3C strings.
///
/// Returns `(traceparent, tracestate)`. Both are `None` when no span
/// is currently active (e.g. no OTLP exporter configured, so the
/// `sidecar.dispatch` `tracing` span has no bridged OTel span) — the
/// caller then falls back to the gateway context on the envelope.
pub fn inject_current_context() -> (Option<String>, Option<String>) {
    inject_context(&Context::current())
}

/// Variant of [`inject_current_context`] taking an explicit context.
pub fn inject_context(cx: &Context) -> (Option<String>, Option<String>) {
    let mut carrier: HashMap<String, String> = HashMap::with_capacity(2);
    global::get_text_map_propagator(|propagator| {
        propagator.inject_context(cx, &mut HashMapInjector(&mut carrier));
    });
    let traceparent = carrier.remove("traceparent");
    let tracestate = carrier.remove("tracestate");
    (traceparent, tracestate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry::trace::{Span as _, Tracer, TracerProvider as _};
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::SdkTracerProvider;

    /// Install the propagator once per test path. Installing twice is
    /// harmless (the global slot accepts the new value), but it must be
    /// live before extract/inject for the wire format to match.
    fn install_propagator() {
        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
    }

    const SAMPLE_TP: &str = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";

    #[test]
    fn extract_w3c_inherits_trace_id() {
        install_propagator();
        let cx = extract_context_from_w3c(Some(SAMPLE_TP), None);
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let span = tracer.start_with_context("child", &cx);
        let span_cx = span.span_context().clone();
        assert!(span_cx.is_valid(), "child span context must be valid");
        assert_eq!(
            format!("{:032x}", span_cx.trace_id()),
            "0af7651916cd43dd8448eb211c80319c",
            "child must inherit the extracted trace id",
        );
    }

    #[test]
    fn extract_w3c_absent_is_root_context() {
        install_propagator();
        let cx = extract_context_from_w3c(None, None);
        assert!(
            !cx.span().span_context().is_valid(),
            "no traceparent ⇒ root (invalid) span context",
        );
    }

    #[test]
    fn remote_span_context_present_and_absent() {
        install_propagator();
        let sc = remote_span_context(Some(SAMPLE_TP), None).expect("valid traceparent ⇒ Some");
        assert_eq!(
            format!("{:032x}", sc.trace_id()),
            "0af7651916cd43dd8448eb211c80319c",
        );
        assert_eq!(format!("{:016x}", sc.span_id()), "b7ad6b7169203331");
        assert!(remote_span_context(None, None).is_none());
    }

    #[test]
    fn inject_with_no_active_span_returns_none_pair() {
        install_propagator();
        let (tp, ts) = inject_context(&Context::new());
        assert!(tp.is_none(), "no active span ⇒ no traceparent");
        assert!(ts.is_none(), "no active span ⇒ no tracestate");
    }

    #[test]
    fn inject_with_active_span_yields_w3c_traceparent() {
        install_propagator();
        let provider = SdkTracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let span = tracer.start("parent");
        let cx = Context::current().with_span(span);
        let (tp, _ts) = inject_context(&cx);
        let tp = tp.expect("active span should inject a traceparent");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "traceparent must be 4 fields: {tp}");
        assert_eq!(parts[1].len(), 32, "trace_id field 32 hex chars");
        assert_eq!(parts[2].len(), 16, "span_id field 16 hex chars");
    }

    const INBOUND_TRACE: &str = "0af7651916cd43dd8448eb211c80319c";
    const INBOUND_SPAN: &str = "b7ad6b7169203331";

    /// The core nesting contract: with the OTel layer active (i.e. an
    /// exporter is configured), a `sidecar.dispatch` span parented on
    /// the inbound gateway context injects a *new* span under the
    /// *same* trace — so `worker.run_batch` nests under the sidecar
    /// span, which nests under the gateway span.
    #[test]
    fn sidecar_span_injects_new_span_under_inbound_trace() {
        use tracing_subscriber::layer::SubscriberExt;
        install_propagator();
        let provider = SdkTracerProvider::builder().build();
        let subscriber = tracing_subscriber::registry()
            .with(tracing_opentelemetry::layer().with_tracer(provider.tracer("test")));

        let inbound_tp = format!("00-{INBOUND_TRACE}-{INBOUND_SPAN}-01");
        let (tp, _ts) = tracing::subscriber::with_default(subscriber, || {
            use tracing_opentelemetry::OpenTelemetrySpanExt;
            let span = tracing::info_span!("sidecar.dispatch");
            let _ = span.set_parent(extract_context_from_w3c(Some(&inbound_tp), None));
            span.in_scope(inject_current_context)
        });

        let tp = tp.expect("active sidecar span should inject a traceparent");
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(parts.len(), 4, "traceparent must be 4 fields: {tp}");
        assert_eq!(
            parts[1], INBOUND_TRACE,
            "sidecar span must continue the inbound trace",
        );
        assert_ne!(
            parts[2], INBOUND_SPAN,
            "sidecar span must have its own span id (the middle hop)",
        );
    }

    /// Degrade-gracefully parity: without the OTel layer (no exporter),
    /// the `sidecar.dispatch` span has no bridged OTel span, so
    /// injection yields `(None, None)` and the dispatcher falls back to
    /// the gateway context — `gateway → worker` linkage is preserved,
    /// no sidecar span is emitted.
    #[test]
    fn no_otel_layer_injects_nothing() {
        install_propagator();
        let subscriber = tracing_subscriber::registry();
        let (tp, ts) = tracing::subscriber::with_default(subscriber, || {
            let span = tracing::info_span!("sidecar.dispatch");
            span.in_scope(inject_current_context)
        });
        assert!(tp.is_none(), "no exporter/layer ⇒ no sidecar traceparent");
        assert!(ts.is_none(), "no exporter/layer ⇒ no sidecar tracestate");
    }
}
