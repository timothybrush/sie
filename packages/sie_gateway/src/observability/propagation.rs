//! W3C Trace Context propagation helpers.
//!
//! The gateway sits at two propagation boundaries:
//!
//! 1. **Inbound HTTP → in-process Context**. The client sends a
//!    `traceparent` (and optional `tracestate`) header. We extract it
//!    via the globally-installed [`TextMapPropagator`] and attach the
//!    resulting context to a new gateway-side span.
//!
//! 2. **In-process Context → outbound NATS envelope**. Before
//!    publishing the [`crate::queue::publisher::WorkItem`] we
//!    serialise the current context (the gateway span) back into the
//!    two header strings and write them into the envelope. The
//!    worker re-extracts on the other side.
//!
//! Both directions go through the **same** [`TextMapPropagator`]
//! instance — the global W3C propagator installed in
//! [`super::tracing::init_tracing`] — so the wire format is
//! guaranteed identical in both directions.

use std::collections::HashMap;

use axum::http::HeaderMap;
use opentelemetry::global;
use opentelemetry::propagation::{Extractor, Injector};
use opentelemetry::Context;

/// Adapter exposing an axum [`HeaderMap`] as an OTel [`Extractor`].
///
/// Borrowed view — no allocations beyond a transient `Vec<&str>`
/// inside `keys()`.
struct HeaderMapExtractor<'a>(&'a HeaderMap);

impl<'a> Extractor for HeaderMapExtractor<'a> {
    fn get(&self, key: &str) -> Option<&str> {
        self.0.get(key).and_then(|v| v.to_str().ok())
    }

    fn keys(&self) -> Vec<&str> {
        self.0.keys().map(|k| k.as_str()).collect()
    }
}

/// Adapter exposing a `HashMap<String, String>` as an OTel
/// [`Injector`]. The hashmap is the propagator-friendly intermediate
/// form for envelope injection: the propagator writes the two W3C
/// headers as `String`s, and we lift them out for the typed envelope
/// fields.
struct HashMapInjector<'a>(&'a mut HashMap<String, String>);

impl<'a> Injector for HashMapInjector<'a> {
    fn set(&mut self, key: &str, value: String) {
        self.0.insert(key.to_string(), value);
    }
}

/// Extract a parent [`Context`] from inbound HTTP request headers.
///
/// Returns the empty (root) context when no `traceparent` header is
/// present, matching W3C semantics: callers should still open their
/// own span; it will simply not be a child of any external trace.
pub fn extract_context_from_headers(headers: &HeaderMap) -> Context {
    global::get_text_map_propagator(|propagator| propagator.extract(&HeaderMapExtractor(headers)))
}

/// Serialise the current OTel [`Context`] (active span) back into the
/// two W3C strings.
///
/// Returns `(traceparent, tracestate)`. Both are `None` when no span
/// is currently active or the propagator chose to skip them (e.g.
/// invalid context). The pair is the on-the-wire shape the work
/// envelope carries — keep `tracestate` independent of `traceparent`
/// because vendor state (`tracestate`) can exist even on a
/// just-created root context. (In practice both are populated or
/// both are empty; the splitting is for clarity.)
pub fn inject_current_context() -> (Option<String>, Option<String>) {
    inject_context(&Context::current())
}

/// Variant of [`inject_current_context`] that takes an explicit
/// context — useful when the caller has already detached / re-
/// attached a span and wants to inject the not-currently-attached
/// parent.
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
    use axum::http::{HeaderName, HeaderValue};
    use opentelemetry::trace::{Span as _, TraceContextExt, Tracer, TracerProvider as _};
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::TracerProvider;

    /// Install the propagator once per test process. Each test path
    /// is independent; installing twice is harmless (the global
    /// slot accepts the new value), but the propagator must be live
    /// before extract/inject for the wire format to match.
    fn install_propagator() {
        opentelemetry::global::set_text_map_propagator(TraceContextPropagator::new());
    }

    #[test]
    fn extract_round_trip_returns_same_traceparent() {
        install_propagator();
        let mut headers = HeaderMap::new();
        let tp = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01";
        headers.insert(
            HeaderName::from_static("traceparent"),
            HeaderValue::from_static(tp),
        );

        let cx = extract_context_from_headers(&headers);
        // Attach the context, create a child span; the child's parent
        // should match the extracted trace id.
        let provider = TracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let span = tracer.start_with_context("child", &cx);
        let span_cx = span.span_context().clone();
        assert!(span_cx.is_valid(), "child span context must be valid");
        let trace_id = format!("{:032x}", span_cx.trace_id());
        assert_eq!(
            trace_id, "0af7651916cd43dd8448eb211c80319c",
            "child must inherit the extracted trace id"
        );
    }

    #[test]
    fn inject_with_no_active_span_returns_none_pair() {
        install_propagator();
        // Root context with no span — propagator should skip both
        // headers.
        let (tp, ts) = inject_context(&Context::new());
        assert!(tp.is_none(), "no active span ⇒ no traceparent");
        assert!(ts.is_none(), "no active span ⇒ no tracestate");
    }

    #[test]
    fn inject_with_active_span_yields_w3c_traceparent() {
        install_propagator();
        let provider = TracerProvider::builder().build();
        let tracer = provider.tracer("test");
        let span = tracer.start("parent");
        // `TraceContextExt::with_span` is the 0.24 API for building a
        // context that carries a span; the older `current_with_span`
        // free function was retired with 0.23.
        let cx = Context::current().with_span(span);
        let (tp, _ts) = inject_context(&cx);
        let tp = tp.expect("active span should inject a traceparent");
        // W3C format: version-traceid-spanid-flags (2-32-16-2 hex).
        let parts: Vec<&str> = tp.split('-').collect();
        assert_eq!(
            parts.len(),
            4,
            "traceparent must be 4 dash-separated fields: {tp}"
        );
        assert_eq!(parts[0].len(), 2, "version field 2 hex chars");
        assert_eq!(parts[1].len(), 32, "trace_id field 32 hex chars");
        assert_eq!(parts[2].len(), 16, "span_id field 16 hex chars");
        assert_eq!(parts[3].len(), 2, "flags field 2 hex chars");
    }
}
