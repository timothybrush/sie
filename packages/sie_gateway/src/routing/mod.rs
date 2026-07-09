//! Direct-dispatch routing.
//!
//! Public API:
//! - [`pick_worker`]: pressure-aware selection with HRW tie-breaks for a `(model, pool)` ring.
//! - [`key::resolve`]: priority-ordered routing-key extraction + xxh3 hashing.
//! - [`fmt_key_hash`]: privacy-safe log formatter (`xxh:XXXXXXXX`).
//! - [`log_raw_keys_enabled`] / [`warn_if_raw_logging_enabled`]: opt-in raw-key logging.
//!
//! **Privacy contract.** Raw routing-key strings are never logged at the
//! default tracing level. All code paths in this module that surface a key
//! must go through [`fmt_key_hash`] unless [`log_raw_keys_enabled`] returns
//! `true` (gated by `SIE_ROUTING_LOG_RAW=1`, with a loud startup warning).
//!
//! **Compile-time gate: `raw-routing-logs` Cargo feature.** The
//! `SIE_ROUTING_LOG_RAW=1` runtime opt-in is itself gated by the
//! `raw-routing-logs` Cargo feature (off by default). In release builds
//! without `raw-routing-logs`, raw keys cannot appear in logs regardless
//! of env var — [`log_raw_keys_enabled`] is a stub that always returns
//! `false`, [`warn_if_raw_logging_enabled`] is a no-op, and the
//! [`key::RoutingKeyResolved::raw_for_debug`] field does not exist on the
//! struct. Production deploys SHOULD build without this feature.

pub mod hrw;
pub mod key;

#[cfg(test)]
mod tests;

// Re-export the most-used items so callers can write
// `crate::routing::pick_worker(..)` directly. The submodule paths
// remain available for code that needs the rest of the type surface.
pub use hrw::pick_worker;

/// Whether a direct-dispatched generation request should skip the
/// first-chunk timeout republish because the pool-fallback lane had no
/// alternate worker at dispatch time.
pub(crate) fn suppress_first_chunk_republish_for_lane(
    was_direct_dispatched: bool,
    pool_fallback_lane_worker_count: usize,
) -> bool {
    was_direct_dispatched && pool_fallback_lane_worker_count <= 1
}

/// Environment variable that opts in to raw routing-key logging.
/// Default (unset / not "1") emits only the `xxh:` prefix.
///
/// Only consulted when the `raw-routing-logs` Cargo feature is enabled;
/// the const is always exported so tests and ops tooling can reference
/// the name regardless of the feature state.
#[allow(dead_code)]
pub const RAW_LOGGING_ENV: &str = "SIE_ROUTING_LOG_RAW";

/// Returns `true` iff the operator has opted in to raw routing-key logging.
///
/// When the `raw-routing-logs` Cargo feature is disabled (the default and
/// the recommended production configuration), this function is a stub that
/// always returns `false` — `SIE_ROUTING_LOG_RAW` is not consulted.
#[cfg(feature = "raw-routing-logs")]
pub fn log_raw_keys_enabled() -> bool {
    std::env::var(RAW_LOGGING_ENV).ok().as_deref() == Some("1")
}

/// Stub: see feature-on variant above. Kept callable so consumers don't
/// need their own cfg gates.
#[cfg(not(feature = "raw-routing-logs"))]
#[allow(dead_code)]
pub fn log_raw_keys_enabled() -> bool {
    false
}

/// Privacy-safe log formatter for a 64-bit routing-key hash.
///
/// Returns `xxh:` followed by the upper 32 bits in lowercase hex. The
/// upper 32 bits are sufficient to disambiguate distinct keys in logs
/// without leaking enough entropy to recover the original string.
pub fn fmt_key_hash(hash: u64) -> String {
    format!("xxh:{:08x}", (hash >> 32) as u32)
}

/// Startup-time loud warning when raw-key logging is enabled. Called
/// once from `main::run_server`. Also asserted by routing tests.
///
/// When the `raw-routing-logs` Cargo feature is disabled, this function is
/// a no-op — callers do not need to gate the call site themselves.
#[cfg(feature = "raw-routing-logs")]
pub fn warn_if_raw_logging_enabled() {
    if log_raw_keys_enabled() {
        tracing::warn!(
            env = RAW_LOGGING_ENV,
            "SIE_ROUTING_LOG_RAW=1 — raw routing keys WILL appear in logs. Disable in production."
        );
    }
}

#[cfg(not(feature = "raw-routing-logs"))]
pub fn warn_if_raw_logging_enabled() {}
