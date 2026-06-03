//! Pure classification for `SIE_GATEWAY_HEALTH_MODE` vs NATS connectivity.
//! Side effects and manager startup live in `main.rs`.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthModeDisposition {
    /// Default supported path: WebSocket worker health.
    WebSocketDefault,
    /// Subscribe to `sie.health.>` when URL + client exist.
    TryNats,
    /// `nats` requested but `SIE_NATS_URL` is unset or empty.
    FallbackWebSocketMissingNatsUrl,
    /// `nats` requested but no usable NATS client is available yet.
    FallbackWebSocketNoNatsClient,
    /// Unknown mode string; fall back to WebSocket like `main` has always done.
    FallbackWebSocketUnsupported,
}

/// Classify how the gateway should initialize worker health transport before any
/// subscription side effects.
pub fn health_mode_disposition(
    mode: &str,
    nats_url_nonempty: bool,
    nats_client_available: bool,
) -> HealthModeDisposition {
    match mode {
        "ws" => HealthModeDisposition::WebSocketDefault,
        "nats" if !nats_url_nonempty => HealthModeDisposition::FallbackWebSocketMissingNatsUrl,
        "nats" if !nats_client_available => HealthModeDisposition::FallbackWebSocketNoNatsClient,
        "nats" => HealthModeDisposition::TryNats,
        _ => HealthModeDisposition::FallbackWebSocketUnsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ws_always_default() {
        assert_eq!(
            health_mode_disposition("ws", false, false),
            HealthModeDisposition::WebSocketDefault
        );
        assert_eq!(
            health_mode_disposition("ws", true, true),
            HealthModeDisposition::WebSocketDefault
        );
    }

    #[test]
    fn nats_requires_url() {
        assert_eq!(
            health_mode_disposition("nats", false, true),
            HealthModeDisposition::FallbackWebSocketMissingNatsUrl
        );
    }

    #[test]
    fn nats_requires_client_when_url_present() {
        assert_eq!(
            health_mode_disposition("nats", true, false),
            HealthModeDisposition::FallbackWebSocketNoNatsClient
        );
    }

    #[test]
    fn nats_with_url_and_client_is_ready() {
        assert_eq!(
            health_mode_disposition("nats", true, true),
            HealthModeDisposition::TryNats
        );
    }

    #[test]
    fn unknown_mode_falls_back() {
        assert_eq!(
            health_mode_disposition("grpc", true, true),
            HealthModeDisposition::FallbackWebSocketUnsupported
        );
    }
}
