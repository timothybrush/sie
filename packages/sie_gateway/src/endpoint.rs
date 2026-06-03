#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InferenceEndpoint {
    Encode,
    Score,
    Extract,
    Embeddings,
    Generate,
    Unknown,
}

impl InferenceEndpoint {
    #[cfg(test)]
    pub(crate) const NON_GENERATION_QUEUE_LABELS: [&'static str; 4] =
        ["encode", "score", "extract", "embeddings"];

    pub(crate) fn from_label(label: &str) -> Self {
        match label {
            "encode" => Self::Encode,
            "score" => Self::Score,
            "extract" => Self::Extract,
            "embeddings" => Self::Embeddings,
            "generate" => Self::Generate,
            _ => Self::Unknown,
        }
    }

    pub(crate) fn uses_generation_gateway_tracing(self) -> bool {
        matches!(self, Self::Generate)
    }

    pub(crate) fn injects_queue_trace_context(self) -> bool {
        matches!(self, Self::Generate)
    }
}

#[cfg(test)]
mod tests {
    use super::InferenceEndpoint;

    #[test]
    fn non_generation_queue_labels_do_not_enable_generation_features() {
        for label in InferenceEndpoint::NON_GENERATION_QUEUE_LABELS {
            let endpoint = InferenceEndpoint::from_label(label);
            assert!(!endpoint.uses_generation_gateway_tracing());
            assert!(!endpoint.injects_queue_trace_context());
        }
    }

    #[test]
    fn generate_is_the_only_generation_trace_endpoint() {
        let endpoint = InferenceEndpoint::from_label("generate");
        assert!(endpoint.uses_generation_gateway_tracing());
        assert!(endpoint.injects_queue_trace_context());
    }

    #[test]
    fn unknown_labels_fail_closed_to_non_generation_behavior() {
        let endpoint = InferenceEndpoint::from_label("chat");
        assert_eq!(endpoint, InferenceEndpoint::Unknown);
        assert!(!endpoint.uses_generation_gateway_tracing());
        assert!(!endpoint.injects_queue_trace_context());
    }
}
