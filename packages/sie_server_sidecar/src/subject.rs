//! NATS subject helpers — model-id normalisation + extraction.
//!
//! Work subject format: `sie.work.{normalized_model_id}.{pool_name}`.

/// Subject token index of the normalized model id (`sie.work.<MODEL>.pool`).
const MODEL_TOKEN_INDEX: usize = 2;

/// Minimum number of dot-delimited tokens a valid work subject must have.
const MIN_SUBJECT_PARTS: usize = 4;

/// Inverse of [`normalize_model_id`]. Best-effort: `__` → `/`, `_dot_` → `.`.
pub fn denormalize_model_id(normalized: &str) -> String {
    normalized.replace("__", "/").replace("_dot_", ".")
}

/// Make a model id safe to embed in a NATS subject token.
pub fn normalize_model_id(model_id: &str) -> String {
    model_id
        .replace('/', "__")
        .replace('.', "_dot_")
        .replace(['*', '>', ' '], "_")
}

/// Extract and denormalise the model id from a work subject.
/// Returns `None` for malformed subjects (fewer than 4 tokens).
pub fn extract_model_id(subject: &str) -> Option<String> {
    let parts: Vec<&str> = subject.split('.').collect();
    if parts.len() < MIN_SUBJECT_PARTS {
        return None;
    }
    Some(denormalize_model_id(parts[MODEL_TOKEN_INDEX]))
}

/// True iff the two NATS subject filters share at least one concrete
/// subject — i.e. NATS would reject creating two WorkQueue consumers
/// with these filters on the same stream.
///
/// Wildcard semantics:
///   - `*` matches exactly one token
///   - `>` matches one or more trailing tokens (only legal as the last
///     token; anything else is treated as a literal here, which still
///     gives a sound — if conservative — answer)
///
/// This is the predicate behind the worker's stale-durable self-heal
/// in `nats_consumer::ensure_stream_and_consumer`: we treat any
/// existing consumer whose filter overlaps ours (and whose name is not
/// ours) as a leftover from a prior bundle/engine deploy, and delete
/// it. Without this, flipping bundles on the same pool wedges
/// JetStream with `consumer filter overlaps` and the worker
/// CrashLoops.
pub fn subjects_overlap(a: &str, b: &str) -> bool {
    let at: Vec<&str> = a.split('.').collect();
    let bt: Vec<&str> = b.split('.').collect();
    if at.is_empty() || bt.is_empty() {
        return false;
    }

    let a_has_gt = at.last() == Some(&">");
    let b_has_gt = bt.last() == Some(&">");
    let a_fixed = if a_has_gt { at.len() - 1 } else { at.len() };
    let b_fixed = if b_has_gt { bt.len() - 1 } else { bt.len() };

    let common = a_fixed.min(b_fixed);
    for i in 0..common {
        let ta = at[i];
        let tb = bt[i];
        if ta != "*" && tb != "*" && ta != tb {
            return false;
        }
    }

    // After matching the fixed prefix, decide whether the trailing
    // tokens (if any) can also align under `>` semantics. `>` requires
    // at least one trailing token on the other side.
    match (a_has_gt, b_has_gt) {
        (false, false) => at.len() == bt.len(),
        (true, false) => bt.len() >= at.len(),
        (false, true) => at.len() >= bt.len(),
        (true, true) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_handles_slashes_and_dots() {
        assert_eq!(normalize_model_id("BAAI/bge-m3"), "BAAI__bge-m3");
        assert_eq!(normalize_model_id("a.b"), "a_dot_b");
        assert_eq!(normalize_model_id("a/b.c"), "a__b_dot_c");
    }

    #[test]
    fn denormalize_is_inverse_of_normalize_for_safe_ids() {
        for id in ["BAAI/bge-m3", "openai/text-embedding-3-small", "x.y"] {
            assert_eq!(denormalize_model_id(&normalize_model_id(id)), id);
        }
    }

    #[test]
    fn extract_from_valid_subject() {
        assert_eq!(
            extract_model_id("sie.work.BAAI__bge-m3.l4"),
            Some("BAAI/bge-m3".to_string())
        );
        assert_eq!(
            extract_model_id("sie.work.a_dot_b.default"),
            Some("a.b".to_string())
        );
    }

    #[test]
    fn extract_returns_none_on_malformed() {
        assert_eq!(extract_model_id(""), None);
        assert_eq!(extract_model_id("sie.work"), None);
        assert_eq!(extract_model_id("sie.work.model"), None);
    }

    // ----- subjects_overlap ---------------------------------------------------

    #[test]
    fn overlap_identical_filters_with_wildcard_overlap() {
        // The exact case that wedged the GKE worker on bundle flips:
        // two consumers (e.g. `default_l4` from a new deploy and a
        // stale durable left behind by a previous bundle) both
        // filtered on this string.
        assert!(subjects_overlap("sie.work.*.l4", "sie.work.*.l4"));
    }

    #[test]
    fn overlap_wildcard_vs_literal_in_same_slot() {
        assert!(subjects_overlap(
            "sie.work.*.l4",
            "sie.work.BAAI__bge-m3.l4"
        ));
        assert!(subjects_overlap(
            "sie.work.BAAI__bge-m3.l4",
            "sie.work.*.l4"
        ));
    }

    #[test]
    fn overlap_disjoint_pools_do_not_overlap() {
        // Different pools → distinct streams in practice, but the
        // predicate must still report no overlap so it's safe to call
        // even on cross-stream listings.
        assert!(!subjects_overlap("sie.work.*.l4", "sie.work.*.h100"));
        assert!(!subjects_overlap("sie.work.foo.l4", "sie.work.foo.eval-l4"));
    }

    #[test]
    fn overlap_different_lengths_without_gt_do_not_overlap() {
        assert!(!subjects_overlap("sie.work.*.l4", "sie.work.*.l4.extra"));
        assert!(!subjects_overlap("sie.work.*", "sie.work.*.l4"));
    }

    #[test]
    fn overlap_gt_swallows_trailing_tokens() {
        // `>` must match AT LEAST one trailing token, so it overlaps
        // with longer filters but not with one of equal length.
        assert!(subjects_overlap("sie.work.>", "sie.work.foo.l4"));
        assert!(subjects_overlap("sie.work.foo.l4", "sie.work.>"));
        assert!(subjects_overlap("sie.>", "sie.work.*.l4"));
        assert!(!subjects_overlap("sie.work.>", "sie.work"));
        assert!(!subjects_overlap("sie.>", "nats.work.*.l4"));
    }

    #[test]
    fn overlap_two_gt_filters_that_share_a_prefix() {
        assert!(subjects_overlap("sie.work.>", "sie.>"));
        assert!(!subjects_overlap("sie.>", "nats.>"));
    }

    #[test]
    fn overlap_empty_inputs_are_safe() {
        assert!(!subjects_overlap("", "sie.work.*.l4"));
        assert!(!subjects_overlap("sie.work.*.l4", ""));
    }
}
