//! Shared `output_types` validation for every native encoder
//! backend.
//!
//! A dense-only backend (i.e. anything that isn't hybrid / ColBERT)
//! must explicitly reject requests for `"sparse"`, `"multivector"`,
//! or any unknown type — silently returning dense for those would
//! corrupt downstream consumers that thought they were getting
//! sparse / ColBERT vectors.

/// Return the first unsupported output type in `requested`, or
/// `None` if every requested type is something this dense-only
/// backend can produce.
///
/// Rules:
/// * `None` → client didn't specify → default to dense → accepted.
/// * `Some(&[])` → client explicitly asked for no outputs →
///   **rejected** (ambiguous — probably a client bug).
/// * `Some([..."dense"...])` with only `dense` (case-insensitive) →
///   accepted.
/// * Anything else (`sparse`, `multivector`, unknown) → rejected,
///   returning that offending value for the error message.
pub fn first_unsupported_output_type(requested: Option<&[String]>) -> Option<String> {
    match requested {
        None => None,
        Some([]) => Some("<empty list>".to_string()),
        Some(list) => list
            .iter()
            .find(|t| !t.eq_ignore_ascii_case("dense"))
            .cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_none_and_dense_only() {
        assert_eq!(first_unsupported_output_type(None), None);
        assert_eq!(
            first_unsupported_output_type(Some(&["dense".to_string()])),
            None
        );
        assert_eq!(
            first_unsupported_output_type(Some(&["DENSE".to_string()])),
            None
        );
    }

    #[test]
    fn rejects_empty_list_and_other_types() {
        assert_eq!(
            first_unsupported_output_type(Some(&[])),
            Some("<empty list>".to_string())
        );
        assert_eq!(
            first_unsupported_output_type(Some(&["sparse".to_string()])),
            Some("sparse".to_string())
        );
        assert_eq!(
            first_unsupported_output_type(Some(&["dense".to_string(), "multivector".to_string()])),
            Some("multivector".to_string())
        );
        assert_eq!(
            first_unsupported_output_type(Some(&["garbage".to_string()])),
            Some("garbage".to_string())
        );
    }
}
