//! Apply instruction / is_query / query_template / doc_template
//! to raw item text before tokenisation.
//!
//! This matches Python's `sie_server.adapters._utils.extract_texts`
//! so a future native encoder sees the exact same tokens the Python
//! path sees. Without this, asymmetric-retrieval models (`BAAI/bge-*`,
//! `intfloat/e5-*`, `nomic-embed-*`, `gte-Qwen2-*-instruct`) produce
//! the wrong embedding when the caller sets `is_query=true` or passes
//! `instruction` — the embedding silently scores against the wrong
//! subspace and recall collapses.
//!
//! # Precedence (identical to Python)
//!
//! 1. **Template path.** If the matching template is non-empty
//!    (`query_template` when `is_query`, `doc_template` otherwise),
//!    run `template.format(text=..., instruction=...)`. Python's
//!    `str.format` here supports `{text}` and `{instruction}`
//!    placeholders (plus `{instruction or ""}` at call site). Any
//!    other braces in the template are treated as literal braces.
//! 2. **Bare instruction.** No template, but `instruction` is set:
//!    the result is `"{instruction} {text}"` (single space join).
//! 3. **Passthrough.** Neither template nor instruction: the text
//!    is used verbatim.
//!
//! Templates are read from `item.options` (per-request override) and
//! fall back to backend-level defaults supplied by the caller.

/// Resolved text-prep knobs for a single [`crate::ipc_types::EncodeBatchItem`]. Kept
/// as a struct so the per-item call sites in `EncoderEngine` stay
/// short.
pub struct TextPrep<'a> {
    /// Optional instruction string applied to the text (retrieval
    /// models like BGE / E5 / GTE treat this as a prompt prefix).
    pub instruction: Option<&'a str>,
    /// `true` if the caller marked this item as a search query
    /// (affects which template is picked).
    pub is_query: bool,
    /// Query-side template (Python `{text}` / `{instruction}` substitution).
    pub query_template: Option<&'a str>,
    /// Document-side template.
    pub doc_template: Option<&'a str>,
}

impl TextPrep<'_> {
    /// Apply the template / instruction rules. Returns the
    /// transformed string, or borrows the input if no changes are
    /// needed.
    pub fn apply(&self, text: &str) -> String {
        let template = if self.is_query {
            self.query_template
        } else {
            self.doc_template
        };
        if let Some(tpl) = template.filter(|t| !t.is_empty()) {
            return format_template(tpl, text, self.instruction.unwrap_or(""));
        }
        if let Some(instr) = self.instruction.filter(|i| !i.is_empty()) {
            return format!("{instr} {text}");
        }
        text.to_string()
    }
}

/// Cheap Python-`str.format`-workalike for the two known
/// placeholders (`{text}` and `{instruction}`). We avoid pulling in
/// a full templating crate because the Python side only ever uses
/// those two names — everything else is treated as literal text.
///
/// `{{` and `}}` are literal `{` / `}` (matching Python behavior).
fn format_template(tpl: &str, text: &str, instruction: &str) -> String {
    let mut out = String::with_capacity(tpl.len() + text.len() + instruction.len());
    let mut chars = tpl.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' => {
                if chars.peek() == Some(&'{') {
                    out.push('{');
                    chars.next();
                    continue;
                }
                // Collect placeholder name until '}'.
                let mut name = String::new();
                let mut closed = false;
                for next in chars.by_ref() {
                    if next == '}' {
                        closed = true;
                        break;
                    }
                    name.push(next);
                }
                if !closed {
                    // Unterminated placeholder — emit as-is, Python
                    // would raise here but we prefer to degrade
                    // gracefully at the edge.
                    out.push('{');
                    out.push_str(&name);
                    continue;
                }
                match name.as_str() {
                    "text" => out.push_str(text),
                    "instruction" => out.push_str(instruction),
                    other => {
                        // Unknown placeholder — preserve literal.
                        out.push('{');
                        out.push_str(other);
                        out.push('}');
                    }
                }
            }
            '}' => {
                if chars.peek() == Some(&'}') {
                    out.push('}');
                    chars.next();
                } else {
                    // Stray '}' — emit literally.
                    out.push('}');
                }
            }
            _ => out.push(c),
        }
    }
    out
}

/// Pull `query_template` / `doc_template` overrides out of an item's
/// `options` field. Mirrors Python's
/// `opts.get("query_template", default)` path.
pub fn extract_templates_from_options<'a>(
    options: Option<&'a serde_json::Value>,
    default_query: Option<&'a str>,
    default_doc: Option<&'a str>,
) -> (Option<&'a str>, Option<&'a str>) {
    let Some(obj) = options.and_then(|v| v.as_object()) else {
        return (default_query, default_doc);
    };
    let query_template = obj
        .get("query_template")
        .and_then(|v| v.as_str())
        .or(default_query);
    let doc_template = obj
        .get("doc_template")
        .and_then(|v| v.as_str())
        .or(default_doc);
    (query_template, doc_template)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_when_neither_template_nor_instruction() {
        let prep = TextPrep {
            instruction: None,
            is_query: false,
            query_template: None,
            doc_template: None,
        };
        assert_eq!(prep.apply("hello"), "hello");
    }

    #[test]
    fn bare_instruction_when_no_template() {
        let prep = TextPrep {
            instruction: Some("Represent this:"),
            is_query: true,
            query_template: None,
            doc_template: None,
        };
        assert_eq!(prep.apply("hello world"), "Represent this: hello world");
    }

    #[test]
    fn query_template_selected_when_is_query() {
        let prep = TextPrep {
            instruction: None,
            is_query: true,
            query_template: Some("query: {text}"),
            doc_template: Some("passage: {text}"),
        };
        assert_eq!(prep.apply("cats"), "query: cats");
    }

    #[test]
    fn doc_template_selected_when_not_query() {
        let prep = TextPrep {
            instruction: None,
            is_query: false,
            query_template: Some("query: {text}"),
            doc_template: Some("passage: {text}"),
        };
        assert_eq!(prep.apply("cats"), "passage: cats");
    }

    #[test]
    fn instruction_substituted_into_template() {
        let prep = TextPrep {
            instruction: Some("retrieve relevant passages"),
            is_query: true,
            query_template: Some("Instruct: {instruction}\nQuery: {text}"),
            doc_template: None,
        };
        assert_eq!(
            prep.apply("what is love"),
            "Instruct: retrieve relevant passages\nQuery: what is love"
        );
    }

    #[test]
    fn empty_template_falls_back_to_bare_instruction() {
        let prep = TextPrep {
            instruction: Some("Represent this"),
            is_query: true,
            query_template: Some(""),
            doc_template: None,
        };
        assert_eq!(prep.apply("hello"), "Represent this hello");
    }

    #[test]
    fn escaped_braces_are_literal() {
        let prep = TextPrep {
            instruction: None,
            is_query: true,
            query_template: Some("{{literal}} {text}"),
            doc_template: None,
        };
        assert_eq!(prep.apply("x"), "{literal} x");
    }

    #[test]
    fn unknown_placeholder_is_preserved() {
        let prep = TextPrep {
            instruction: None,
            is_query: true,
            query_template: Some("{foo}:{text}"),
            doc_template: None,
        };
        assert_eq!(prep.apply("x"), "{foo}:x");
    }

    #[test]
    fn extract_templates_from_options_falls_back() {
        let opts = serde_json::json!({"query_template": "q: {text}"});
        let (q, d) = extract_templates_from_options(Some(&opts), None, Some("passage: {text}"));
        assert_eq!(q, Some("q: {text}"));
        assert_eq!(d, Some("passage: {text}"));
    }

    #[test]
    fn extract_templates_returns_defaults_when_options_missing() {
        let (q, d) =
            extract_templates_from_options(None, Some("q: {text}"), Some("passage: {text}"));
        assert_eq!(q, Some("q: {text}"));
        assert_eq!(d, Some("passage: {text}"));
    }
}
