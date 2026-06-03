//! Shared grammar (structured-output) parser and safety caps.
//!
//! Both the SIE-native ``/v1/generate/{model}`` JSON parser and the
//! OpenAI ``/v1/chat/completions`` ``response_format`` translator funnel
//! through :func:`parse_grammar`. The parser enforces:
//!
//! * Payload size cap (64 KiB)
//! * JSON Schema nesting-depth cap (16)
//! * Regex length cap (4 KiB)
//! * JSON Schema reject-list (``$ref``, ``$dynamicRef``, ``if/then/else``,
//!   ``unevaluatedProperties``, ``dependentSchemas``)
//! * Mutual exclusivity of ``json_schema`` and ``regex``
//!
//! All failures return a 400 :class:`Response` carrying the OpenAI
//! error envelope with ``code`` (``grammar_invalid`` |
//! ``unsupported_field``) and ``param`` naming the offending key path.
//! The worker is downstream and assumes the gateway has already
//! filtered — it does not re-check these caps.
//!
//! ``$ref`` rejection note: the spec wording is "external ``$ref``";
//! v1 rejects **all** ``$ref`` (including internal ``#/...``) for
//! implementation simplicity. The error message and the OpenAPI spec
//! both call this out; this can be refined if customer pressure justifies
//! the JSON-pointer resolver.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::Value;

use crate::http_error::{json_openai_error, openai_code as oai_code, openai_type as oai_type};
use crate::metrics;
use crate::queue::publisher::GrammarSpec;

/// Helper that maps a :class:`GrammarSpec` variant back to the string
/// the model's ``capabilities.grammar`` list uses
/// (``"json_schema"`` | ``"regex"`` | ``"ebnf"``).
pub fn grammar_kind_label(g: &GrammarSpec) -> &'static str {
    match g {
        GrammarSpec::JsonSchema { .. } => "json_schema",
        GrammarSpec::Regex { .. } => "regex",
        GrammarSpec::Ebnf { .. } => "ebnf",
    }
}

/// Reject the request when the model's
/// ``tasks.generate.capabilities.grammar`` list does not advertise the
/// requested ``grammar.kind``. ``capabilities`` is the YAML-derived
/// list as exposed by :class:`ModelInfoExtras`. A ``None`` capabilities
/// list (model has no ``generate`` task at all) also rejects — a
/// non-generation model cannot accept ``grammar`` regardless. Returns
/// ``Ok(())`` when the request is permitted.
#[allow(clippy::result_large_err)]
pub fn check_capability(
    grammar: &GrammarSpec,
    capabilities: Option<&[String]>,
    model: &str,
) -> Result<(), Response> {
    let kind = grammar_kind_label(grammar);
    let allowed = capabilities.is_some_and(|caps| caps.iter().any(|c| c == kind));
    if allowed {
        return Ok(());
    }
    metrics::record_grammar_reject("capability");
    let param = format!("grammar.{kind}");
    let message = if capabilities.is_none() {
        format!("Model '{model}' does not support grammar (no generate task)")
    } else {
        format!("Model '{model}' does not declare '{kind}' grammar support")
    };
    Err((
        StatusCode::BAD_REQUEST,
        Json(json_openai_error(
            message,
            oai_type::INVALID_REQUEST,
            Some(&param),
            oai_code::UNSUPPORTED_FIELD,
        )),
    )
        .into_response())
}

/// Maximum size of the raw ``grammar`` object after JSON serialisation,
/// in bytes. 64 KiB comfortably fits the typical extraction schemas
/// while keeping the worker compile budget bounded.
pub const MAX_GRAMMAR_BYTES: usize = 64 * 1024;

/// Maximum JSON Schema nesting depth. Counted by recursive walks over
/// ``properties`` / ``items`` / ``oneOf`` / ``anyOf`` / ``allOf`` /
/// ``additionalProperties``. Pathological schemas (think
/// ``{"items":{"items":{"items":...}}}``) trigger this before Outlines
/// gets a chance to OOM on compile.
pub const MAX_SCHEMA_DEPTH: usize = 16;

/// Maximum total node count visited during the schema walk. Depth alone
/// doesn't stop a *wide* schema (one shallow object with hundreds of
/// thousands of trivial properties), where every key still pays for a
/// `format!` allocation and a recursion frame. 16 384 is far above any
/// legitimate schema while still bounding the walker's CPU/allocations
/// to single-digit milliseconds in the worst case.
pub const MAX_SCHEMA_NODES: usize = 16 * 1024;

/// Maximum regex length, in characters. Long regexes drive the
/// Outlines compile time non-linearly; 4 KiB is generous for legitimate
/// use cases (license keys, product codes, etc.).
pub const MAX_REGEX_LEN: usize = 4 * 1024;

/// Maximum EBNF/CFG grammar source length, in characters. Outlines'
/// EBNF compiler is exponential in the worst case; we cap the source
/// well below :const:`MAX_GRAMMAR_BYTES` so the overall payload cap
/// still leaves room for label/strict siblings without surprising the
/// caller with a payload-size rejection on a grammar that fits.
pub const MAX_EBNF_LEN: usize = 8 * 1024;

/// JSON Schema keywords rejected at the gateway. These are features
/// Outlines either does not implement or implements at prohibitive
/// compile cost. Each rejection names the keyword in ``param`` so
/// callers can fix their schema without trial-and-error.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "$ref",
    "$dynamicRef",
    "if",
    "then",
    "else",
    "unevaluatedProperties",
    "dependentSchemas",
];

/// Result of :func:`parse_grammar`. ``Err`` carries an already-built
/// 400 response so the caller can return it directly (mirrors
/// :class:`ChatParamsResult` in ``proxy.rs``).
pub enum GrammarParseResult {
    Ok(GrammarSpec),
    Err(Response),
}

fn bad_request(message: String, param: &str, code: &'static str) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json_openai_error(
            message,
            oai_type::INVALID_REQUEST,
            Some(param),
            code,
        )),
    )
        .into_response()
}

/// Parse the ``grammar`` object as it appears under
/// ``/v1/generate/{model}`` request bodies. Caller is responsible for
/// the capability gate (model config ``tasks.generate.capabilities.grammar``
/// list); this function only enforces the wire-shape contract.
///
/// Wire shape (mutually-exclusive variants under ``grammar:``):
///
/// ```json
/// { "json_schema": {"type": "object", ...} }
/// // or
/// { "regex": "[A-Z]{3}-\\d{4}" }
/// // or
/// { "ebnf": "root ::= \"hello\" | \"goodbye\"" }
/// ```
///
/// Optional sibling keys ``label`` and ``strict`` are surfaced to the
/// worker via the resulting :class:`GrammarSpec`. They never affect the
/// cache key — see :func:`sie_server.types.grammar.hash_grammar`.
pub fn parse_grammar(v: &Value) -> GrammarParseResult {
    let Some(obj) = v.as_object() else {
        metrics::record_grammar_reject("malformed");
        return GrammarParseResult::Err(bad_request(
            "'grammar' must be a JSON object".to_string(),
            "grammar",
            oai_code::INVALID_REQUEST,
        ));
    };

    // §2.5 step 1: payload size cap. Run before any recursion so a
    // billion-key schema cannot exhaust the walker's stack before the
    // size check fires.
    //
    // ``serde_json::to_vec`` is the cheapest way to get a byte count
    // for a tree we already hold; we deliberately don't compare against
    // the request body length because the gateway's outer
    // ``MAX_PROXY_BODY`` covers that and `grammar:` is one field of
    // many.
    let serialized_len = serde_json::to_vec(v).map(|b| b.len()).unwrap_or(0);
    if serialized_len > MAX_GRAMMAR_BYTES {
        metrics::record_grammar_reject("payload_size");
        return GrammarParseResult::Err(bad_request(
            format!(
                "grammar payload {serialized_len} bytes exceeds limit ({MAX_GRAMMAR_BYTES} bytes)"
            ),
            "grammar",
            oai_code::INVALID_REQUEST,
        ));
    }

    let has_schema = obj.contains_key("json_schema");
    let has_regex = obj.contains_key("regex");
    let has_ebnf = obj.contains_key("ebnf");
    let variants_present = [has_schema, has_regex, has_ebnf]
        .iter()
        .filter(|p| **p)
        .count();
    if variants_present > 1 {
        metrics::record_grammar_reject("mutex");
        return GrammarParseResult::Err(bad_request(
            "'grammar.json_schema', 'grammar.regex' and 'grammar.ebnf' are mutually exclusive"
                .to_string(),
            "grammar",
            oai_code::INVALID_REQUEST,
        ));
    }
    if variants_present == 0 {
        metrics::record_grammar_reject("malformed");
        return GrammarParseResult::Err(bad_request(
            "'grammar' must contain exactly one of 'json_schema', 'regex' or 'ebnf'".to_string(),
            "grammar",
            oai_code::INVALID_REQUEST,
        ));
    }

    let label = obj.get("label").and_then(|v| v.as_str()).map(String::from);
    let strict = obj.get("strict").and_then(|v| v.as_bool());

    if has_schema {
        let schema = obj.get("json_schema").expect("checked above");
        if let Err(resp) = walk_schema(schema, "grammar.json_schema", 0) {
            return GrammarParseResult::Err(resp);
        }
        GrammarParseResult::Ok(GrammarSpec::JsonSchema {
            value: schema.clone(),
            label,
            strict,
        })
    } else if has_regex {
        let regex_val = obj.get("regex").expect("checked above");
        let Some(regex) = regex_val.as_str() else {
            metrics::record_grammar_reject("malformed");
            return GrammarParseResult::Err(bad_request(
                "'grammar.regex' must be a string".to_string(),
                "grammar.regex",
                oai_code::INVALID_REQUEST,
            ));
        };
        if regex.len() > MAX_REGEX_LEN {
            metrics::record_grammar_reject("regex_len");
            return GrammarParseResult::Err(bad_request(
                format!(
                    "regex length {} exceeds limit ({MAX_REGEX_LEN})",
                    regex.len()
                ),
                "grammar.regex",
                oai_code::INVALID_REQUEST,
            ));
        }
        GrammarParseResult::Ok(GrammarSpec::Regex {
            value: regex.to_string(),
            label,
            strict,
        })
    } else {
        // ``ebnf`` branch — string source; no further structural walk
        // (the gateway does not parse EBNF; Outlines/XGrammar is the
        // authority). MAX_GRAMMAR_BYTES at the envelope level plus
        // MAX_EBNF_LEN at the source level bound compile cost.
        let ebnf_val = obj.get("ebnf").expect("checked above");
        let Some(ebnf) = ebnf_val.as_str() else {
            metrics::record_grammar_reject("malformed");
            return GrammarParseResult::Err(bad_request(
                "'grammar.ebnf' must be a string".to_string(),
                "grammar.ebnf",
                oai_code::INVALID_REQUEST,
            ));
        };
        if ebnf.len() > MAX_EBNF_LEN {
            metrics::record_grammar_reject("ebnf_len");
            return GrammarParseResult::Err(bad_request(
                format!("ebnf length {} exceeds limit ({MAX_EBNF_LEN})", ebnf.len()),
                "grammar.ebnf",
                oai_code::INVALID_REQUEST,
            ));
        }
        GrammarParseResult::Ok(GrammarSpec::Ebnf {
            value: ebnf.to_string(),
            label,
            strict,
        })
    }
}

/// Recursive walk over a JSON-Schema-shaped value. Enforces depth and
/// rejects the unsupported keywords listed in
/// :const:`UNSUPPORTED_KEYWORDS`.
///
/// ``path`` is the dotted accessor for whatever produced 400s name in
/// ``param``. Array elements append ``[N]``; object members append
/// ``.<key>``.
#[allow(clippy::result_large_err)]
fn walk_schema(v: &Value, path: &str, depth: usize) -> Result<(), Response> {
    let mut visited: usize = 0;
    walk_schema_inner(v, path, depth, &mut visited)
}

#[allow(clippy::result_large_err)]
fn walk_schema_inner(
    v: &Value,
    path: &str,
    depth: usize,
    visited: &mut usize,
) -> Result<(), Response> {
    *visited = visited.saturating_add(1);
    if *visited > MAX_SCHEMA_NODES {
        metrics::record_grammar_reject("node_count");
        return Err(bad_request(
            format!("JSON Schema node count exceeds limit ({MAX_SCHEMA_NODES})"),
            path,
            oai_code::INVALID_REQUEST,
        ));
    }
    if depth > MAX_SCHEMA_DEPTH {
        metrics::record_grammar_reject("depth");
        return Err(bad_request(
            format!("JSON Schema depth exceeds limit ({MAX_SCHEMA_DEPTH})"),
            path,
            oai_code::INVALID_REQUEST,
        ));
    }

    match v {
        Value::Object(map) => {
            // Reject before descending so the message names the keyword
            // at the shallowest occurrence.
            for &kw in UNSUPPORTED_KEYWORDS {
                if map.contains_key(kw) {
                    metrics::record_grammar_reject("unsupported_keyword");
                    let param = format!("{path}.{kw}");
                    let message = if kw == "$ref" {
                        // Grammar v1 blanket-rejects all ``$ref``,
                        // including internal ``#/...``. The spec
                        // wording is "external ``$ref``"; later revisions may
                        // refine. Documented in OpenAPI.
                        "'$ref' is not supported (the grammar surface rejects all $ref including internal)"
                            .to_string()
                    } else {
                        format!("JSON Schema keyword '{kw}' is not supported")
                    };
                    return Err(bad_request(message, &param, oai_code::UNSUPPORTED_FIELD));
                }
            }
            for (k, child) in map {
                let child_path = format!("{path}.{k}");
                let child_depth = if is_schema_nesting_key(k) {
                    depth + 1
                } else {
                    depth
                };
                walk_schema_inner(child, &child_path, child_depth, visited)?;
            }
        }
        Value::Array(arr) => {
            for (i, child) in arr.iter().enumerate() {
                let child_path = format!("{path}[{i}]");
                walk_schema_inner(child, &child_path, depth, visited)?;
            }
        }
        // Scalars (string / number / bool / null) cannot host
        // unsupported keywords and do not contribute to depth.
        _ => {}
    }

    Ok(())
}

fn is_schema_nesting_key(key: &str) -> bool {
    matches!(
        key,
        "properties"
            | "patternProperties"
            | "additionalProperties"
            | "unevaluatedProperties"
            | "items"
            | "prefixItems"
            | "contains"
            | "propertyNames"
            | "oneOf"
            | "anyOf"
            | "allOf"
            | "not"
            | "definitions"
            | "$defs"
            | "dependentSchemas"
            | "if"
            | "then"
            | "else"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use serde_json::json;

    async fn err_body(resp: Response) -> serde_json::Value {
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let body = to_bytes(resp.into_body(), 64 * 1024).await.unwrap();
        serde_json::from_slice(&body).unwrap()
    }

    fn ok_or_panic(r: GrammarParseResult) -> GrammarSpec {
        match r {
            GrammarParseResult::Ok(g) => g,
            GrammarParseResult::Err(_) => panic!("expected Ok"),
        }
    }

    async fn err_or_panic(r: GrammarParseResult) -> serde_json::Value {
        match r {
            GrammarParseResult::Ok(_) => panic!("expected Err"),
            GrammarParseResult::Err(resp) => err_body(resp).await,
        }
    }

    #[test]
    fn test_parse_grammar_accepts_small_json_schema() {
        let v = json!({
            "json_schema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            "label": "tiny",
        });
        let spec = ok_or_panic(parse_grammar(&v));
        match spec {
            GrammarSpec::JsonSchema { label, .. } => {
                assert_eq!(label.as_deref(), Some("tiny"));
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_grammar_accepts_small_regex() {
        let v = json!({"regex": r"[A-Z]{3}-\d{4}"});
        let spec = ok_or_panic(parse_grammar(&v));
        match spec {
            GrammarSpec::Regex { value, .. } => assert_eq!(value, r"[A-Z]{3}-\d{4}"),
            other => panic!("expected Regex, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_oversized_payload() {
        // Build a schema strictly larger than the cap.
        let mut props = serde_json::Map::new();
        // Each property contributes ~30 bytes; (64 * 1024 / 30) ≈ 2240
        // is comfortably above. Pad to be safe.
        for i in 0..4000 {
            props.insert(
                format!("field_{i:05}"),
                json!({"type": "string", "description": "filler"}),
            );
        }
        let v = json!({"json_schema": {"type": "object", "properties": props}});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar");
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("exceeds limit"), "msg: {msg}");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_deeply_nested_schema() {
        // Build a schema with depth > MAX_SCHEMA_DEPTH.
        let mut leaf = json!({"type": "string"});
        for _ in 0..(MAX_SCHEMA_DEPTH + 5) {
            leaf = json!({"type": "object", "properties": {"x": leaf}});
        }
        let v = json!({"json_schema": leaf});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        let param = body["error"]["param"].as_str().unwrap_or("");
        assert!(
            param.starts_with("grammar.json_schema"),
            "expected depth error path under grammar.json_schema, got {param}"
        );
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_dollar_ref() {
        let v = json!({
            "json_schema": {
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/Foo"}},
            }
        });
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "unsupported_field");
        assert_eq!(
            body["error"]["param"],
            "grammar.json_schema.properties.a.$ref"
        );
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("$ref"), "msg should mention $ref: {msg}");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_each_unsupported_keyword() {
        for kw in [
            "$dynamicRef",
            "if",
            "then",
            "else",
            "unevaluatedProperties",
            "dependentSchemas",
        ] {
            let mut schema = serde_json::Map::new();
            schema.insert("type".to_string(), json!("object"));
            schema.insert(kw.to_string(), json!({"x": true}));
            let v = json!({"json_schema": Value::Object(schema)});
            let body = err_or_panic(parse_grammar(&v)).await;
            assert_eq!(
                body["error"]["code"], "unsupported_field",
                "keyword {kw} should reject"
            );
            assert_eq!(
                body["error"]["param"],
                format!("grammar.json_schema.{kw}"),
                "param path for {kw}"
            );
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_oversized_regex() {
        let big = "a".repeat(MAX_REGEX_LEN + 1);
        let v = json!({"regex": big});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar.regex");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_mutual_exclusivity_violation() {
        let v = json!({
            "json_schema": {"type": "object"},
            "regex": "abc",
        });
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar");
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(
            msg.contains("mutually exclusive"),
            "msg should mention mutual exclusivity: {msg}"
        );
    }

    #[test]
    fn test_parse_grammar_accepts_small_ebnf() {
        let v = json!({
            "ebnf": "root ::= \"hello\" | \"goodbye\"",
            "label": "greeting",
        });
        let spec = ok_or_panic(parse_grammar(&v));
        match spec {
            GrammarSpec::Ebnf { value, label, .. } => {
                assert!(value.contains("root ::="));
                assert_eq!(label.as_deref(), Some("greeting"));
            }
            other => panic!("expected Ebnf, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_oversized_ebnf() {
        let big = "a".repeat(MAX_EBNF_LEN + 1);
        let v = json!({"ebnf": big});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar.ebnf");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_non_string_ebnf() {
        let v = json!({"ebnf": 42});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar.ebnf");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_ebnf_plus_other_variant() {
        let v = json!({
            "ebnf": "root ::= \"a\"",
            "regex": "[a-z]+",
        });
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["code"], "invalid_request");
        assert_eq!(body["error"]["param"], "grammar");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_empty_object() {
        let v = json!({});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_non_object() {
        let v = json!("oops");
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar");
    }

    #[tokio::test]
    async fn test_parse_grammar_rejects_non_string_regex() {
        let v = json!({"regex": 42});
        let body = err_or_panic(parse_grammar(&v)).await;
        assert_eq!(body["error"]["param"], "grammar.regex");
    }

    #[test]
    fn test_parse_grammar_passes_through_label_and_strict() {
        let v = json!({
            "json_schema": {"type": "string"},
            "label": "name_v1",
            "strict": true,
        });
        match ok_or_panic(parse_grammar(&v)) {
            GrammarSpec::JsonSchema { label, strict, .. } => {
                assert_eq!(label.as_deref(), Some("name_v1"));
                assert_eq!(strict, Some(true));
            }
            other => panic!("expected JsonSchema, got {other:?}"),
        }
    }

    // ── capability gate ────────────────────────────────────────────

    #[tokio::test]
    async fn test_check_capability_accepts_listed_kind() {
        let g = GrammarSpec::JsonSchema {
            value: json!({"type": "string"}),
            label: None,
            strict: None,
        };
        let caps = vec!["json_schema".to_string(), "regex".to_string()];
        assert!(check_capability(&g, Some(&caps), "m").is_ok());
    }

    #[tokio::test]
    async fn test_check_capability_rejects_unlisted_kind() {
        let g = GrammarSpec::Regex {
            value: "[a-z]+".to_string(),
            label: None,
            strict: None,
        };
        let caps = vec!["json_schema".to_string()];
        let resp = check_capability(&g, Some(&caps), "Qwen/X").expect_err("expected reject");
        let body = err_body(resp).await;
        assert_eq!(body["error"]["code"], "unsupported_field");
        assert_eq!(body["error"]["param"], "grammar.regex");
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("Qwen/X"), "msg should mention model: {msg}");
        assert!(msg.contains("regex"), "msg should mention kind: {msg}");
    }

    #[tokio::test]
    async fn test_check_capability_accepts_ebnf_when_listed() {
        let g = GrammarSpec::Ebnf {
            value: "root ::= \"a\"".to_string(),
            label: None,
            strict: None,
        };
        let caps = vec!["ebnf".to_string()];
        assert!(check_capability(&g, Some(&caps), "m").is_ok());
    }

    #[tokio::test]
    async fn test_check_capability_rejects_ebnf_when_unlisted() {
        let g = GrammarSpec::Ebnf {
            value: "root ::= \"a\"".to_string(),
            label: None,
            strict: None,
        };
        let caps = vec!["json_schema".to_string(), "regex".to_string()];
        let resp = check_capability(&g, Some(&caps), "Qwen/X").expect_err("expected reject");
        let body = err_body(resp).await;
        assert_eq!(body["error"]["code"], "unsupported_field");
        assert_eq!(body["error"]["param"], "grammar.ebnf");
    }

    #[tokio::test]
    async fn test_check_capability_rejects_empty_list() {
        let g = GrammarSpec::Regex {
            value: "[a-z]+".to_string(),
            label: None,
            strict: None,
        };
        let caps: Vec<String> = Vec::new();
        assert!(check_capability(&g, Some(&caps), "m").is_err());
    }

    #[tokio::test]
    async fn test_check_capability_rejects_none_capabilities() {
        // Model has no ``generate`` task at all — anything grammar-shaped
        // must reject.
        let g = GrammarSpec::JsonSchema {
            value: json!({}),
            label: None,
            strict: None,
        };
        let resp = check_capability(&g, None, "m").expect_err("expected reject");
        let body = err_body(resp).await;
        let msg = body["error"]["message"].as_str().unwrap_or("");
        assert!(msg.contains("does not support grammar"), "msg: {msg}");
    }

    /// Depth budget is generous enough for realistic extraction schemas
    /// (4-5 levels of objects/arrays). Exercise a moderate schema to
    /// guard against an off-by-one that would falsely reject sensible
    /// inputs.
    #[test]
    fn test_parse_grammar_accepts_realistic_extraction_schema() {
        let v = json!({
            "json_schema": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "qty": {"type": "integer"},
                            },
                            "required": ["name", "qty"],
                        }
                    },
                    "total": {"type": "number"}
                },
                "required": ["items"],
                "additionalProperties": false
            }
        });
        let _ = ok_or_panic(parse_grammar(&v));
    }
}
