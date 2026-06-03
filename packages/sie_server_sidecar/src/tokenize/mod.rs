//! Rust-side tokenisation for the Python-IPC backend path.
//!
//! The sidecar tokenises text for models whose adapter has declared a
//! tokeniser on the `EnsureModelReady` handshake (see
//! [`registry::TokenizerRegistry::register_from_descriptor`]) and
//! attaches the result as [`crate::ipc_types::PreparedTokens`] on
//! `EncodeBatchItem` / `ScoreBatchItem`. The Python adapter checks the
//! `tokenizer_id` hash; on match it uses the tokens directly, on
//! mismatch / absence it falls back to its own HF `AutoTokenizer`
//! call — bit-for-bit the same wire contract as today.
//!
//! # Parity with Python
//!
//! The Python adapters that currently tokenise inline
//! (`RoPEFlashAdapter`, `XLMRobertaFlashAdapter`, `ModernBertFlashAdapter`,
//! `NomicFlashAdapter`, ...) all call the HF fast tokenizer with the
//! same kwargs:
//!
//! ```python
//! tokenizer(texts, max_length=N, truncation=True, padding=False)
//! ```
//!
//! Ragged per-item `input_ids` — no batch padding. The packed flash
//! path handles cross-sequence layout itself via `cu_seqlens`. The
//! Rust side mirrors that policy in [`tokenize_no_pad`]: it loads the
//! same `tokenizer.json` via the HF `tokenizers` crate, forces
//! truncation to `max_seq_len`, and disables padding. `input_ids`
//! vectors therefore have their natural per-item lengths and match
//! Python's output byte-for-byte (see the byte-identity unit test).
//!
//! # What this module DOESN'T do (v1 scope)
//!
//! - No query / doc template application (`"query: {text}"` etc). If
//!   the model has a template configured, the dispatcher elides
//!   `prepared_tokens` for that request and Python tokenises as
//!   today. Template support is an additive v2 extension.
//! - No sentence-pair tokenisation (cross-encoder score). Those
//!   models fall back to Python tokenise for v1.
//! - No image / audio / multimodal preprocessing.
//!
//! Anything not covered above stays on the Python path — v1 is
//! strictly additive and non-destructive.

pub mod registry;

use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, TruncationStrategy};

pub use registry::{TokenizerEntry, TokenizerRegistry};

/// Per-item ragged token output, matching Python's
/// `tokenizer(texts, max_length, truncation=True, padding=False)`.
///
/// Each inner `Vec<u32>` is the `input_ids` for the corresponding
/// text; `attention_mask[i]` has the same length as `input_ids[i]`
/// (all-ones for single-sequence text, since there is no padding);
/// `token_type_ids[i]` is present when the tokenizer produces
/// segments (zero-filled for most BERT-family encoders), otherwise
/// the vector is empty and the caller treats it as "not emitted".
#[derive(Debug, Clone, Default)]
pub struct RaggedTokens {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub token_type_ids: Vec<Vec<u32>>,
}

impl RaggedTokens {
    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    /// True iff every `token_type_ids[i]` is all-zeros. BERT-family
    /// tokenizers emit a zero-filled segment vector even when the
    /// model doesn't use segments; the dispatcher uses this to skip
    /// emitting the field over the wire and save bytes.
    pub fn token_type_ids_all_zero(&self) -> bool {
        self.token_type_ids
            .iter()
            .all(|row| row.iter().all(|&t| t == 0))
    }
}

/// Load a HF fast-tokenizer from `tokenizer.json` configured for the
/// IPC-path policy: **no padding, truncate to `max_seq_len`**.
///
/// This is the only HF tokenizer loader in the workspace — the
/// flash-attn adapters on the Python side explicitly require ragged
/// input, so we never pad on the sidecar; Python pads ad-hoc when a
/// model needs it.
///
/// # Padding preservation
///
/// Do NOT clobber `pad_id` / `pad_token` on an existing
/// `PaddingParams` block in `tokenizer.json` (XLM-R's `pad_id=1`
/// would regress to `0` otherwise). Mutate in place if present,
/// install defaults only when absent. Strategy is explicitly set to
/// `PaddingStrategy::Fixed(0)` in the "absent" case so batch-time
/// padding never silently re-activates.
pub fn load_no_pad_tokenizer(
    path: &std::path::Path,
    max_seq_len: usize,
) -> Result<Tokenizer, String> {
    let mut tok = Tokenizer::from_file(path).map_err(|e| format!("load tokenizer: {e}"))?;

    // Disable padding. The existing `PaddingParams` (if any) still
    // carries `pad_id` / `pad_token` that the adapter might need at
    // pad time; we just set the strategy to `Fixed(0)` so the
    // encode_batch call itself emits no padding.
    if let Some(existing) = tok.get_padding_mut() {
        existing.strategy = PaddingStrategy::Fixed(0);
    } else {
        tok.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::Fixed(0),
            ..Default::default()
        }));
    }

    tok.with_truncation(Some(TruncationParams {
        max_length: max_seq_len,
        strategy: TruncationStrategy::LongestFirst,
        ..Default::default()
    }))
    .map_err(|e| format!("set truncation: {e}"))?;

    Ok(tok)
}

/// Batch-tokenise `texts` into ragged per-item `input_ids` +
/// `attention_mask` + `token_type_ids`, with no padding.
///
/// Mirrors Python's
/// ```python
/// tokenizer(texts, max_length=N, truncation=True, padding=False)
/// ```
///
/// Returns a default (empty) [`RaggedTokens`] when `texts` is empty.
pub fn tokenize_no_pad(tok: &Tokenizer, texts: &[&str]) -> Result<RaggedTokens, String> {
    if texts.is_empty() {
        return Ok(RaggedTokens::default());
    }
    let encodings = tok
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| format!("tokenize batch: {e}"))?;

    let mut input_ids = Vec::with_capacity(encodings.len());
    let mut attention_mask = Vec::with_capacity(encodings.len());
    let mut token_type_ids = Vec::with_capacity(encodings.len());
    for enc in encodings {
        input_ids.push(enc.get_ids().to_vec());
        attention_mask.push(enc.get_attention_mask().to_vec());
        token_type_ids.push(enc.get_type_ids().to_vec());
    }

    Ok(RaggedTokens {
        input_ids,
        attention_mask,
        token_type_ids,
    })
}

/// Compute the stable, cross-language content hash of a loaded
/// tokenizer.
///
/// The hash is taken over the tokenizer's canonical JSON
/// serialisation (`Tokenizer::to_string(false)`), **not** the raw
/// bytes of the `tokenizer.json` file. Two reasons:
///
/// 1. The Python side can reproduce it exactly: HF's `tokenizers`
///    crate is the same library wrapped by the Python package, so
///    `tokenizer.backend_tokenizer.to_str(pretty=False).encode()`
///    yields byte-identical input to BLAKE3.
/// 2. The raw on-disk file can vary in insignificant ways (trailing
///    whitespace, key ordering after some HF hub round-trips)
///    without the actual tokenizer configuration changing. Hashing
///    the parsed form avoids false drift alarms.
///
/// Uses BLAKE3 truncated to 32 hex chars (128 bits) — overkill for
/// accidental-drift detection, fits comfortably in the wire budget,
/// and is trivially reproducible on the Python side via the
/// `blake3` PyPI package.
///
/// Callers should hash the tokenizer **before** applying any
/// IPC-specific padding / truncation config (the
/// `load_no_pad_tokenizer` call), so the id is determined by the
/// tokenizer's own declared settings, not by our runtime overrides.
/// The registry does this correctly.
pub fn tokenizer_content_hash(tok: &Tokenizer) -> String {
    // `to_string(false)` → compact, non-pretty JSON. The Python
    // `tokenizers` binding exposes the same method as
    // `.to_str(pretty=False)`; both wrap `Tokenizer::to_string` in
    // the underlying Rust crate.
    let canonical = tok
        .to_string(false)
        .expect("tokenizer.json round-trip serialisation must not fail");
    let digest = blake3::hash(canonical.as_bytes());
    digest.to_hex().as_str()[..32].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ragged_empty_by_default() {
        let r = RaggedTokens::default();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn tokenizer_content_hash_is_deterministic_and_short() {
        // Tiny WordLevel tokenizer used throughout the registry
        // tests; re-inlined here so this unit doesn't depend on the
        // registry's test-only constant.
        const TINY: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "world": 2},
                "unk_token": "[UNK]"
            }
        }"#;
        let dir = tempfile::TempDir::new().unwrap();
        let path = dir.path().join("tokenizer.json");
        std::fs::write(&path, TINY).unwrap();

        let tok_a = Tokenizer::from_file(&path).unwrap();
        let tok_b = Tokenizer::from_file(&path).unwrap();
        let a = tokenizer_content_hash(&tok_a);
        let b = tokenizer_content_hash(&tok_b);
        assert_eq!(a, b, "hash must be deterministic for the same input");
        assert_eq!(a.len(), 32, "hash must be exactly 32 hex chars (128 bits)");

        // Sanity: mutating the tokenizer (different vocab) changes
        // the canonical serialisation and therefore the hash.
        const TINY2: &str = r#"{
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "WordLevel",
                "vocab": {"[UNK]": 0, "hello": 1, "there": 2},
                "unk_token": "[UNK]"
            }
        }"#;
        let path2 = dir.path().join("tokenizer2.json");
        std::fs::write(&path2, TINY2).unwrap();
        let tok_c = Tokenizer::from_file(&path2).unwrap();
        assert_ne!(a, tokenizer_content_hash(&tok_c));
    }

    #[test]
    fn token_type_ids_all_zero_detects_bert_style_segments() {
        let empty = RaggedTokens::default();
        assert!(
            empty.token_type_ids_all_zero(),
            "empty counts as all-zero (no rows to violate)"
        );

        let bert_like = RaggedTokens {
            input_ids: vec![vec![1, 2, 3], vec![4, 5]],
            attention_mask: vec![vec![1, 1, 1], vec![1, 1]],
            token_type_ids: vec![vec![0, 0, 0], vec![0, 0]],
        };
        assert!(bert_like.token_type_ids_all_zero());

        let pair_like = RaggedTokens {
            input_ids: vec![vec![1, 2, 3]],
            attention_mask: vec![vec![1, 1, 1]],
            token_type_ids: vec![vec![0, 1, 1]],
        };
        assert!(
            !pair_like.token_type_ids_all_zero(),
            "sentence-pair segment ids must not be collapsed"
        );
    }
}
