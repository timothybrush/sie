//! Debug/ops helper: print the cross-language `tokenizer_id` hash
//! for a given `tokenizer.json`.
//!
//! Useful when diagnosing a Python ↔ Rust tokenizer-id mismatch at
//! runtime. The Python companion computes the same value via:
//!
//! ```python
//! from tokenizers import Tokenizer
//! import blake3
//! tok = Tokenizer.from_file("/path/to/tokenizer.json")
//! print(blake3.blake3(tok.to_str(pretty=False).encode()).hexdigest()[:32])
//! ```
//!
//! Usage:
//!
//! ```bash
//! cargo run --example print_tokenizer_hash -- /path/to/tokenizer.json
//! ```

use std::env;

use sie_server_sidecar::tokenize::tokenizer_content_hash;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = env::args().skip(1);
    let path = args.next().ok_or("usage: print_tokenizer_hash <path>")?;
    let tok = Tokenizer::from_file(&path).map_err(|e| format!("load {path}: {e}"))?;
    println!("{}", tokenizer_content_hash(&tok));
    Ok(())
}
