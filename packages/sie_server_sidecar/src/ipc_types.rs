//! IPC wire types — re-exported from [`crate::protocol::ipc_types`] so existing
//! sidecar call sites can keep using `crate::ipc_types`.
//!
//! The Python mirror lives in `sie_server/src/sie_server/ipc_types.py`.
//! CI runs a two-way parity check between the two.
//!
//! Wire format: `[4-byte BE length][msgpack body]`, where `body` is a
//! msgpack **map** encoding `RequestEnvelope` / `ResponseEnvelope`.
//! `rmp_serde::to_vec_named` + `serde(default)` for forward-compat
//! field absorption.

pub use crate::protocol::ipc_types::*;
