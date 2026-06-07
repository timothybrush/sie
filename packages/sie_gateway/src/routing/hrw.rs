//! Highest-Random-Weight (rendezvous) hashing for direct-dispatch routing.
//!
//! Given a request hash and a snapshot of candidate workers, pick the
//! single worker whose combined hash with the request key is largest.
//! This is `O(n)` per pick, stable across all callers without
//! coordination, and degrades gracefully when workers come and go
//! (only `1/n` of keys move on membership change).
//!
//! Snapshots are immutable. The
//! [`crate::state::worker_registry::WorkerRegistry`] owns the source
//! of truth; this module just consumes a borrowed slice of
//! `(worker_id, worker_id_hash)` pairs.

use xxhash_rust::xxh3::xxh3_64;

use super::key::RoutingKeyResolved;

/// One entry on the HRW ring. Pre-hashing `worker_id` once at snapshot
/// build time keeps `pick_worker` cheap.
#[derive(Debug, Clone)]
pub struct RingEntry {
    pub worker_id: String,
    pub worker_id_hash: u64,
}

impl RingEntry {
    pub fn new(worker_id: impl Into<String>) -> Self {
        let worker_id = worker_id.into();
        let worker_id_hash = xxh3_64(worker_id.as_bytes());
        Self {
            worker_id,
            worker_id_hash,
        }
    }
}

/// Immutable worker ring for one `(model, pool, machine_profile, bundle)` lane.
/// `WorkerRegistry::ring_snapshot_for` rebuilds this per request from the
/// registry snapshot; the broader worker view is what lives behind `ArcSwap`,
/// so reads stay lock-free even though this lane ring is freshly constructed.
#[derive(Debug, Clone, Default)]
pub struct RingSnapshot {
    pub entries: Vec<RingEntry>,
}

impl RingSnapshot {
    pub fn new(worker_ids: impl IntoIterator<Item = String>) -> Self {
        Self {
            entries: worker_ids.into_iter().map(RingEntry::new).collect(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Clippy lints ``len`` without ``is_empty``; expose both even
    /// though :func:`pick` already early-returns on an empty ring.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// HRW pick: highest `combine(request_hash, worker_id_hash)` wins.
///
/// Returns `None` when the snapshot is empty (caller falls back to the
/// pool subject) or when the resolved key has no hash (caller may
/// round-robin or also fall back).
///
/// Ties on `combine` are broken lexicographically by `worker_id` so the
/// pick is identical across gateway replicas even when the snapshot's
/// source iteration order is nondeterministic (e.g. derived from a
/// `HashMap`).
pub fn pick_worker<'a>(snapshot: &'a RingSnapshot, key: &RoutingKeyResolved) -> Option<&'a str> {
    if snapshot.entries.is_empty() {
        return None;
    }
    let request_hash = key.hash?;
    snapshot
        .entries
        .iter()
        .max_by(|a, b| {
            combine(request_hash, a.worker_id_hash)
                .cmp(&combine(request_hash, b.worker_id_hash))
                .then_with(|| a.worker_id.cmp(&b.worker_id))
        })
        .map(|e| e.worker_id.as_str())
}

/// Combine the request key hash with a worker-id hash.
///
/// Uses an asymmetric mix (`worker_id_hash.rotate_left(32) ^ request_hash`,
/// re-hashed) so the function is *not* invariant under the transformation
/// `(req, wid) → (req ⊕ x, wid ⊕ x)`. A plain `xxh3(req ^ wid)` mix gives
/// identical outputs for any pair with equal XOR — which means hash
/// collisions are not bounded by xxh3's collision resistance, only by the
/// symmetry of the input. Rotating one input before XOR breaks that
/// symmetry while keeping the function deterministic and branch-free.
#[inline]
fn combine(request_hash: u64, worker_id_hash: u64) -> u64 {
    let mixed = (worker_id_hash.rotate_left(32) ^ request_hash).to_le_bytes();
    xxh3_64(&mixed)
}

/// Test-only re-export of [`combine`] for regression tests that need to
/// assert the mix is not XOR-symmetric.
#[cfg(test)]
pub(crate) fn combine_for_test(request_hash: u64, worker_id_hash: u64) -> u64 {
    combine(request_hash, worker_id_hash)
}
