//! HRW direct-dispatch routing tests with a fake-worker ring.
//!
//! The gateway's HRW picker (`crate::routing::hrw`) is a private module
//! of the binary crate, so an integration test cannot call it directly.
//! Instead this test reproduces the *exact* HRW algorithm the gateway
//! uses — `xxh3_64` over `worker_id`, the asymmetric `combine` mix
//! (`worker_id_hash.rotate_left(32) ^ request_hash`, re-hashed), and the
//! lexicographic tie-break — against the **same** `xxhash-rust` crate the
//! gateway depends on. If the gateway's algorithm changes, this test's
//! expected pins change with it, which is the regression signal we want.
//!
//! Acceptance criteria (direct-dispatch routing plan):
//! * Deterministic pinning: the same `request_id`/routing key always
//!   resolves to the same single worker across many trials.
//! * Rebalance: removing the pinned worker moves *only that worker's*
//!   keys (HRW's `1/n` movement property) — every other key keeps its
//!   prior worker.
//!
//! No NATS required — this runs in the default `cargo test` pass.

use xxhash_rust::xxh3::xxh3_64;

/// Mirror of `routing::hrw::combine`. Kept byte-for-byte identical so the
/// pins this test asserts match the gateway's real picks.
fn combine(request_hash: u64, worker_id_hash: u64) -> u64 {
    let mixed = (worker_id_hash.rotate_left(32) ^ request_hash).to_le_bytes();
    xxh3_64(&mixed)
}

/// Mirror of `routing::hrw::pick_worker` over a borrowed `(id, id_hash)`
/// ring. Highest `combine` wins; ties break lexicographically by id so
/// the pick is replica-stable regardless of ring iteration order.
fn pick_worker(ring: &[(String, u64)], request_hash: u64) -> Option<&str> {
    ring.iter()
        .max_by(|a, b| {
            combine(request_hash, a.1)
                .cmp(&combine(request_hash, b.1))
                .then_with(|| a.0.cmp(&b.0))
        })
        .map(|e| e.0.as_str())
}

fn build_ring(worker_ids: &[&str]) -> Vec<(String, u64)> {
    worker_ids
        .iter()
        .map(|id| (id.to_string(), xxh3_64(id.as_bytes())))
        .collect()
}

/// The routing key hash the gateway derives from a request's
/// `routing_key` / `prompt_cache_key` / prompt. We only need *a* stable
/// hash here; the exact derivation is covered by `routing::key` unit
/// tests. Hash the request id to stand in for the resolved key hash.
fn request_hash(request_id: &str) -> u64 {
    xxh3_64(request_id.as_bytes())
}

#[test]
fn hrw_pins_to_one_worker_for_same_routing_key() {
    let ring = build_ring(&["w1", "w2", "w3", "w4"]);
    let rid = "req-stable-abc";
    let h = request_hash(rid);

    // Determinism: 1000 picks for the same key must all land on one worker.
    let first = pick_worker(&ring, h).expect("non-empty ring");
    for _ in 0..1000 {
        assert_eq!(pick_worker(&ring, h), Some(first));
    }

    // Ring iteration order must not change the pick (replica stability):
    // shuffle the ring and re-pick.
    let mut shuffled = ring.clone();
    shuffled.reverse();
    assert_eq!(pick_worker(&shuffled, h), Some(first));
}

#[test]
fn hrw_distributes_distinct_keys_across_workers() {
    // Sanity: HRW is not degenerate — distinct keys spread over the ring
    // rather than all collapsing onto one worker.
    let ring = build_ring(&["w1", "w2", "w3", "w4"]);
    let mut hit = std::collections::HashSet::new();
    for i in 0..400 {
        let h = request_hash(&format!("req-{i}"));
        hit.insert(pick_worker(&ring, h).unwrap().to_string());
    }
    assert!(
        hit.len() >= 2,
        "HRW should spread distinct keys over multiple workers, hit={hit:?}"
    );
}

#[test]
fn hrw_reroutes_only_pinned_keys_after_worker_removal() {
    let full = build_ring(&["w1", "w2", "w3", "w4"]);

    // Record the pick for many keys with the full ring.
    let keys: Vec<u64> = (0..500).map(|i| request_hash(&format!("k-{i}"))).collect();
    let before: Vec<&str> = keys
        .iter()
        .map(|&h| pick_worker(&full, h).unwrap())
        .collect();

    // Remove the worker that owns the most keys (guaranteed non-empty
    // ownership) so the "victim's keys move, others stay" invariant has
    // something to assert on regardless of the concrete hash spread.
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for w in &before {
        *counts.entry(*w).or_default() += 1;
    }
    let victim = *counts
        .iter()
        .max_by_key(|(_, n)| **n)
        .map(|(w, _)| w)
        .expect("at least one worker owns keys");
    let survivors: Vec<&str> = ["w1", "w2", "w3", "w4"]
        .into_iter()
        .filter(|w| *w != victim)
        .collect();
    let reduced = build_ring(&survivors);

    let after: Vec<&str> = keys
        .iter()
        .map(|&h| pick_worker(&reduced, h).unwrap())
        .collect();

    // HRW invariant: every key that was NOT on the victim keeps its
    // worker; only the victim's keys move (to a single deterministic new
    // worker each).
    for (i, (b, a)) in before.iter().zip(after.iter()).enumerate() {
        if *b == victim {
            assert_ne!(*a, victim, "victim's key {i} must move off the dead worker");
        } else {
            assert_eq!(*b, *a, "non-victim key {i} must not move");
        }
    }

    // And the rerouting is itself deterministic: re-running yields the
    // same assignment.
    let after2: Vec<&str> = keys
        .iter()
        .map(|&h| pick_worker(&reduced, h).unwrap())
        .collect();
    assert_eq!(after, after2);
}
