use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tracing::info;

use crate::metrics;

const DEMAND_EXPIRY_SECS: u64 = 120;

type DemandKey = (String, String, String);
type DemandEntry = (String, String, String, Arc<Notify>, Arc<AtomicU64>);

/// Tracks pending demand per (pool, machine_profile, bundle) with refreshable expiry deadlines.
/// Each active demand entry has a background task that clears the metric
/// after 120s of inactivity. Calling `record` resets the timer.
pub struct DemandTracker {
    /// Map from (pool_lowercase, machine_profile_lowercase, bundle_lowercase) to
    /// (original_pool, original_machine_profile, original_bundle, notify, generation).
    entries: DashMap<DemandKey, DemandEntry>,
}

impl Default for DemandTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl DemandTracker {
    pub fn new() -> Self {
        Self {
            entries: DashMap::new(),
        }
    }

    /// Record pending demand for a (pool, machine_profile, bundle) lane.
    /// Sets the PENDING_DEMAND gauge to 1.0 and starts/refreshes
    /// the 120s auto-expiry timer.
    pub fn record(self: &Arc<Self>, pool: &str, machine_profile: &str, bundle: &str) {
        let key = (
            pool.to_lowercase(),
            machine_profile.to_lowercase(),
            bundle.to_lowercase(),
        );

        // Use entry() API for atomic get-or-insert to avoid TOCTOU race
        match self.entries.entry(key.clone()) {
            Entry::Occupied(entry) => {
                let (
                    ref orig_pool,
                    ref orig_machine_profile,
                    ref orig_bundle,
                    ref notify,
                    ref generation,
                ) = *entry.get();
                // Set gauge with the original-case labels to avoid creating duplicate series
                metrics::PENDING_DEMAND
                    .with_label_values(&[orig_pool, orig_machine_profile, orig_bundle])
                    .set(1.0);
                generation.fetch_add(1, Ordering::AcqRel);
                notify.notify_one();
            }
            Entry::Vacant(entry) => {
                let notify = Arc::new(Notify::new());
                let generation = Arc::new(AtomicU64::new(0));
                entry.insert((
                    pool.to_string(),
                    machine_profile.to_string(),
                    bundle.to_string(),
                    Arc::clone(&notify),
                    Arc::clone(&generation),
                ));

                // Set gauge with the canonical label casing (first caller defines it)
                metrics::PENDING_DEMAND
                    .with_label_values(&[pool, machine_profile, bundle])
                    .set(1.0);

                let pool_owned = pool.to_string();
                let machine_profile_owned = machine_profile.to_string();
                let bundle_owned = bundle.to_string();
                let tracker = Arc::clone(self);

                tokio::spawn(async move {
                    let key = (
                        pool_owned.to_lowercase(),
                        machine_profile_owned.to_lowercase(),
                        bundle_owned.to_lowercase(),
                    );
                    loop {
                        let is_current_entry = tracker
                            .entries
                            .get(&key)
                            .map(|entry| Arc::ptr_eq(&entry.4, &generation))
                            .unwrap_or(false);
                        if !is_current_entry {
                            return;
                        }
                        let observed_generation = generation.load(Ordering::Acquire);
                        tokio::select! {
                            _ = tokio::time::sleep(Duration::from_secs(DEMAND_EXPIRY_SECS)) => {
                                let removed = tracker.entries.remove_if(&key, |_, entry| {
                                    Arc::ptr_eq(&entry.4, &generation)
                                        && entry.4.load(Ordering::Acquire) == observed_generation
                                });
                                if let Some((_, (orig_pool, orig_machine_profile, orig_bundle, _, _))) = removed {
                                    metrics::PENDING_DEMAND
                                        .with_label_values(&[&orig_pool, &orig_machine_profile, &orig_bundle])
                                        .set(0.0);
                                    info!(
                                        pool = %orig_pool,
                                        machine_profile = %orig_machine_profile,
                                        bundle = %orig_bundle,
                                        "pending demand expired (no requests for 120s)"
                                    );
                                    return;
                                }
                                let is_current_entry = tracker
                                    .entries
                                    .get(&key)
                                    .map(|entry| Arc::ptr_eq(&entry.4, &generation))
                                    .unwrap_or(false);
                                if !is_current_entry {
                                    return;
                                }
                                continue;
                            }
                            _ = notify.notified() => {
                                continue;
                            }
                        }
                    }
                });
            }
        }
    }

    /// Explicitly clear demand for a (pool, machine_profile, bundle) lane.
    /// Removes the entry and zeros the gauge.
    pub fn clear(&self, pool: &str, machine_profile: &str, bundle: &str) {
        let key = (
            pool.to_lowercase(),
            machine_profile.to_lowercase(),
            bundle.to_lowercase(),
        );
        if let Some((_, (orig_pool, orig_machine_profile, orig_bundle, notify, _))) =
            self.entries.remove(&key)
        {
            notify.notify_one();
            metrics::PENDING_DEMAND
                .with_label_values(&[&orig_pool, &orig_machine_profile, &orig_bundle])
                .set(0.0);
        } else {
            // No tracked entry, but clear the gauge anyway with the provided values
            metrics::PENDING_DEMAND
                .with_label_values(&[pool, machine_profile, bundle])
                .set(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_record_sets_demand_gauge() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("default", "l4-spot", "default");
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["default", "l4-spot", "default"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);
        tracker.clear("default", "l4-spot", "default");
    }

    #[tokio::test]
    async fn test_demand_entry_exists_after_record() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("test-pool-expire", "test-gpu-expire", "test-bundle-expire");
        assert!(tracker.entries.contains_key(&(
            "test-pool-expire".to_string(),
            "test-gpu-expire".to_string(),
            "test-bundle-expire".to_string()
        )));
        tracker.clear("test-pool-expire", "test-gpu-expire", "test-bundle-expire");
    }

    #[tokio::test]
    async fn test_record_refreshes_existing_timer() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("default", "l4-refresh", "default");
        // Record again -- should not create a second entry
        tracker.record("default", "l4-refresh", "default");
        assert!(tracker.entries.contains_key(&(
            "default".to_string(),
            "l4-refresh".to_string(),
            "default".to_string()
        )));
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["default", "l4-refresh", "default"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);
        tracker.clear("default", "l4-refresh", "default");
    }

    #[tokio::test]
    async fn test_record_refreshes_expiry_generation() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("default", "l4-generation", "default");
        let key = (
            "default".to_string(),
            "l4-generation".to_string(),
            "default".to_string(),
        );
        let before = tracker
            .entries
            .get(&key)
            .expect("entry should exist")
            .4
            .load(Ordering::Acquire);

        tracker.record("default", "l4-generation", "default");

        let after = tracker
            .entries
            .get(&key)
            .expect("entry should still exist")
            .4
            .load(Ordering::Acquire);
        assert_eq!(after, before + 1);
        tracker.clear("default", "l4-generation", "default");
    }

    #[tokio::test]
    async fn test_clear_removes_entry_and_zeros_gauge() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("default", "l4-clear", "default");
        tracker.clear("default", "l4-clear", "default");
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["default", "l4-clear", "default"])
            .get();
        assert!((val - 0.0).abs() < f64::EPSILON);
        assert!(!tracker.entries.contains_key(&(
            "default".to_string(),
            "l4-clear".to_string(),
            "default".to_string()
        )));
    }

    #[tokio::test]
    async fn test_clear_nonexistent_is_noop() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        // Should not panic
        tracker.clear("nonexistent", "nonexistent", "nonexistent");
    }

    #[tokio::test]
    async fn test_case_insensitive_key_matching() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("Default", "L4-Spot", "Premium");
        // Second record with same case should refresh, not create new
        tracker.record("Default", "L4-Spot", "Premium");
        assert_eq!(tracker.entries.len(), 1);
        // Gauge was set with the original case labels
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["Default", "L4-Spot", "Premium"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);
        tracker.clear("DEFAULT", "L4-SPOT", "PREMIUM");
        assert_eq!(tracker.entries.len(), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn test_timer_refresh_extends_expiry() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());

        // Record demand
        tracker.record("default", "l4-timer", "default-timer");
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["default", "l4-timer", "default-timer"])
            .get();
        assert!((val - 1.0).abs() < f64::EPSILON);

        // Advance 100s (< 120s expiry), then refresh.
        tokio::time::advance(Duration::from_secs(100)).await;
        tokio::task::yield_now().await;
        tracker.record("default", "l4-timer", "default-timer");

        // Let the spawned task process the notify and loop back to start
        // a new sleep(120s) before we advance time further.
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Advance another 100s. The spawned task restarted its 120s sleep
        // after the notify wakeup, so this 100s advance does NOT expire it.
        tokio::time::advance(Duration::from_secs(100)).await;
        tokio::task::yield_now().await;

        // Gauge should STILL be 1.0 because the timer was refreshed
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["default", "l4-timer", "default-timer"])
            .get();
        assert!(
            (val - 1.0).abs() < f64::EPSILON,
            "expected gauge=1.0 after refresh, got {val}"
        );

        // Advance 25s more (125s since last record, past the 120s expiry).
        tokio::time::advance(Duration::from_secs(25)).await;
        tokio::task::yield_now().await;
        tokio::task::yield_now().await;

        // Gauge should now be 0.0 (expired)
        let val = metrics::PENDING_DEMAND
            .with_label_values(&["default", "l4-timer", "default-timer"])
            .get();
        assert!(
            (val - 0.0).abs() < f64::EPSILON,
            "expected gauge=0.0 after expiry, got {val}"
        );
    }

    #[tokio::test]
    async fn test_multiple_gpu_bundle_pairs_independent() {
        let _ = &*metrics::REGISTRY;
        let tracker = Arc::new(DemandTracker::new());
        tracker.record("default", "l4-multi", "default");
        tracker.record("tenant-a", "l4-multi", "default");
        tracker.record("default", "a100-multi", "premium");
        assert_eq!(tracker.entries.len(), 3);
        tracker.clear("default", "l4-multi", "default");
        assert_eq!(tracker.entries.len(), 2);
        assert!(tracker.entries.contains_key(&(
            "tenant-a".to_string(),
            "l4-multi".to_string(),
            "default".to_string()
        )));
        tracker.clear("tenant-a", "l4-multi", "default");
        tracker.clear("default", "a100-multi", "premium");
    }
}
