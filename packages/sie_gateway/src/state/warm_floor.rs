use std::collections::{HashMap, HashSet};

use tracing::warn;

use crate::metrics;
use crate::state::pool_manager::machine_profiles_for_pool;
use crate::types::pool::Pool;

/// Tracks the warm-floor lanes the emitter set on the previous tick so it
/// can clear lanes that disappear (pool deleted, `minimum_worker_count`
/// dropped to 0, or the profile/bundle set changed). Keyed by pool name; each
/// value is the set of `(machine_profile, bundle)` lanes that pool emitted
/// last tick.
pub type WarmFloorLanes = HashMap<String, HashSet<(String, String)>>;

/// One per-lane warm-floor mutation the reconciler decided on this tick.
/// Returned (in addition to being applied to the `POOL_WARM_FLOOR` gauge)
/// so the pure reconcile step is unit-testable without scraping `/metrics`.
#[derive(Debug, Clone, PartialEq)]
pub enum WarmFloorAction {
    /// Lane should hold `value` warm machines (the pool's
    /// `minimum_worker_count`).
    Set {
        pool: String,
        machine_profile: String,
        bundle: String,
        value: f64,
    },
    /// Lane is no longer wanted; drive its gauge to 0 so KEDA can scale it
    /// back down. `or vector(0)` in the trigger keeps scale-from-zero intact.
    Clear {
        pool: String,
        machine_profile: String,
        bundle: String,
    },
}

/// Lowercased bundle label for a pool, mirroring the `pending_demand`
/// convention: an unset bundle maps to `"default"`.
fn bundle_label(pool: &Pool) -> String {
    pool.spec
        .bundle
        .as_deref()
        .unwrap_or("default")
        .to_lowercase()
}

/// Reconcile the warm-floor gauge for the current set of pools.
///
/// For every pool with `minimum_worker_count > 0` and at least one nameable
/// machine profile, emit one `Set` per `(machine_profile, bundle)` lane. Any lane
/// present last tick but absent now is emitted as a `Clear`. `prev` is
/// updated in place to the lanes emitted this tick so the next call can
/// detect disappearances.
///
/// A pool with no nameable machine profile (e.g. the bare `default` pool
/// with empty `gpus`/`gpu_caps`) cannot name a lane; it emits nothing, and
/// if it nonetheless requested a floor we log one `warn!` that the floor is
/// unenforceable without a machine profile.
///
/// Pool names are already normalized lowercase on create; profiles come
/// lowercased from [`machine_profiles_for_pool`]; the bundle is lowercased
/// here — so the labels match the `pending_demand` lane conventions exactly.
pub fn reconcile_warm_floor(pools: &[Pool], prev: &mut WarmFloorLanes) -> Vec<WarmFloorAction> {
    let mut actions = Vec::new();
    let mut current: WarmFloorLanes = HashMap::new();

    for pool in pools {
        if pool.spec.minimum_worker_count == 0 {
            continue;
        }
        let profiles = machine_profiles_for_pool(pool);
        if profiles.is_empty() {
            warn!(
                pool = %pool.spec.name,
                minimum_worker_count = pool.spec.minimum_worker_count,
                "warm floor requested but pool has no machine profile; floor is unenforceable"
            );
            continue;
        }
        let bundle = bundle_label(pool);
        let value = f64::from(pool.spec.minimum_worker_count);
        let lanes = current.entry(pool.spec.name.clone()).or_default();
        for profile in profiles {
            lanes.insert((profile.clone(), bundle.clone()));
            actions.push(WarmFloorAction::Set {
                pool: pool.spec.name.clone(),
                machine_profile: profile,
                bundle: bundle.clone(),
                value,
            });
        }
    }

    // Clear lanes we set last tick that are no longer present this tick.
    for (pool, lanes) in prev.iter() {
        let now = current.get(pool);
        for (machine_profile, bundle) in lanes {
            let still_present =
                now.is_some_and(|set| set.contains(&(machine_profile.clone(), bundle.clone())));
            if !still_present {
                actions.push(WarmFloorAction::Clear {
                    pool: pool.clone(),
                    machine_profile: machine_profile.clone(),
                    bundle: bundle.clone(),
                });
            }
        }
    }

    *prev = current;
    actions
}

/// Reconcile and apply the warm-floor gauge in one step: compute the actions
/// for this tick (updating `prev`) and push each one onto `POOL_WARM_FLOOR`.
pub fn reconcile_and_emit(pools: &[Pool], prev: &mut WarmFloorLanes) {
    for action in reconcile_warm_floor(pools, prev) {
        match action {
            WarmFloorAction::Set {
                pool,
                machine_profile,
                bundle,
                value,
            } => metrics::set_pool_warm_floor(&pool, &machine_profile, &bundle, value),
            WarmFloorAction::Clear {
                pool,
                machine_profile,
                bundle,
            } => metrics::clear_pool_warm_floor(&pool, &machine_profile, &bundle),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::pool::{PoolSpec, PoolStatus};

    fn pool(name: &str, minimum_worker_count: u32, gpus: &[&str], bundle: Option<&str>) -> Pool {
        let gpus = gpus
            .iter()
            .map(|g| (g.to_string(), 0u32))
            .collect::<HashMap<_, _>>();
        Pool {
            spec: PoolSpec {
                name: name.to_string(),
                bundle: bundle.map(|b| b.to_string()),
                gpus,
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count,
                pinned_models: Vec::new(),
            },
            status: PoolStatus::default(),
        }
    }

    fn set_action(pool: &str, profile: &str, bundle: &str, value: f64) -> WarmFloorAction {
        WarmFloorAction::Set {
            pool: pool.to_string(),
            machine_profile: profile.to_string(),
            bundle: bundle.to_string(),
            value,
        }
    }

    fn clear_action(pool: &str, profile: &str, bundle: &str) -> WarmFloorAction {
        WarmFloorAction::Clear {
            pool: pool.to_string(),
            machine_profile: profile.to_string(),
            bundle: bundle.to_string(),
        }
    }

    #[test]
    fn single_profile_emits_one_lane() {
        let mut prev = WarmFloorLanes::new();
        let actions = reconcile_warm_floor(&[pool("tenant", 2, &["l4"], None)], &mut prev);
        assert_eq!(actions, vec![set_action("tenant", "l4", "default", 2.0)]);
        assert_eq!(prev.get("tenant").unwrap().len(), 1);
    }

    #[test]
    fn multi_profile_emits_one_lane_each() {
        let mut prev = WarmFloorLanes::new();
        let actions = reconcile_warm_floor(&[pool("tenant", 2, &["l4", "a100"], None)], &mut prev);
        assert_eq!(actions.len(), 2);
        assert!(actions.contains(&set_action("tenant", "l4", "default", 2.0)));
        assert!(actions.contains(&set_action("tenant", "a100", "default", 2.0)));
    }

    #[test]
    fn bundle_label_is_lowercased() {
        let mut prev = WarmFloorLanes::new();
        let actions =
            reconcile_warm_floor(&[pool("tenant", 1, &["L4"], Some("SGLang"))], &mut prev);
        assert_eq!(actions, vec![set_action("tenant", "l4", "sglang", 1.0)]);
    }

    #[test]
    fn zero_minimum_worker_count_emits_nothing() {
        let mut prev = WarmFloorLanes::new();
        let actions = reconcile_warm_floor(&[pool("tenant", 0, &["l4"], None)], &mut prev);
        assert!(actions.is_empty());
        assert!(prev.is_empty());
    }

    #[test]
    fn no_profile_pool_emits_nothing() {
        let mut prev = WarmFloorLanes::new();
        let actions = reconcile_warm_floor(&[pool("default", 3, &[], None)], &mut prev);
        assert!(actions.is_empty());
        assert!(prev.is_empty());
    }

    #[test]
    fn deleting_pool_clears_its_prior_lanes() {
        let mut prev = WarmFloorLanes::new();
        reconcile_warm_floor(&[pool("tenant", 2, &["l4"], None)], &mut prev);
        // Pool disappears next tick.
        let actions = reconcile_warm_floor(&[], &mut prev);
        assert_eq!(actions, vec![clear_action("tenant", "l4", "default")]);
        assert!(prev.is_empty());
    }

    #[test]
    fn zeroing_minimum_worker_count_clears_lane() {
        let mut prev = WarmFloorLanes::new();
        reconcile_warm_floor(&[pool("tenant", 2, &["l4"], None)], &mut prev);
        let actions = reconcile_warm_floor(&[pool("tenant", 0, &["l4"], None)], &mut prev);
        assert_eq!(actions, vec![clear_action("tenant", "l4", "default")]);
        assert!(prev.is_empty());
    }

    #[test]
    fn changing_profile_clears_old_lane_and_sets_new() {
        let mut prev = WarmFloorLanes::new();
        reconcile_warm_floor(&[pool("tenant", 2, &["l4"], None)], &mut prev);
        let actions = reconcile_warm_floor(&[pool("tenant", 2, &["a100"], None)], &mut prev);
        assert!(actions.contains(&set_action("tenant", "a100", "default", 2.0)));
        assert!(actions.contains(&clear_action("tenant", "l4", "default")));
    }

    #[test]
    fn steady_state_keeps_lane_set_no_clear() {
        let mut prev = WarmFloorLanes::new();
        reconcile_warm_floor(&[pool("tenant", 2, &["l4"], None)], &mut prev);
        let actions = reconcile_warm_floor(&[pool("tenant", 2, &["l4"], None)], &mut prev);
        assert_eq!(actions, vec![set_action("tenant", "l4", "default", 2.0)]);
    }
}
