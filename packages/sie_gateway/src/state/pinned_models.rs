use std::collections::{HashMap, HashSet};

use crate::metrics;
use crate::types::pool::Pool;

/// Per-pool view of which models are currently loaded on the pool's healthy
/// workers. Keyed by pool name; each value is the union of loaded model ids
/// across every healthy worker assigned to / serving that pool. The caller
/// builds this from the pool's `status.assigned_workers` cross-referenced
/// with the worker registry's health + loaded models, so the reconcile step
/// stays a pure function over a plain snapshot (mirrors how `warm_floor`
/// takes `&[Pool]`).
pub type PoolLoadedModels = HashMap<String, HashSet<String>>;

/// Tracks the `(pool, model)` pinned lanes the reconciler emitted on the
/// previous tick so it can clear lanes that disappear (pool deleted or the
/// model unpinned). Keyed by pool name; each value is the set of pinned model
/// ids that pool emitted last tick.
pub type PinnedLanes = HashMap<String, HashSet<String>>;

/// One per-lane pinned-model-loaded mutation the reconciler decided on this
/// tick. Returned (in addition to being applied to the
/// `POOL_PINNED_MODEL_LOADED` gauge) so the pure reconcile step is
/// unit-testable without scraping `/metrics`.
#[derive(Debug, Clone, PartialEq)]
pub enum PinnedModelAction {
    /// Lane should report `value` (1.0 = the pinned model is loaded on at
    /// least one healthy assigned worker, 0.0 = not loaded).
    Set {
        pool: String,
        model: String,
        value: f64,
    },
    /// Lane is no longer pinned (pool deleted or model unpinned); drop its
    /// gauge series so it does not linger on `/metrics`.
    Clear { pool: String, model: String },
}

/// Reconcile the pinned-model-loaded gauge for the current set of pools.
///
/// For every pool with a non-empty `pinned_models`, emit one `Set` per pinned
/// model: `1.0` iff that model appears in the pool's loaded-model view
/// (`loaded`), `0.0` otherwise. Any `(pool, model)` lane present last tick but
/// absent now (pool deleted or model unpinned) is emitted as a `Clear`. `prev`
/// is updated in place to the lanes emitted this tick so the next call can
/// detect disappearances. Pools with an empty `pinned_models` emit nothing.
pub fn reconcile_pinned_models(
    pools: &[Pool],
    loaded: &PoolLoadedModels,
    prev: &mut PinnedLanes,
) -> Vec<PinnedModelAction> {
    let mut actions = Vec::new();
    let mut current: PinnedLanes = HashMap::new();

    for pool in pools {
        if pool.spec.pinned_models.is_empty() {
            continue;
        }
        let pool_loaded = loaded.get(&pool.spec.name);
        let lanes = current.entry(pool.spec.name.clone()).or_default();
        for model in &pool.spec.pinned_models {
            // Skip duplicate pins (the API dedupes, but a static pool spec
            // could still carry repeats) so we emit one lane per model.
            if !lanes.insert(model.clone()) {
                continue;
            }
            // Workers report `loaded_models` verbatim and the routing path
            // already compares model ids case-insensitively, so match the same
            // way here rather than an exact `contains`, which would report a
            // false 0 for a case-variant worker id.
            let is_loaded =
                pool_loaded.is_some_and(|set| set.iter().any(|m| m.eq_ignore_ascii_case(model)));
            actions.push(PinnedModelAction::Set {
                pool: pool.spec.name.clone(),
                model: model.clone(),
                value: if is_loaded { 1.0 } else { 0.0 },
            });
        }
    }

    // Clear lanes we set last tick that are no longer present this tick.
    for (pool, models) in prev.iter() {
        let now = current.get(pool);
        for model in models {
            let still_present = now.is_some_and(|set| set.contains(model));
            if !still_present {
                actions.push(PinnedModelAction::Clear {
                    pool: pool.clone(),
                    model: model.clone(),
                });
            }
        }
    }

    *prev = current;
    actions
}

/// Reconcile and apply the pinned-model-loaded gauge in one step: compute the
/// actions for this tick (updating `prev`) and push each one onto
/// `POOL_PINNED_MODEL_LOADED`.
pub fn reconcile_and_emit(pools: &[Pool], loaded: &PoolLoadedModels, prev: &mut PinnedLanes) {
    for action in reconcile_pinned_models(pools, loaded, prev) {
        match action {
            PinnedModelAction::Set { pool, model, value } => {
                metrics::set_pool_pinned_model_loaded(&pool, &model, value)
            }
            PinnedModelAction::Clear { pool, model } => {
                metrics::clear_pool_pinned_model_loaded(&pool, &model)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::pool::{PoolSpec, PoolStatus};

    fn pool(name: &str, pinned: &[&str]) -> Pool {
        Pool {
            spec: PoolSpec {
                name: name.to_string(),
                bundle: None,
                gpus: HashMap::new(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
                pinned_models: pinned.iter().map(|m| m.to_string()).collect(),
            },
            status: PoolStatus::default(),
        }
    }

    fn loaded(entries: &[(&str, &[&str])]) -> PoolLoadedModels {
        entries
            .iter()
            .map(|(pool, models)| {
                (
                    pool.to_string(),
                    models.iter().map(|m| m.to_string()).collect(),
                )
            })
            .collect()
    }

    fn set_action(pool: &str, model: &str, value: f64) -> PinnedModelAction {
        PinnedModelAction::Set {
            pool: pool.to_string(),
            model: model.to_string(),
            value,
        }
    }

    fn clear_action(pool: &str, model: &str) -> PinnedModelAction {
        PinnedModelAction::Clear {
            pool: pool.to_string(),
            model: model.to_string(),
        }
    }

    #[test]
    fn loaded_model_reports_one() {
        let mut prev = PinnedLanes::new();
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["BAAI/bge-m3"])],
            &loaded(&[("tenant", &["BAAI/bge-m3"])]),
            &mut prev,
        );
        assert_eq!(actions, vec![set_action("tenant", "BAAI/bge-m3", 1.0)]);
        assert_eq!(prev.get("tenant").unwrap().len(), 1);
    }

    #[test]
    fn unloaded_model_reports_zero() {
        let mut prev = PinnedLanes::new();
        // Pinned but not in the pool's loaded view → 0.
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["BAAI/bge-m3"])],
            &loaded(&[("tenant", &["other/model"])]),
            &mut prev,
        );
        assert_eq!(actions, vec![set_action("tenant", "BAAI/bge-m3", 0.0)]);
    }

    #[test]
    fn no_loaded_view_for_pool_reports_zero() {
        let mut prev = PinnedLanes::new();
        // No entry for the pool at all (e.g. no healthy assigned workers) → 0.
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["BAAI/bge-m3"])],
            &PoolLoadedModels::new(),
            &mut prev,
        );
        assert_eq!(actions, vec![set_action("tenant", "BAAI/bge-m3", 0.0)]);
    }

    #[test]
    fn empty_pinned_set_emits_nothing() {
        let mut prev = PinnedLanes::new();
        let actions = reconcile_pinned_models(
            &[pool("tenant", &[])],
            &loaded(&[("tenant", &["BAAI/bge-m3"])]),
            &mut prev,
        );
        assert!(actions.is_empty());
        assert!(prev.is_empty());
    }

    #[test]
    fn multi_model_multi_pool() {
        let mut prev = PinnedLanes::new();
        let actions = reconcile_pinned_models(
            &[pool("a", &["m1", "m2"]), pool("b", &["m3"])],
            &loaded(&[("a", &["m1"]), ("b", &["m3"])]),
            &mut prev,
        );
        assert_eq!(actions.len(), 3);
        assert!(actions.contains(&set_action("a", "m1", 1.0)));
        assert!(actions.contains(&set_action("a", "m2", 0.0)));
        assert!(actions.contains(&set_action("b", "m3", 1.0)));
    }

    #[test]
    fn deleting_pool_clears_its_prior_lanes() {
        let mut prev = PinnedLanes::new();
        reconcile_pinned_models(
            &[pool("tenant", &["BAAI/bge-m3"])],
            &loaded(&[("tenant", &["BAAI/bge-m3"])]),
            &mut prev,
        );
        // Pool disappears next tick.
        let actions = reconcile_pinned_models(&[], &PoolLoadedModels::new(), &mut prev);
        assert_eq!(actions, vec![clear_action("tenant", "BAAI/bge-m3")]);
        assert!(prev.is_empty());
    }

    #[test]
    fn unpinning_model_clears_lane() {
        let mut prev = PinnedLanes::new();
        reconcile_pinned_models(
            &[pool("tenant", &["m1", "m2"])],
            &loaded(&[("tenant", &["m1", "m2"])]),
            &mut prev,
        );
        // m2 unpinned next tick.
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["m1"])],
            &loaded(&[("tenant", &["m1", "m2"])]),
            &mut prev,
        );
        assert!(actions.contains(&set_action("tenant", "m1", 1.0)));
        assert!(actions.contains(&clear_action("tenant", "m2")));
        assert_eq!(prev.get("tenant").unwrap().len(), 1);
    }

    #[test]
    fn steady_state_keeps_lane_set_no_clear() {
        let mut prev = PinnedLanes::new();
        reconcile_pinned_models(
            &[pool("tenant", &["m1"])],
            &loaded(&[("tenant", &["m1"])]),
            &mut prev,
        );
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["m1"])],
            &loaded(&[("tenant", &["m1"])]),
            &mut prev,
        );
        assert_eq!(actions, vec![set_action("tenant", "m1", 1.0)]);
    }

    #[test]
    fn load_state_flips_to_one_when_model_appears() {
        let mut prev = PinnedLanes::new();
        reconcile_pinned_models(
            &[pool("tenant", &["m1"])],
            &PoolLoadedModels::new(),
            &mut prev,
        );
        // Worker loads m1 on the next tick.
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["m1"])],
            &loaded(&[("tenant", &["m1"])]),
            &mut prev,
        );
        assert_eq!(actions, vec![set_action("tenant", "m1", 1.0)]);
    }

    #[test]
    fn case_insensitive_loaded_match() {
        let mut prev = PinnedLanes::new();
        // Pinned id is the canonical `BAAI/bge-m3`; the worker reports a
        // case-variant `baai/bge-m3` (workers store loaded ids verbatim). The
        // gauge must still report loaded (1.0) and keep the canonical label.
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["BAAI/bge-m3"])],
            &loaded(&[("tenant", &["baai/bge-m3"])]),
            &mut prev,
        );
        assert_eq!(actions, vec![set_action("tenant", "BAAI/bge-m3", 1.0)]);
    }

    #[test]
    fn profile_variant_matches_exact_loaded_variant() {
        let mut prev = PinnedLanes::new();
        // Workers report the concrete `{base}:{profile}` variant id in their
        // loaded set, so a profile-qualified pin matches it exactly.
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["org/model:fp8"])],
            &loaded(&[("tenant", &["org/model:fp8"])]),
            &mut prev,
        );
        assert_eq!(actions, vec![set_action("tenant", "org/model:fp8", 1.0)]);
    }

    #[test]
    fn profile_variant_not_loaded_when_only_base_or_other_variant_present() {
        let mut prev = PinnedLanes::new();
        // A profile variant is its own concrete load, so a pinned `:fp8` reads
        // not-loaded when only the base model or a different profile is resident.
        let actions = reconcile_pinned_models(
            &[pool("tenant", &["org/model:fp8"])],
            &loaded(&[("tenant", &["org/model", "org/model:bf16"])]),
            &mut prev,
        );
        assert_eq!(actions, vec![set_action("tenant", "org/model:fp8", 0.0)]);
    }
}
