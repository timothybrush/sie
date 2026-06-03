//! Per-model scheduler registry.
//!
//! The registry owns one [`Scheduler`] per `model_id` and hands it
//! back on demand:
//!
//! * [`SchedulerRegistry::get_or_create`] returns `(Arc<Scheduler<..>>, created)`.
//!   A `created == true` flag signals the dispatcher that this was
//!   the first call for the model and a per-model drain loop needs
//!   to be spawned now; subsequent calls return the same shared
//!   [`Arc`] with `created == false`, so the scheduler's adaptive
//!   controller state and per-LoRA batcher map persist across
//!   requests.
//! * Schedulers are lazily created — models that never receive
//!   traffic never allocate one.
//!
//! There is no per-model scheduler env list any more. Active models route
//! through the Rust scheduler when their worker pool is the sidecar pool.
//! Every model that lands on a worker-sidecar goes through the
//! scheduler.
//!
//! Kept generic over `<I: HasCost, T>` so this module is testable in
//! isolation against the concrete dispatcher item / work metadata
//! types.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use super::batch_config::BatchConfig;
use super::batch_former::HasCost;
use super::engine::Scheduler;

/// Per-model scheduler registry.
pub struct SchedulerRegistry<I: HasCost, T> {
    /// Default caps + per-scheduler defaults every lazily created
    /// [`Scheduler`] starts with. Operators override by setting
    /// model-specific tunables at the `Scheduler::builder()` seam
    /// when per-model tuning lands.
    default_config: BatchConfig,
    schedulers: RwLock<HashMap<String, Arc<Scheduler<I, T>>>>,
}

impl<I, T> std::fmt::Debug for SchedulerRegistry<I, T>
where
    I: HasCost,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerRegistry")
            .field("default_config", &self.default_config)
            .finish_non_exhaustive()
    }
}

impl<I, T> SchedulerRegistry<I, T>
where
    I: HasCost + Send + Sync + 'static,
    T: Send + Sync + 'static,
{
    /// Build with the given default per-scheduler config.
    #[must_use]
    pub fn new(default_config: BatchConfig) -> Self {
        Self {
            default_config,
            schedulers: RwLock::new(HashMap::new()),
        }
    }

    /// Return the scheduler for `model_id`, lazily creating it on
    /// first call.
    ///
    /// The second tuple element is `true` only on the call that
    /// actually materialised the scheduler. The dispatcher uses it
    /// to spawn exactly one drain loop per model — see
    /// `crate::dispatcher::Dispatcher::resolve_scheduler`.
    ///
    /// Thread-safe: the hot path is a read-lock + hashmap lookup;
    /// only the very first call for a given `model_id` takes the
    /// write lock, and concurrent callers double-check under it.
    pub async fn get_or_create(&self, model_id: &str) -> (Arc<Scheduler<I, T>>, bool) {
        if let Some(s) = self.schedulers.read().await.get(model_id) {
            return (Arc::clone(s), false);
        }
        let mut map = self.schedulers.write().await;
        if let Some(s) = map.get(model_id) {
            return (Arc::clone(s), false);
        }
        let sched = Arc::new(Scheduler::new(self.default_config));
        map.insert(model_id.to_owned(), Arc::clone(&sched));
        (sched, true)
    }

    /// List the model ids that currently have an instantiated
    /// scheduler. Useful for dashboards and for the shadow-trace
    /// replay which needs to enumerate live per-model state.
    pub async fn active_models(&self) -> Vec<String> {
        self.schedulers.read().await.keys().cloned().collect()
    }

    /// Number of active schedulers. Cheap — one read-lock acquire.
    pub async fn active_count(&self) -> usize {
        self.schedulers.read().await.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    struct StubItem;
    impl HasCost for StubItem {
        fn cost(&self) -> u64 {
            1
        }
        fn original_index(&self) -> usize {
            0
        }
    }

    #[tokio::test]
    async fn get_or_create_marks_first_call_as_created() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (_, created1) = reg.get_or_create("foo").await;
        assert!(created1, "first call for a model must report created=true");
        let (_, created2) = reg.get_or_create("foo").await;
        assert!(
            !created2,
            "subsequent call for the same model must report created=false"
        );
    }

    #[tokio::test]
    async fn get_or_create_returns_same_arc_on_repeat_calls() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (a, _) = reg.get_or_create("foo").await;
        let (b, _) = reg.get_or_create("foo").await;
        // Pointer equality — lazy creation must memoize the Arc so
        // adaptive-controller state persists across dispatcher calls.
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn different_models_get_different_schedulers() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        let (a, _) = reg.get_or_create("foo").await;
        let (b, _) = reg.get_or_create("bar").await;
        assert!(!Arc::ptr_eq(&a, &b));
    }

    #[tokio::test]
    async fn active_models_reports_materialised_schedulers() {
        let reg: SchedulerRegistry<StubItem, ()> = SchedulerRegistry::new(BatchConfig::default());
        assert!(reg.active_models().await.is_empty());
        let _ = reg.get_or_create("foo").await;
        let _ = reg.get_or_create("bar").await;
        let mut active = reg.active_models().await;
        active.sort();
        assert_eq!(active, vec!["bar".to_string(), "foo".to_string()]);
        assert_eq!(reg.active_count().await, 2);
    }
}
