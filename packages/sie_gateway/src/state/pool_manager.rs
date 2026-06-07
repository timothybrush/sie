use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::state::k8s_pool_backend::K8sPoolBackend;
use crate::types::pool::{AssignedWorker, Pool, PoolSpec, PoolState, PoolStatus};

pub const DEFAULT_POOL_NAME: &str = "default";
const DEFAULT_LEASE_DURATION_S: f64 = 1200.0; // 20 minutes
const TIMESTAMP_TOLERANCE_S: f64 = 0.001;
type WorkerAssignment = (String, String, String, String, String);

#[derive(Debug)]
pub struct DefaultPoolProtectedError;

impl std::fmt::Display for DefaultPoolProtectedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cannot delete the default pool '{}'", DEFAULT_POOL_NAME)
    }
}

impl std::error::Error for DefaultPoolProtectedError {}

#[derive(Debug)]
pub struct DefaultPoolMutationError;

impl std::fmt::Display for DefaultPoolMutationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Cannot modify the default pool '{}'", DEFAULT_POOL_NAME)
    }
}

impl std::error::Error for DefaultPoolMutationError {}

#[derive(Debug)]
pub struct InvalidMachineProfileError {
    pub invalid_profiles: Vec<String>,
    pub valid_profiles: Vec<String>,
}

impl std::fmt::Display for InvalidMachineProfileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Unknown machine profiles: {:?}. Valid profiles: {:?}",
            self.invalid_profiles, self.valid_profiles
        )
    }
}

impl std::error::Error for InvalidMachineProfileError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolAdmissionStatus {
    pub cap: u32,
    pub assigned_count: usize,
}

pub struct PoolManager {
    pools: RwLock<HashMap<String, Pool>>,
    lease_duration_s: f64,
    configured_profiles: Vec<String>,
    /// Optional K8s ConfigMap backend for pool persistence.
    k8s_backend: Option<Arc<K8sPoolBackend>>,
}

impl PoolManager {
    pub fn new(configured_profiles: Vec<String>) -> Self {
        Self {
            pools: RwLock::new(HashMap::new()),
            lease_duration_s: DEFAULT_LEASE_DURATION_S,
            configured_profiles,
            k8s_backend: None,
        }
    }

    /// Attach a K8s pool backend for persistent pool storage.
    #[allow(dead_code)]
    pub fn with_k8s_backend(mut self, backend: Arc<K8sPoolBackend>) -> Self {
        self.k8s_backend = Some(backend);
        self
    }

    /// Restore pools from the K8s backend on startup.
    pub async fn restore_from_k8s(&self) -> Result<usize, String> {
        let backend = match &self.k8s_backend {
            Some(b) => b,
            None => return Ok(0),
        };

        let k8s_pools = backend.list_pools().await?;
        let mut pools = self.pools.write().await;
        let mut count = 0;
        for pool in k8s_pools {
            if !Self::should_restore_pool_from_k8s(&pool) {
                continue;
            }
            if !pools.contains_key(&pool.spec.name) {
                info!(pool = %pool.spec.name, "restored pool from K8s");
                pools.insert(pool.spec.name.clone(), pool);
                count += 1;
            }
        }
        Ok(count)
    }

    fn should_restore_pool_from_k8s(pool: &Pool) -> bool {
        pool.spec.name != DEFAULT_POOL_NAME
    }

    fn now_secs() -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
    }

    /// Compute the Lease TTL in whole seconds from the pool spec or global default.
    fn lease_ttl_seconds(pool: &Pool) -> i32 {
        pool.spec
            .ttl_seconds
            .map(|s| s.min(i32::MAX as u64) as i32)
            .unwrap_or((DEFAULT_LEASE_DURATION_S as u64).min(i32::MAX as u64) as i32)
    }

    pub async fn create_default_pool(&self) {
        if self.configured_profiles.is_empty() {
            info!("no machine profiles configured, skipping default pool creation");
            return;
        }

        let gpus: HashMap<String, u32> = self
            .configured_profiles
            .iter()
            .map(|p| (p.clone(), 0))
            .collect();

        match self
            .create_pool(DEFAULT_POOL_NAME, gpus, None, None, 0)
            .await
        {
            Ok(_) => {
                info!(
                    profiles = ?self.configured_profiles,
                    "created default pool"
                );
            }
            Err(e) => {
                warn!(error = %e, "failed to create default pool");
            }
        }
    }

    pub async fn create_pool(
        &self,
        name: &str,
        gpus: HashMap<String, u32>,
        bundle: Option<String>,
        ttl_seconds: Option<u64>,
        minimum_worker_count: u32,
    ) -> Result<Pool, Box<dyn std::error::Error + Send + Sync>> {
        self.create_pool_with_caps(
            name,
            gpus,
            HashMap::new(),
            bundle,
            ttl_seconds,
            minimum_worker_count,
        )
        .await
    }

    pub async fn create_pool_with_caps(
        &self,
        name: &str,
        mut gpus: HashMap<String, u32>,
        gpu_caps: HashMap<String, u32>,
        bundle: Option<String>,
        ttl_seconds: Option<u64>,
        minimum_worker_count: u32,
    ) -> Result<Pool, Box<dyn std::error::Error + Send + Sync>> {
        for gpu_type in gpu_caps.keys() {
            if !gpus
                .keys()
                .any(|required_gpu| required_gpu.eq_ignore_ascii_case(gpu_type))
            {
                gpus.insert(gpu_type.clone(), 0);
            }
        }

        // Validate profiles
        if !self.configured_profiles.is_empty() {
            let invalid: Vec<String> = gpus
                .keys()
                .chain(gpu_caps.keys())
                .filter(|k| {
                    !self
                        .configured_profiles
                        .iter()
                        .any(|p| p.eq_ignore_ascii_case(k))
                })
                .cloned()
                .collect();

            if !invalid.is_empty() {
                return Err(Box::new(InvalidMachineProfileError {
                    invalid_profiles: invalid,
                    valid_profiles: self.configured_profiles.clone(),
                }));
            }
        }

        let now = Self::now_secs();
        let mut pools = self.pools.write().await;

        // Idempotent: return existing pool if found
        if let Some(existing) = pools.get_mut(name) {
            let spec_changed = existing.spec.bundle != bundle
                || existing.spec.gpus != gpus
                || existing.spec.gpu_caps != gpu_caps
                || existing.spec.ttl_seconds != ttl_seconds
                || existing.spec.minimum_worker_count != minimum_worker_count;

            if name == DEFAULT_POOL_NAME && spec_changed {
                return Err(Box::new(DefaultPoolMutationError));
            }

            existing.status.last_renewed = now;
            let event = if spec_changed {
                existing.spec.bundle = bundle;
                existing.spec.gpus = gpus;
                existing.spec.gpu_caps = gpu_caps;
                existing.spec.ttl_seconds = ttl_seconds;
                existing.spec.minimum_worker_count = minimum_worker_count;
                existing.status.state = PoolState::Pending;
                existing.status.assigned_workers.clear();
                "updated"
            } else {
                "renewed"
            };
            let result = existing.clone();
            drop(pools); // release write lock before K8s call

            // Persist spec updates and renew the K8s Lease (best-effort,
            // skip default pool).
            if name != DEFAULT_POOL_NAME {
                if let Some(ref backend) = self.k8s_backend {
                    if spec_changed {
                        if let Err(e) = backend.save_pool(&result).await {
                            warn!(error = %e, pool = name, "failed to persist pool update to K8s");
                        }
                    }
                    let ttl = Self::lease_ttl_seconds(&result);
                    if let Err(e) = backend.create_or_renew_lease(name, ttl).await {
                        warn!(error = %e, pool = name, "failed to renew K8s Lease");
                    }
                }
            }

            crate::metrics::POOL_EVENTS
                .with_label_values(&[event])
                .inc();
            return Ok(result);
        }

        let pool = Pool {
            spec: PoolSpec {
                name: name.to_string(),
                bundle,
                gpus,
                gpu_caps,
                ttl_seconds,
                minimum_worker_count,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: now,
                last_renewed: now,
            },
        };

        pools.insert(name.to_string(), pool.clone());
        drop(pools); // release write lock before K8s call
        info!(pool = name, "created pool");
        crate::metrics::POOL_EVENTS
            .with_label_values(&["created"])
            .inc();

        // Persist to K8s backend (best-effort)
        if let Some(ref backend) = self.k8s_backend {
            if let Err(e) = backend.save_pool(&pool).await {
                warn!(error = %e, pool = name, "failed to persist pool to K8s");
            }

            // Create K8s Lease for crash-safe TTL (skip default pool)
            if name != DEFAULT_POOL_NAME {
                let ttl = Self::lease_ttl_seconds(&pool);
                if let Err(e) = backend.create_or_renew_lease(name, ttl).await {
                    warn!(error = %e, pool = name, "failed to create K8s Lease");
                }
            }
        }

        Ok(pool)
    }

    pub async fn get_pool(&self, name: &str) -> Option<Pool> {
        let pools = self.pools.read().await;
        pools.get(name).cloned()
    }

    pub async fn capped_lane_status(
        &self,
        pool_name: &str,
        machine_profile: &str,
        bundle: &str,
    ) -> Option<PoolAdmissionStatus> {
        if machine_profile.is_empty() {
            return None;
        }

        let pools = self.pools.read().await;
        let pool = pools.get(pool_name)?;
        let cap = pool
            .spec
            .gpu_caps
            .iter()
            .find(|(profile, _)| profile.eq_ignore_ascii_case(machine_profile))
            .map(|(_, cap)| *cap)?;
        let assigned_count = pool
            .status
            .assigned_workers
            .iter()
            .filter(|worker| worker.gpu.eq_ignore_ascii_case(machine_profile))
            .filter(|worker| worker.bundle.eq_ignore_ascii_case(bundle))
            .count();

        Some(PoolAdmissionStatus {
            cap,
            assigned_count,
        })
    }

    /// Return the assigned worker names for a capped concrete lane.
    ///
    /// `None` means the pool/profile is uncapped, so callers should not apply
    /// an admission filter. `Some(empty)` means admission is enabled for this
    /// profile but no worker is currently admitted. This mirrors the
    /// worker-side pool-admission gate: named pools with missing status fail
    /// closed, while the default pool remains fail-open.
    pub async fn admitted_worker_names_for_capped_lane(
        &self,
        pool_name: &str,
        machine_profile: &str,
        bundle: &str,
    ) -> Option<HashSet<String>> {
        if machine_profile.trim().is_empty() {
            return None;
        }

        let pools = self.pools.read().await;
        let Some(pool) = pools.get(pool_name) else {
            return if pool_name.eq_ignore_ascii_case(DEFAULT_POOL_NAME) {
                None
            } else {
                Some(HashSet::new())
            };
        };
        let cap = pool
            .spec
            .gpu_caps
            .iter()
            .find(|(profile, _)| profile.eq_ignore_ascii_case(machine_profile))
            .map(|(_, cap)| *cap)?;
        if cap == 0 {
            return Some(HashSet::new());
        }

        Some(
            pool.status
                .assigned_workers
                .iter()
                .filter(|worker| worker.gpu.eq_ignore_ascii_case(machine_profile))
                .filter(|worker| worker.bundle.eq_ignore_ascii_case(bundle))
                .map(|worker| worker.name.clone())
                .collect(),
        )
    }

    /// Return the pool's only configured machine profile, if the spec is
    /// unambiguous across both requirements (`gpus`) and admission caps
    /// (`gpu_caps`). Used by cold queue routing when a caller pins only a
    /// pool: the NATS subject still needs a concrete machine-profile token so
    /// KEDA can observe `pending_demand{pool,machine_profile,bundle}`.
    pub async fn single_machine_profile_for_pool(&self, pool_name: &str) -> Option<String> {
        let pools = self.pools.read().await;
        let pool = pools.get(pool_name)?;
        let mut profiles: Vec<String> = Vec::new();
        for profile in pool.spec.gpus.keys().chain(pool.spec.gpu_caps.keys()) {
            if profiles
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(profile))
            {
                continue;
            }
            profiles.push(profile.to_lowercase());
        }
        if profiles.len() == 1 {
            profiles.pop()
        } else {
            None
        }
    }

    pub async fn list_pools(&self) -> Vec<Pool> {
        let pools = self.pools.read().await;
        pools.values().cloned().collect()
    }

    pub async fn delete_pool(&self, name: &str) -> Result<bool, DefaultPoolProtectedError> {
        if name == DEFAULT_POOL_NAME {
            return Err(DefaultPoolProtectedError);
        }

        let mut pools = self.pools.write().await;
        let removed = pools.remove(name).is_some();
        drop(pools); // release write lock before K8s call

        if removed {
            info!(pool = name, "deleted pool");
            crate::metrics::POOL_EVENTS
                .with_label_values(&["deleted"])
                .inc();
            if let Some(ref backend) = self.k8s_backend {
                if let Err(e) = backend.delete_pool(name).await {
                    warn!(error = %e, pool = name, "failed to delete pool from K8s");
                }
                if let Err(e) = backend.delete_lease(name).await {
                    warn!(error = %e, pool = name, "failed to delete K8s Lease");
                }
            }
        }
        Ok(removed)
    }

    pub async fn renew_pool(&self, name: &str) -> bool {
        let found = {
            let mut pools = self.pools.write().await;
            if let Some(pool) = pools.get_mut(name) {
                pool.status.last_renewed = Self::now_secs();
                true
            } else {
                false
            }
        }; // write lock dropped here

        // Renew the K8s Lease (best-effort, skip default pool)
        if found && name != DEFAULT_POOL_NAME {
            if let Some(ref backend) = self.k8s_backend {
                if let Err(e) = backend.renew_lease(name).await {
                    warn!(error = %e, pool = name, "failed to renew K8s Lease");
                }
            }
        }

        if found {
            crate::metrics::POOL_EVENTS
                .with_label_values(&["renewed"])
                .inc();
        }
        found
    }

    pub async fn assign_workers(
        &self,
        pool_name: &str,
        available_workers: &[WorkerAssignment], // (name, url, gpu, bundle, queue_pool)
    ) -> bool {
        let mut pools = self.pools.write().await;
        let pool = match pools.get_mut(pool_name) {
            Some(p) => p,
            None => return false,
        };

        let filtered: Vec<&WorkerAssignment> = available_workers
            .iter()
            .filter(|(_, _, _, _, queue_pool)| worker_consumes_pool(queue_pool, pool_name))
            .filter(|(_, _, _, bundle, _)| match pool.spec.bundle.as_ref() {
                Some(bundle_filter) => bundle == bundle_filter,
                None => true,
            })
            .collect();

        // Group by GPU type (lowercase)
        let mut workers_by_gpu: HashMap<String, Vec<&WorkerAssignment>> = HashMap::new();
        for w in &filtered {
            workers_by_gpu
                .entry(w.2.to_lowercase())
                .or_default()
                .push(w);
        }
        for workers in workers_by_gpu.values_mut() {
            // HA gateway replicas discover the same K8s endpoint set and each
            // computes capped admission locally. Stable ordering keeps the
            // admitted pod set convergent no matter how HashMap iteration lands.
            workers.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        }

        let mut assigned: Vec<AssignedWorker> = Vec::new();
        let mut all_met = true;
        let gpu_caps: HashMap<String, u32> = pool
            .spec
            .gpu_caps
            .iter()
            .map(|(gpu, cap)| (gpu.to_lowercase(), *cap))
            .collect();

        for (gpu_type, required_count) in &pool.spec.gpus {
            let gpu_lower = gpu_type.to_lowercase();
            let available = workers_by_gpu.get_mut(&gpu_lower);
            let available_count = available.as_ref().map(|workers| workers.len()).unwrap_or(0);
            let required = *required_count as usize;

            if available_count < required {
                all_met = false;
            }

            if let Some(workers) = available {
                let cap = gpu_caps
                    .get(&gpu_lower)
                    .map(|cap| *cap as usize)
                    .unwrap_or(usize::MAX);
                let take = workers.len().min(cap);
                for w in workers.drain(..take) {
                    assigned.push(AssignedWorker {
                        name: w.0.clone(),
                        url: w.1.clone(),
                        gpu: w.2.clone(),
                        bundle: w.3.clone(),
                    });
                }
            }
        }

        let state = if pool_name == DEFAULT_POOL_NAME {
            if assigned.is_empty() {
                PoolState::Pending
            } else {
                PoolState::Active
            }
        } else if all_met && assigned.len() as u32 >= pool.spec.minimum_worker_count {
            PoolState::Active
        } else {
            PoolState::Pending
        };
        let new_status = PoolStatus {
            state,
            assigned_workers: assigned,
            created_at: pool.status.created_at,
            last_renewed: pool.status.last_renewed,
        };
        let status_changed = pool.status != new_status;
        pool.status = new_status.clone();
        let backend = self.k8s_backend.clone();
        drop(pools);

        if status_changed && pool_name != DEFAULT_POOL_NAME {
            if let Some(ref backend) = backend {
                if let Err(e) = backend.update_pool_status(pool_name, &new_status).await {
                    warn!(error = %e, pool = pool_name, "failed to persist pool assignment status to K8s");
                }
            }
        }

        all_met
    }

    #[allow(dead_code)]
    pub async fn get_all_assigned_urls(&self) -> HashSet<String> {
        let pools = self.pools.read().await;
        let mut urls = HashSet::new();
        for pool in pools.values() {
            if pool.status.state == PoolState::Active {
                for w in &pool.status.assigned_workers {
                    urls.insert(w.url.clone());
                }
            }
        }
        urls
    }

    /// Apply a pool received from a remote gateway (via K8s watch).
    /// Inserts the pool if it does not exist, or updates it if the incoming
    /// pool has a more recent `last_renewed` timestamp. Status-only updates may
    /// keep `last_renewed` unchanged, because assignment changes should not
    /// extend the pool lease.
    /// Does NOT write back to K8s (the event already came from K8s).
    pub async fn apply_remote_pool(&self, pool: Pool) {
        let name = pool.spec.name.clone();

        // Skip the default pool -- each gateway manages its own default pool
        if name == DEFAULT_POOL_NAME {
            return;
        }

        let mut pools = self.pools.write().await;
        if let Some(existing) = pools.get(&name) {
            let renewed_delta = pool.status.last_renewed - existing.status.last_renewed;
            if renewed_delta < -TIMESTAMP_TOLERANCE_S {
                return;
            }

            if renewed_delta.abs() <= TIMESTAMP_TOLERANCE_S {
                if pool.spec != existing.spec {
                    return;
                }
                if pool.status == existing.status {
                    return;
                }
            }
        }

        info!(pool = %name, "applied remote pool from K8s watch");
        pools.insert(name, pool);
    }

    /// Remove a pool that was deleted by a remote gateway (via K8s watch).
    /// Does NOT write back to K8s (the event already came from K8s).
    pub async fn remove_remote_pool(&self, name: &str) {
        // Never delete the default pool via watch events
        if name == DEFAULT_POOL_NAME {
            return;
        }

        let mut pools = self.pools.write().await;
        if pools.remove(name).is_some() {
            info!(pool = %name, "removed remote pool via K8s watch");
        }
    }

    pub async fn check_expired_leases(&self) -> Vec<String> {
        let now = Self::now_secs();
        let mut expired = Vec::new();

        {
            let pools = self.pools.read().await;
            for (name, pool) in pools.iter() {
                if name == DEFAULT_POOL_NAME {
                    continue;
                }
                let ttl = pool
                    .spec
                    .ttl_seconds
                    .map(|s| s as f64)
                    .unwrap_or(self.lease_duration_s);
                if now - pool.status.last_renewed > ttl {
                    expired.push(name.clone());
                }
            }
        }

        // Also check K8s Leases for pools that may have been abandoned by a
        // crashed gateway (their software TTL timer never fires).
        if let Some(ref backend) = self.k8s_backend {
            match backend.list_expired_leases().await {
                Ok(k8s_expired) => {
                    for name in k8s_expired {
                        if name == DEFAULT_POOL_NAME || expired.contains(&name) {
                            continue;
                        }
                        let pools = self.pools.read().await;
                        let in_memory = pools.contains_key(&name);
                        drop(pools);

                        if in_memory {
                            info!(pool = %name, "K8s Lease expired (crash-safe TTL)");
                            expired.push(name);
                        } else {
                            // Orphaned Lease from a crashed gateway -- clean up
                            // K8s resources without adding to the expired vector.
                            info!(pool = %name, "orphaned K8s Lease cleanup");
                            if let Err(e) = backend.delete_pool(&name).await {
                                warn!(pool = %name, error = %e, "failed to delete orphaned pool from K8s");
                            }
                            if let Err(e) = backend.delete_lease(&name).await {
                                warn!(pool = %name, error = %e, "failed to delete orphaned Lease from K8s");
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(error = %e, "failed to list expired K8s Leases");
                }
            }
        }

        for name in &expired {
            {
                let mut pools = self.pools.write().await;
                if let Some(pool) = pools.get_mut(name) {
                    pool.status.state = PoolState::Expired;
                }
                pools.remove(name);
            }
            // Persist deletion to K8s backend (ConfigMap + Lease)
            if let Some(ref backend) = self.k8s_backend {
                if let Err(e) = backend.delete_pool(name).await {
                    warn!(pool = %name, error = %e, "failed to delete expired pool from K8s");
                }
                if let Err(e) = backend.delete_lease(name).await {
                    warn!(pool = %name, error = %e, "failed to delete expired Lease from K8s");
                }
            }
            info!(pool = %name, "cleaned up expired pool");
            crate::metrics::POOL_EVENTS
                .with_label_values(&["expired"])
                .inc();
        }

        expired
    }
}

fn worker_consumes_pool(worker_pool: &str, pool_name: &str) -> bool {
    normalize_pool_name(worker_pool) == normalize_pool_name(pool_name)
}

fn normalize_pool_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn worker(
        name: &str,
        url: &str,
        gpu: &str,
        bundle: &str,
    ) -> (String, String, String, String, String) {
        worker_in_pool(name, url, gpu, bundle, DEFAULT_POOL_NAME)
    }

    fn worker_in_pool(
        name: &str,
        url: &str,
        gpu: &str,
        bundle: &str,
        pool: &str,
    ) -> (String, String, String, String, String) {
        (
            name.to_string(),
            url.to_string(),
            gpu.to_string(),
            bundle.to_string(),
            pool.to_string(),
        )
    }

    #[tokio::test]
    async fn test_create_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);

        let pool = pm.create_pool("test", gpus, None, None, 0).await.unwrap();
        assert_eq!(pool.spec.name, "test");
        assert_eq!(pool.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_create_pool_idempotent() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);

        let pool1 = pm
            .create_pool("test", gpus.clone(), None, None, 0)
            .await
            .unwrap();
        let pool2 = pm.create_pool("test", gpus, None, None, 0).await.unwrap();
        assert_eq!(pool1.spec.name, pool2.spec.name);
    }

    #[tokio::test]
    async fn test_create_pool_existing_updates_spec() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            "test",
        )];
        pm.assign_workers("test", &workers).await;

        let mut updated_gpus = HashMap::new();
        updated_gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        let pool = pm
            .create_pool_with_caps("test", updated_gpus, gpu_caps, None, Some(60), 0)
            .await
            .unwrap();

        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(pool.spec.gpu_caps.get("l4-spot"), Some(&2));
        assert_eq!(pool.spec.ttl_seconds, Some(60));
        assert_eq!(pool.status.state, PoolState::Pending);
        assert!(pool.status.assigned_workers.is_empty());
    }

    #[tokio::test]
    async fn test_create_pool_rejects_default_mutation() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        let result = pm.create_pool(DEFAULT_POOL_NAME, gpus, None, None, 0).await;

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_default_pool_fails() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        let result = pm.delete_pool(DEFAULT_POOL_NAME).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_delete_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let deleted = pm.delete_pool("test").await.unwrap();
        assert!(deleted);

        let pool = pm.get_pool("test").await;
        assert!(pool.is_none());
    }

    #[tokio::test]
    async fn test_assign_workers() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let workers = vec![
            worker_in_pool("w1", "http://w1:8080", "l4-spot", "default", "test"),
            worker_in_pool("w2", "http://w2:8080", "l4-spot", "default", "test"),
            worker_in_pool("w3", "http://w3:8080", "a100", "default", "test"),
        ];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 2);
    }

    #[tokio::test]
    async fn test_assign_workers_required_count_does_not_cap_assignment() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 3);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let workers: Vec<(String, String, String, String, String)> = (1..=5)
            .map(|i| {
                worker_in_pool(
                    &format!("w{}", i),
                    &format!("http://w{}:8080", i),
                    "l4-spot",
                    "default",
                    "test",
                )
            })
            .collect();

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 5);
    }

    #[tokio::test]
    async fn test_assign_workers_honors_gpu_cap() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 3);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 4);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        let workers: Vec<(String, String, String, String, String)> = (1..=5)
            .map(|i| {
                worker_in_pool(
                    &format!("w{}", i),
                    &format!("http://w{}:8080", i),
                    "l4-spot",
                    "default",
                    "test",
                )
            })
            .collect();

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 4);
    }

    #[tokio::test]
    async fn test_assign_workers_cap_selection_is_deterministic() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        let workers = vec![
            worker_in_pool("w3", "http://w3:8080", "l4-spot", "default", "test"),
            worker_in_pool("w1", "http://w1:8080", "l4-spot", "default", "test"),
            worker_in_pool("w2", "http://w2:8080", "l4-spot", "default", "test"),
        ];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        let names: Vec<&str> = pool
            .status
            .assigned_workers
            .iter()
            .map(|worker| worker.name.as_str())
            .collect();
        assert_eq!(names, vec!["w1", "w2"]);
    }

    #[tokio::test]
    async fn test_assign_workers_ignores_other_queue_pools_when_capped() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 1);
        pm.create_pool_with_caps("bench", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        let workers = vec![
            worker("default-worker", "http://w1:8080", "l4-spot", "default"),
            worker_in_pool(
                "bench-worker",
                "http://w2:8080",
                "l4-spot",
                "default",
                "bench",
            ),
        ];

        let all_met = pm.assign_workers("bench", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("bench").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 1);
        assert_eq!(pool.status.assigned_workers[0].name, "bench-worker");
    }

    #[tokio::test]
    async fn test_assign_workers_ignores_other_queue_pools_without_caps() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("bench", gpus, None, None, 0).await.unwrap();

        let workers = vec![worker(
            "default-worker",
            "http://w1:8080",
            "l4-spot",
            "default",
        )];

        let all_met = pm.assign_workers("bench", &workers).await;
        assert!(!all_met);

        let pool = pm.get_pool("bench").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Pending);
        assert!(pool.status.assigned_workers.is_empty());
    }

    #[tokio::test]
    async fn test_gpu_cap_without_requirement_defaults_required_to_zero() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let gpus = HashMap::new();
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        let workers: Vec<(String, String, String, String, String)> = (1..=3)
            .map(|i| {
                worker_in_pool(
                    &format!("w{}", i),
                    &format!("http://w{}:8080", i),
                    "l4-spot",
                    "default",
                    "test",
                )
            })
            .collect();

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 2);
    }

    #[tokio::test]
    async fn test_zero_required_pool_can_be_active_without_workers() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let workers = Vec::new();
        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert!(pool.status.assigned_workers.is_empty());
    }

    #[tokio::test]
    async fn test_capped_lane_status_reports_cap_and_assigned_count() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            "test",
        )];
        pm.assign_workers("test", &workers).await;

        let lane_status = pm
            .capped_lane_status("test", "L4-SPOT", "default")
            .await
            .expect("lane should be capped");
        assert_eq!(lane_status.cap, 2);
        assert_eq!(lane_status.assigned_count, 1);
        assert!(pm
            .capped_lane_status("test", "a100", "default")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_admitted_worker_names_for_capped_lane_filters_assignment() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 1);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        let workers = vec![
            worker_in_pool("w1", "http://w1:8080", "l4-spot", "default", "test"),
            worker_in_pool("w2", "http://w2:8080", "l4-spot", "default", "test"),
        ];
        pm.assign_workers("test", &workers).await;

        let admitted = pm
            .admitted_worker_names_for_capped_lane("test", "L4-SPOT", "default")
            .await
            .expect("profile should be capped");
        assert_eq!(admitted, HashSet::from(["w1".to_string()]));
    }

    #[tokio::test]
    async fn test_admitted_worker_names_for_capped_lane_respects_bundle() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 2);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        let workers = vec![
            worker_in_pool("w1", "http://w1:8080", "l4-spot", "default", "test"),
            worker_in_pool("w2", "http://w2:8080", "l4-spot", "sglang", "test"),
        ];
        pm.assign_workers("test", &workers).await;

        let admitted = pm
            .admitted_worker_names_for_capped_lane("test", "l4-spot", "default")
            .await
            .expect("profile should be capped");
        assert_eq!(admitted, HashSet::from(["w1".to_string()]));

        let default_status = pm
            .capped_lane_status("test", "l4-spot", "default")
            .await
            .expect("default lane should be capped");
        assert_eq!(default_status.assigned_count, 1);
        let missing_status = pm
            .capped_lane_status("test", "l4-spot", "rerank")
            .await
            .expect("missing bundle lane should still be capped");
        assert_eq!(missing_status.assigned_count, 0);
    }

    #[tokio::test]
    async fn test_admitted_worker_names_for_uncapped_lane_returns_none() {
        let pm = PoolManager::new(vec!["l4-spot".to_string(), "a100".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("a100".to_string(), 1);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        assert!(pm
            .admitted_worker_names_for_capped_lane("test", "l4-spot", "default")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_admitted_worker_names_for_zero_cap_and_missing_named_pool_fail_closed() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let mut gpu_caps = HashMap::new();
        gpu_caps.insert("l4-spot".to_string(), 0);
        pm.create_pool_with_caps("test", gpus, gpu_caps, None, None, 0)
            .await
            .unwrap();

        assert_eq!(
            pm.admitted_worker_names_for_capped_lane("test", "l4-spot", "default")
                .await,
            Some(HashSet::new())
        );
        assert_eq!(
            pm.admitted_worker_names_for_capped_lane("missing", "l4-spot", "default")
                .await,
            Some(HashSet::new())
        );
        assert!(pm
            .admitted_worker_names_for_capped_lane(DEFAULT_POOL_NAME, "l4-spot", "default")
            .await
            .is_none());
    }

    #[tokio::test]
    async fn test_assign_workers_partial() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 3);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            "test",
        )];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(!all_met);

        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_minimum_worker_count_enforced() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        // Require at least 3 workers total
        pm.create_pool("test", gpus, None, None, 3).await.unwrap();

        // Provide only 1 matching worker — GPU requirement met but worker count too low
        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            "test",
        )];

        let all_met = pm.assign_workers("test", &workers).await;
        // GPU count met (1/1) but minimum_worker_count (3) not met → stays Pending
        assert!(all_met); // all GPU types met
        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Pending);
        assert_eq!(pool.status.assigned_workers.len(), 1);
    }

    #[tokio::test]
    async fn test_minimum_worker_count_met() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 2);
        // Require at least 2 workers
        pm.create_pool("test", gpus, None, None, 2).await.unwrap();

        let workers = vec![
            worker_in_pool("w1", "http://w1:8080", "l4-spot", "default", "test"),
            worker_in_pool("w2", "http://w2:8080", "l4-spot", "default", "test"),
        ];

        let all_met = pm.assign_workers("test", &workers).await;
        assert!(all_met);
        let pool = pm.get_pool("test").await.unwrap();
        assert_eq!(pool.status.state, PoolState::Active);
        assert_eq!(pool.status.assigned_workers.len(), 2);
    }

    #[tokio::test]
    async fn test_renew_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);
        pm.create_pool("test", gpus, None, None, 0).await.unwrap();

        let renewed = pm.renew_pool("test").await;
        assert!(renewed);

        let not_renewed = pm.renew_pool("nonexistent").await;
        assert!(!not_renewed);
    }

    #[tokio::test]
    async fn test_invalid_machine_profile() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("invalid-gpu".to_string(), 1);

        let result = pm.create_pool("test", gpus, None, None, 0).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_per_pool_ttl_respected() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 1);

        // Pool with short TTL (1 second)
        pm.create_pool("short-ttl", gpus.clone(), None, Some(1), 0)
            .await
            .unwrap();

        // Pool with no TTL (uses global default = 1200s)
        pm.create_pool("default-ttl", gpus, None, None, 0)
            .await
            .unwrap();

        // Backdate last_renewed so the short-TTL pool is expired
        {
            let mut pools = pm.pools.write().await;
            if let Some(p) = pools.get_mut("short-ttl") {
                p.status.last_renewed -= 5.0; // 5s ago, exceeds 1s TTL
            }
            if let Some(p) = pools.get_mut("default-ttl") {
                p.status.last_renewed -= 5.0; // 5s ago, well within 1200s TTL
            }
        }

        let expired = pm.check_expired_leases().await;
        assert!(expired.contains(&"short-ttl".to_string()));
        assert!(!expired.contains(&"default-ttl".to_string()));

        // short-ttl should be removed
        assert!(pm.get_pool("short-ttl").await.is_none());
        // default-ttl should still exist
        assert!(pm.get_pool("default-ttl").await.is_some());
    }

    #[tokio::test]
    async fn test_create_default_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string(), "a100-40gb".to_string()]);

        pm.create_default_pool().await;

        let pool = pm.get_pool(DEFAULT_POOL_NAME).await;
        assert!(pool.is_some());

        let pool = pool.unwrap();
        assert_eq!(pool.spec.gpus.len(), 2);
        assert_eq!(pool.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(pool.spec.gpus.get("a100-40gb"), Some(&0));
        assert!(pool.spec.gpu_caps.is_empty());
    }

    #[tokio::test]
    async fn test_default_pool_state_tracks_eligible_workers() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        pm.create_default_pool().await;

        let no_workers: Vec<WorkerAssignment> = Vec::new();
        pm.assign_workers(DEFAULT_POOL_NAME, &no_workers).await;
        let pending = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();
        assert_eq!(pending.status.state, PoolState::Pending);
        assert!(pending.status.assigned_workers.is_empty());

        let workers = vec![worker_in_pool(
            "w1",
            "http://w1:8080",
            "l4-spot",
            "default",
            DEFAULT_POOL_NAME,
        )];
        pm.assign_workers(DEFAULT_POOL_NAME, &workers).await;
        let active = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();
        assert_eq!(active.status.state, PoolState::Active);
        assert_eq!(active.status.assigned_workers.len(), 1);
    }

    #[tokio::test]
    async fn test_default_pool_requires_explicit_worker_pool() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        pm.create_default_pool().await;

        let workers = vec![
            worker_in_pool("empty", "http://w1:8080", "l4-spot", "default", ""),
            worker_in_pool(
                "_default",
                "http://w2:8080",
                "l4-spot",
                "default",
                "_default",
            ),
        ];
        pm.assign_workers(DEFAULT_POOL_NAME, &workers).await;

        let pool = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();
        assert_eq!(pool.status.state, PoolState::Pending);
        assert!(pool.status.assigned_workers.is_empty());
    }

    #[tokio::test]
    async fn test_restore_from_k8s_no_backend() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        // No K8s backend → returns Ok(0)
        let count = pm.restore_from_k8s().await.unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_restore_from_k8s_skips_default_pool() {
        let pool = Pool {
            spec: PoolSpec {
                name: DEFAULT_POOL_NAME.to_string(),
                bundle: None,
                gpus: HashMap::new(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: 9999.0,
                last_renewed: 9999.0,
            },
        };

        assert!(!PoolManager::should_restore_pool_from_k8s(&pool));
    }

    #[tokio::test]
    async fn test_k8s_backend_field_defaults_none() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        assert!(pm.k8s_backend.is_none());
    }

    #[tokio::test]
    async fn test_apply_remote_pool_skips_default() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        let original = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();

        let pool = Pool {
            spec: PoolSpec {
                name: DEFAULT_POOL_NAME.to_string(),
                bundle: None,
                gpus: HashMap::new(),
                gpu_caps: HashMap::new(),
                ttl_seconds: None,
                minimum_worker_count: 0,
            },
            status: PoolStatus {
                state: PoolState::Pending,
                assigned_workers: Vec::new(),
                created_at: 9999.0,
                last_renewed: 9999.0,
            },
        };

        pm.apply_remote_pool(pool).await;

        // Should still have the locally-created default pool, not the remote one
        let found = pm.get_pool(DEFAULT_POOL_NAME).await.unwrap();
        // The remote pool had empty gpus; the original default pool has l4-spot=0
        assert_eq!(found.spec.gpus.get("l4-spot"), Some(&0));
        assert_eq!(found.status.created_at, original.status.created_at);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_same_timestamp_applies_status_update() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let local = pm
            .create_pool("bench", gpus.clone(), None, Some(60), 0)
            .await
            .unwrap();

        let mut remote = local.clone();
        remote.status.state = PoolState::Active;
        remote.status.assigned_workers = vec![AssignedWorker {
            name: "worker-a".to_string(),
            url: "http://worker-a:8080".to_string(),
            gpu: "l4-spot".to_string(),
            bundle: "default".to_string(),
        }];

        pm.apply_remote_pool(remote).await;

        let found = pm.get_pool("bench").await.unwrap();
        assert_eq!(found.status.state, PoolState::Active);
        assert_eq!(found.status.assigned_workers.len(), 1);
        assert_eq!(found.spec.gpus, gpus);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_same_timestamp_keeps_local_spec() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let local = pm
            .create_pool("bench", gpus, Some("default".to_string()), Some(60), 0)
            .await
            .unwrap();

        let mut remote = local.clone();
        remote.spec.bundle = Some("other".to_string());
        remote.status.state = PoolState::Active;

        pm.apply_remote_pool(remote).await;

        let found = pm.get_pool("bench").await.unwrap();
        assert_eq!(found.spec.bundle, Some("default".to_string()));
        assert_eq!(found.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_apply_remote_pool_timestamp_tolerance_keeps_local_spec() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);

        let mut gpus = HashMap::new();
        gpus.insert("l4-spot".to_string(), 0);
        let local = pm
            .create_pool("bench", gpus, Some("default".to_string()), Some(60), 0)
            .await
            .unwrap();

        let mut remote = local.clone();
        remote.status.last_renewed += TIMESTAMP_TOLERANCE_S / 2.0;
        remote.spec.bundle = Some("other".to_string());
        remote.status.state = PoolState::Active;

        pm.apply_remote_pool(remote).await;

        let found = pm.get_pool("bench").await.unwrap();
        assert_eq!(found.spec.bundle, Some("default".to_string()));
        assert_eq!(found.status.state, PoolState::Pending);
    }

    #[tokio::test]
    async fn test_remove_remote_pool_skips_default() {
        let pm = PoolManager::new(vec!["l4-spot".to_string()]);
        pm.create_default_pool().await;

        pm.remove_remote_pool(DEFAULT_POOL_NAME).await;

        assert!(pm.get_pool(DEFAULT_POOL_NAME).await.is_some());
    }

    #[tokio::test]
    async fn test_remove_remote_pool_nonexistent() {
        let pm = PoolManager::new(vec![]);

        // Should not panic or error
        pm.remove_remote_pool("nonexistent").await;
    }
}
