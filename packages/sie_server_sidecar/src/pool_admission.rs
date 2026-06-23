//! Worker-side pool admission gate.
//!
//! The gateway owns pool assignment. The sidecar polls `/v1/pools`, opens its
//! physical JetStream consumer only when the physical pool or at least one
//! backed logical pool assigns this worker, then checks every work item's
//! `admission_pool` before Python IPC so direct NATS publishes cannot bypass
//! the gateway's admission decision.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use serde_json::{Map, Value};
use tokio::sync::Mutex;
use tokio::time::{interval, MissedTickBehavior};
use tracing::{debug, info};

use crate::config::WorkerConfig;
use crate::shutdown::Shutdown;

const HTTP_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug)]
struct GateState {
    last_check_at: Option<Instant>,
    last_success_at: Option<Instant>,
    admitted: bool,
    last_reason: String,
}

/// Runtime admission guard shared by the pool pull loop and generation
/// worker-direct pull loop.
#[derive(Debug)]
pub struct PoolAdmissionGate {
    pool_name: String,
    worker_id: String,
    machine_profile: String,
    gateway_url: String,
    api_key: Option<String>,
    check_interval: Duration,
    pause: Duration,
    stale_after: Duration,
    http: reqwest::Client,
    admitted: AtomicBool,
    assigned_logical_pools: RwLock<HashSet<String>>,
    state: Mutex<GateState>,
}

impl PoolAdmissionGate {
    pub fn from_worker_config(config: &WorkerConfig) -> anyhow::Result<Option<Arc<Self>>> {
        if !config.pool_admission_enabled {
            info!("pool-admission: disabled by SIE_POOL_ADMISSION_ENABLED=false");
            return Ok(None);
        }
        let Some(gateway_url) = config.gateway_url.as_deref().map(str::trim) else {
            debug!("pool-admission: disabled because SIE_GATEWAY_URL is unset");
            return Ok(None);
        };
        if gateway_url.is_empty() {
            debug!("pool-admission: disabled because SIE_GATEWAY_URL is empty");
            return Ok(None);
        }

        let http = reqwest::Client::builder().timeout(HTTP_TIMEOUT).build()?;
        let initially_admitted = normalize_pool_name(&config.pool) == "default";
        let gate = Self {
            pool_name: config.pool.clone(),
            worker_id: config.worker_id.clone(),
            machine_profile: config.machine_profile.clone(),
            gateway_url: gateway_url.trim_end_matches('/').to_string(),
            api_key: config.gateway_api_key.clone(),
            check_interval: Duration::from_millis(config.pool_admission_check_interval_ms),
            pause: Duration::from_millis(config.pool_admission_pause_ms),
            stale_after: Duration::from_millis(config.pool_admission_stale_after_ms),
            http,
            admitted: AtomicBool::new(initially_admitted),
            assigned_logical_pools: RwLock::new(HashSet::new()),
            state: Mutex::new(GateState {
                last_check_at: None,
                last_success_at: None,
                admitted: initially_admitted,
                last_reason: "initial".to_string(),
            }),
        };
        info!(
            pool = %gate.pool_name,
            worker_id = %gate.worker_id,
            machine_profile = %gate.machine_profile,
            check_interval_ms = gate.check_interval.as_millis() as u64,
            pause_ms = gate.pause.as_millis() as u64,
            stale_after_ms = gate.stale_after.as_millis() as u64,
            "pool-admission: enabled"
        );
        Ok(Some(Arc::new(gate)))
    }

    pub async fn run(self: Arc<Self>, shutdown: Arc<Shutdown>) {
        let mut ticker = interval(self.check_interval);
        ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
        loop {
            tokio::select! {
                biased;
                _ = shutdown.wait() => return,
                _ = ticker.tick() => self.refresh().await,
            }
        }
    }

    pub fn pause_duration(&self) -> Duration {
        self.pause
    }

    pub fn admitted(&self) -> bool {
        self.admitted.load(Ordering::Acquire)
    }

    pub fn pull_admitted(&self) -> bool {
        self.admitted()
            || self
                .assigned_logical_pools
                .read()
                .is_ok_and(|pools| !pools.is_empty())
    }

    pub fn admits_work_item_pool(&self, admission_pool: &str) -> bool {
        let requested = normalize_pool_name(admission_pool);
        if requested.is_empty() || requested == normalize_pool_name(&self.pool_name) {
            return self.admitted();
        }
        self.assigned_logical_pools
            .read()
            .is_ok_and(|pools| pools.contains(&requested))
    }

    async fn refresh(&self) {
        let now = Instant::now();
        {
            let mut state = self.state.lock().await;
            state.last_check_at = Some(now);
        }

        let fetched = self.fetch_pools().await;
        let mut state = self.state.lock().await;
        match fetched {
            Ok(pools) => {
                state.last_success_at = Some(Instant::now());
                self.set_assigned_logical_pools(assigned_logical_pools_for_worker(
                    &pools,
                    &self.worker_id,
                    &self.pool_name,
                ));
                let pool = pool_by_name(&pools, &self.pool_name);
                let (admitted, reason) = decide_pool_admission(
                    pool,
                    &self.pool_name,
                    &self.worker_id,
                    &self.machine_profile,
                );
                self.set_admitted(&mut state, admitted, reason);
            }
            Err(error) => {
                if !self.handle_error(&mut state, Instant::now(), error) {
                    self.set_assigned_logical_pools(HashSet::new());
                }
            }
        }
    }

    async fn fetch_pools(&self) -> Result<Vec<Value>, String> {
        let url = format!("{}/v1/pools", self.gateway_url);
        let mut req = self.http.get(&url).header("accept", "application/json");
        if let Some(token) = self.api_key.as_deref().filter(|token| !token.is_empty()) {
            req = req.bearer_auth(token);
        }
        let resp = req.send().await.map_err(|e| e.to_string())?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!(
                "status {} from pool list endpoint: {}",
                status.as_u16(),
                body.chars().take(120).collect::<String>()
            ));
        }
        let body = resp.json::<Value>().await.map_err(|e| e.to_string())?;
        parse_pool_list(&body).ok_or_else(|| "pool list response missing 'pools' array".to_string())
    }

    fn handle_error(&self, state: &mut GateState, now: Instant, error: String) -> bool {
        if state
            .last_success_at
            .is_some_and(|last| now.duration_since(last) <= self.stale_after)
        {
            debug!(
                pool = %self.pool_name,
                worker_id = %self.worker_id,
                error = %error,
                "pool-admission: keeping cached decision during stale window"
            );
            return true;
        }

        // Keep the default pool available during gateway/status glitches.
        // Named pools fail closed because their caps are isolation contracts.
        let fail_open = normalize_pool_name(&self.pool_name) == "default";
        self.set_admitted(state, fail_open, format!("status_error:{error}"));
        false
    }

    fn set_admitted(&self, state: &mut GateState, admitted: bool, reason: String) {
        self.admitted.store(admitted, Ordering::Release);
        if admitted == state.admitted && reason == state.last_reason {
            return;
        }
        state.admitted = admitted;
        state.last_reason = reason.clone();
        info!(
            pool = %self.pool_name,
            worker_id = %self.worker_id,
            machine_profile = %self.machine_profile,
            reason = %reason,
            "pool-admission: {}",
            if admitted { "granted" } else { "paused" }
        );
    }

    fn set_assigned_logical_pools(&self, pools: HashSet<String>) {
        if let Ok(mut assigned) = self.assigned_logical_pools.write() {
            *assigned = pools;
        }
    }
}

fn normalize_pool_name(name: &str) -> String {
    name.trim().to_ascii_lowercase()
}

fn parse_pool_list(body: &Value) -> Option<Vec<Value>> {
    body.get("pools")
        .and_then(Value::as_array)
        .cloned()
        .or_else(|| body.as_array().cloned())
}

fn pool_by_name<'a>(pools: &'a [Value], pool_name: &str) -> Option<&'a Value> {
    let pool_name = normalize_pool_name(pool_name);
    pools.iter().find(|pool| {
        pool.get("spec")
            .and_then(|spec| spec.get("name"))
            .and_then(Value::as_str)
            .is_some_and(|name| normalize_pool_name(name) == pool_name)
    })
}

fn pool_queue_matches(pool: &Value, home_pool: &str) -> bool {
    let spec = pool.get("spec");
    let queue_pool = spec
        .and_then(|spec| spec.get("queue_pool"))
        .and_then(Value::as_str)
        .or_else(|| {
            spec.and_then(|spec| spec.get("name"))
                .and_then(Value::as_str)
        })
        .unwrap_or("default");
    normalize_pool_name(queue_pool) == normalize_pool_name(home_pool)
}

fn pool_assigns_worker(pool: &Value, worker_id: &str) -> bool {
    pool.get("status")
        .and_then(|status| status.get("assigned_workers"))
        .and_then(Value::as_array)
        .is_some_and(|workers| {
            workers.iter().any(|worker| {
                worker
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| name == worker_id)
            })
        })
}

fn assigned_logical_pools_for_worker(
    pools: &[Value],
    worker_id: &str,
    home_pool: &str,
) -> HashSet<String> {
    pools
        .iter()
        .filter(|pool| pool_queue_matches(pool, home_pool))
        .filter(|pool| pool_assigns_worker(pool, worker_id))
        .filter_map(|pool| {
            pool.get("spec")
                .and_then(|spec| spec.get("name"))
                .and_then(Value::as_str)
                .map(normalize_pool_name)
        })
        .collect()
}

fn object_field<'a>(value: &'a Value, key: &str) -> Option<&'a Map<String, Value>> {
    value.get(key).and_then(Value::as_object)
}

fn lookup_case_insensitive<'a>(mapping: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    let key_lower = key.to_ascii_lowercase();
    mapping
        .iter()
        .find(|(candidate, _)| candidate.to_ascii_lowercase() == key_lower)
        .map(|(_, value)| value)
}

fn cap_as_i64(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|n| i64::try_from(n).ok()))
        .or_else(|| value.as_str().and_then(|s| s.parse::<i64>().ok()))
}

fn decide_pool_admission(
    pool: Option<&Value>,
    pool_name: &str,
    worker_id: &str,
    machine_profile: &str,
) -> (bool, String) {
    let Some(pool) = pool else {
        return (
            normalize_pool_name(pool_name) == "default",
            "pool_missing".to_string(),
        );
    };

    let Some(spec) = object_field(pool, "spec") else {
        return (true, "uncapped".to_string());
    };
    let Some(caps) = spec.get("gpu_caps").and_then(Value::as_object) else {
        return (true, "uncapped".to_string());
    };
    if caps.is_empty() {
        return (true, "uncapped".to_string());
    }

    let cap = if machine_profile.trim().is_empty() {
        None
    } else {
        lookup_case_insensitive(caps, machine_profile)
    };
    let Some(cap) = cap else {
        return (true, "profile_uncapped".to_string());
    };
    match cap_as_i64(cap) {
        Some(n) if n > 0 => {}
        Some(_) => return (false, "cap_exhausted".to_string()),
        None => return (false, "malformed_gpu_cap".to_string()),
    }

    let Some(status) = object_field(pool, "status") else {
        return (false, "malformed_assigned_workers".to_string());
    };
    let Some(assigned) = status.get("assigned_workers").and_then(Value::as_array) else {
        return (false, "malformed_assigned_workers".to_string());
    };

    for worker in assigned {
        if worker
            .as_object()
            .and_then(|w| w.get("name"))
            .and_then(Value::as_str)
            == Some(worker_id)
        {
            return (true, "assigned".to_string());
        }
    }
    (false, "not_assigned".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn pool_missing_fails_open_only_for_default_pool() {
        assert_eq!(
            decide_pool_admission(None, "default", "w1", "l4"),
            (true, "pool_missing".to_string())
        );
        assert_eq!(
            decide_pool_admission(None, "eval", "w1", "l4"),
            (false, "pool_missing".to_string())
        );
    }

    #[test]
    fn uncapped_pool_is_admitted() {
        let pool = json!({"spec": {"gpu_caps": {}}, "status": {"assigned_workers": []}});
        assert_eq!(
            decide_pool_admission(Some(&pool), "eval", "w1", "l4"),
            (true, "uncapped".to_string())
        );
    }

    #[test]
    fn capped_pool_requires_worker_assignment() {
        let pool = json!({
            "spec": {"gpu_caps": {"L4": 2}},
            "status": {"assigned_workers": [{"name": "w1", "url": "http://w1", "gpu": "l4"}]}
        });
        assert_eq!(
            decide_pool_admission(Some(&pool), "eval", "w1", "l4"),
            (true, "assigned".to_string())
        );
        assert_eq!(
            decide_pool_admission(Some(&pool), "eval", "w2", "l4"),
            (false, "not_assigned".to_string())
        );
    }

    #[test]
    fn capped_pool_profile_without_cap_is_admitted() {
        let pool = json!({
            "spec": {"gpu_caps": {"h100": 1}},
            "status": {"assigned_workers": []}
        });
        assert_eq!(
            decide_pool_admission(Some(&pool), "eval", "w1", "l4"),
            (true, "profile_uncapped".to_string())
        );
    }

    #[test]
    fn exhausted_or_malformed_caps_fail_closed() {
        let exhausted = json!({
            "spec": {"gpu_caps": {"l4": 0}},
            "status": {"assigned_workers": [{"name": "w1"}]}
        });
        assert_eq!(
            decide_pool_admission(Some(&exhausted), "eval", "w1", "l4"),
            (false, "cap_exhausted".to_string())
        );

        let malformed = json!({
            "spec": {"gpu_caps": {"l4": "many"}},
            "status": {"assigned_workers": [{"name": "w1"}]}
        });
        assert_eq!(
            decide_pool_admission(Some(&malformed), "eval", "w1", "l4"),
            (false, "malformed_gpu_cap".to_string())
        );
    }

    #[test]
    fn malformed_assignment_list_fails_closed_for_capped_profile() {
        let pool = json!({
            "spec": {"gpu_caps": {"l4": 1}},
            "status": {"assigned_workers": {"name": "w1"}}
        });
        assert_eq!(
            decide_pool_admission(Some(&pool), "eval", "w1", "l4"),
            (false, "malformed_assigned_workers".to_string())
        );
    }

    #[test]
    fn logical_pool_assignments_follow_home_queue_and_worker() {
        let pools = vec![
            json!({
                "spec": {"name": "tenant-a", "queue_pool": "default"},
                "status": {"assigned_workers": [{"name": "worker-1"}]}
            }),
            json!({
                "spec": {"name": "tenant-b", "queue_pool": "default"},
                "status": {"assigned_workers": [{"name": "worker-2"}]}
            }),
            json!({
                "spec": {"name": "dedicated", "queue_pool": "customer-a"},
                "status": {"assigned_workers": [{"name": "worker-1"}]}
            }),
        ];

        let assigned = assigned_logical_pools_for_worker(&pools, "worker-1", "default");
        assert!(assigned.contains("tenant-a"));
        assert!(!assigned.contains("tenant-b"));
        assert!(!assigned.contains("dedicated"));
    }

    #[test]
    fn gate_admits_same_physical_pool_or_assigned_logical_pool() {
        let gate = PoolAdmissionGate {
            pool_name: "default".to_string(),
            worker_id: "worker-1".to_string(),
            machine_profile: "l4".to_string(),
            gateway_url: "http://gateway".to_string(),
            api_key: None,
            check_interval: Duration::from_secs(1),
            pause: Duration::from_millis(10),
            stale_after: Duration::from_secs(30),
            http: reqwest::Client::new(),
            admitted: AtomicBool::new(true),
            assigned_logical_pools: RwLock::new(HashSet::from(["tenant-a".to_string()])),
            state: Mutex::new(GateState {
                last_check_at: None,
                last_success_at: None,
                admitted: true,
                last_reason: "test".to_string(),
            }),
        };

        assert!(gate.admits_work_item_pool(""));
        assert!(gate.admits_work_item_pool("default"));
        assert!(gate.admits_work_item_pool("TENANT-A"));
        assert!(!gate.admits_work_item_pool("tenant-b"));
    }

    #[test]
    fn gate_can_pull_for_assigned_logical_pool_without_physical_admission() {
        let gate = PoolAdmissionGate {
            pool_name: "customer-queue".to_string(),
            worker_id: "worker-1".to_string(),
            machine_profile: "l4".to_string(),
            gateway_url: "http://gateway".to_string(),
            api_key: None,
            check_interval: Duration::from_secs(1),
            pause: Duration::from_millis(10),
            stale_after: Duration::from_secs(30),
            http: reqwest::Client::new(),
            admitted: AtomicBool::new(false),
            assigned_logical_pools: RwLock::new(HashSet::from(["tenant-a".to_string()])),
            state: Mutex::new(GateState {
                last_check_at: None,
                last_success_at: None,
                admitted: false,
                last_reason: "test".to_string(),
            }),
        };

        assert!(gate.pull_admitted());
        assert!(gate.admits_work_item_pool("tenant-a"));
        assert!(!gate.admits_work_item_pool(""));
        assert!(!gate.admits_work_item_pool("customer-queue"));
        assert!(!gate.admits_work_item_pool("tenant-b"));
    }
}
