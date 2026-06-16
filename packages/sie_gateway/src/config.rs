use std::collections::HashMap;
use std::env;

use serde::Deserialize;

use crate::types::pool::PoolSpec;

#[derive(Debug, Clone)]
pub struct Config {
    // Server
    pub host: String,
    pub port: u16,
    pub metrics_port: Option<u16>,

    // Discovery
    pub worker_urls: Vec<String>,
    pub use_kubernetes: bool,
    pub k8s_namespace: String,
    pub k8s_service: String,
    pub k8s_port: u16,

    // Features
    pub health_mode: String,

    // NATS
    pub nats_url: String,
    /// Trusted-producer allowlist for `sie.config.models._all`. Defaults
    /// to `["sie-config"]`. Incoming `ConfigNotification`s whose
    /// `producer_id` is not in this list are dropped (neither the epoch
    /// counter nor the registry is touched). Override via
    /// `SIE_NATS_CONFIG_TRUSTED_PRODUCERS` (CSV), or disable validation
    /// entirely with `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER=true` (intended
    /// for local dev / single-node test clusters only).
    pub nats_config_trusted_producers: Vec<String>,

    // Auth
    pub auth_mode: String,
    pub auth_tokens: Vec<String>,
    pub admin_token: String,
    /// Opt-in bypass for operational surfaces (`/`, `/health`,
    /// `/metrics`, `/ws/*`) when auth is enabled. Kubernetes probes
    /// (`/healthz`, `/readyz`) are always exempt regardless. Defaults
    /// to `false` (fail-closed); set `SIE_AUTH_EXEMPT_OPERATIONAL=true`
    /// only when these endpoints are already network-isolated (e.g.
    /// internal ClusterIP with no ingress).
    pub auth_exempt_operational: bool,

    // Logging
    pub log_level: String,
    pub json_logs: bool,

    // Feature toggles
    pub enable_pools: bool,
    pub hot_reload: bool,
    pub watch_polling: bool,
    pub multi_router: bool,

    // Tuning
    pub request_timeout: f64,
    pub max_stream_pending: u64,
    pub stream_max_age_s: u64,

    // Configured GPUs (survives scale-to-zero)
    pub configured_gpus: Vec<String>,
    // Pre-computed lowercase→original map for GPU profile resolution (avoids HashMap rebuild per request)
    pub gpu_profile_map: HashMap<String, String>,

    // Helm-declared queue pools that are long-lived admission boundaries.
    // API-created pools remain lease-based; these specs do not expire.
    pub static_queue_pools: Vec<PoolSpec>,

    // Job/friendly model aliases: lowercase alias → canonical model id. Lets a
    // caller request `model: "code"` and have it resolve to the recommended
    // model before the registry lookup. Ships with built-in defaults (see
    // `build_model_aliases`); extend/override via `SIE_GATEWAY_MODEL_ALIASES`
    // (JSON map). Mirrors the `SIE_GATEWAY_GPU_ALIASES` mechanism.
    pub model_aliases: HashMap<String, String>,

    // Model registry paths (filesystem seed; same volume mounted into sie-config
    // for consistency, but the gateway never writes to them).
    pub bundles_dir: String,
    pub models_dir: String,

    // sie-config control plane URL. In-cluster Helm default is something like
    // `http://<release>-sie-config.<ns>.svc.cluster.local:8080`. When unset the
    // gateway runs without a bootstrap (useful in tests and single-process
    // examples); production Helm always sets this.
    pub config_service_url: Option<String>,

    // Admin token the gateway presents as a bearer credential when calling
    // `sie-config`'s bootstrap endpoints (`GET /v1/configs/export` and
    // `GET /v1/configs/epoch`). Reuses SIE_ADMIN_TOKEN because both services
    // share one admin secret in-cluster.
    pub config_service_token: Option<String>,

    // Payload store (local path, s3://bucket/prefix, gs://bucket/prefix, or
    // abfs(s)://container@account.dfs.core.windows.net/prefix)
    pub payload_store_url: String,
}

fn env_bool(key: &str) -> bool {
    match env::var(key) {
        Ok(v) => matches!(v.to_lowercase().as_str(), "true" | "1" | "yes"),
        Err(_) => false,
    }
}

fn env_int(key: &str, fallback: u16) -> u16 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn env_optional_int(key: &str) -> Option<u16> {
    env::var(key).ok().and_then(|s| s.parse().ok())
}

fn env_float(key: &str, fallback: f64) -> f64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn env_u64(key: &str, fallback: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(fallback)
}

fn env_csv(key: &str) -> Vec<String> {
    match env::var(key) {
        Ok(s) if !s.is_empty() => s
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        _ => Vec::new(),
    }
}

fn env_json_string_map(key: &str) -> HashMap<String, String> {
    match env::var(key) {
        Ok(s) if !s.trim().is_empty() => {
            serde_json::from_str::<HashMap<String, String>>(&s).unwrap_or_default()
        }
        _ => HashMap::new(),
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StaticQueuePoolEnvSpec {
    #[serde(default)]
    bundle: Option<String>,
    #[serde(default)]
    gpus: HashMap<String, u32>,
    #[serde(default, alias = "gpu_caps")]
    gpu_caps: HashMap<String, u32>,
    #[serde(default, alias = "minimum_worker_count")]
    minimum_worker_count: u32,
}

fn env_static_queue_pools(key: &str) -> Vec<PoolSpec> {
    let parsed = match env::var(key) {
        Ok(s) if !s.trim().is_empty() => {
            match serde_json::from_str::<HashMap<String, StaticQueuePoolEnvSpec>>(&s) {
                Ok(parsed) => parsed,
                Err(error) => {
                    panic!("failed to parse {key}: {error}; fix or unset {key}")
                }
            }
        }
        _ => HashMap::new(),
    };

    let mut specs: Vec<PoolSpec> = parsed
        .into_iter()
        .filter_map(|(name, spec)| {
            let name = name.trim().to_string();
            if name.is_empty() {
                return None;
            }
            let bundle = spec.bundle.and_then(|bundle| {
                let bundle = bundle.trim().to_string();
                if bundle.is_empty() {
                    None
                } else {
                    Some(bundle)
                }
            });
            Some(PoolSpec {
                name,
                bundle,
                gpus: spec.gpus,
                gpu_caps: spec.gpu_caps,
                ttl_seconds: None,
                minimum_worker_count: spec.minimum_worker_count,
            })
        })
        .collect();
    specs.sort_by(|a, b| a.name.cmp(&b.name));
    specs
}

/// Job/friendly model aliases: lowercase alias → model id (or `bundle:/model`).
///
/// Ships with built-in defaults so `model="code"` works without operator
/// config (and makes "each agent job routes to the right model" true by
/// default). `SIE_GATEWAY_MODEL_ALIASES` (a JSON map) extends or overrides
/// them. Empty alias/target pairs are skipped; aliases are stored lowercased
/// because resolution lowercases the lookup.
///
/// A target may be a bare model id (`Org/Model`), a concrete profile variant
/// (`Org/Model:profile`), or a bundle-qualified spec (`bundle:/Org/Model`).
/// The bundle form lets an operator pin a precision / profile bundle for a job
/// — e.g. map `sql` to a BF16 bundle that avoids the FP8 SQL-accuracy
/// regression (ADR 0001). `resolve_model_spec_with_aliases` (proxy.rs) applies
/// the bundle and preserves concrete profile variants.
fn build_model_aliases(overrides: HashMap<String, String>) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = HashMap::new();
    // Built-in: the code-generation job → the model with a MEASURED
    // HumanEval/MBPP pass@1 baseline that also serves reliably
    // (Qwen3-4B-Instruct-2507: 0.866 / 0.74). Qwen3.5-4B is stronger on paper
    // but its NEXTN/hybrid serving path does not come up reliably yet, so it is
    // not the default until measured + its serving init is fixed.
    map.insert(
        "code".to_string(),
        "Qwen/Qwen3-4B-Instruct-2507".to_string(),
    );
    // Text-to-SQL job → an ebnf-grammar-capable LLM for the "any LLM + SQL
    // grammar" path. Repoint to SQLCoder once that model is onboarded.
    map.insert("sql".to_string(), "Qwen/Qwen3-4B-Instruct-2507".to_string());
    // CHECK POLICY job → a generative guard model that emits a safe/unsafe
    // verdict. Granite Guardian 3.0 2B (Apache-2.0, ungated) measured on
    // ToxicChat via the generation gate; serves on the same SGLang path.
    map.insert(
        "guard".to_string(),
        "ibm-granite/granite-guardian-3.0-2b".to_string(),
    );
    for (alias, target) in overrides {
        let alias = alias.trim().to_lowercase();
        let target = target.trim().to_string();
        if alias.is_empty() || target.is_empty() {
            continue;
        }
        map.insert(alias, target);
    }
    map
}

fn build_gpu_profile_map(
    configured_gpus: &[String],
    aliases: HashMap<String, String>,
) -> HashMap<String, String> {
    let mut map: HashMap<String, String> = configured_gpus
        .iter()
        .map(|g| (g.to_lowercase(), g.clone()))
        .collect();

    for (alias, profile) in aliases {
        let alias = alias.trim();
        let profile = profile.trim();
        if alias.is_empty() || profile.is_empty() {
            continue;
        }
        map.entry(alias.to_lowercase())
            .or_insert_with(|| profile.to_string());
    }

    map
}

fn env_default(key: &str, fallback: &str) -> String {
    match env::var(key) {
        Ok(v) if !v.is_empty() => v,
        _ => fallback.to_string(),
    }
}

impl Config {
    pub fn load() -> Self {
        let mut auth_tokens = env_csv("SIE_AUTH_TOKENS");
        if auth_tokens.is_empty() {
            auth_tokens = env_csv("SIE_AUTH_TOKEN");
        }
        let configured_gpus = env_csv("SIE_GATEWAY_CONFIGURED_GPUS");
        let gpu_profile_map = build_gpu_profile_map(
            &configured_gpus,
            env_json_string_map("SIE_GATEWAY_GPU_ALIASES"),
        );
        let model_aliases = build_model_aliases(env_json_string_map("SIE_GATEWAY_MODEL_ALIASES"));

        Self {
            host: "0.0.0.0".to_string(),
            port: 8080,
            metrics_port: env_optional_int("SIE_METRICS_PORT"),

            worker_urls: env_csv("SIE_GATEWAY_WORKERS"),
            use_kubernetes: env_bool("SIE_GATEWAY_KUBERNETES"),
            k8s_namespace: env_default("SIE_GATEWAY_K8S_NAMESPACE", "default"),
            k8s_service: env_default("SIE_GATEWAY_K8S_SERVICE", "sie-worker"),
            k8s_port: env_int("SIE_GATEWAY_K8S_PORT", 8080),

            health_mode: env_default("SIE_GATEWAY_HEALTH_MODE", "ws"),

            nats_url: env::var("SIE_NATS_URL").unwrap_or_default(),
            nats_config_trusted_producers: {
                // Explicit opt-in to the legacy "trust anyone" behavior.
                if env_bool("SIE_NATS_CONFIG_TRUST_ANY_PRODUCER") {
                    Vec::new()
                } else {
                    let custom = env_csv("SIE_NATS_CONFIG_TRUSTED_PRODUCERS");
                    if custom.is_empty() {
                        vec!["sie-config".to_string()]
                    } else {
                        custom
                    }
                }
            },

            auth_mode: env_default("SIE_AUTH_MODE", "none"),
            auth_tokens,
            admin_token: env::var("SIE_ADMIN_TOKEN").unwrap_or_default(),
            auth_exempt_operational: env_bool("SIE_AUTH_EXEMPT_OPERATIONAL"),

            log_level: env_default("SIE_LOG_LEVEL", "info"),
            json_logs: env_bool("SIE_LOG_JSON"),

            enable_pools: env_bool("SIE_GATEWAY_ENABLE_POOLS"),
            hot_reload: env_bool("SIE_GATEWAY_HOT_RELOAD"),
            watch_polling: env_bool("SIE_GATEWAY_WATCH_POLLING")
                || env_bool("SIE_GATEWAY_POLLING_WATCHER"),
            multi_router: env_bool("SIE_MULTI_ROUTER"),

            request_timeout: env_float("SIE_GATEWAY_REQUEST_TIMEOUT", 30.0),
            max_stream_pending: env_u64("SIE_GATEWAY_MAX_STREAM_PENDING", 50_000),
            stream_max_age_s: env_u64("SIE_STREAM_MAX_AGE_S", 120),

            configured_gpus,
            gpu_profile_map,
            static_queue_pools: env_static_queue_pools("SIE_GATEWAY_STATIC_QUEUE_POOLS"),
            model_aliases,

            bundles_dir: env_default("SIE_BUNDLES_DIR", "bundles"),
            models_dir: env_default("SIE_MODELS_DIR", "models"),

            config_service_url: {
                let raw = env::var("SIE_CONFIG_SERVICE_URL").unwrap_or_default();
                if raw.is_empty() {
                    None
                } else {
                    Some(raw)
                }
            },
            config_service_token: {
                let raw = env::var("SIE_ADMIN_TOKEN").unwrap_or_default();
                if raw.is_empty() {
                    None
                } else {
                    Some(raw)
                }
            },

            payload_store_url: env_default("SIE_PAYLOAD_STORE_URL", ""),
        }
    }

    /// Report auth configuration soundness. Returns `(level, message)`
    /// pairs that callers log at startup. Catches fail-open
    /// misconfigurations (e.g. tokens set while `SIE_AUTH_MODE=none`),
    /// unknown modes, missing tokens, and explicit operational bypasses.
    ///
    /// Does not mutate `self` and does not refuse startup; matches the
    /// gateway's "log and continue" posture for config issues.
    pub fn audit_auth(&self) -> Vec<(AuditLevel, String)> {
        let mut issues = Vec::new();
        let mode = self.auth_mode.as_str();
        let has_tokens = !self.auth_tokens.is_empty();
        let has_admin = !self.admin_token.is_empty();

        let is_enabled = matches!(mode, "static" | "token");
        let is_disabled = matches!(mode, "none" | "");

        if !is_enabled && !is_disabled {
            issues.push((
                AuditLevel::Error,
                format!(
                    "SIE_AUTH_MODE='{}' is not recognized; expected 'none', 'static', or 'token'. Auth is currently DISABLED (fail-open) because of the unknown mode — fix SIE_AUTH_MODE.",
                    mode
                ),
            ));
        }

        if !is_enabled && (has_tokens || has_admin) {
            issues.push((
                AuditLevel::Error,
                "SIE_AUTH_TOKEN(S) or SIE_ADMIN_TOKEN is set but SIE_AUTH_MODE is not 'static'/'token'. Auth is DISABLED; the tokens are dead configuration. Set SIE_AUTH_MODE=token to enforce auth.".to_string(),
            ));
        }

        if is_enabled && !has_tokens {
            issues.push((
                AuditLevel::Error,
                "Auth is enabled but SIE_AUTH_TOKEN(S) is empty. All non-probe requests will be rejected with 500.".to_string(),
            ));
        }

        if is_enabled && !has_admin {
            issues.push((
                AuditLevel::Warn,
                "Auth is enabled but SIE_ADMIN_TOKEN is unset. Admin-only endpoints (config writes, pool mutations) will refuse with 403 until an admin token is configured.".to_string(),
            ));
        }

        if is_enabled && self.auth_exempt_operational {
            issues.push((
                AuditLevel::Warn,
                "SIE_AUTH_EXEMPT_OPERATIONAL=true: status page, /health, /metrics, and /ws/* bypass auth. Use only when those endpoints are already network-isolated.".to_string(),
            ));
        }

        issues
    }

    /// Report NATS config-delta producer-trust soundness. Mirrors the
    /// pattern of `audit_auth` but scoped to the
    /// `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER` /
    /// `SIE_NATS_CONFIG_TRUSTED_PRODUCERS` pair. Emitted at startup.
    pub fn audit_nats_producer_trust(&self) -> Vec<(AuditLevel, String)> {
        let mut issues = Vec::new();
        // Both flags cannot be observed independently from `self` because
        // the load step collapses them into a single `Vec<String>`. We
        // detect the conflict by re-reading the env: "trust any" wins on
        // collapse, so if the allowlist env is *also* set we warn that it
        // is silently ignored. This is cheap and only runs once at boot.
        let trust_any = env_bool("SIE_NATS_CONFIG_TRUST_ANY_PRODUCER");
        let has_custom_allowlist = !env_csv("SIE_NATS_CONFIG_TRUSTED_PRODUCERS").is_empty();
        if trust_any && has_custom_allowlist {
            issues.push((
                AuditLevel::Warn,
                "SIE_NATS_CONFIG_TRUST_ANY_PRODUCER=true overrides SIE_NATS_CONFIG_TRUSTED_PRODUCERS; the allowlist is ignored. Unset one.".to_string(),
            ));
        }
        if self.nats_config_trusted_producers.is_empty() {
            issues.push((
                AuditLevel::Warn,
                "NATS config-delta producer validation is DISABLED; any publisher on sie.config.models._all will be accepted. Intended for local dev / single-node test clusters.".to_string(),
            ));
        }
        issues
    }
}

/// Severity for a `Config::audit_auth` finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditLevel {
    Warn,
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Serialize env-var tests to avoid races (env vars are process-global).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn with_env<F: FnOnce()>(vars: &[(&str, &str)], f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let old: Vec<(&str, Option<String>)> =
            vars.iter().map(|(k, _)| (*k, env::var(k).ok())).collect();
        for (k, v) in vars {
            env::set_var(k, v);
        }
        f();
        for (k, old_val) in old {
            match old_val {
                Some(v) => env::set_var(k, v),
                None => env::remove_var(k),
            }
        }
    }

    fn without_env<F: FnOnce()>(keys: &[&str], f: F) {
        let _guard = ENV_LOCK.lock().unwrap();
        let old: Vec<(&str, Option<String>)> =
            keys.iter().map(|k| (*k, env::var(k).ok())).collect();
        for k in keys {
            env::remove_var(k);
        }
        f();
        for (k, old_val) in old {
            if let Some(v) = old_val {
                env::set_var(k, v);
            }
        }
    }

    // ── env_bool ───────────────────────────────────────────────────

    #[test]
    fn test_env_bool_true_values() {
        for val in &["true", "1", "yes", "True", "YES", "TRUE"] {
            with_env(&[("_TEST_BOOL", val)], || {
                assert!(env_bool("_TEST_BOOL"), "expected true for '{}'", val);
            });
        }
    }

    #[test]
    fn test_env_bool_false_values() {
        for val in &["false", "0", "no", "anything"] {
            with_env(&[("_TEST_BOOL", val)], || {
                assert!(!env_bool("_TEST_BOOL"), "expected false for '{}'", val);
            });
        }
    }

    #[test]
    fn test_env_bool_missing() {
        without_env(&["_TEST_BOOL_MISSING"], || {
            assert!(!env_bool("_TEST_BOOL_MISSING"));
        });
    }

    // ── env_int ────────────────────────────────────────────────────

    #[test]
    fn test_env_int_valid() {
        with_env(&[("_TEST_INT", "9090")], || {
            assert_eq!(env_int("_TEST_INT", 80), 9090);
        });
    }

    #[test]
    fn test_env_int_invalid_uses_fallback() {
        with_env(&[("_TEST_INT", "not_a_number")], || {
            assert_eq!(env_int("_TEST_INT", 80), 80);
        });
    }

    #[test]
    fn test_env_int_missing_uses_fallback() {
        without_env(&["_TEST_INT_MISSING"], || {
            assert_eq!(env_int("_TEST_INT_MISSING", 8080), 8080);
        });
    }

    // ── env_float ──────────────────────────────────────────────────

    #[test]
    fn test_env_float_valid() {
        with_env(&[("_TEST_FLOAT", "2.5")], || {
            assert!((env_float("_TEST_FLOAT", 1.0) - 2.5).abs() < f64::EPSILON);
        });
    }

    #[test]
    fn test_env_float_fallback() {
        without_env(&["_TEST_FLOAT_MISSING"], || {
            assert!((env_float("_TEST_FLOAT_MISSING", 30.0) - 30.0).abs() < f64::EPSILON);
        });
    }

    // ── env_csv ────────────────────────────────────────────────────

    #[test]
    fn test_env_csv_multiple() {
        with_env(
            &[("_TEST_CSV", "http://a:80, http://b:80, http://c:80")],
            || {
                let result = env_csv("_TEST_CSV");
                assert_eq!(result, vec!["http://a:80", "http://b:80", "http://c:80"]);
            },
        );
    }

    #[test]
    fn test_env_csv_empty() {
        with_env(&[("_TEST_CSV", "")], || {
            assert!(env_csv("_TEST_CSV").is_empty());
        });
    }

    #[test]
    fn test_env_csv_missing() {
        without_env(&["_TEST_CSV_MISSING"], || {
            assert!(env_csv("_TEST_CSV_MISSING").is_empty());
        });
    }

    #[test]
    fn test_env_csv_trims_whitespace() {
        with_env(&[("_TEST_CSV", "  a , b , c  ")], || {
            assert_eq!(env_csv("_TEST_CSV"), vec!["a", "b", "c"]);
        });
    }

    #[test]
    fn test_env_csv_filters_empty_entries() {
        with_env(&[("_TEST_CSV", "a,,b,")], || {
            assert_eq!(env_csv("_TEST_CSV"), vec!["a", "b"]);
        });
    }

    #[test]
    fn test_env_json_string_map() {
        with_env(&[("_TEST_JSON_MAP", r#"{"l4":"l4-spot"}"#)], || {
            let result = env_json_string_map("_TEST_JSON_MAP");
            assert_eq!(result.get("l4"), Some(&"l4-spot".to_string()));
        });
    }

    #[test]
    fn test_env_json_string_map_invalid_is_empty() {
        with_env(&[("_TEST_JSON_MAP", "not-json")], || {
            assert!(env_json_string_map("_TEST_JSON_MAP").is_empty());
        });
    }

    #[test]
    fn test_env_static_queue_pools_accepts_helm_shape() {
        with_env(
            &[(
                "_TEST_STATIC_QUEUE_POOLS",
                r#"{
                    "companyA": {
                        "gpus": {"l4": 0},
                        "gpuCaps": {},
                        "minimumWorkerCount": 1
                    },
                    "companyB": {
                        "bundle": "sglang",
                        "gpus": {"a100": 1},
                        "gpu_caps": {"a100": 2},
                        "minimum_worker_count": 2
                    }
                }"#,
            )],
            || {
                let result = env_static_queue_pools("_TEST_STATIC_QUEUE_POOLS");

                assert_eq!(result.len(), 2);
                assert_eq!(result[0].name, "companyA");
                assert_eq!(result[0].gpus.get("l4"), Some(&0));
                assert!(result[0].gpu_caps.is_empty());
                assert_eq!(result[0].minimum_worker_count, 1);
                assert_eq!(result[0].ttl_seconds, None);

                assert_eq!(result[1].name, "companyB");
                assert_eq!(result[1].bundle.as_deref(), Some("sglang"));
                assert_eq!(result[1].gpus.get("a100"), Some(&1));
                assert_eq!(result[1].gpu_caps.get("a100"), Some(&2));
                assert_eq!(result[1].minimum_worker_count, 2);
            },
        );
    }

    #[test]
    fn test_env_static_queue_pools_minimum_worker_count_defaults_to_zero() {
        with_env(
            &[(
                "_TEST_STATIC_QUEUE_POOLS",
                r#"{"companyC": {"gpus": {"l4": 0}}}"#,
            )],
            || {
                let result = env_static_queue_pools("_TEST_STATIC_QUEUE_POOLS");
                assert_eq!(result.len(), 1);
                assert_eq!(result[0].minimum_worker_count, 0);
            },
        );
    }

    #[test]
    fn test_env_static_queue_pools_invalid_panics() {
        with_env(&[("_TEST_STATIC_QUEUE_POOLS", "not-json")], || {
            let result = std::panic::catch_unwind(|| {
                let _ = env_static_queue_pools("_TEST_STATIC_QUEUE_POOLS");
            });
            let panic = result.expect_err("invalid static queue pool JSON must fail fast");
            let message = panic
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| panic.downcast_ref::<&str>().copied())
                .unwrap_or("");
            assert!(message.contains("failed to parse _TEST_STATIC_QUEUE_POOLS"));
        });
    }

    #[test]
    fn test_build_gpu_profile_map_preserves_canonical_and_aliases() {
        let mut aliases = HashMap::new();
        aliases.insert("l4".to_string(), "l4-spot".to_string());

        let result = build_gpu_profile_map(&["l4-spot".to_string()], aliases);

        assert_eq!(result.get("l4-spot"), Some(&"l4-spot".to_string()));
        assert_eq!(result.get("l4"), Some(&"l4-spot".to_string()));
    }

    #[test]
    fn test_build_model_aliases_has_builtin_code_default() {
        let result = build_model_aliases(HashMap::new());
        assert_eq!(
            result.get("code"),
            Some(&"Qwen/Qwen3-4B-Instruct-2507".to_string())
        );
        assert_eq!(
            result.get("sql"),
            Some(&"Qwen/Qwen3-4B-Instruct-2507".to_string())
        );
        assert_eq!(
            result.get("guard"),
            Some(&"ibm-granite/granite-guardian-3.0-2b".to_string())
        );
    }

    #[test]
    fn test_build_model_aliases_env_overrides_and_extends() {
        let mut overrides = HashMap::new();
        overrides.insert("code".to_string(), "Org/Coder".to_string()); // override default
        overrides.insert("SQL".to_string(), "Org/SQLModel".to_string()); // extend + lowercased
        overrides.insert("blank".to_string(), "".to_string()); // skipped (empty target)

        let result = build_model_aliases(overrides);

        assert_eq!(result.get("code"), Some(&"Org/Coder".to_string()));
        assert_eq!(result.get("sql"), Some(&"Org/SQLModel".to_string()));
        assert!(!result.contains_key("blank"));
    }

    #[test]
    fn test_build_gpu_profile_map_does_not_override_canonical_profile() {
        let mut aliases = HashMap::new();
        aliases.insert("l4".to_string(), "l4-spot".to_string());

        let result = build_gpu_profile_map(&["l4".to_string(), "l4-spot".to_string()], aliases);

        assert_eq!(result.get("l4"), Some(&"l4".to_string()));
    }

    #[test]
    fn test_config_load_uses_gpu_aliases() {
        with_env(
            &[
                ("SIE_GATEWAY_CONFIGURED_GPUS", "l4-spot"),
                ("SIE_GATEWAY_GPU_ALIASES", r#"{"l4":"l4-spot"}"#),
            ],
            || {
                let cfg = Config::load();
                assert_eq!(cfg.configured_gpus, vec!["l4-spot"]);
                assert_eq!(cfg.gpu_profile_map.get("l4"), Some(&"l4-spot".to_string()));
            },
        );
    }

    // ── env_default ────────────────────────────────────────────────

    #[test]
    fn test_env_default_set() {
        with_env(&[("_TEST_DEFAULT", "custom_value")], || {
            assert_eq!(env_default("_TEST_DEFAULT", "fallback"), "custom_value");
        });
    }

    #[test]
    fn test_env_default_empty_uses_fallback() {
        with_env(&[("_TEST_DEFAULT", "")], || {
            assert_eq!(env_default("_TEST_DEFAULT", "fallback"), "fallback");
        });
    }

    #[test]
    fn test_env_default_missing_uses_fallback() {
        without_env(&["_TEST_DEFAULT_MISSING"], || {
            assert_eq!(env_default("_TEST_DEFAULT_MISSING", "fallback"), "fallback");
        });
    }

    // ── env_u64, env_usize ─────────────────────────────────────────

    #[test]
    fn test_env_u64() {
        with_env(&[("_TEST_U64", "12345")], || {
            assert_eq!(env_u64("_TEST_U64", 0), 12345);
        });
    }

    // ── Config.load integration ───────────────────────────────────

    #[test]
    fn test_config_service_url_unset_is_none() {
        without_env(&["SIE_CONFIG_SERVICE_URL"], || {
            let cfg = Config::load();
            assert!(cfg.config_service_url.is_none());
        });
    }

    #[test]
    fn test_config_service_url_from_env() {
        with_env(
            &[(
                "SIE_CONFIG_SERVICE_URL",
                "http://sie-config.sie.svc.cluster.local:8080",
            )],
            || {
                let cfg = Config::load();
                assert_eq!(
                    cfg.config_service_url.as_deref(),
                    Some("http://sie-config.sie.svc.cluster.local:8080"),
                );
            },
        );
    }

    #[test]
    fn test_config_service_url_empty_is_none() {
        with_env(&[("SIE_CONFIG_SERVICE_URL", "")], || {
            let cfg = Config::load();
            assert!(cfg.config_service_url.is_none());
        });
    }

    #[test]
    fn test_payload_store_url_default() {
        without_env(&["SIE_PAYLOAD_STORE_URL"], || {
            let cfg = Config::load();
            assert_eq!(cfg.payload_store_url, "");
        });
    }

    #[test]
    fn test_payload_store_url_from_env() {
        with_env(
            &[("SIE_PAYLOAD_STORE_URL", "s3://my-bucket/payloads")],
            || {
                let cfg = Config::load();
                assert_eq!(cfg.payload_store_url, "s3://my-bucket/payloads");
            },
        );
    }

    #[test]
    fn test_metrics_port_from_env() {
        with_env(&[("SIE_METRICS_PORT", "9090")], || {
            let cfg = Config::load();
            assert_eq!(cfg.metrics_port, Some(9090));
        });
    }

    #[test]
    fn test_metrics_port_unset_is_none() {
        without_env(&["SIE_METRICS_PORT"], || {
            let cfg = Config::load();
            assert_eq!(cfg.metrics_port, None);
        });
    }

    #[test]
    fn test_metrics_port_invalid_is_none() {
        with_env(&[("SIE_METRICS_PORT", "not-a-number")], || {
            let cfg = Config::load();
            assert_eq!(cfg.metrics_port, None);
        });
    }

    #[test]
    fn test_metrics_port_out_of_range_is_none() {
        with_env(&[("SIE_METRICS_PORT", "70000")], || {
            let cfg = Config::load();
            assert_eq!(cfg.metrics_port, None);
        });
    }

    #[test]
    fn test_stream_max_age_default_matches_worker_contract() {
        without_env(&["SIE_STREAM_MAX_AGE_S"], || {
            let cfg = Config::load();
            assert_eq!(cfg.stream_max_age_s, 120);
        });
    }

    #[test]
    fn test_stream_max_age_from_env() {
        with_env(&[("SIE_STREAM_MAX_AGE_S", "240")], || {
            let cfg = Config::load();
            assert_eq!(cfg.stream_max_age_s, 240);
        });
    }

    #[test]
    fn test_admin_token_populates_config_service_token() {
        with_env(&[("SIE_ADMIN_TOKEN", "super-secret")], || {
            let cfg = Config::load();
            assert_eq!(cfg.admin_token, "super-secret");
            assert_eq!(cfg.config_service_token.as_deref(), Some("super-secret"));
        });
    }

    #[test]
    fn test_admin_token_unset_leaves_config_service_token_none() {
        without_env(&["SIE_ADMIN_TOKEN"], || {
            let cfg = Config::load();
            assert!(cfg.admin_token.is_empty());
            assert!(cfg.config_service_token.is_none());
        });
    }

    // ── audit_auth ─────────────────────────────────────────────────

    fn cfg_with_auth(
        mode: &str,
        tokens: Vec<&str>,
        admin: &str,
        exempt_operational: bool,
    ) -> Config {
        let mut cfg = Config {
            host: String::new(),
            port: 0,
            metrics_port: None,
            worker_urls: Vec::new(),
            use_kubernetes: false,
            k8s_namespace: String::new(),
            k8s_service: String::new(),
            k8s_port: 0,
            health_mode: String::new(),
            nats_url: String::new(),
            nats_config_trusted_producers: Vec::new(),
            auth_mode: mode.to_string(),
            auth_tokens: tokens.into_iter().map(String::from).collect(),
            admin_token: admin.to_string(),
            auth_exempt_operational: exempt_operational,
            log_level: String::new(),
            json_logs: false,
            enable_pools: false,
            hot_reload: false,
            watch_polling: false,
            multi_router: false,
            request_timeout: 0.0,
            max_stream_pending: 0,
            stream_max_age_s: 0,
            configured_gpus: Vec::new(),
            gpu_profile_map: HashMap::new(),
            static_queue_pools: Vec::new(),
            model_aliases: HashMap::new(),
            bundles_dir: String::new(),
            models_dir: String::new(),
            config_service_url: None,
            config_service_token: None,
            payload_store_url: String::new(),
        };
        // Silence the "unused mut" warning on the path where we don't mutate.
        let _ = &mut cfg;
        cfg
    }

    #[test]
    fn test_audit_auth_none_with_tokens_is_error() {
        let cfg = cfg_with_auth("none", vec!["t1"], "admin", false);
        let issues = cfg.audit_auth();
        assert!(
            issues
                .iter()
                .any(|(lvl, msg)| *lvl == AuditLevel::Error && msg.contains("DISABLED")),
            "expected error about tokens + disabled auth, got {:?}",
            issues
        );
    }

    #[test]
    fn test_audit_auth_token_mode_accepted() {
        let cfg = cfg_with_auth("token", vec!["t1"], "admin", false);
        let issues = cfg.audit_auth();
        assert!(
            issues.iter().all(|(lvl, _)| *lvl != AuditLevel::Error),
            "unexpected errors: {:?}",
            issues
        );
    }

    #[test]
    fn test_audit_auth_static_mode_accepted() {
        let cfg = cfg_with_auth("static", vec!["t1"], "admin", false);
        let issues = cfg.audit_auth();
        assert!(issues.iter().all(|(lvl, _)| *lvl != AuditLevel::Error));
    }

    #[test]
    fn test_audit_auth_enabled_without_tokens_is_error() {
        let cfg = cfg_with_auth("token", vec![], "", false);
        let issues = cfg.audit_auth();
        assert!(issues
            .iter()
            .any(|(lvl, msg)| *lvl == AuditLevel::Error && msg.contains("SIE_AUTH_TOKEN")));
    }

    #[test]
    fn test_audit_auth_enabled_without_admin_token_is_warn() {
        let cfg = cfg_with_auth("token", vec!["t1"], "", false);
        let issues = cfg.audit_auth();
        assert!(issues
            .iter()
            .any(|(lvl, msg)| *lvl == AuditLevel::Warn && msg.contains("SIE_ADMIN_TOKEN")));
    }

    #[test]
    fn test_audit_auth_unknown_mode_is_error() {
        let cfg = cfg_with_auth("bearer", vec!["t1"], "admin", false);
        let issues = cfg.audit_auth();
        assert!(issues
            .iter()
            .any(|(lvl, msg)| *lvl == AuditLevel::Error && msg.contains("not recognized")));
    }

    #[test]
    fn test_audit_auth_exempt_operational_is_warn() {
        let cfg = cfg_with_auth("token", vec!["t1"], "admin", true);
        let issues = cfg.audit_auth();
        assert!(issues
            .iter()
            .any(|(lvl, msg)| *lvl == AuditLevel::Warn
                && msg.contains("SIE_AUTH_EXEMPT_OPERATIONAL")));
    }

    #[test]
    fn test_audit_auth_clean_none_no_findings() {
        let cfg = cfg_with_auth("none", vec![], "", false);
        let issues = cfg.audit_auth();
        assert!(issues.is_empty(), "expected no findings, got {:?}", issues);
    }

    /// Guard: the gateway does not own a config store. Setting
    /// `SIE_CONFIG_STORE_DIR` or `SIE_CONFIG_RESTORE` must not resurrect a
    /// config-store path or mutate `Config`. If a future change reintroduces
    /// a field that reads either variable, this test has to be updated
    /// deliberately.
    #[test]
    fn test_removed_config_store_env_vars_are_ignored() {
        with_env(
            &[
                ("SIE_CONFIG_STORE_DIR", "/var/lib/gateway/config-store"),
                ("SIE_CONFIG_RESTORE", "true"),
            ],
            || {
                let cfg = Config::load();
                // No field on Config reads either var; this test guards against
                // a future accidental re-introduction being done via a field we
                // forgot to check. If someone adds config_store_dir back, this
                // test has to be updated deliberately.
                assert!(cfg.config_service_url.is_none());
                // The Debug impl for Config MUST NOT contain the removed paths.
                let dbg = format!("{:?}", cfg);
                assert!(
                    !dbg.contains("config_store_dir"),
                    "Config resurrected config_store_dir: {}",
                    dbg
                );
                assert!(
                    !dbg.contains("config_restore"),
                    "Config resurrected config_restore: {}",
                    dbg
                );
            },
        );
    }
}
