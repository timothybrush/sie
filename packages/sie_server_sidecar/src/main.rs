//! sie-server-sidecar — sidecar driving the NATS queue for a sie-server pod.
//!
//! See `docs/architecture-guide.md` for the design. This binary is the
//! entrypoint; the library modules live in `lib.rs`.

use anyhow::Context;
use clap::Parser;
use tracing::{error, info};

use sie_server_sidecar::config::WorkerConfig;
use sie_server_sidecar::config_subscriber::trusted_producers_from_env;
use sie_server_sidecar::dispatcher::default_max_concurrent_batches;
use sie_server_sidecar::run;

#[derive(Parser, Debug)]
#[command(author, version, about = "SIE server sidecar", long_about = None)]
struct Cli {
    #[arg(long, env = "SIE_NATS_URL")]
    nats_url: String,

    #[arg(long, env = "SIE_POOL")]
    pool: String,

    #[arg(long, env = "SIE_BUNDLE")]
    bundle: String,

    #[arg(long, env = "SIE_IPC_SOCKET_PATH", default_value = "/tmp/sie-ipc.sock")]
    ipc_socket_path: String,

    /// Number of concurrent IPC connections to the Python adapter process.
    /// Defaults to `SIE_MAX_CONCURRENT_BATCHES` when unset (falling
    /// back to the dispatcher's own default of 4), so the IPC pool
    /// never becomes the binding constraint on the dispatcher's
    /// concurrency cap.
    #[arg(long, env = "SIE_IPC_POOL_SIZE")]
    ipc_pool_size: Option<usize>,

    #[arg(long, env = "SIE_PAYLOAD_STORE_URL")]
    payload_store_url: Option<String>,

    /// Gateway URL used by the worker-side pool admission gate.
    #[arg(long, env = "SIE_GATEWAY_URL")]
    gateway_url: Option<String>,

    /// Bearer token for gateway pool-status reads.
    #[arg(long, env = "SIE_GATEWAY_API_KEY")]
    gateway_api_key: Option<String>,

    /// Enable/disable the worker-side pool admission gate.
    #[arg(long, env = "SIE_POOL_ADMISSION_ENABLED")]
    pool_admission_enabled: Option<String>,

    /// Pool admission check cadence, in seconds.
    #[arg(
        long,
        env = "SIE_POOL_ADMISSION_CHECK_INTERVAL_S",
        default_value = "5.0"
    )]
    pool_admission_check_interval_s: f64,

    /// Sleep duration while the worker is not admitted, in seconds.
    #[arg(long, env = "SIE_POOL_ADMISSION_PAUSE_S", default_value = "1.0")]
    pool_admission_pause_s: f64,

    /// How long to reuse the last successful admission decision after a
    /// transient gateway/status error, in seconds.
    #[arg(long, env = "SIE_POOL_ADMISSION_STALE_AFTER_S", default_value = "30.0")]
    pool_admission_stale_after_s: f64,

    #[arg(long, env = "SIE_WORKER_METRICS_PORT", default_value = "9095")]
    metrics_port: u16,

    #[arg(long, env = "SIE_WORKER_ID")]
    worker_id: Option<String>,

    #[arg(long, env = "SIE_WORKER_PING_INTERVAL_MS", default_value = "2000")]
    ping_interval_ms: u64,

    /// Multiplier applied to `--ping-interval-ms` to derive the
    /// staleness threshold for `/readyz`. Default `3`, matching the
    /// sidecar readiness contract. `0` is treated
    /// as the default (see `WorkerConfig::ready_stale_mult`).
    #[arg(long, env = "SIE_WORKER_READYZ_STALE_MULT", default_value = "3")]
    ready_stale_mult: u32,

    /// Machine-profile label echoed in NATS health heartbeats and
    /// embedded in this worker's queue subject lane. Requests carrying
    /// `X-SIE-MACHINE-PROFILE` constrain route resolution to this value.
    /// Required. Helm sets it explicitly from the worker pool's
    /// machineProfile.
    #[arg(long, env = "SIE_MACHINE_PROFILE")]
    machine_profile: String,

    /// GPU count surfaced in heartbeats. Informational; the
    /// gateway coerces 0 to 1. Defaults to 1.
    #[arg(long, env = "SIE_GPU_COUNT", default_value = "1")]
    gpu_count: i32,

    /// Optional bundle-config hash echoed in heartbeats. Empty
    /// when the operator hasn't pinned a value.
    #[arg(long, env = "SIE_BUNDLE_CONFIG_HASH", default_value = "")]
    bundle_config_hash: String,

    /// Optional sie-config base URL. When set, the sidecar polls
    /// /v1/configs/epoch and reconciles missed config deltas from
    /// /v1/configs/export.
    #[arg(long, env = "SIE_CONFIG_SERVICE_URL")]
    config_service_url: Option<String>,

    /// Bearer token for sie-config export reads. Defaults from the shared
    /// SIE_ADMIN_TOKEN secret in Helm when config auth is enabled.
    #[arg(long, env = "SIE_ADMIN_TOKEN")]
    config_service_token: Option<String>,

    /// Worker-side config epoch poll interval in milliseconds.
    #[arg(
        long,
        env = "SIE_WORKER_CONFIG_POLL_INTERVAL_MS",
        default_value = "30000"
    )]
    config_poll_interval_ms: u64,

    /// Slow full-export reconcile interval in milliseconds. Set to 0 to
    /// disable after the startup export. Kept separate from the epoch poll
    /// so no-config-store deployments can still recover missed live deltas.
    #[arg(
        long,
        env = "SIE_WORKER_CONFIG_FULL_EXPORT_INTERVAL_MS",
        default_value = "300000"
    )]
    config_full_export_interval_ms: u64,

    /// Heartbeat interval (ms) for the NATS health publisher.
    /// Default 5_000 ms; must stay << gateway
    /// `heartbeat_timeout` (30 s) so the staleness check has
    /// margin.
    #[arg(long, env = "SIE_HEALTH_PUBLISH_INTERVAL_MS", default_value = "5000")]
    health_publish_interval_ms: u64,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_tracing();

    let cli = Cli::parse();
    let pool = validate_lane_segment("SIE_POOL", cli.pool)?;
    let bundle = validate_lane_segment("SIE_BUNDLE", cli.bundle)?;
    let machine_profile = validate_lane_segment("SIE_MACHINE_PROFILE", cli.machine_profile)?;
    let config = WorkerConfig {
        nats_url: cli.nats_url,
        pool,
        bundle,
        ipc_socket_path: cli.ipc_socket_path.into(),
        ipc_pool_size: cli
            .ipc_pool_size
            .filter(|&n| n > 0)
            .unwrap_or_else(default_max_concurrent_batches),
        payload_store_url: cli.payload_store_url,
        gateway_url: cli.gateway_url.filter(|url| !url.trim().is_empty()),
        gateway_api_key: cli.gateway_api_key.filter(|token| !token.trim().is_empty()),
        pool_admission_enabled: env_bool_value(cli.pool_admission_enabled.as_deref(), true),
        pool_admission_check_interval_ms: seconds_to_millis(
            cli.pool_admission_check_interval_s,
            1_000,
        ),
        pool_admission_pause_ms: seconds_to_millis(cli.pool_admission_pause_s, 100),
        pool_admission_stale_after_ms: seconds_to_millis(cli.pool_admission_stale_after_s, 0),
        metrics_port: cli.metrics_port,
        worker_id: cli.worker_id.unwrap_or_else(|| {
            std::env::var("HOSTNAME").unwrap_or_else(|_| format!("worker-{}", uuid::Uuid::new_v4()))
        }),
        ping_interval_ms: cli.ping_interval_ms,
        ready_stale_mult: cli.ready_stale_mult,
        machine_profile,
        gpu_count: cli.gpu_count,
        bundle_config_hash: cli.bundle_config_hash,
        config_service_url: cli.config_service_url.filter(|url| !url.trim().is_empty()),
        config_service_token: cli
            .config_service_token
            .filter(|token| !token.trim().is_empty()),
        config_poll_interval_ms: cli.config_poll_interval_ms.max(1_000),
        config_full_export_interval_ms: cli.config_full_export_interval_ms,
        nats_config_trusted_producers: trusted_producers_from_env(),
        health_publish_interval_ms: cli.health_publish_interval_ms,
    };

    info!(
        pool = %config.pool,
        bundle = %config.bundle,
        ipc = %config.ipc_socket_path.display(),
        ipc_pool_size = config.ipc_pool_size,
        worker_id = %config.worker_id,
        "sie-server-sidecar starting"
    );

    if let Err(e) = run(config).await.context("server sidecar run loop") {
        error!(error = ?e, "sie-server-sidecar exited with error");
        std::process::exit(1);
    }

    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    fmt().with_env_filter(filter).json().init();
}

fn validate_lane_segment(name: &str, raw: String) -> anyhow::Result<String> {
    let value = raw.trim();
    if value.is_empty() {
        anyhow::bail!("{name} must not be empty");
    }
    if value
        .chars()
        .any(|c| c.is_whitespace() || matches!(c, '.' | '*' | '>'))
    {
        anyhow::bail!("{name} must not contain '.', '*', '>', or whitespace");
    }
    Ok(value.to_string())
}

fn env_bool_value(raw: Option<&str>, default: bool) -> bool {
    let Some(raw) = raw else {
        return default;
    };
    let trimmed = raw.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return default;
    }
    !matches!(trimmed.as_str(), "0" | "false" | "no" | "off")
}

fn seconds_to_millis(seconds: f64, min_ms: u64) -> u64 {
    if !seconds.is_finite() || seconds <= 0.0 {
        return min_ms;
    }
    ((seconds * 1_000.0).round() as u64).max(min_ms)
}

#[cfg(test)]
mod tests {
    use super::validate_lane_segment;

    #[test]
    fn validate_lane_segment_trims_valid_values() {
        assert_eq!(
            validate_lane_segment("SIE_POOL", " default-pool ".to_string()).unwrap(),
            "default-pool"
        );
    }

    #[test]
    fn validate_lane_segment_rejects_subject_wildcards_and_dot_tokens() {
        for raw in ["", "   ", "foo.bar", "foo*", "foo>", "foo bar"] {
            assert!(validate_lane_segment("SIE_POOL", raw.to_string()).is_err());
        }
    }
}
