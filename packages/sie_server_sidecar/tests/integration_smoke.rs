//! End-to-end smoke test for the worker-sidecar.
//!
//! What this proves:
//!   * The Rust `sie-server-sidecar` binary compiles and starts.
//!   * It can reach a real NATS JetStream server and create the expected
//!     pool stream + durable pull consumer.
//!   * It can open the UDS to the Python `IpcServer` and exchange
//!     msgpack-framed RPCs (Ping, EnsureModelReady, ProcessEncodeBatch).
//!   * It can direct-dispatch generation work to `ProcessGenerate`, stream
//!     generation events back over IPC, publish the raw response, and ACK.
//!   * It publishes a `WorkResult` to the reply subject whose shape matches
//!     what `sie_gateway` expects.
//!
//! What it does **not** prove:
//!   * Real inference. The Python side runs a stub executor
//!     (`sie_server._ipc_test_harness`) that returns a canned
//!     `result_msgpack` without loading any model.
//!   * Anything about the gateway publisher / collector.
//!   * Production configuration of S3, metrics scraping, etc.
//!
//! External deps required: `nats-server` on `$PATH` and `uv` (for the
//! Python harness). If either is missing the test is skipped so clean
//! dev machines / CI without these tools still `cargo test` green.

use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use sie_server_sidecar::subject::normalize_model_id;
use sie_server_sidecar::work_types::{WorkItem, WorkResult};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;
use tokio::process::{Child, Command};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio::time::{sleep, timeout};

static SMOKE_TEST_SEMAPHORE: OnceLock<Arc<Semaphore>> = OnceLock::new();

fn msg_value(value: serde_json::Value) -> rmpv::Value {
    let bytes = rmp_serde::to_vec_named(&value).expect("msgpack encode fixture value");
    rmp_serde::from_slice(&bytes).expect("msgpack decode fixture value")
}

fn text_item(text: impl Into<String>) -> rmpv::Value {
    let text = text.into();
    msg_value(serde_json::json!({ "text": text }))
}

fn document_item(data: Vec<u8>, format: &str) -> rmpv::Value {
    rmpv::Value::Map(vec![(
        rmpv::Value::from("document"),
        rmpv::Value::Map(vec![
            (rmpv::Value::from("data"), rmpv::Value::Binary(data)),
            (rmpv::Value::from("format"), rmpv::Value::from(format)),
        ]),
    )])
}

fn msg_value_key_eq(key: &rmpv::Value, expected: &str) -> bool {
    match key {
        rmpv::Value::String(s) => s.as_str() == Some(expected),
        rmpv::Value::Binary(b) => std::str::from_utf8(b).ok() == Some(expected),
        _ => false,
    }
}

fn msg_map_get<'a>(value: &'a rmpv::Value, key: &str) -> Option<&'a rmpv::Value> {
    let rmpv::Value::Map(entries) = value else {
        return None;
    };
    entries
        .iter()
        .find(|(k, _)| msg_value_key_eq(k, key))
        .map(|(_, v)| v)
}

fn msg_as_bool(value: &rmpv::Value) -> Option<bool> {
    match value {
        rmpv::Value::Boolean(value) => Some(*value),
        _ => None,
    }
}

fn msg_as_str(value: &rmpv::Value) -> Option<&str> {
    match value {
        rmpv::Value::String(value) => value.as_str(),
        rmpv::Value::Binary(value) => std::str::from_utf8(value).ok(),
        _ => None,
    }
}

fn msg_as_u64(value: &rmpv::Value) -> Option<u64> {
    match value {
        rmpv::Value::Integer(value) => value.as_u64(),
        _ => None,
    }
}

async fn smoke_test_guard() -> OwnedSemaphorePermit {
    SMOKE_TEST_SEMAPHORE
        .get_or_init(|| Arc::new(Semaphore::new(1)))
        .clone()
        .acquire_owned()
        .await
        .expect("smoke test semaphore is open")
}

fn pool_work_subject(pool: &str, machine_profile: &str, bundle: &str, model_id: &str) -> String {
    format!(
        "sie.work.{}.{}.{}.{}",
        pool,
        normalize_model_id(machine_profile),
        normalize_model_id(bundle),
        normalize_model_id(model_id)
    )
}

fn worker_work_subject(
    pool: &str,
    machine_profile: &str,
    bundle: &str,
    model_id: &str,
    worker_id: &str,
) -> String {
    format!(
        "sie.work.{}.{}.{}.{}.{}",
        pool,
        normalize_model_id(machine_profile),
        normalize_model_id(bundle),
        normalize_model_id(model_id),
        normalize_model_id(worker_id)
    )
}

// ---------------------------------------------------------------------------
// Skip helpers — keep the test happy on bare-bones dev machines
// ---------------------------------------------------------------------------

fn which(bin: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths).find_map(|dir| {
            let p = dir.join(bin);
            if p.is_file() {
                Some(p)
            } else {
                None
            }
        })
    })
}

fn skip_unless_tools_available() -> bool {
    let mut missing = vec![];
    if which("nats-server").is_none() {
        missing.push("nats-server");
    }
    if which("mise").is_none() && which("uv").is_none() {
        missing.push("mise/uv");
    }
    if !missing.is_empty() {
        eprintln!(
            "integration_smoke: skipping — missing tools on $PATH: {}",
            missing.join(", ")
        );
        return true;
    }
    false
}

// ---------------------------------------------------------------------------
// NATS harness
// ---------------------------------------------------------------------------

fn find_free_tcp_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    listener.local_addr().unwrap().port()
}

struct NatsHarness {
    child: Child,
    url: String,
    port: u16,
}

impl NatsHarness {
    async fn start() -> Self {
        let port = find_free_tcp_port();
        let store_dir = tempfile::tempdir().expect("jetstream dir");
        let child = Command::new("nats-server")
            .arg("-p")
            .arg(port.to_string())
            .arg("-js")
            .arg("-sd")
            .arg(store_dir.path())
            .arg("-m")
            .arg("0")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .expect("spawn nats-server");

        // Keep the temp dir alive for the server's lifetime.
        std::mem::forget(store_dir);

        let url = format!("nats://127.0.0.1:{port}");
        wait_for_tcp(port, Duration::from_secs(10))
            .await
            .expect("nats-server ready");

        Self { child, url, port }
    }
}

async fn wait_for_tcp(port: u16, budget: Duration) -> Result<(), String> {
    let deadline = Instant::now() + budget;
    while Instant::now() < deadline {
        if TcpStream::connect(("127.0.0.1", port)).is_ok() {
            return Ok(());
        }
        sleep(Duration::from_millis(50)).await;
    }
    Err(format!("timeout waiting for 127.0.0.1:{port}"))
}

/// Raw HTTP/1.1 GET /metrics against the worker's Prometheus endpoint.
/// A hand-rolled client avoids pulling reqwest into dev-deps for this
/// single assertion.
fn scrape_metrics(port: u16) -> std::io::Result<String> {
    use std::io::{Read, Write};
    let mut sock = TcpStream::connect(("127.0.0.1", port))?;
    sock.set_read_timeout(Some(Duration::from_secs(5)))?;
    sock.set_write_timeout(Some(Duration::from_secs(5)))?;
    sock.write_all(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")?;
    let mut raw = String::new();
    sock.read_to_string(&mut raw)?;
    // Strip HTTP headers — body starts after the first blank line.
    match raw.find("\r\n\r\n") {
        Some(idx) => Ok(raw[idx + 4..].to_string()),
        None => Ok(raw),
    }
}

impl Drop for NatsHarness {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

// ---------------------------------------------------------------------------
// Gateway pool-status harness
// ---------------------------------------------------------------------------

struct GatewayPoolHarness {
    url: String,
    assigned: Arc<AtomicBool>,
    handle: tokio::task::JoinHandle<()>,
}

impl GatewayPoolHarness {
    async fn start(
        pool: &'static str,
        worker_id: &'static str,
        machine_profile: &'static str,
    ) -> Self {
        Self::start_with_queue_pool(pool, pool, worker_id, machine_profile).await
    }

    async fn start_with_queue_pool(
        pool: &'static str,
        queue_pool: &'static str,
        worker_id: &'static str,
        machine_profile: &'static str,
    ) -> Self {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .await
            .expect("bind fake gateway");
        let addr = listener.local_addr().expect("fake gateway local addr");
        let assigned = Arc::new(AtomicBool::new(false));
        let assigned_for_task = Arc::clone(&assigned);
        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                let assigned = Arc::clone(&assigned_for_task);
                tokio::spawn(async move {
                    let mut buf = vec![0u8; 2048];
                    let Ok(n) = socket.read(&mut buf).await else {
                        return;
                    };
                    let request = String::from_utf8_lossy(&buf[..n]);
                    let ok_path = request.lines().next().is_some_and(|line| {
                        let mut parts = line.split_whitespace();
                        matches!(parts.next(), Some("GET"))
                            && parts.next().is_some_and(|target| {
                                target == "/v1/pools" || target.starts_with("/v1/pools?")
                            })
                    });
                    let (status, body) = if ok_path {
                        let assigned_workers = if assigned.load(Ordering::Acquire) {
                            format!(
                                r#"[{{"name":"{worker_id}","url":"http://{worker_id}","gpu":"{machine_profile}"}}]"#
                            )
                        } else {
                            "[]".to_string()
                        };
                        (
                            "200 OK",
                            format!(
                                r#"{{"pools":[{{"spec":{{"name":"{pool}","queue_pool":"{queue_pool}","gpu_caps":{{"{machine_profile}":1}}}},"status":{{"state":"active","assigned_workers":{assigned_workers}}}}}]}}"#
                            ),
                        )
                    } else {
                        ("404 Not Found", "{}".to_string())
                    };
                    let response = format!(
                        "HTTP/1.1 {status}\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    );
                    let _ = socket.write_all(response.as_bytes()).await;
                });
            }
        });

        Self {
            url: format!("http://{addr}"),
            assigned,
            handle,
        }
    }

    fn set_assigned(&self, value: bool) {
        self.assigned.store(value, Ordering::Release);
    }
}

impl Drop for GatewayPoolHarness {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

// ---------------------------------------------------------------------------
// Python IPC harness
// ---------------------------------------------------------------------------

struct PythonHarness {
    child: Child,
    socket_path: PathBuf,
}

impl PythonHarness {
    async fn start(socket_path: PathBuf) -> Self {
        Self::start_with_extra_args(socket_path, 0, Vec::new()).await
    }

    /// Like [`Self::start`] but injects a fixed `per_request_delay_ms`
    /// into every `process_*_batch` RPC on the Python side. Tests use
    /// this to make concurrent in-flight RPCs observable on the
    /// Prometheus `sie_worker_ipc_pool_inflight` gauge.
    async fn start_with_delay_ms(socket_path: PathBuf, per_request_delay_ms: u64) -> Self {
        Self::start_with_extra_args(socket_path, per_request_delay_ms, Vec::new()).await
    }

    async fn start_with_descriptor(
        socket_path: PathBuf,
        tokenizer_path: &std::path::Path,
        max_seq_len: u32,
    ) -> Self {
        Self::start_with_extra_args(
            socket_path,
            0,
            vec![
                "--tokenizer-path".to_string(),
                tokenizer_path.display().to_string(),
                "--max-seq-len".to_string(),
                max_seq_len.to_string(),
            ],
        )
        .await
    }

    async fn start_with_fake_generate(socket_path: PathBuf, model_id: &str) -> Self {
        Self::start_with_extra_args(
            socket_path,
            0,
            vec!["--fake-generate-model".to_string(), model_id.to_string()],
        )
        .await
    }

    async fn start_with_fake_generate_after_capability_polls(
        socket_path: PathBuf,
        model_id: &str,
        hidden_polls: u32,
    ) -> Self {
        Self::start_with_extra_args(
            socket_path,
            0,
            vec![
                "--fake-generate-model".to_string(),
                model_id.to_string(),
                "--fake-generate-hidden-polls".to_string(),
                hidden_polls.to_string(),
            ],
        )
        .await
    }

    async fn start_with_extra_args(
        socket_path: PathBuf,
        per_request_delay_ms: u64,
        extra_args: Vec<String>,
    ) -> Self {
        let (program, base_args) = if which("mise").is_some() {
            (
                "mise",
                vec![
                    "exec".to_string(),
                    "--".to_string(),
                    "uv".to_string(),
                    "run".to_string(),
                    "--no-sync".to_string(),
                ],
            )
        } else {
            ("uv", vec!["run".to_string(), "--no-sync".to_string()])
        };

        let workspace = workspace_root();
        let delay_str = per_request_delay_ms.to_string();
        let mut harness_args = vec![
            "python".to_string(),
            "-m".to_string(),
            "sie_server._ipc_test_harness".to_string(),
            "--socket".to_string(),
            socket_path.to_str().unwrap().to_string(),
            "--worker-id".to_string(),
            "smoke-harness".to_string(),
            "--log-level".to_string(),
            "INFO".to_string(),
            "--per-request-delay-ms".to_string(),
            delay_str,
        ];
        harness_args.extend(extra_args);

        let mut cmd = Command::new(program);
        cmd.current_dir(&workspace)
            .args(&base_args)
            .args(&harness_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = cmd.spawn().expect("spawn python harness");

        let stdout = child.stdout.take().expect("stdout");
        let mut reader = BufReader::new(stdout).lines();

        let ready_deadline = Instant::now() + Duration::from_secs(45);
        let mut ready = false;
        while Instant::now() < ready_deadline {
            match timeout(Duration::from_millis(500), reader.next_line()).await {
                Ok(Ok(Some(line))) => {
                    eprintln!("[python-harness] {line}");
                    if line.contains("HARNESS_READY") {
                        ready = true;
                        break;
                    }
                }
                Ok(Ok(None)) => break,
                Ok(Err(e)) => panic!("read python stdout: {e}"),
                Err(_) => {
                    if let Some(status) = child.try_wait().expect("try_wait") {
                        panic!("python harness exited early: status={status:?}");
                    }
                }
            }
        }
        if !ready {
            panic!(
                "python harness never printed HARNESS_READY within 45s (socket={})",
                socket_path.display()
            );
        }

        // Drain remaining stdout in background so the pipe doesn't fill up.
        tokio::spawn(async move {
            while let Ok(Some(line)) = reader.next_line().await {
                eprintln!("[python-harness] {line}");
            }
        });

        // Drain stderr too.
        if let Some(stderr) = child.stderr.take() {
            let mut stderr_reader = BufReader::new(stderr).lines();
            tokio::spawn(async move {
                while let Ok(Some(line)) = stderr_reader.next_line().await {
                    eprintln!("[python-harness:stderr] {line}");
                }
            });
        }

        Self { child, socket_path }
    }
}

impl Drop for PythonHarness {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

fn workspace_root() -> PathBuf {
    // tests/ → package root; package root → workspace root is two levels up
    // (packages/sie_server_sidecar).
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("canonicalize workspace root")
}

// ---------------------------------------------------------------------------
// sie-server-sidecar binary harness
// ---------------------------------------------------------------------------

struct WorkerHarness {
    child: Child,
}

impl WorkerHarness {
    fn spawn(
        nats_url: &str,
        ipc_socket: &std::path::Path,
        pool: &str,
        bundle: &str,
        metrics_port: u16,
        payload_store_url: Option<&str>,
    ) -> Self {
        Self::spawn_with_env(
            nats_url,
            ipc_socket,
            pool,
            bundle,
            metrics_port,
            payload_store_url,
            &[],
        )
    }

    /// Same as [`Self::spawn`] but allows injecting additional env vars
    /// (used e.g. to set `SIE_IPC_POOL_SIZE` for pool integration tests).
    fn spawn_with_env(
        nats_url: &str,
        ipc_socket: &std::path::Path,
        pool: &str,
        bundle: &str,
        metrics_port: u16,
        payload_store_url: Option<&str>,
        extra_env: &[(&str, &str)],
    ) -> Self {
        let exe = env!("CARGO_BIN_EXE_sie-server-sidecar");
        let mut cmd = Command::new(exe);
        cmd.env("SIE_NATS_URL", nats_url)
            .env("SIE_POOL", pool)
            .env("SIE_MACHINE_PROFILE", pool)
            .env("SIE_BUNDLE", bundle)
            .env("SIE_IPC_SOCKET_PATH", ipc_socket)
            .env("SIE_WORKER_METRICS_PORT", metrics_port.to_string())
            .env("SIE_WORKER_ID", "smoke-worker")
            .env("SIE_WORKER_PING_INTERVAL_MS", "500")
            .env("RUST_LOG", "info,sie_server_sidecar=debug,async_nats=warn")
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true);
        if let Some(url) = payload_store_url {
            cmd.env("SIE_PAYLOAD_STORE_URL", url);
        }
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let child = cmd.spawn().expect("spawn sie-server-sidecar");
        Self { child }
    }
}

impl Drop for WorkerHarness {
    fn drop(&mut self) {
        let _ = self.child.start_kill();
    }
}

// ---------------------------------------------------------------------------
// Short-socket helper — macOS AF_UNIX has a ~104-char limit on paths, and the
// `$TMPDIR` on this platform is ~50 chars. We create a dir under /tmp (short)
// instead.
// ---------------------------------------------------------------------------

struct ShortSocket {
    path: PathBuf,
    _dir: tempfile::TempDir,
}

impl ShortSocket {
    fn new(name: &str) -> Self {
        let dir = tempfile::Builder::new()
            .prefix("siews-")
            .tempdir_in("/tmp")
            .expect("create short socket dir");
        let path = dir.path().join(name);
        Self { path, _dir: dir }
    }
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_encode_request_round_trips_through_rust_worker() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    // 1. Start NATS.
    let nats = NatsHarness::start().await;
    eprintln!("nats: {} (port {})", nats.url, nats.port);

    // 2. Start Python IPC harness.
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;
    eprintln!("python harness: socket={}", python.socket_path.display());

    // 3. Start the worker-sidecar.
    let pool = "smoke";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);

    // Wait until the worker metrics endpoint responds; this proves the binary
    // started the NATS, IPC, and metrics stack.
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");

    // Let the worker create the consumer before publishing. Without this the
    // publish could arrive before the stream exists.
    sleep(Duration::from_millis(500)).await;

    // Connect our own NATS client for publish/subscribe.
    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");

    let reply_subject = format!("_INBOX.smoke.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    // 6. Publish a synthetic encode WorkItem on the pool subject.
    let model_id = "BAAI/bge-m3";
    let subject = pool_work_subject(pool, pool, bundle, model_id);
    let request_id = "smoke-req-1";

    // Stamp `timestamp` at `now - 0.25s` so the worker sees a realistic
    // queue latency (Python computes `(time.time() - wi.timestamp) * 1000`).
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "encode".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: String::new(),
        machine_profile: pool.into(),
        item: Some(text_item("hello rust worker")),
        payload_ref: None,
        output_types: Some(vec!["dense".into()]),
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: None,
        output_schema: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "smoke-gw".into(),
        reply_subject: reply_subject.clone(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s - 0.25,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode WorkItem");

    // JetStream publish so the stream actually captures the message.
    let js = async_nats::jetstream::new(client.clone());
    let (stream, sequence) = publish_jetstream_with_retry(&js, &subject, payload).await;
    eprintln!("published WorkItem, stream={} seq={}", stream, sequence);

    // 7. Wait for the reply.
    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for WorkResult")
        .expect("reply stream closed");

    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    eprintln!("got WorkResult: {result:?}");

    assert!(
        result.success,
        "expected success, got error={:?}",
        result.error
    );
    assert_eq!(result.request_id, request_id);
    assert_eq!(result.work_item_id, format!("{request_id}.0"));
    assert_eq!(result.item_index, 0);
    assert!(!result.result_msgpack.is_empty(), "result_msgpack is empty");

    // Decode the canned payload from the stub executor.
    let canned: serde_json::Value =
        rmp_serde::from_slice(&result.result_msgpack).expect("decode canned");
    assert_eq!(canned["smoke"], "ok");
    assert_eq!(canned["source"], "ipc_test_harness");

    // worker_id must be set by publisher::build_work_result.
    assert_eq!(result.worker_id.as_deref(), Some("smoke-worker"));

    // Parity #1: Rust-computed timing fields must be populated on the
    // success/ack path.
    //
    // - `queue_ms` comes from `(now - WorkItem.timestamp) * 1000`; we
    //   stamp `timestamp = now() - 0.25s` so we expect ~250ms + scheduling.
    // - `processing_ms` is set to 0.0 to match Python's placeholder.
    // - `payload_fetch_ms` is omitted because the WorkItem has an inline
    //   `item` (no payload_ref resolution).
    let queue_ms = result.queue_ms.expect("queue_ms should be populated");
    assert!(
        (100.0..5_000.0).contains(&queue_ms),
        "queue_ms out of expected range: {queue_ms}",
    );
    assert_eq!(
        result.processing_ms,
        Some(0.0),
        "processing_ms should match Python placeholder",
    );
    assert!(
        result.payload_fetch_ms.is_none(),
        "payload_fetch_ms should be None for inline items, got {:?}",
        result.payload_fetch_ms,
    );

    // Parity #2: the worker must expose Prometheus metrics that mirror
    // Python's `sie_pull_loop_*` surface, so existing dashboards keep
    // working. We just confirm the names show up after a successful
    // encode cycle — exact counts live in the lib-side unit tests.
    let body = scrape_metrics(metrics_port).expect("scrape /metrics");
    for expected in [
        "sie_pull_loop_items_fetched",
        "sie_pull_loop_batch_process_seconds",
        "sie_worker_messages_received_total",
        "sie_worker_messages_acked_total",
    ] {
        assert!(
            body.contains(expected),
            "/metrics body missing expected metric {expected}:\n{body}"
        );
    }

    // Drop everything in reverse order (workers/python/nats) via Drop impls.
    drop(_worker);
    drop(python);
    drop(nats);
    // Keep sock alive just a tick so the drain has a moment.
    let _ = sock;

    // Brief courtesy wait so kill-on-drop children get reaped before the
    // test harness returns — avoids noisy zombie warnings.
    sleep(Duration::from_millis(200)).await;

    // Silence unused warning on Arc in case the compiler complains.
    let _ = Arc::new(());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_encode_direct_dispatch_round_trips_through_worker_stream() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    let nats = NatsHarness::start().await;
    eprintln!("nats: {} (port {})", nats.url, nats.port);

    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;
    eprintln!("python harness: socket={}", python.socket_path.display());

    let pool = "smoke-direct";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());

    let reply_subject = format!("_INBOX.smoke-direct.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let model_id = "BAAI/bge-m3";
    let subject = worker_work_subject(pool, pool, bundle, model_id, "smoke-worker");
    publish_work_item(
        &js,
        &subject,
        "smoke-direct-encode-1",
        model_id,
        "BAAI__bge-m3",
        pool,
        &reply_subject,
    )
    .await;

    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for direct-dispatch WorkResult")
        .expect("reply stream closed");
    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    assert!(
        result.success,
        "expected direct-dispatch success, got error={:?}",
        result.error
    );
    assert_eq!(result.request_id, "smoke-direct-encode-1");
    assert_eq!(result.worker_id.as_deref(), Some("smoke-worker"));

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}

/// End-to-end smoke test for generation direct-dispatch through the
/// sidecar. The Python harness exposes one fake generation model and
/// answers `ProcessGenerate` with the same event sequence a real
/// StreamingProcessor would use: progress ACK, raw publish, final ACK.
///
/// This is intentionally below the full Tilt/SGLang layer. It proves
/// the sidecar-only contract that endpoint tests cannot isolate:
/// worker-specific stream creation, direct generation subject routing,
/// streaming IPC frames, raw NATS publish, and JetStream settlement.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_generate_direct_dispatch_round_trips_through_rust_worker() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    let nats = NatsHarness::start().await;
    eprintln!("nats: {} (port {})", nats.url, nats.port);

    let model_id = "Qwen/Qwen3-0.6B";
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start_with_fake_generate(sock.path.clone(), model_id).await;
    eprintln!("python harness: socket={}", python.socket_path.display());

    let pool = "smoke-gen";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let reply_subject = format!("_INBOX.smoke-gen.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let request_id = "smoke-generate-1";
    let subject = worker_work_subject(pool, pool, bundle, model_id, "smoke-worker");
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "generate".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: String::new(),
        machine_profile: pool.into(),
        item: None,
        payload_ref: None,
        output_types: None,
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: None,
        output_schema: None,
        generate: Some(serde_json::json!({
            "prompt": "hello",
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
        })),
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "smoke-gw".into(),
        reply_subject: reply_subject.clone(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s - 0.25,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode generate WorkItem");
    let js = async_nats::jetstream::new(client.clone());
    let (stream, sequence) = publish_jetstream_with_retry(&js, &subject, payload).await;
    eprintln!(
        "published generate WorkItem, stream={} seq={}",
        stream, sequence
    );

    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for raw generate reply")
        .expect("reply stream closed");
    let body: serde_json::Value =
        rmp_serde::from_slice(&reply.payload).expect("decode raw generate payload");
    eprintln!("got raw generate payload: {body}");

    assert_eq!(body["smoke"], "generate");
    assert_eq!(body["source"], "ipc_test_harness");
    assert_eq!(body["model_id"], model_id);
    assert_eq!(body["request_id"], request_id);
    assert_eq!(body["work_item_id"], format!("{request_id}.0"));

    let mut acked = 0.0;
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        let metrics = scrape_metrics(metrics_port).expect("scrape /metrics");
        acked = scrape_scalar(&metrics, "sie_worker_messages_acked_total").unwrap_or(0.0);
        if acked >= 1.0 {
            break;
        }
        sleep(Duration::from_millis(100)).await;
    }
    assert!(
        acked >= 1.0,
        "generate ProcessGenerate ACK event did not settle the JetStream message",
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}

/// Worker direct-dispatch is not gated by WorkerCapabilities. The harness hides
/// generation capability from the first probe, but the worker-specific stream
/// must still be available for direct generation work immediately.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_generation_direct_dispatch_is_active_before_capability_reconcile() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    let nats = NatsHarness::start().await;
    eprintln!("nats: {} (port {})", nats.url, nats.port);

    let model_id = "Qwen/Qwen3-0.6B";
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start_with_fake_generate_after_capability_polls(
        sock.path.clone(),
        model_id,
        1,
    )
    .await;
    eprintln!("python harness: socket={}", python.socket_path.display());

    let pool = "smoke-gen-hot-add";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker =
        WorkerHarness::spawn_with_env(&nats.url, &sock.path, pool, bundle, metrics_port, None, &[]);
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let reply_subject = format!("_INBOX.smoke-gen-hot-add.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let request_id = "smoke-generate-hot-add-1";
    let subject = worker_work_subject(pool, pool, bundle, model_id, "smoke-worker");
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "generate".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: String::new(),
        machine_profile: pool.into(),
        item: None,
        payload_ref: None,
        output_types: None,
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: None,
        output_schema: None,
        generate: Some(serde_json::json!({
            "prompt": "hello after config",
            "max_new_tokens": 4,
            "temperature": 0.0,
            "top_p": 1.0,
        })),
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "smoke-gw".into(),
        reply_subject: reply_subject.clone(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode generate WorkItem");
    let js = async_nats::jetstream::new(client.clone());
    let (stream, sequence) = publish_jetstream_with_retry(&js, &subject, payload).await;
    eprintln!(
        "published hot-add generate WorkItem, stream={} seq={}",
        stream, sequence
    );

    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for raw generate reply")
        .expect("reply stream closed");
    let body: serde_json::Value =
        rmp_serde::from_slice(&reply.payload).expect("decode raw generate payload");
    assert_eq!(body["smoke"], "generate");
    assert_eq!(body["model_id"], model_id);
    assert_eq!(body["request_id"], request_id);

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}

/// End-to-end smoke test for the payload-store offload path.
///
/// The gateway normally offloads large item payloads to a
/// `LocalPayloadStore` (which writes to a shared volume) and stamps the
/// resulting absolute path on the WorkItem as `payload_ref` with `item =
/// None`. The worker is expected to pull the bytes back, decode them
/// from msgpack, and only then dispatch the batch to Python.
///
/// What this proves on top of the inline-item test:
///   * The worker-sidecar honors `SIE_PAYLOAD_STORE_URL`.
///   * `LocalPayloadStore::get` resolves the absolute path and returns
///     the bytes the gateway wrote.
///   * The dispatcher feeds the decoded item into `ProcessEncodeBatch`
///     and gets a successful outcome back.
///   * `payload_fetch_ms` is populated (> 0) on the success path.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_payload_ref_request_round_trips_through_rust_worker() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    // 1. NATS.
    let nats = NatsHarness::start().await;
    eprintln!("nats: {} (port {})", nats.url, nats.port);

    // 2. Python IPC harness.
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;
    eprintln!("python harness: socket={}", python.socket_path.display());

    // 3. A shared "payload store" directory — stand-in for the gateway's
    //    LocalPayloadStore base_dir. We use /tmp for short paths.
    let payload_dir = tempfile::Builder::new()
        .prefix("siews-payloads-")
        .tempdir_in("/tmp")
        .expect("create payload dir");
    let payload_store_url = payload_dir
        .path()
        .to_str()
        .expect("payload dir utf-8")
        .to_string();

    // Mimic the gateway's offload protocol: msgpack-encode the item and
    // persist it as `{request_id}_{item_index}.bin`. The `payload_ref`
    // is the absolute path, which is what `LocalPayloadStore::put`
    // returns in production.
    let request_id = "smoke-payload-ref-1";
    let item_key = format!("{request_id}_0.bin");
    let item_path = payload_dir.path().join(&item_key);
    let item_body = serde_json::json!({"text": "hello from payload store"});
    let item_bytes = rmp_serde::to_vec_named(&item_body).expect("msgpack encode item");
    tokio::fs::write(&item_path, &item_bytes)
        .await
        .expect("write payload");
    let payload_ref = item_path.to_str().expect("payload path utf-8").to_string();
    eprintln!("wrote payload: {payload_ref} ({} bytes)", item_bytes.len());

    // 4. worker-sidecar wired to the payload store.
    let pool = "smoke";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(
        &nats.url,
        &sock.path,
        pool,
        bundle,
        metrics_port,
        Some(&payload_store_url),
    );
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    // 5. Publish a WorkItem with item=None and payload_ref set.
    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let reply_subject = format!("_INBOX.smoke.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let model_id = "BAAI/bge-m3";
    let subject = pool_work_subject(pool, pool, bundle, model_id);
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "encode".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: String::new(),
        machine_profile: pool.into(),
        item: None,
        payload_ref: Some(payload_ref.clone()),
        output_types: Some(vec!["dense".into()]),
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: None,
        output_schema: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "smoke-gw".into(),
        reply_subject: reply_subject.clone(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode WorkItem");

    let js = async_nats::jetstream::new(client.clone());
    let (stream, sequence) = publish_jetstream_with_retry(&js, &subject, payload).await;
    eprintln!(
        "published payload-ref WorkItem, stream={} seq={}",
        stream, sequence
    );

    // 6. Reply must arrive, decode, and reflect a successful publish_and_ack.
    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for WorkResult")
        .expect("reply stream closed");
    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    eprintln!("got WorkResult: {result:?}");

    assert!(
        result.success,
        "expected success, got error={:?}",
        result.error
    );
    assert_eq!(result.request_id, request_id);
    assert_eq!(result.work_item_id, format!("{request_id}.0"));
    assert_eq!(result.item_index, 0);

    // The canned stub payload must still round-trip.
    let canned: serde_json::Value =
        rmp_serde::from_slice(&result.result_msgpack).expect("decode canned");
    assert_eq!(canned["smoke"], "ok");
    assert_eq!(canned["source"], "ipc_test_harness");

    // Parity: payload_fetch_ms is populated for the offloaded path.
    // The read is from local tmpfs so the duration is tiny — just
    // assert it exists and is non-negative. A hard upper bound would
    // flake under CI load.
    let fetch_ms = result
        .payload_fetch_ms
        .expect("payload_fetch_ms should be populated when item is offloaded");
    assert!(
        (0.0..5_000.0).contains(&fetch_ms),
        "payload_fetch_ms out of expected range: {fetch_ms}"
    );

    // Drop in reverse startup order.
    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    let _ = payload_dir;

    sleep(Duration::from_millis(200)).await;
}

/// Regression: document bytes must remain msgpack `bin` from the offloaded
/// WorkItem payload through the Rust dispatcher and into the Python IPC item.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_extract_payload_ref_preserves_document_bytes_through_ipc() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    let nats = NatsHarness::start().await;
    eprintln!("nats: {} (port {})", nats.url, nats.port);

    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;
    eprintln!("python harness: socket={}", python.socket_path.display());

    let payload_dir = tempfile::Builder::new()
        .prefix("siews-doc-payloads-")
        .tempdir_in("/tmp")
        .expect("create payload dir");
    let payload_store_url = payload_dir
        .path()
        .to_str()
        .expect("payload dir utf-8")
        .to_string();

    let request_id = "smoke-doc-payload-ref-1";
    let pdf_bytes = b"%PDF-1.4 tiny sidecar regression".to_vec();
    let item_path = payload_dir.path().join(format!("{request_id}_0.bin"));
    let item_body = document_item(pdf_bytes.clone(), "pdf");
    let item_bytes = rmp_serde::to_vec_named(&item_body).expect("msgpack encode document item");
    tokio::fs::write(&item_path, &item_bytes)
        .await
        .expect("write document payload");
    let payload_ref = item_path.to_str().expect("payload path utf-8").to_string();
    eprintln!(
        "wrote document payload: {payload_ref} ({} bytes)",
        item_bytes.len()
    );

    let pool = "smoke-doc";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(
        &nats.url,
        &sock.path,
        pool,
        bundle,
        metrics_port,
        Some(&payload_store_url),
    );
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let reply_subject = format!("_INBOX.smoke-doc.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let model_id = "docling";
    let subject = pool_work_subject(pool, pool, bundle, model_id);
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "extract".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: String::new(),
        machine_profile: pool.into(),
        item: None,
        payload_ref: Some(payload_ref.clone()),
        output_types: None,
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: Some(vec!["document".into()]),
        output_schema: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "smoke-gw".into(),
        reply_subject: reply_subject.clone(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode extract WorkItem");
    let js = async_nats::jetstream::new(client.clone());
    let (stream, sequence) = publish_jetstream_with_retry(&js, &subject, payload).await;
    eprintln!(
        "published document payload-ref WorkItem, stream={} seq={}",
        stream, sequence
    );

    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for extract WorkResult")
        .expect("reply stream closed");
    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    assert!(
        result.success,
        "expected success, got error={:?}",
        result.error
    );
    assert_eq!(result.request_id, request_id);
    assert_eq!(result.work_item_id, format!("{request_id}.0"));

    let echo: rmpv::Value =
        rmp_serde::from_slice(&result.result_msgpack).expect("decode extract echo");
    assert_eq!(msg_map_get(&echo, "smoke").and_then(msg_as_str), Some("ok"));
    assert_eq!(
        msg_map_get(&echo, "source").and_then(msg_as_str),
        Some("ipc_test_harness")
    );
    let document = msg_map_get(&echo, "extract_document").expect("extract_document echo");
    assert_eq!(
        msg_map_get(document, "present").and_then(msg_as_bool),
        Some(true)
    );
    assert_eq!(
        msg_map_get(document, "data_is_bytes").and_then(msg_as_bool),
        Some(true),
        "Python IPC harness did not receive document.data as bytes; echo={echo:?}",
    );
    assert_eq!(
        msg_map_get(document, "data_len").and_then(msg_as_u64),
        Some(pdf_bytes.len() as u64)
    );
    assert_eq!(
        msg_map_get(document, "format").and_then(msg_as_str),
        Some("pdf")
    );
    assert_eq!(
        msg_map_get(document, "data"),
        Some(&rmpv::Value::Binary(pdf_bytes)),
        "document.data changed before reaching Python IPC; echo={echo:?}",
    );

    let fetch_ms = result
        .payload_fetch_ms
        .expect("payload_fetch_ms should be populated when item is offloaded");
    assert!(
        (0.0..5_000.0).contains(&fetch_ms),
        "payload_fetch_ms out of expected range: {fetch_ms}"
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    let _ = payload_dir;
    sleep(Duration::from_millis(200)).await;
}

// ---------------------------------------------------------------------------
// Concurrent / stress scenarios for the pull loop
//
// These exist because we regressed from "N concurrent requests all succeed
// in ~tens of ms" to "1 succeeds in ~30 ms, N-1 hit 30 s gateway timeout" by
// recreating the async-nats pull stream every iteration with a 1-20 ms
// expiry. Under concurrency any message the server had routed to the
// dropped stream became `ack_pending` until `ack_wait=30s` elapsed.
//
// The floor the tests defend is:
//
//   * a **burst** of N concurrent publishes resolves end-to-end in a wall
//     clock comfortably below `ack_wait` (30 s), with **no redelivery
//     count > 1** on any message (i.e. we never hit the stream-drop path);
//   * the `sie_worker_nats_redelivery_total` counter stays at zero.
//
// Both assertions were false before the long-lived-stream fix and are
// true after it. This is effectively the cheapest stress test that would
// have caught the production regression without the whole helm stack.
// ---------------------------------------------------------------------------

/// Scrape `sie_worker_nats_redelivery_total` from the worker's /metrics
/// body. The counter is present unconditionally (emitted even at 0) by
/// [`MetricsRegistry::new`], so a missing line is a test failure rather
/// than "not enough load yet".
fn redelivery_total(metrics_body: &str) -> u64 {
    for line in metrics_body.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("sie_worker_nats_redelivery_total") {
            let rest = rest.trim_start();
            let val = rest.split_whitespace().next().unwrap_or("0");
            return val.parse::<f64>().unwrap_or(0.0) as u64;
        }
    }
    panic!("sie_worker_nats_redelivery_total missing from /metrics body:\n{metrics_body}");
}

async fn publish_jetstream_with_retry(
    js: &async_nats::jetstream::Context,
    subject: &str,
    payload: Vec<u8>,
) -> (String, u64) {
    let mut last_error = String::new();
    for attempt in 0..50 {
        let ack = js
            .publish(subject.to_string(), payload.clone().into())
            .await
            .expect("jetstream publish");
        match ack.await {
            Ok(ack) => return (ack.stream, ack.sequence),
            Err(e) => {
                last_error = format!("{e:?}");
                if !last_error.contains("StreamNotFound") {
                    panic!("await publish ack: {last_error}");
                }
                eprintln!(
                    "publish ack stream not ready yet (attempt={attempt}, subject={subject})"
                );
                sleep(Duration::from_millis(100)).await;
            }
        }
    }
    panic!("stream was never ready for {subject}: {last_error}");
}

async fn publish_work_item(
    js: &async_nats::jetstream::Context,
    subject: &str,
    request_id: &str,
    model_id: &str,
    normalized: &str,
    pool: &str,
    reply_subject: &str,
) {
    publish_work_item_with_admission_pool(
        js,
        subject,
        request_id,
        model_id,
        normalized,
        pool,
        "",
        reply_subject,
    )
    .await;
}

#[allow(clippy::too_many_arguments)]
async fn publish_work_item_with_admission_pool(
    js: &async_nats::jetstream::Context,
    subject: &str,
    request_id: &str,
    model_id: &str,
    normalized: &str,
    pool: &str,
    admission_pool: &str,
    reply_subject: &str,
) {
    let _ = normalized; // subject is pre-built by caller
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "encode".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: admission_pool.into(),
        machine_profile: pool.into(),
        item: Some(text_item(format!("concurrent-{request_id}"))),
        payload_ref: None,
        output_types: Some(vec!["dense".into()]),
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: None,
        output_schema: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "stress-gw".into(),
        reply_subject: reply_subject.into(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode WorkItem");
    let _ = publish_jetstream_with_retry(js, subject, payload).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pool_admission_pauses_until_gateway_assigns_worker() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    let pool = "gated";
    let worker_id = "smoke-worker";
    let gateway = GatewayPoolHarness::start(pool, worker_id, pool).await;

    let nats = NatsHarness::start().await;
    let sock = ShortSocket::new("ipc.sock");
    let _python = PythonHarness::start(sock.path.clone()).await;

    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn_with_env(
        &nats.url,
        &sock.path,
        pool,
        bundle,
        metrics_port,
        None,
        &[
            ("SIE_GATEWAY_URL", gateway.url.as_str()),
            ("SIE_POOL_ADMISSION_CHECK_INTERVAL_S", "1"),
            ("SIE_POOL_ADMISSION_PAUSE_S", "0.1"),
            ("SIE_POOL_ADMISSION_STALE_AFTER_S", "0"),
            ("SIE_NAK_DELAY_S", "0.1"),
        ],
    );
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());

    let reply_subject = format!("_INBOX.gated.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let model_id = "BAAI/bge-m3";
    let normalized = "BAAI__bge-m3";
    let subject = pool_work_subject(pool, pool, bundle, model_id);
    publish_work_item(
        &js,
        &subject,
        "gated-req-1",
        model_id,
        normalized,
        pool,
        &reply_subject,
    )
    .await;

    match timeout(Duration::from_millis(700), sub.next()).await {
        Err(_) => {}
        Ok(Some(msg)) => panic!(
            "worker pulled while not assigned; payload bytes={}",
            msg.payload.len()
        ),
        Ok(None) => panic!("reply subscription closed while worker was not assigned"),
    }

    gateway.set_assigned(true);
    let reply = timeout(Duration::from_secs(5), sub.next())
        .await
        .expect("timed out waiting for WorkResult after assignment")
        .expect("reply stream closed after assignment");
    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    assert!(
        result.success,
        "expected success after assignment, got error={:?}",
        result.error
    );
    assert_eq!(result.request_id, "gated-req-1");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn logical_pool_admission_naks_until_assigned_worker_serves_item() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    let physical_pool = "default";
    let logical_pool = "tenant-a";
    let worker_id = "smoke-worker";
    let gateway = GatewayPoolHarness::start_with_queue_pool(
        logical_pool,
        physical_pool,
        worker_id,
        physical_pool,
    )
    .await;

    let nats = NatsHarness::start().await;
    let sock = ShortSocket::new("ipc.sock");
    let _python = PythonHarness::start(sock.path.clone()).await;

    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn_with_env(
        &nats.url,
        &sock.path,
        physical_pool,
        bundle,
        metrics_port,
        None,
        &[
            ("SIE_GATEWAY_URL", gateway.url.as_str()),
            ("SIE_POOL_ADMISSION_CHECK_INTERVAL_S", "1"),
            ("SIE_POOL_ADMISSION_PAUSE_S", "0.1"),
            ("SIE_POOL_ADMISSION_STALE_AFTER_S", "0"),
            ("SIE_NAK_DELAY_S", "0.1"),
        ],
    );
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());
    let reply_subject = format!("_INBOX.logical.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let model_id = "BAAI/bge-m3";
    let normalized = "BAAI__bge-m3";
    let subject = pool_work_subject(physical_pool, physical_pool, bundle, model_id);
    publish_work_item_with_admission_pool(
        &js,
        &subject,
        "logical-req-1",
        model_id,
        normalized,
        physical_pool,
        logical_pool,
        &reply_subject,
    )
    .await;

    match timeout(Duration::from_millis(700), sub.next()).await {
        Err(_) => {}
        Ok(Some(msg)) => panic!(
            "worker served logical pool before assignment; payload bytes={}",
            msg.payload.len()
        ),
        Ok(None) => panic!("reply subscription closed before logical assignment"),
    }

    gateway.set_assigned(true);
    let reply = timeout(Duration::from_secs(5), sub.next())
        .await
        .expect("timed out waiting for WorkResult after logical assignment")
        .expect("reply stream closed after logical assignment");
    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    assert!(
        result.success,
        "expected success after logical assignment, got error={:?}",
        result.error
    );
    assert_eq!(result.request_id, "logical-req-1");
}

/// Fire N concurrent publishes on distinct reply subjects and assert that
/// **every one** gets a reply within a budget well short of `ack_wait`.
///
/// Regression guard: the pre-fix code reliably failed this at N >= 2 with
/// 1 reply arriving fast and the rest timing out after 30 s. Anything
/// near that pattern re-appearing should trip this test.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stress_concurrent_requests_do_not_stall_on_ack_wait() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    const CONCURRENCY: usize = 16;
    // Budget must be comfortably below `ack_wait=30s`. The pre-fix code
    // took ~30.002 s to service the N-1 stragglers; 10 s gives us a 20 s
    // margin while being forgiving of CI noise.
    const REPLY_BUDGET: Duration = Duration::from_secs(10);

    let nats = NatsHarness::start().await;
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;

    let pool = "stress";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());

    let model_id = "BAAI/bge-m3";
    let normalized = "BAAI__bge-m3";
    let subject = pool_work_subject(pool, pool, bundle, model_id);

    // One subscription per request on a unique inbox; collect all N in a
    // concurrent `JoinSet` so the publishes are temporally clustered.
    let mut join = tokio::task::JoinSet::new();
    let start = Instant::now();
    for i in 0..CONCURRENCY {
        let reply_subject = format!("_INBOX.stress.{}", uuid::Uuid::new_v4());
        let mut sub = client
            .subscribe(reply_subject.clone())
            .await
            .expect("subscribe reply");
        let subject = subject.clone();
        let model_id = model_id.to_string();
        let normalized = normalized.to_string();
        let pool = pool.to_string();
        let js = js.clone();
        join.spawn(async move {
            let request_id = format!("stress-req-{i}");
            publish_work_item(
                &js,
                &subject,
                &request_id,
                &model_id,
                &normalized,
                &pool,
                &reply_subject,
            )
            .await;
            let msg = timeout(REPLY_BUDGET, sub.next())
                .await
                .unwrap_or_else(|_| {
                    panic!(
                        "request {request_id} did not receive a WorkResult within {:?} \
                         (pre-fix regression would stall here at ~30s)",
                        REPLY_BUDGET,
                    )
                })
                .expect("reply stream closed");
            let result: WorkResult =
                rmp_serde::from_slice(&msg.payload).expect("decode WorkResult");
            assert!(
                result.success,
                "request {} got success=false: {:?}",
                request_id, result.error,
            );
            assert_eq!(result.request_id, request_id);
            result
        });
    }

    let mut seen = 0usize;
    while let Some(res) = join.join_next().await {
        res.expect("stress task panicked");
        seen += 1;
    }
    assert_eq!(seen, CONCURRENCY, "missing replies");

    let elapsed = start.elapsed();
    // Global wall-clock guard. Even with 1 ms per request serially this
    // should finish fast; 10 s is the ack_wait safety rail.
    assert!(
        elapsed < Duration::from_secs(15),
        "concurrent burst took {:?} (pre-fix would be ~30s)",
        elapsed,
    );

    // Metrics check: redelivery_total must stay at 0 because we never let
    // messages sit past ack_wait.
    let metrics = scrape_metrics(metrics_port).expect("scrape /metrics");
    let redelivered = redelivery_total(&metrics);
    assert_eq!(
        redelivered, 0,
        "sie_worker_nats_redelivery_total = {redelivered}, expected 0 \
         (a non-zero value means the pull loop is letting messages sit \
         past ack_wait, which is the regression we're guarding against)"
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}

/// Sustained-load variant: many publishes spread over a few seconds (so
/// the adaptive quantum controller actually warms up). Guards against a
/// regression where the quantum grows without bound and steady-state
/// throughput collapses.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stress_sustained_load_finishes_within_budget() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    const TOTAL: usize = 200;
    const INTER_PUBLISH_MS: u64 = 10; // 100 req/s offered load
    const BUDGET: Duration = Duration::from_secs(15);

    let nats = NatsHarness::start().await;
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;

    let pool = "stress-sustained";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());

    let model_id = "BAAI/bge-m3";
    let normalized = "BAAI__bge-m3";
    let subject = pool_work_subject(pool, pool, bundle, model_id);

    let mut join = tokio::task::JoinSet::new();
    let start = Instant::now();
    for i in 0..TOTAL {
        let reply_subject = format!("_INBOX.stress-sus.{}", uuid::Uuid::new_v4());
        let mut sub = client
            .subscribe(reply_subject.clone())
            .await
            .expect("subscribe reply");
        let subject = subject.clone();
        let model_id = model_id.to_string();
        let normalized = normalized.to_string();
        let pool = pool.to_string();
        let js = js.clone();
        join.spawn(async move {
            let request_id = format!("sustained-{i}");
            publish_work_item(
                &js,
                &subject,
                &request_id,
                &model_id,
                &normalized,
                &pool,
                &reply_subject,
            )
            .await;
            let msg = timeout(Duration::from_secs(10), sub.next())
                .await
                .unwrap_or_else(|_| panic!("sustained request {request_id} timed out"))
                .expect("reply stream closed");
            let result: WorkResult =
                rmp_serde::from_slice(&msg.payload).expect("decode WorkResult");
            assert!(
                result.success,
                "request {request_id} failed: {:?}",
                result.error
            );
        });
        sleep(Duration::from_millis(INTER_PUBLISH_MS)).await;
    }

    let mut seen = 0usize;
    while let Some(res) = join.join_next().await {
        res.expect("sustained task panicked");
        seen += 1;
    }
    assert_eq!(seen, TOTAL, "missing replies under sustained load");

    let elapsed = start.elapsed();
    assert!(
        elapsed < BUDGET,
        "sustained load of {TOTAL} req took {:?}, budget {:?}",
        elapsed,
        BUDGET,
    );

    let metrics = scrape_metrics(metrics_port).expect("scrape /metrics");
    let redelivered = redelivery_total(&metrics);
    // Under sustained load with a healthy worker we expect zero
    // redeliveries; allow 1% headroom for scheduler noise.
    assert!(
        redelivered <= (TOTAL / 100) as u64,
        "sie_worker_nats_redelivery_total = {redelivered}, over 1% of offered load",
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}

/// Scrape a single-series Prometheus gauge / counter value by metric name.
/// Returns `None` if the line isn't present (e.g. label-variant unseen
/// yet), in which case callers typically treat it as "zero observed".
fn scrape_scalar(metrics_body: &str, name: &str) -> Option<f64> {
    for line in metrics_body.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.is_empty() {
            continue;
        }
        let rest = match line.strip_prefix(name) {
            Some(r) => r,
            None => continue,
        };
        // Word-boundary guard — `foo` must not match `foo_extended`.
        match rest.chars().next() {
            Some(c) if c.is_whitespace() || c == '{' => {}
            _ => continue,
        }
        let val = rest.split_whitespace().next_back().unwrap_or("nan");
        return val.parse::<f64>().ok();
    }
    None
}

/// Scrape the total sample count of a Prometheus Histogram. Used to
/// confirm that the pool-acquire path was actually exercised.
fn histogram_count(metrics_body: &str, name: &str) -> u64 {
    let needle = format!("{name}_count");
    scrape_scalar(metrics_body, &needle).unwrap_or(0.0) as u64
}

/// Pool integration test. Drives a burst of requests across **multiple
/// distinct model IDs** with Python artificially slowed to 150ms/RPC,
/// and asserts the IPC pool is what enables concurrency between model
/// groups.
///
/// Design notes:
/// - The Rust dispatcher intentionally **batches requests targeting
///   the same model into a single IPC call** (that's the whole point
///   of the adaptive batcher). A naive "fire 8 requests to the same
///   model" test would therefore observe `inflight=1` even with a
///   4-slot pool, because there's only one ProcessEncodeBatch RPC to
///   issue for that group. The test used to fail for exactly that
///   reason.
/// - To actually exercise the pool we need ≥ 2 **distinct model
///   groups** running concurrently through the dispatcher. With N
///   models, the dispatcher's `ensure_model_ready` + `process_encode_batch`
///   for each group become independent RPCs that the pool should run
///   in parallel.
/// - Python-side delay (via `--per-request-delay-ms`) stretches each
///   RPC long enough for Prometheus scrapes to catch a non-zero
///   `sie_worker_ipc_pool_inflight` gauge mid-burst.
///
/// Assertions:
/// 1. `sie_worker_ipc_pool_size` matches the configured pool size.
/// 2. All requests succeed.
/// 3. The peak observed `sie_worker_ipc_pool_inflight` during the
///    burst is `>= 2` — the regression guard for the single-mutex era.
/// 4. End-to-end latency is under a budget that only holds if the
///    model groups ran in parallel (vs. serialized).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pool_enables_concurrent_in_flight_ipc() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    const POOL_SIZE: usize = 4;
    // Four distinct models → four independent dispatcher model groups
    // → at least four parallel IPC calls if the pool works.
    const MODELS: &[(&str, &str)] = &[
        ("BAAI/bge-m3", "BAAI__bge-m3"),
        ("BAAI/bge-small-en", "BAAI__bge-small-en"),
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers__all-MiniLM-L6-v2",
        ),
        ("intfloat/e5-base-v2", "intfloat__e5-base-v2"),
    ];
    const PYTHON_DELAY_MS: u64 = 150;
    // Serialized floor: MODELS.len() groups × (ensure_model_ready +
    // process_encode_batch) × PYTHON_DELAY_MS ≈ 4 × 2 × 150ms = 1.2s
    // minimum if everything ran on one socket.
    //  Parallel floor with 4-slot pool: ~2 × 150ms (ensure then process)
    // for each group running in parallel + dispatcher/NATS overhead,
    // comfortably under 1.5s in practice.
    const BUDGET: Duration = Duration::from_millis(1500);

    let nats = NatsHarness::start().await;
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start_with_delay_ms(sock.path.clone(), PYTHON_DELAY_MS).await;

    let pool = "pool-integ";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let pool_size_str = POOL_SIZE.to_string();
    let _worker = WorkerHarness::spawn_with_env(
        &nats.url,
        &sock.path,
        pool,
        bundle,
        metrics_port,
        None,
        &[
            ("SIE_IPC_POOL_SIZE", pool_size_str.as_str()),
            ("SIE_MAX_CONCURRENT_BATCHES", pool_size_str.as_str()),
        ],
    );
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let pre = scrape_metrics(metrics_port).expect("initial scrape");
    let observed_size = scrape_scalar(&pre, "sie_worker_ipc_pool_size")
        .expect("sie_worker_ipc_pool_size gauge missing");
    assert_eq!(
        observed_size as usize, POOL_SIZE,
        "configured SIE_IPC_POOL_SIZE did not propagate to the pool metric"
    );

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());

    // Poller: scrape the /metrics endpoint as fast as we reasonably
    // can (10ms cadence) while the burst is in flight, tracking the
    // peak inflight value seen.
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering as AtomicOrdering};
    let peak_inflight = Arc::new(AtomicI64::new(0));
    let stop_poller = Arc::new(AtomicBool::new(false));
    let peak_clone = Arc::clone(&peak_inflight);
    let stop_clone = Arc::clone(&stop_poller);
    let poller = tokio::spawn(async move {
        while !stop_clone.load(AtomicOrdering::Relaxed) {
            if let Ok(body) = scrape_metrics(metrics_port) {
                if let Some(v) = scrape_scalar(&body, "sie_worker_ipc_pool_inflight") {
                    let v = v as i64;
                    let mut cur = peak_clone.load(AtomicOrdering::Relaxed);
                    while v > cur
                        && peak_clone
                            .compare_exchange(
                                cur,
                                v,
                                AtomicOrdering::Relaxed,
                                AtomicOrdering::Relaxed,
                            )
                            .is_err()
                    {
                        cur = peak_clone.load(AtomicOrdering::Relaxed);
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    });

    // Fire 2 requests per model = 8 total requests across 4 groups.
    let mut join = tokio::task::JoinSet::new();
    let start = Instant::now();
    for (idx, (model_id, normalized)) in MODELS.iter().enumerate() {
        for rep in 0..2 {
            let reply_subject = format!("_INBOX.pool.{}", uuid::Uuid::new_v4());
            let mut sub = client
                .subscribe(reply_subject.clone())
                .await
                .expect("subscribe reply");
            let subject = pool_work_subject(pool, pool, bundle, model_id);
            let model_id = (*model_id).to_string();
            let normalized = (*normalized).to_string();
            let pool = pool.to_string();
            let js = js.clone();
            let request_id = format!("pool-req-{idx}-{rep}");
            join.spawn(async move {
                publish_work_item(
                    &js,
                    &subject,
                    &request_id,
                    &model_id,
                    &normalized,
                    &pool,
                    &reply_subject,
                )
                .await;
                let msg = timeout(Duration::from_secs(10), sub.next())
                    .await
                    .unwrap_or_else(|_| panic!("pool request {request_id} timed out"))
                    .expect("reply stream closed");
                let result: WorkResult =
                    rmp_serde::from_slice(&msg.payload).expect("decode WorkResult");
                assert!(
                    result.success,
                    "request {request_id} failed: {:?}",
                    result.error
                );
            });
        }
    }
    let total = MODELS.len() * 2;
    let mut seen = 0usize;
    while let Some(res) = join.join_next().await {
        res.expect("pool task panicked");
        seen += 1;
    }
    let elapsed = start.elapsed();
    assert_eq!(seen, total);

    stop_poller.store(true, AtomicOrdering::Relaxed);
    let _ = poller.await;

    // --- Assertion 1: peak inflight observed > 1.
    let peak = peak_inflight.load(AtomicOrdering::Relaxed);
    assert!(
        peak >= 2,
        "peak sie_worker_ipc_pool_inflight = {peak}; expected >= 2 with \
         {} distinct model groups in flight. inflight=1 throughout means \
         the pool regressed to single-connection serialization.",
        MODELS.len(),
    );

    // --- Assertion 2: latency budget — only meetable with parallel IPC.
    assert!(
        elapsed < BUDGET,
        "concurrent burst across {} model groups took {:?}; budget {:?}. \
         Serialized (pool=1) would take ≥ {} × 2 × {}ms = {:?} in Python \
         delay alone. A budget miss here means the IPC pool regressed.",
        MODELS.len(),
        elapsed,
        BUDGET,
        MODELS.len(),
        PYTHON_DELAY_MS,
        Duration::from_millis((MODELS.len() as u64) * 2 * PYTHON_DELAY_MS),
    );

    // --- Assertion 3: the acquire histogram recorded samples.
    let post = scrape_metrics(metrics_port).expect("post scrape");
    let acquire_count = histogram_count(&post, "sie_worker_ipc_pool_acquire_wait_seconds");
    assert!(
        acquire_count >= MODELS.len() as u64,
        "sie_worker_ipc_pool_acquire_wait_seconds_count = {acquire_count}; \
         expected >= {} (one per model group's process_encode_batch RPC)",
        MODELS.len(),
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}

// ---------------------------------------------------------------------------
// Rust-side prepared_tokens round-trip
// ---------------------------------------------------------------------------

/// End-to-end smoke test for the Rust-side `prepared_tokens` hand-off.
///
/// What this proves on top of the baseline encode test:
///   * The Python `EnsureModelReady` descriptor is honored by the worker:
///     the named model's `tokenizer.json` is loaded and registered.
///   * For a plain-text encode request with no instruction/template and
///     `is_query = false`, the dispatcher pre-tokenizes the text in Rust
///     and attaches a `PreparedTokens` payload to the outgoing
///     `ProcessEncodeBatch` IPC call.
///   * The Python IPC stub (`_ipc_test_harness`) receives
///     `prepared_tokens` on the `EncodeBatchItem`, and echoes back the
///     tokenizer id, the first `input_ids` sequence, and the configured
///     `max_seq_len` so we can assert on it here.
///
/// What it does **not** prove:
///   * That the Python fast-path actually accepts the hand-off and skips
///     its own tokenizer — that's covered in Python unit tests. The
///     harness runs no real adapter, so we just verify the Rust side
///     populated the wire field end-to-end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_prepared_tokens_round_trip_through_rust_worker() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    // 1. Materialise a tiny WordLevel tokenizer on disk. The vocab is
    //    chosen so "hello world" tokenizes deterministically to [1, 2].
    const TINY_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": { "type": "Whitespace" },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": { "[UNK]": 0, "hello": 1, "world": 2 },
    "unk_token": "[UNK]"
  }
}"#;
    let tok_dir = tempfile::Builder::new()
        .prefix("siews-tok-")
        .tempdir_in("/tmp")
        .expect("tokenizer tempdir");
    let tok_path = tok_dir.path().join("tokenizer.json");
    std::fs::write(&tok_path, TINY_TOKENIZER_JSON).expect("write tokenizer.json");

    // 2. NATS + Python IPC harness.
    let nats = NatsHarness::start().await;
    eprintln!("nats: {} (port {})", nats.url, nats.port);

    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start_with_descriptor(sock.path.clone(), &tok_path, 64).await;
    eprintln!("python harness: socket={}", python.socket_path.display());

    // 3. Start the worker-sidecar. The Python harness returns a
    //    ModelDescriptor pointing at our tiny tokenizer. Pick a custom
    //    max_seq_len (64) so we can assert it round-trips through
    //    PreparedTokens.max_seq_len.
    let model_id = "tiny-prepared-tokens";
    let pool = "smoke-pt";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();

    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);

    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let reply_subject = format!("_INBOX.smoke-pt.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    // 4. Publish a plain-text encode WorkItem that satisfies the
    //    Rust-tokenisation safety rules (text-only, no instruction,
    //    is_query = false) so the dispatcher actually pre-tokenizes.
    let subject = pool_work_subject(pool, pool, bundle, model_id);
    let request_id = "smoke-pt-req-1";
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "encode".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: String::new(),
        machine_profile: pool.into(),
        item: Some(text_item("hello world")),
        payload_ref: None,
        output_types: Some(vec!["dense".into()]),
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: None,
        output_schema: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "smoke-gw".into(),
        reply_subject: reply_subject.clone(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s - 0.25,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode WorkItem");
    let js = async_nats::jetstream::new(client.clone());
    let _ = publish_jetstream_with_retry(&js, &subject, payload).await;

    // 5. Collect the reply and decode the canned echo from the harness.
    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for WorkResult")
        .expect("reply stream closed");
    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    assert!(
        result.success,
        "expected success, got error={:?}",
        result.error
    );
    assert_eq!(result.request_id, request_id);
    assert!(!result.result_msgpack.is_empty(), "result_msgpack is empty");

    let canned: serde_json::Value =
        rmp_serde::from_slice(&result.result_msgpack).expect("decode canned");
    eprintln!("canned echo: {canned}");

    // 6. Assertions on the echoed prepared_tokens block — this is the
    //    evidence that the Rust dispatcher actually populated
    //    `prepared_tokens` on the IPC request and that the Python stub
    //    deserialised it back into an `EncodeBatchItem`.
    let pt = canned
        .get("prepared_tokens")
        .expect("canned payload missing `prepared_tokens`");
    assert_eq!(
        pt.get("present").and_then(|v| v.as_bool()),
        Some(true),
        "Rust dispatcher did not attach prepared_tokens for a plain-text encode. \
         Check ModelDescriptor ingestion or the safety gates in \
         Dispatcher::maybe_prepare_encode_tokens. Full echo: {canned}",
    );

    let tokenizer_id = pt
        .get("tokenizer_id")
        .and_then(|v| v.as_str())
        .expect("tokenizer_id should be a string when prepared_tokens is present");
    assert_eq!(
        tokenizer_id.len(),
        32,
        "tokenizer_id should be a 32-char blake3 prefix, got {tokenizer_id:?}",
    );
    assert!(
        tokenizer_id.chars().all(|c| c.is_ascii_hexdigit()),
        "tokenizer_id should be hex-only, got {tokenizer_id:?}",
    );

    assert_eq!(
        pt.get("max_seq_len").and_then(|v| v.as_u64()),
        Some(64),
        "max_seq_len did not round-trip from ModelDescriptor. echo: {canned}",
    );

    // "hello world" against the tiny vocab = [1, 2]. The WordLevel model
    // has no special tokens / post-processor, so this is exact.
    let ids = pt
        .get("input_ids_first_seq")
        .and_then(|v| v.as_array())
        .expect("input_ids_first_seq should be a list");
    let ids: Vec<u64> = ids
        .iter()
        .map(|v| v.as_u64().expect("input id is a non-negative int"))
        .collect();
    assert_eq!(
        ids,
        vec![1u64, 2u64],
        "tokenized ids for `hello world` should be [1, 2] against the tiny \
         vocab (hello=1, world=2). Got {ids:?} — tokenizer loading or \
         tokenize_no_pad semantics regressed.",
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    let _ = tok_dir;
    sleep(Duration::from_millis(200)).await;
}

// ---------------------------------------------------------------------------
// Rust scheduler smoke tests.
//
// Since the scheduler is always-on now (no `SIE_RUST_SCHEDULER_MODELS`
// gate), every request routes through batch formation → `RunBatch` IPC →
// the Python stub's `process_*_batch` handler. The tests below cover the
// surface the original encode smoke test doesn't exercise:
//
//   * the score path (which forces `LoraKey::base` regardless of
//     `options["lora"]`),
//   * the extract path (ExtractBatchItem shape is distinct),
//   * concurrent-item coalescing into a single RunBatch (the whole
//     reason the scheduler exists).
//
// We also scrape `/metrics` to assert the dispatcher's
// `sie_worker_scheduler_enqueued_items_total{op=..}` counter actually
// increments — that's the direct evidence the request went through the
// scheduler path, not the legacy `process_*_batch` fallback.
// ---------------------------------------------------------------------------

/// Scrape a Prometheus counter value by metric name + exact label suffix.
/// Returns 0.0 when the series isn't present (useful for before/after
/// deltas where the first scrape may show the counter uninitialised).
fn scrape_counter(body: &str, metric: &str, label_suffix: &str) -> f64 {
    for line in body.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let Some(rest) = line.strip_prefix(metric) else {
            continue;
        };
        if !rest.starts_with(label_suffix) {
            continue;
        }
        // After `metric{..labels..}` there's a space then the value.
        let after_labels = match rest.find('}') {
            Some(i) => &rest[i + 1..],
            None => rest,
        };
        let val = after_labels.split_whitespace().next().unwrap_or("0");
        return val.parse::<f64>().unwrap_or(0.0);
    }
    0.0
}

/// Publish a score WorkItem that carries an inline query + one score
/// item (no payload_ref indirection). Mirrors `publish_work_item` but
/// for the score shape.
async fn publish_score_work_item(
    js: &async_nats::jetstream::Context,
    subject: &str,
    request_id: &str,
    model_id: &str,
    pool: &str,
    reply_subject: &str,
) {
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "score".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: String::new(),
        machine_profile: pool.into(),
        item: None,
        payload_ref: None,
        output_types: None,
        instruction: None,
        is_query: false,
        options: None,
        query_item: Some(text_item("what is the capital of France?")),
        query_payload_ref: None,
        score_items: Some(vec![text_item("Paris is the capital.")]),
        labels: None,
        output_schema: None,
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "scheduler-smoke-gw".into(),
        reply_subject: reply_subject.into(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode score WorkItem");
    let _ = publish_jetstream_with_retry(js, subject, payload).await;
}

/// Publish an extract WorkItem. Uses `output_schema` + `labels` so the
/// ExtractBatchItem shape exercises more fields than the minimal path.
async fn publish_extract_work_item(
    js: &async_nats::jetstream::Context,
    subject: &str,
    request_id: &str,
    model_id: &str,
    pool: &str,
    reply_subject: &str,
) {
    let now_s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();
    let work_item = WorkItem {
        work_item_id: format!("{request_id}.0"),
        request_id: request_id.into(),
        item_index: 0,
        total_items: 1,
        operation: "extract".into(),
        model_id: model_id.into(),
        profile_id: String::new(),
        engine: String::new(),
        pool_name: pool.into(),
        admission_pool: String::new(),
        machine_profile: pool.into(),
        item: Some(text_item("Barack Obama was born in Hawaii.")),
        payload_ref: None,
        output_types: None,
        instruction: None,
        is_query: false,
        options: None,
        query_item: None,
        query_payload_ref: None,
        score_items: None,
        labels: Some(vec!["PERSON".into(), "LOCATION".into()]),
        output_schema: Some(serde_json::json!({"type": "entities"})),
        generate: None,
        routing_key: None,
        prompt_cache_key: None,
        bundle_config_hash: String::new(),
        router_id: "scheduler-smoke-gw".into(),
        reply_subject: reply_subject.into(),
        traceparent: None,
        tracestate: None,
        timestamp: now_s,
    };
    let payload = rmp_serde::to_vec_named(&work_item).expect("encode extract WorkItem");
    let _ = publish_jetstream_with_retry(js, subject, payload).await;
}

/// Score: publish a score WorkItem, expect a reply, and assert the
/// `sie_worker_scheduler_enqueued_items_total{op="score"}` counter
/// stepped from 0 → 1. That's direct evidence the request went
/// through `Dispatcher::enqueue_score_into_scheduler` (not the legacy
/// `handle_score` path) and that the drain loop shipped the batch
/// over `RunBatch`.
///
/// Regression guard: if anyone rewires the dispatcher so score bypasses
/// the scheduler (or silently falls back to `process_score_batch`), the
/// scheduler metric stays at 0 and this test fails loudly.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_scheduler_routes_score_request_end_to_end() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    let nats = NatsHarness::start().await;
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;

    let pool = "sched-score";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());

    let model_id = "BAAI/bge-m3";
    let subject = pool_work_subject(pool, pool, bundle, model_id);

    let reply_subject = format!("_INBOX.sched-score.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let request_id = "sched-score-1";
    publish_score_work_item(&js, &subject, request_id, model_id, pool, &reply_subject).await;

    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for score WorkResult")
        .expect("reply stream closed");
    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    assert!(
        result.success,
        "score via scheduler returned error: {:?}",
        result.error
    );
    assert_eq!(result.request_id, request_id);

    // The scheduler counter is labelled `{model, op}`. The model label
    // is the dispatcher's `model_label(model_id)` which for an
    // unconfigured model collapses to the raw id.
    let body = scrape_metrics(metrics_port).expect("scrape /metrics");
    let enqueued = scrape_counter(
        &body,
        "sie_worker_scheduler_enqueued_items_total",
        &format!("{{model=\"{model_id}\",operation=\"score\"}}"),
    );
    assert!(
        enqueued >= 1.0,
        "sie_worker_scheduler_enqueued_items_total for score must be >= 1 \
         (actual: {enqueued}). A zero means score bypassed the Rust \
         scheduler — check Dispatcher::enqueue_score_into_scheduler."
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}

/// Extract: same shape as the score test but for the extract path.
/// Covers the third SchedulerItem variant and its ExtractBatchItem
/// wire shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_scheduler_routes_extract_request_end_to_end() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    let nats = NatsHarness::start().await;
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;

    let pool = "sched-extract";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());

    let model_id = "BAAI/bge-m3";
    let subject = pool_work_subject(pool, pool, bundle, model_id);

    let reply_subject = format!("_INBOX.sched-extract.{}", uuid::Uuid::new_v4());
    let mut sub = client
        .subscribe(reply_subject.clone())
        .await
        .expect("subscribe reply");

    let request_id = "sched-extract-1";
    publish_extract_work_item(&js, &subject, request_id, model_id, pool, &reply_subject).await;

    let reply = timeout(Duration::from_secs(30), sub.next())
        .await
        .expect("timed out waiting for extract WorkResult")
        .expect("reply stream closed");
    let result: WorkResult = rmp_serde::from_slice(&reply.payload).expect("decode WorkResult");
    assert!(
        result.success,
        "extract via scheduler returned error: {:?}",
        result.error
    );
    assert_eq!(result.request_id, request_id);

    let body = scrape_metrics(metrics_port).expect("scrape /metrics");
    let enqueued = scrape_counter(
        &body,
        "sie_worker_scheduler_enqueued_items_total",
        &format!("{{model=\"{model_id}\",operation=\"extract\"}}"),
    );
    assert!(
        enqueued >= 1.0,
        "sie_worker_scheduler_enqueued_items_total for extract must be >= 1 \
         (actual: {enqueued}). A zero means extract bypassed the Rust \
         scheduler — check Dispatcher::enqueue_extract_into_scheduler."
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}

/// Concurrent encodes: publish N items fast enough that several land in
/// the scheduler's 15 ms coalesce window, then verify (a) every reply
/// comes back successfully and (b) the scheduler counter records all
/// N items. We can't cleanly assert "these were in a single RunBatch"
/// from the outside without instrumenting Python, but the enqueue
/// counter at least proves the items went through the scheduler and
/// didn't fall through to the legacy per-op path.
///
/// Also scrapes `sie_pull_loop_batch_process_seconds{op="encode"}_count`
/// — under the scheduler each drain-loop tick contributes one
/// observation, so this stays > 0 after the test just like on the
/// legacy path. Dashboards keyed off that histogram continue working.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn smoke_scheduler_coalesces_concurrent_encodes() {
    if skip_unless_tools_available() {
        return;
    }
    let _guard = smoke_test_guard().await;

    const CONCURRENCY: usize = 8;
    const REPLY_BUDGET: Duration = Duration::from_secs(15);

    let nats = NatsHarness::start().await;
    let sock = ShortSocket::new("ipc.sock");
    let python = PythonHarness::start(sock.path.clone()).await;

    let pool = "sched-burst";
    let bundle = "default";
    let metrics_port = find_free_tcp_port();
    let _worker = WorkerHarness::spawn(&nats.url, &sock.path, pool, bundle, metrics_port, None);
    wait_for_tcp(metrics_port, Duration::from_secs(30))
        .await
        .expect("worker metrics port");
    sleep(Duration::from_millis(500)).await;

    let client = async_nats::connect(&nats.url)
        .await
        .expect("client connect");
    let js = async_nats::jetstream::new(client.clone());

    let model_id = "BAAI/bge-m3";
    let normalized = "BAAI__bge-m3";
    let subject = pool_work_subject(pool, pool, bundle, model_id);

    let mut join = tokio::task::JoinSet::new();
    for i in 0..CONCURRENCY {
        let reply_subject = format!("_INBOX.sched-burst.{}", uuid::Uuid::new_v4());
        let mut sub = client
            .subscribe(reply_subject.clone())
            .await
            .expect("subscribe reply");
        let subject = subject.clone();
        let model_id = model_id.to_string();
        let normalized = normalized.to_string();
        let pool = pool.to_string();
        let js = js.clone();
        join.spawn(async move {
            let request_id = format!("sched-burst-{i}");
            publish_work_item(
                &js,
                &subject,
                &request_id,
                &model_id,
                &normalized,
                &pool,
                &reply_subject,
            )
            .await;
            let msg = timeout(REPLY_BUDGET, sub.next())
                .await
                .unwrap_or_else(|_| panic!("sched-burst {request_id} timed out"))
                .expect("reply stream closed");
            let result: WorkResult =
                rmp_serde::from_slice(&msg.payload).expect("decode WorkResult");
            assert!(
                result.success,
                "sched-burst {request_id} got success=false: {:?}",
                result.error
            );
        });
    }

    let mut seen = 0usize;
    while let Some(res) = join.join_next().await {
        res.expect("sched-burst task panicked");
        seen += 1;
    }
    assert_eq!(seen, CONCURRENCY, "missing replies");

    // The enqueue counter must reflect every item we submitted. It's
    // incremented inside `enqueue_encode_into_scheduler` at enqueue
    // time (not after the drain completes), so this runs no race
    // against the drain loop finishing.
    let body = scrape_metrics(metrics_port).expect("scrape /metrics");
    let enqueued = scrape_counter(
        &body,
        "sie_worker_scheduler_enqueued_items_total",
        &format!("{{model=\"{model_id}\",operation=\"encode\"}}"),
    );
    assert!(
        enqueued >= CONCURRENCY as f64,
        "sie_worker_scheduler_enqueued_items_total for encode should be \
         >= {CONCURRENCY} (actual: {enqueued}). Missing samples mean \
         items bypassed the Rust scheduler path.",
    );

    // `batch_items` is a histogram; the `_count` series tracks the
    // number of observations. Each flushed batch is one observation,
    // so at minimum we need a single sample (possibly more if the
    // items landed across multiple 15 ms coalesce windows). The
    // corresponding `_sum` must then be ≥ CONCURRENCY since every
    // item contributes one to the size sum.
    //
    // Label order matters: the `prometheus` crate emits labels in
    // alphabetical order (lora < model < operation), not in the
    // registration order. Dashboards that hard-code label order
    // against the registration tuple break; our scrape assertions
    // need to match the on-the-wire order exactly.
    let batch_items_count = scrape_counter(
        &body,
        "sie_worker_scheduler_batch_items_count",
        &format!("{{lora=\"base\",model=\"{model_id}\",operation=\"encode\"}}"),
    );
    assert!(
        batch_items_count >= 1.0,
        "sie_worker_scheduler_batch_items_count must be >= 1 \
         (actual: {batch_items_count}). A zero means the drain \
         loop never ran or the histogram isn't being observed.",
    );
    let batch_items_sum = scrape_counter(
        &body,
        "sie_worker_scheduler_batch_items_sum",
        &format!("{{lora=\"base\",model=\"{model_id}\",operation=\"encode\"}}"),
    );
    assert!(
        batch_items_sum >= CONCURRENCY as f64,
        "sie_worker_scheduler_batch_items_sum should be >= {CONCURRENCY} \
         (actual: {batch_items_sum}). Every submitted item contributes \
         1 to the histogram sum; smaller means items were dropped.",
    );

    // Live models_total gauge: after touching exactly one model, it
    // should have incremented to at least 1. We don't assert
    // equality because other scheduler tests running in the same
    // harness do not share this worker (each test spawns its
    // own), but we defensively use `>= 1` so the gauge isn't racy
    // against any parallel test reuse.
    let models_total = scrape_counter(&body, "sie_worker_scheduler_models_total", "");
    assert!(
        models_total >= 1.0,
        "sie_worker_scheduler_models_total must be >= 1 (actual: {models_total}). \
         The gauge should tick up on first-traffic scheduler creation.",
    );

    drop(_worker);
    drop(python);
    drop(nats);
    let _ = sock;
    sleep(Duration::from_millis(200)).await;
}
