# SIE Server Sidecar Architecture

This is the canonical reference for `packages/sie_server_sidecar`.

## Status

The package is production-oriented but still deployment-gated by Helm:

- Source package: `packages/sie_server_sidecar`
- Cargo package: `sie-server-sidecar`
- Binary: `sie-server-sidecar`
- Kubernetes container: `worker-sidecar`
- Container image: `ghcr.io/superlinked/sie-server-sidecar`
- Prometheus metric prefix: `sie_worker_`

The Kubernetes container is named `worker-sidecar`, while sidecar-only build
artifacts use the `sie-server-sidecar` family to match the colocated
`sie-server` adapter process. Prometheus metric families use the
`sie_worker_*` prefix because they describe worker-side runtime behavior.

GPU-independent preparation code lives in this package:

- `src/protocol/` owns the Rust IPC schema.
- `src/prep/` owns GPU-independent prep/framing helpers.
- `src/ipc_types.rs` remains as a re-export so sidecar code can keep using
  `crate::ipc_types`.

`tools/ci/check_ipc_types_parity.py` checks the Rust protocol schema against
`packages/sie_server/src/sie_server/ipc_types.py`.

## Runtime Split

A queue-mode worker pod has two cooperating containers:

```text
gateway -> NATS JetStream -> worker-sidecar -> UDS IPC -> sie-server adapter
                                       |
                                       +-> NATS Core reply subject
```

Responsibilities:

| Component | Owns |
| --- | --- |
| `sie_gateway` | HTTP/API edge, model and pool resolution, JetStream publish, inbox collection, DLQ listener |
| `worker-sidecar` | JetStream pull consumer, subject validation, payload fetch, batch formation, adaptive scheduling, Rust tokenization fast path, result framing, ACK/NAK, reply publish, sidecar metrics/readiness |
| `sie-server` | Python adapter execution, model registry and GPU lifecycle, adapter-specific preprocessing/fallbacks, IPC server, direct HTTP API |

The sidecar never loads model weights and has no GPU dependency. It talks to
the colocated adapter over a Unix domain socket.

## Request Path

1. Gateway publishes a msgpack `WorkItem` on `sie.work.{pool}.{machine_profile}.{bundle}.{model}`.
2. The sidecar's long-lived JetStream pull stream receives messages from
   stream `WORK_POOL_{pool}`.
3. The dispatcher extracts the model from the subject and rejects malformed
   subjects before trusting payload data.
4. The dispatcher validates `reply_subject`; non-empty values must start with
   `_INBOX.`.
5. The sidecar groups by model, calls `EnsureModelReady`, and resolves payload
   references if present.
6. The Rust scheduler batches by `(operation, lora_key)` and sends `RunBatch`
   over UDS IPC when the adapter declares `supports_run_batch`.
7. Python returns `BatchOutcome` with per-item `ItemOutcome`s.
8. The sidecar frames/publishes `WorkResult` and ACKs the JetStream message.
   Publish failure skips ACK so JetStream redelivers.

Failure rules:

- Bad msgpack, malformed subject, transient backend failures, and draining
  generally NAK with a delay.
- Unsafe reply subjects are ACK-dropped to avoid publish amplification.
- Unknown operations publish an error outcome and ACK when the reply publish
  succeeds.
- IPC transport errors retry once transparently before the group is NAKed.
- Gateway owns DLQ publication by listening to JetStream max-delivery
  advisories; the sidecar only ACKs or NAKs.

## NATS Contract

Worker stream and consumer settings must stay aligned with the gateway and SDK:

| Field | Value |
| --- | --- |
| Stream | `WORK_POOL_{pool}` |
| Stream subjects | `sie.work.{pool}.*.*.*` |
| Consumer filter | `sie.work.{pool}.{machine_profile}.{bundle}.*` |
| Retention | WorkQueue |
| Storage | Memory |
| Consumer | `{pool}_{machine_profile}_{bundle}` |
| Ack policy | Explicit |
| Ack wait | 30 s |
| Max deliver | `SIE_MAX_DELIVER`, default 20 |
| Max ack pending | `SIE_MAX_ACK_PENDING`, default 1000 |

On existing streams, the gateway and sidecar reconcile the subject list to
exactly `sie.work.{pool}.*.*.*`. Legacy subjects such as `sie.work.*.{pool}` are
intentionally removed from stream configuration. This release is a cutover to
the lane-aware subject shape, not a mixed-version bridge: all gateways and
workers in the cluster must use the new shape, and any old queued work should be
drained or purged before rollout.

The pull stream must be long-lived. Dropping and recreating pull streams inside
the fetch loop can make async-nats mark unread messages as delivered, causing
redelivery only after the 30 s `ack_wait`. The implementation creates one pull
stream and polls it continuously.
Integration tests guard this with concurrent and sustained-load scenarios that
watch `sie_worker_nats_redelivery_total`.

The sidecar also runs a slow stream/durable reconciler. It periodically
re-ensures `WORK_POOL_{pool}` and `{pool}_{machine_profile}_{bundle}` so NATS
metadata drift can self-heal without adding reconnect-event work or locks to
the fetch loop. It does not rebuild the active pull stream; terminal
pull-stream errors still use the normal pull-loop recovery path.

Generation models use a worker-specific JetStream subject
`sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}`. The sidecar
creates and polls that worker stream only after Python `WorkerCapabilities`
reports at least one generation model. If live config reconciliation later adds
a generation model to an already-running sidecar, a capability reconciler
activates the direct stream and cancel subscriber exactly once.

When `SIE_GATEWAY_URL` is set, the sidecar runs the pool admission gate before
pulling from either the pool stream or the generation worker stream. Capped
named pools fail closed unless `/v1/pools/{pool}` lists this pod in
`status.assigned_workers`; uncapped pools and profile-uncapped workers continue
pulling. The default pool fails open during transient gateway/status errors so
baseline capacity remains available.

## IPC Contract

Framing is:

```text
[4-byte big-endian length][msgpack named-map body]
```

The method surface is:

- `Ping`
- `EnsureModelReady`
- `ProcessEncodeBatch`
- `ProcessScoreBatch`
- `ProcessExtractBatch`
- `RunBatch`
- `Drain`

`RunBatch` is the hot scheduler path. It carries a homogeneous-op batch; mixed
ops are rejected by Python and pinned by parity fixtures under `tests/parity/`.
Envelope identity fields on `RunBatchItem` preserve `work_item_id`,
`request_id`, and `item_index` even when the operation-specific payload is
missing.

`ModelDescriptor` is returned by `EnsureModelReady` and lets the sidecar learn
per-model capabilities at runtime:

- tokenizer path and tokenizer content hash
- maximum sequence length
- supported output types
- `supports_run_batch`

## Scheduler And Batching

The Rust scheduler is live for queue-mode traffic that reaches this sidecar.
There is no sidecar-local active model list that gates the path; routing is a
gateway/pool decision.

Preserved Python semantics:

- cost, count, max-wait, and coalesce-window flush triggers
- oversize item flushes alone
- cost-sorted sub-batch packing
- per-model adaptive controller
- score runs on the base LoRA path
- per-item outcomes fan out without dropping the rest of the batch

Production-tuned sidecar defaults:

| Knob | Default | Notes |
| --- | ---: | --- |
| `SIE_RUST_PIPELINE_DEPTH` | 2 | one active adapter batch plus one queued behind it |
| `SIE_BATCHER_COALESCE_MS` | 5 | intentionally lower than Python reference |
| `SIE_BATCHER_MAX_BATCH_REQUESTS` | 12 | intentionally lower than Python reference |
| `SIE_ADAPTIVE_MIN_QUANTUM_MS` | 2 | pull-loop coalesce floor |
| `SIE_ADAPTIVE_MAX_QUANTUM_MS` | 15 | pull-loop coalesce ceiling |
| `SIE_ADAPTIVE_TARGET_P50_MS` | 50 | pull-loop latency target |

These worker-local defaults cap coalescing and queue depth for CPU-side
responsiveness; they are locked by Rust tests and should be retuned together.

`SIE_PULL_QUANTUM_INCLUDE_QUEUE_MS=1` opts into controller input that includes
queue wait. Default is off because saturation queue time can collapse the pull
quantum even when the pull loop is not the bottleneck.

## Prep And Result Framing

The sidecar owns GPU-independent work:

- text template application for the safe Rust-tokenization path
- HF fast-tokenizer loading from the adapter-declared tokenizer path
- `PreparedTokens` attachment for encode IPC items where safe
- dense, sparse, multivector, and score `RawOutput` framing
- `msgpack_numpy` sentinel-compatible payload bytes

Python remains the correctness fallback. If tokens are absent or the
`tokenizer_id` mismatches, Python retokenizes. If a result cannot be represented
by a typed `RawOutput`, Python-framed `result_msgpack` passes through.

Current worker-side boundaries:

- Encode tokenization fast path exists, but individual text adapters still need
  byte-identity validation before consuming it everywhere.
- Score pair construction stays adapter-side because query/document pair policy
  is model-specific.
- Extract tokenization stays Python-side.
- Native Candle/tch-rs adapter work remains future work behind the same IPC boundary.

## Readiness And Metrics

The sidecar serves HTTP on `SIE_WORKER_METRICS_PORT`, default `9095`:

- `/healthz`: always returns 200 while the process is alive.
- `/readyz`: returns 200 only after a successful IPC `Ping`, while the most
  recent ping is fresh, and before drain starts.
- `/metrics`: Prometheus metrics.

In Helm, the `worker-sidecar` container exposes container port `metrics`, the
worker Service exposes it as service port `worker-metrics`, and the
ServiceMonitor scrapes that endpoint. Metric families remain `sie_worker_*`
and `sie_pull_loop_*`; do not rename them with the Kubernetes container.

Freshness is `SIE_WORKER_PING_INTERVAL_MS * SIE_WORKER_READYZ_STALE_MULT`.
Defaults are 2000 ms and 3.

Helm can route probes to the sidecar with
`workers.common.workerSidecar.probes.enabled`. This is separate from enabling the
sidecar container itself.

## Important Environment Variables

| Variable | Purpose |
| --- | --- |
| `SIE_NATS_URL` | NATS endpoint; required |
| `SIE_POOL` | pool segment in `sie.work.{pool}.{machine_profile}.{bundle}.{model}` |
| `SIE_MACHINE_PROFILE` | machine-profile segment; required and set explicitly by Helm |
| `SIE_BUNDLE` | bundle/runtime segment and consumer-name segment |
| `SIE_IPC_SOCKET_PATH` | UDS path to the adapter |
| `SIE_IPC_POOL_SIZE` | concurrent IPC connections, default matches dispatch concurrency |
| `SIE_NATS_FETCH_BUDGET` | pull-stream credit and fallback model budget |
| `SIE_NATS_PULL_EXPIRES_S` | server-side pull expiry, default 5 s and must stay below ack wait |
| `SIE_NATS_CONSUMER_RECONCILE_INTERVAL_MS` | slow stream/durable reconcile interval, default 30000; non-zero values below 10000 are clamped; `0` disables |
| `SIE_GENERATION_CAPABILITY_RECONCILE_INTERVAL_MS` | generation direct-dispatch activation check interval after startup reports no generation models, default 30000; non-zero values below 5000 are clamped; `0` disables |
| `SIE_STREAM_MAX_AGE_S` | worker stream max age if worker creates the stream first |
| `SIE_PAYLOAD_STORE_URL` | optional shared payload-store backend (`s3://`, `gs://`, `abfs://`, `abfss://`, or a local path shared with the gateway) |
| `SIE_GATEWAY_URL` | optional gateway base URL used by the pool admission gate |
| `SIE_GATEWAY_API_KEY` | optional bearer token for gateway pool-status reads |
| `SIE_POOL_ADMISSION_ENABLED` | pool admission gate toggle, default true when `SIE_GATEWAY_URL` is set |
| `SIE_POOL_ADMISSION_CHECK_INTERVAL_S` | pool admission status check cadence, default 5 s |
| `SIE_POOL_ADMISSION_PAUSE_S` | sleep while this pod is not admitted to pull, default 1 s |
| `SIE_POOL_ADMISSION_STALE_AFTER_S` | cached-decision window after transient gateway/status errors, default 30 s |
| `SIE_WORKER_METRICS_PORT` | sidecar HTTP port |
| `SIE_HEALTH_PUBLISH_ENABLED` | NATS heartbeat publisher toggle |
| `SIE_HEALTH_PUBLISH_INTERVAL_MS` | NATS heartbeat publish interval |
| `SIE_BUNDLE_CONFIG_HASH` | optional initial local bundle hash; normally populated by live apply/export reconciliation |
| `SIE_CONFIG_SERVICE_URL` | optional `sie-config` base URL for worker-side epoch/export reconciliation |
| `SIE_ADMIN_TOKEN` | optional bearer token for `GET /v1/configs/export` when config auth is enabled |
| `SIE_WORKER_CONFIG_POLL_INTERVAL_MS` | worker config epoch poll interval, default 30000 |
| `SIE_WORKER_CONFIG_FULL_EXPORT_INTERVAL_MS` | slow full-export reconcile interval, default 300000; `0` disables after startup |
| `SIE_NATS_CONFIG_TRUSTED_PRODUCERS` | comma-separated producer allowlist for `sie.config.models.<bundle>`; binary default `sie-config`; Helm sets the release-scoped config Deployment name |
| `SIE_NATS_CONFIG_TRUST_ANY_PRODUCER` | disables producer validation for local/dev only |

Operator-provided Helm `extraEnv` is rendered after chart defaults so explicit
overrides still win.

## Live Config Apply

The sidecar subscribes to `sie.config.models.<SIE_BUNDLE>`. Each notification is
accepted only after producer allowlist, bundle, epoch, and size checks pass.
Accepted deltas are forwarded to Python over IPC `ApplyModelConfig`; Python
validates the YAML as `ModelConfig`, mutates its local `ModelRegistry`, clears
the model descriptor cache, and returns the bundle hash it computes locally.

When `SIE_CONFIG_SERVICE_URL` is set, the sidecar also runs an export
reconciler. It performs a startup `GET /v1/configs/export`, then polls
`GET /v1/configs/epoch`. If the control-plane epoch is ahead, or if the slow
full-export reconcile is due for an epoch-0/no-config-store deployment, it replays
only export entries whose `affected_bundles` include this worker's bundle. The
advertised hash advances only after every relevant export entry applies
successfully, so a partial replay does not produce a false worker ACK.

The NATS health publisher reads that hash dynamically, so the gateway's
`/v1/configs/models/{id}/status` endpoint observes worker convergence without
an explicit ACK message. The dispatcher also checks each incoming work item's
expected `bundle_config_hash` against the current local hash plus a small recent
hash history. Unknown hashes are NAKed before `EnsureModelReady` or Python IPC,
so workers behind the gateway redeliver work after their live NATS subscriber or
export reconciler catches up, while accepted in-flight work remains processable
after an append-only config advance.

Config apply also nudges the generation capability reconciler. This covers the
case where a bundle starts encode-only and later receives a generation model:
the sidecar keeps the encode-only hot path until Python exposes the generation
task, then creates the worker-specific generation stream.

## Known Caveats

- **Stream `max_age` race.** Gateway and worker can both create the stream.
  Whichever side creates it first wins. Keep gateway and worker settings aligned
  when changing retention.
- **Durable config drift.** The sidecar cleans up overlapping consumers, but a
  pure durable config change may still need operator cleanup if JetStream
  rejects an update.
- **Config replay scope.** Worker-side export reconciliation replays model
  configs, not bundle definitions or adapter code. Bundle changes still require
  a matching `sie-config`/worker image rollout; the running sidecar can only
  apply configs for adapters already present in the co-located Python image.
- **Subject shape coupling.** The worker extracts the model from subject token
  `parts[5]` in `sie.work.{pool}.{machine_profile}.{bundle}.{model}`. Gateway
  subject changes must update worker parsing in lock-step.
- **Unit integration scope.** Rust integration tests use real NATS and a stub
  Python IPC server; local Tilt E2E covers the real gateway-to-worker path.
- **Sidecar enablement changes deployment behavior.** Keeping
  `workers.common.workerSidecar.enabled` off leaves queue-mode worker pods without
  a queue consumer. Flipping that default changes Helm behavior and should be a
  deliberate deployment decision.

## Testing

Use the repo tasks:

```bash
mise run server-sidecar-check
mise run server-sidecar-fmt -- --check
mise run server-sidecar-clippy
mise run server-sidecar-test
mise exec -- python tools/ci/check_ipc_types_parity.py
mise run test -- packages/sie_server/tests/test_parity_run_batch.py
```

What is covered:

- Rust unit tests for scheduler, adaptive controller, IPC framing/retry,
  payload confinement, metrics, output framing, tokenizer registry, publisher,
  shutdown, and NATS helpers. `mise run server-sidecar-test` runs the crate once with
  default features and once with `cloud-storage` so S3/GCS/Azure payload-store
  code is compiled and tested in the normal worker path.
- Rust integration smoke tests with real `nats-server`, real `sie-server-sidecar`, and
  a Python IPC harness.
- Python parity tests for `RunBatch` fixtures under `tests/parity/`.
- IPC type parity between Rust and Python schema declarations.

## Architecture Decisions

- Use a sidecar around Python adapters, not a full native rewrite.
- Keep the adapter boundary over UDS msgpack IPC.
- Use a long-lived async-nats pull stream.
- Keep the runtime binary name stable as `sie-server-sidecar`; render the Kubernetes
  sidecar container as `worker-sidecar`; publish the container image as
  `sie-server-sidecar`.
- Keep queue/runtime/prep/framing work in Rust while adapters remain in Python.
- Treat native Candle/tch-rs backends as future work behind the same adapter
  boundary.
