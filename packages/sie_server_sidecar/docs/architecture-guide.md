# SIE Server Sidecar Architecture

`packages/sie_server_sidecar` contains the Rust queue runtime used by
sidecar-enabled SIE worker pods.

The sidecar runs next to a SIE backend in the same worker pod. It consumes NATS
JetStream work, talks to the backend over Unix-domain-socket IPC, and publishes
results back to NATS Core.

```text
gateway -> NATS JetStream -> worker-sidecar -> UDS IPC -> worker backend
                                      |
                                      +-> NATS Core reply subject
```

References:

- [Worker runtime](../../../product/design/worker-runtime.md)
- [Deployment topologies](../../../product/design/deployment-topologies.md)
- [Gateway architecture guide](../../sie_gateway/docs/architecture-guide.md)
- [Sidecar README](../README.md)

## Component Boundaries

`sie_gateway` accepts client inference requests, resolves the model, bundle,
machine profile, and pool, publishes work to JetStream, collects worker results,
and owns DLQ publication from JetStream max-delivery advisories.

`worker-sidecar` owns JetStream consumption, subject validation, payload fetch,
batch formation, adaptive scheduling, Rust tokenization when enabled by adapter
descriptors, result framing, ACK/NAK decisions, reply publication, HTTP probes,
Prometheus metrics, NATS health publication, pool admission, and bundle-scoped
config apply.

`sie-server` owns model registry behavior, lazy model loading, GPU residency,
adapter execution, adapter-specific preprocessing and postprocessing, generation
execution, and the IPC server.

The sidecar does not load model weights and does not link GPU libraries.

Public names:

- Rust package and binary: `sie-server-sidecar`
- Kubernetes container: `worker-sidecar`
- Container image: `ghcr.io/superlinked/sie-server-sidecar`
- Metrics prefix: `sie_worker_*`

## Request Flow

Encode, score, and extract queue traffic follows this path:

1. The gateway publishes a msgpack `WorkItem` to a lane-aware JetStream subject.
2. The sidecar consumes the message from the matching durable consumer.
3. The sidecar validates the NATS subject and reply subject.
4. The sidecar resolves payload references when the work item uses external
   payload storage.
5. The sidecar groups work by model.
6. The sidecar calls `EnsureModelReady` over IPC.
7. The scheduler batches compatible items by operation and LoRA key.
8. The sidecar sends scheduled batches over IPC with `RunBatch`.
9. The backend returns per-item outcomes.
10. The sidecar publishes a `WorkResult` to the reply subject.
11. The sidecar ACKs the JetStream message after successful reply publication.

Message settlement:

- Unsafe reply subjects are ACK-dropped.
- Malformed subjects and bad msgpack payloads are NAKed.
- Unknown operations publish per-item error outcomes when reply publication
  succeeds.
- Active local model loads are held with JetStream progress ACKs until
  `EnsureModelReady` returns ready; this preserves the delivery budget during
  cold starts. The progress delay is clamped below the pool consumer `ack_wait`
  even if the retry NAK delay is configured higher.
- Transient backend, decode, stream, and drain failures are NAKed.
- Reply publication failure leaves the JetStream message unacked.

Source: [`dispatcher.rs`](../src/dispatcher.rs),
[`publisher.rs`](../src/publisher.rs), and [`work_types.rs`](../src/work_types.rs).

## NATS Subjects

Pool work uses this subject shape:

```text
sie.work.{pool}.{machine_profile}.{bundle}.{model}
```

Generation direct dispatch uses this subject shape:

```text
sie.work.{pool}.{machine_profile}.{bundle}.{model}.{worker_id}
```

The pool stream subject is:

```text
sie.work.{pool}.*.*.*
```

The pool durable filter is:

```text
sie.work.{pool}.{machine_profile}.{bundle}.*
```

The sidecar extracts the model from the sixth subject token. Model IDs, bundle
IDs, machine profiles, pool names, and worker IDs are normalized to single NATS
subject tokens by the gateway and sidecar helpers.

The sidecar creates one long-lived async-nats pull stream for pool work. A slow
reconciler re-ensures stream and durable metadata outside the fetch loop.

The sidecar also creates and polls the worker-specific direct-dispatch stream.
Generation uses that stream for HRW placement; capped logical batch pools use it
when the gateway must target one assigned worker on a shared backing queue.

Source: [`nats_consumer.rs`](../src/nats_consumer.rs),
[`subject.rs`](../src/subject.rs), and the gateway
[queue publisher](../../sie_gateway/src/queue/publisher.rs).

## Pool Admission

When `SIE_GATEWAY_URL` is configured and pool admission is enabled, the sidecar
polls `GET /v1/pools` before pulling from the pool stream or the
worker-specific direct-dispatch stream. It may pull when its physical `SIE_POOL` is
admitted directly, or when at least one assigned logical pool is backed by that
same queue; each work item is then checked against its `admission_pool` before
backend IPC. The gateway normally direct-dispatches capped logical batch work to
an assigned worker, so unassigned peers do not spend the item's JetStream
delivery budget. Work items rejected by this per-item logical gate are still
NAKed and counted in `sie_worker_pool_admission_naks_total` as a defensive
backstop for stale assignments, direct NATS publishes, and rolling-version
mismatches.

Capped named pools require the worker to appear in
`status.assigned_workers`. The default pool continues pulling during transient
gateway/status errors.

Source: [`pool_admission.rs`](../src/pool_admission.rs).

## Generation

Generation work uses `ProcessGenerate` over IPC. The sidecar passes the
msgpack-encoded `WorkItem` to the backend and handles streaming IPC events.

The gateway streaming path selects either a pool subject or a worker-specific
subject. Worker-specific routing uses HRW over eligible workers. The gateway
stores a pool fallback subject for streaming work and uses it for republish
paths such as NAK handling and first-chunk timeout handling.

The sidecar subscribes to `cancel.>`. Subjects shaped as
`cancel.{router_id}.{request_id}` are forwarded to the backend through
`SignalGenerateCancel`.

Source: [`lib.rs`](../src/lib.rs), [`dispatcher.rs`](../src/dispatcher.rs),
[`ipc_client.rs`](../src/ipc_client.rs), and
[`packages/sie_gateway/src/handlers/proxy.rs`](../../sie_gateway/src/handlers/proxy.rs).

## IPC

IPC framing is:

```text
[4-byte big-endian length][msgpack named-map body]
```

The Rust and Python protocol copies define the same method names:

- `Ping`
- `EnsureModelReady`
- `ProcessEncodeBatch`
- `ProcessScoreBatch`
- `ProcessExtractBatch`
- `ProcessGenerate`
- `WorkerCapabilities`
- `SignalGenerateCancel`
- `RunBatch`
- `ApplyModelConfig`
- `ReplaceModelConfigs`
- `Drain`

`tools/ci/check_ipc_types_parity.py` checks the Rust protocol schema against
`packages/sie_server/src/sie_server/ipc_types.py`.

`EnsureModelReady` returns readiness state and an optional `ModelDescriptor`.
The descriptor carries tokenizer path, tokenizer content hash, maximum sequence
length, output types, default text templates, and `supports_run_batch`.
`loading_started` and `loading_in_progress` mean the same worker is actively
loading the model; the dispatcher progress-ACKs and rechecks instead of NAKing
those deliveries. `retry_later` remains a NAK path.

Source: [`protocol/ipc_types.rs`](../src/protocol/ipc_types.rs),
[`ipc_client.rs`](../src/ipc_client.rs), and
[`packages/sie_server/src/sie_server/ipc_types.py`](../../sie_server/src/sie_server/ipc_types.py).

## Scheduling

The sidecar scheduler handles queue-mode encode, score, and extract batches for
sidecar-enabled worker pools.

Batching keys include operation and LoRA key. Batch formation uses item cost,
request count, coalescing windows, adaptive pull timing, and model-specific
scheduler state. Oversize items flush alone. Per-item outcomes are published
without dropping the rest of the batch.

Routing remains gateway-owned. The sidecar does not keep a separate local
active-model routing list.

Source: [`scheduler/`](../src/scheduler/), [`latency.rs`](../src/latency.rs),
and [`dispatcher.rs`](../src/dispatcher.rs).

## Preparation And Framing

The sidecar performs GPU-independent preparation and framing:

- text template application for Rust tokenization;
- HF fast-tokenizer loading from adapter-declared paths;
- `PreparedTokens` attachment for encode IPC items;
- payload-store fetch and inline replacement for offloaded work items;
- dense, sparse, multivector, score, and generated-output framing;
- `msgpack_numpy` sentinel-compatible payload handling.

The backend retokenizes when prepared tokens are absent or the tokenizer hash
does not match. Score pair construction stays backend-owned. Extract
tokenization stays backend-owned.

Large generation and vision payloads are fetched from payload storage and
inlined before `ProcessGenerate`, so the backend receives a self-contained msgpack
work item.

Source: [`prep/`](../src/prep/), [`tokenize/`](../src/tokenize/),
[`output/`](../src/output/), [`payload_store.rs`](../src/payload_store.rs), and
[`dispatcher.rs`](../src/dispatcher.rs).

## Readiness, Health, And Metrics

The sidecar HTTP server exposes:

- `/healthz`: process liveness;
- `/readyz`: readiness based on successful IPC `Ping`, heartbeat freshness, and
  drain state;
- `/metrics`: Prometheus metrics.

Readiness state is shared with the NATS health publisher. The health publisher
emits worker identity, bundle, machine profile, readiness, and the current
bundle config hash. It also mirrors the latest `loaded_models` list reported by
the backend IPC heartbeat so gateway pool/model gauges reflect live residency in
NATS health mode.

Source: [`metrics.rs`](../src/metrics.rs),
[`readiness.rs`](../src/readiness.rs), and
[`health_publisher.rs`](../src/health_publisher.rs).

## Live Config

The sidecar subscribes to bundle-scoped config deltas:

```text
sie.config.models.{bundle}
```

Each notification is checked for trusted producer, bundle, epoch, and payload
size. Accepted deltas are forwarded to the colocated backend through
`ApplyModelConfig`. The backend returns the applied bundle config hash. The
sidecar stores that hash in `ConfigApplyState`.

When `SIE_CONFIG_SERVICE_URL` is configured, the export reconciler fetches
`/v1/configs/epoch` and `/v1/configs/export` from `sie-config`. Bundle-relevant
exports are sent to the backend through `ReplaceModelConfigs`.

Export reconciliation skips unchanged periodic exports. Exports older than the
local epoch are skipped unless the reconciler is handling an epoch-rewind
recovery. Partial or rejected replacements do not update the advertised bundle
hash.

Grouped encode, score, and extract work carries the gateway's expected bundle
config hash for the resolved pool and bundle. The dispatcher compares that hash
with the current local hash and the accepted recent-hash window before
`EnsureModelReady`. Unknown hashes are NAKed.

Live config apply updates model configuration in the colocated backend registry/catalog.
It does not update adapter code or bundle definitions inside the running worker
image.

Source: [`config_subscriber.rs`](../src/config_subscriber.rs),
[`config_reconciler.rs`](../src/config_reconciler.rs),
[`dispatcher.rs`](../src/dispatcher.rs), and the gateway
[config guide](../../sie_gateway/docs/architecture-guide.md#43-live-deltas).

## Source Links

- Package summary: [README](../README.md)
- Helm values: [`deploy/helm/sie-cluster/values.yaml`](../../../deploy/helm/sie-cluster/values.yaml)
- Runtime config parsing: [`config.rs`](../src/config.rs) and [`main.rs`](../src/main.rs)
- IPC parity check: [`tools/ci/check_ipc_types_parity.py`](../../../tools/ci/check_ipc_types_parity.py)
- Gateway queue contract: [gateway architecture guide](../../sie_gateway/docs/architecture-guide.md)
