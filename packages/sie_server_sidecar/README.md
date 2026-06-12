# SIE Server Sidecar

`sie-server-sidecar` builds the `sie-server-sidecar` binary and `sie-server-sidecar`
container image.

The sidecar owns the queue-mode runtime around an inference adapter pod: NATS
JetStream consumption, batching and scheduling, IPC to the adapter, payload
fetching, result framing, ACK/NAK behavior, metrics, and readiness.

It does not load model weights or link GPU libraries. The colocated
`sie-server` container remains the Python adapter/model-execution process.

Public and runtime names:

- Kubernetes container: `worker-sidecar`
- Binary: `sie-server-sidecar`
- Image: `ghcr.io/superlinked/sie-server-sidecar`
- Metrics: `sie_worker_*` (worker-side runtime metrics)

See [`docs/architecture-guide.md`](docs/architecture-guide.md) for the runtime
contract and deployment caveats.
