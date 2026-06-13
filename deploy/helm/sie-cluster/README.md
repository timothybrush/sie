# SIE Cluster Helm Chart

Deploy SIE (Search Inference Engine) to Kubernetes with autoscaling and observability.

## Quick Start

```bash
helm install sie-cluster oci://ghcr.io/superlinked/charts/sie-cluster \
  --namespace sie \
  --create-namespace
```

## Architecture

```
┌─────────────┐     ┌─────────────────────────────────────┐     ┌──────────────┐
│   Client    │────▶│      Gateway (1 replica; 2+ for HA)  │◀───▶│  sie-config  │
└─────────────┘     └───────────────┬─────────────────────┘     │ (singleton)  │
                                    │                           └──────────────┘
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │ L4 Pool │     │A100 Pool│     │ CPU Pool│
              │ 0-N     │     │ 0-N     │     │ 0-N     │
              └─────────┘     └─────────┘     └─────────┘
```

- **Gateway**: Stateless request proxy that routes to workers based on GPU type and model affinity. Consumes config via GET/NATS from `sie-config`.
- **sie-config**: Authoritative control plane for model/bundle configuration. Serves `/v1/configs/*` writes and publishes NATS deltas to the gateway and workers. Deployed as a singleton (`replicas: 1`, `strategy: Recreate`).
- **Worker Pools**: StatefulSets per enabled worker group, each with KEDA autoscaling. Routing, metrics, and KEDA scale on the full `(queuePool, machineProfile, bundle)` lane.

## Cold Start Expectations

When scaling from zero, expect the following latencies:

| Step | Duration | Notes |
|-------|----------|-------|
| **Node provisioning** | 2-5 min | GKE/EKS spins up GPU node (spot may be slower) |
| **Container startup** | 20-40s | Pull image, start process, health checks |
| **Model loading** | 10-120s | Download weights (if not cached), load to GPU |
| **Total cold start** | 3-7 min | First request to a scaled-to-zero pool |

### Reducing Cold Start Time

1. **Use cluster cache**: Pre-populate object storage with model weights (`--cluster-cache`)
2. **Set minReplicas=1**: Keep one warm replica per critical GPU type
3. **Use reserved capacity**: Avoid spot for latency-sensitive workloads
4. **Pre-warm models**: Call `/v1/encode/{model}` on startup to load weights

### Client Handling

When a pool is scaling from zero, the gateway returns:
- **503 Service Unavailable** with `X-SIE-Error-Code: PROVISIONING`
- **Retry-After: 60** header
- Client should retry after the indicated delay

The SDK handles this automatically with configurable retries.

## Cluster model cache

Pre-populate shared object storage with model weights so worker pods don't re-download from HuggingFace on every cold start. The Python SDK pulls from the cache first and falls back to HF on miss.

**AWS (Terraform-managed bucket):**

```bash
# 1. Provision the bucket via the AWS Terraform module (opt-in via create_model_cache=true)
cd deploy/terraform/aws/examples/dev-g6-spot
terraform apply

# 2. One-time populate from your laptop
sie-admin cache weights sync --bundle default \
  --dest $(terraform output -raw model_cache_bucket_url)/

# 3. Wire into Helm
helm upgrade --install sie-cluster . \
  --set workers.common.clusterCache.enabled=true \
  --set workers.common.clusterCache.url=$(terraform output -raw model_cache_bucket_url)
```

The Terraform output already includes the `/models` prefix, so the same URL is used for both `sie-admin --target` and `clusterCache.url`.

**Other clouds / BYO bucket:** point `workers.common.clusterCache.url` at any `s3://...`, `gs://...`, `abfs://...`, or `abfss://...` URL the workload identity can read; populate it with the same `sie-admin cache weights sync --dest ...` command.

## Autoscaling

KEDA-based autoscaling with scale-to-zero support:

```yaml
autoscaling:
  enabled: true
  # Scale-to-zero after 10 min idle
  cooldownPeriod: 600
  # Check metrics every 15s
  pollingInterval: 15
```

### Scale-from-Zero Trigger

The gateway exposes `sie_gateway_pending_demand{pool="...",machine_profile="...",bundle="..."}`
when requests arrive for queue lanes with no available workers. KEDA uses this
to trigger scale-up even when there are 0 workers (and thus no worker metrics).
For `X-SIE-Pool` requests that omit `X-SIE-MACHINE-PROFILE`, the gateway can
emit a lane-specific signal only when the registered pool spec contains exactly
one machine profile. Multi-profile tenant pools should send both headers.

### Scaling Metrics

| Metric | Source | Purpose |
|--------|--------|---------|
| `sie_gateway_pending_demand` | Gateway | Trigger scale from 0 |
| `sie_gateway_worker_queue_depth` | Gateway worker registry | Scale up on queued work in the exact pool/profile/bundle lane |
| `sie_gateway_active_lease_gpus` | Gateway pool manager | Hold capacity for active pool leases in the exact pool/profile/bundle lane |
| `sie_gateway_rejected_requests_total` | Gateway | Scale up on sustained capacity/no-worker rejections after retryable cold-load reasons are excluded |

### Worker-sidecar Metrics

When `workers.common.workerSidecar.enabled=true`, each worker Pod includes a
`worker-sidecar` container. The sidecar image is
`ghcr.io/superlinked/sie-server-sidecar`, the Rust binary is
`sie-server-sidecar`, and Prometheus families use the `sie_worker_*` prefix
because they describe worker-side runtime behavior.

The chart wires sidecar scraping as:

- `worker-sidecar` container port `metrics` on `SIE_WORKER_METRICS_PORT`
  (default `9095`)
- worker Service port `worker-metrics`
- worker ServiceMonitor endpoint `worker-metrics`

Alert rules that need the sidecar scrape target match
`endpoint="worker-metrics"` rather than an old `job="sie-worker"` label.

## Configuration

See `values.yaml` for all options. Key settings:

**Important**: All worker pools are disabled by default. You must explicitly enable
the pools you need in your values override.

The values below are an illustrative shape; concrete per-cluster sizes
live in each cluster's own values file (e.g. the tester cluster's
rtx6000 default bundle scales 3–10 — see
`deploy/terraform/aws/internal-examples/tester-cluster/DEPLOY.md`).

```yaml
# Worker pool configuration (must explicitly enable pools)
# The map key is the Kubernetes capacity family/resource name. When
# machineProfile is omitted, it defaults to that map key. machineProfile is the
# runtime lane label used by routing and metrics. queuePool is the logical
# queue namespace / tenant boundary; by default all worker groups use the
# shared `default` queue pool so SDK calls can pass just gpu="<profile>".
# Set queuePool on a worker group only for a dedicated tenant/logical pool,
# then target it explicitly as gpu="<queuePool>/<machineProfile>".
# With poolAdmission enabled, named non-default queue pools must also exist as
# gateway-visible pool objects. For deploy-owned pools, declare them under
# queueRouting.staticQueuePools. For API-owned pools, create/renew them through
# /v1/pools.
# Each worker group renders its own StatefulSet + ScaledObject named
# worker-<pool>-<bundle>.
workers:
  pools:
    l4:
      enabled: true       # Enable this pool (disabled by default)
      bundles:
        default:
          minReplicas: 0  # Scale to zero
          maxReplicas: 10
    rtx6000:
      enabled: true
      bundles:
        default:          # embedding/rerank baseline
          minReplicas: 1
          maxReplicas: 5
        sglang:           # generation, scale-from-zero on same GPUs
          minReplicas: 0
          maxReplicas: 5

# Gateway configuration
gateway:
  replicaCount: 2  # HA by default

# Autoscaling
autoscaling:
  enabled: true
  cooldownPeriod: 600  # 10 min before scale-down
```

### Queue Pool Patterns

Use one of these patterns deliberately:

- **Shared baseline pool**: leave `workers.common.queuePool: default`.
  Workers render with `SIE_POOL=default` and their own
  `SIE_MACHINE_PROFILE`; SDK calls use `gpu="<machineProfile>"`.
- **Static custom queue namespace**: set a worker group's `queuePool` to a
  named value and declare the same name under
  `queueRouting.staticQueuePools`. These pool objects are synthesized by the
  gateway at startup and do not expire. Queue pool names are rendered and
  routed in lowercase. Example:

  ```yaml
  queueRouting:
    staticQueuePools:
      company-a:
        gpus:
          l4: 0
        gpuCaps: {}
  workers:
    pools:
      l4:
        queuePool: company-a
  ```

  `gpuCaps: {}` means uncapped admission for matching workers. Use
  `gpuCaps: {l4: 10}` to cap admission for that machine profile.
- **Dynamic isolated pool**: set a worker group's `queuePool` to the dedicated
  pool name, keep `queueRouting.poolAdmission.enabled=true`, and create/renew
  the matching pool through `/v1/pools`. SDK calls use
  `gpu="<queuePool>/<machineProfile>"`.

Missing named pools intentionally do not fail open. Falling back from
`pool=default,machineProfile=l4` to `pool=l4,machineProfile=l4` would cross the
logical capacity boundary without an explicit caller request.

For emergency or legacy static namespaces that are not backed by either
`queueRouting.staticQueuePools` or `/v1/pools`, disabling
`queueRouting.poolAdmission.enabled` lets workers pull without the admission
gate. Prefer declaring static pools instead, so capped/dynamic pools keep their
fail-closed isolation behavior.

### Upgrading from the legacy single-bundle pool schema

Releases up to and including 0.4.x used a flat schema where each pool
declared a single `bundle:` plus `minReplicas:`/`maxReplicas:` at the
pool level. That shape is no longer accepted — `bundles:` is required
(see schema docs in `values.yaml`).

The rename also changes resource names from `worker-<pool>` to
`worker-<pool>-<bundle>` for StatefulSets, KEDA ScaledObjects, PDBs, and
the image-prepull DaemonSet. `helm upgrade` creates the new resources
but does not delete the old ones. The legacy resources are
distinguishable from the new ones by the absence of the
`sie.superlinked.com/bundle` label:

```bash
NS=sie  # release namespace

# Pre-refactor worker family (no bundle label) — delete before/after upgrade
kubectl -n "$NS" delete statefulset,pdb,daemonset \
  -l 'app.kubernetes.io/component=worker,!sie.superlinked.com/bundle'

# Pre-refactor image-prepull DaemonSets
kubectl -n "$NS" delete daemonset \
  -l 'app.kubernetes.io/component=image-prepull,!sie.superlinked.com/bundle'

# Pre-refactor KEDA ScaledObjects (the chart's apply Job only creates;
# it never deletes obsolete entries)
kubectl -n "$NS" delete scaledobject \
  -l 'app.kubernetes.io/component=worker,!sie.superlinked.com/bundle'
```

Run these once per cluster after the upgrade settles. Leftover
ScaledObjects will keep trying to scale deleted StatefulSets and spam
KEDA logs; leftover PDBs will block node drains.

## Ingress

Enable the Ingress with `ingress.enabled=true` and route traffic to the gateway by
hostname. Use the list-valued `ingress.hosts` to front the gateway with one or more
hostnames — each entry becomes an Ingress rule (and, when TLS is enabled, a SAN on
the cert):

```yaml
ingress:
  enabled: true
  className: nginx
  hosts:
    - sie.example.com
    - api.example.com
```

The singular `ingress.host` is the backward-compatible single-host shorthand; it is
ignored whenever `ingress.hosts` is non-empty. With neither set the chart renders a
host-less catch-all Ingress. All hosts share the single `ingress.tlsConfig.secretName`
(one multi-SAN certificate).

## TLS / HTTPS

The chart supports four TLS modes for the Ingress (set via `ingress.tlsConfig.mode`):

- `byo` — bring your own `kubernetes.io/tls` Secret (default, backward compatible).
- `cert-manager` — chart annotates the Ingress; [cert-manager](https://cert-manager.io/) provisions and renews the certificate. Default flavour is ACME (HTTP-01 challenge to Let's Encrypt); you can also point at an existing internal Issuer/ClusterIssuer.
- `self-signed` — chart bootstraps a self-signed root CA, a CA ClusterIssuer, and a leaf cert for the Ingress. Intended for air-gapped / on-prem / VPC-isolated clusters where Let's Encrypt is unreachable.
- `disabled` — no TLS resources rendered. Use when TLS is terminated upstream (cloud load balancer, sidecar, service mesh).

> **Exactly one cert-manager per cluster.** cert-manager's CRDs, webhooks, and `cert-manager` ClusterRoleBinding are cluster-scoped singletons. Two controllers racing on the same CRDs corrupt issuance state. The chart enforces this with a pre-install Job that aborts when bundled cert-manager would collide with an existing install — see "Bundling cert-manager" below.

Only HTTP-01 ACME challenges are supported by the chart. DNS-01 / wildcard certs (which require cloud-provider IRSA / Workload Identity for Route53 / Cloud DNS) are out of scope — set them up manually outside the chart and reference the resulting Secret via `mode: byo`.

### `mode: byo` — bring-your-own certificate

Create the TLS Secret yourself (e.g. from a corporate CA, ACM cert exported to a Secret, or an existing wildcard cert), then point the chart at it:

```bash
kubectl -n sie create secret tls sie-tls --cert=path/to/tls.crt --key=path/to/tls.key
```

```yaml
ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tlsConfig:
    enabled: true
    mode: byo            # default
    secretName: sie-tls  # default
```

**When to use this**: you already manage TLS centrally, or you have a wildcard cert from a corporate CA, or you need DNS-01 / non-ACME issuance.

### `mode: cert-manager` — automated issuance via cert-manager

Prerequisite: either install cert-manager once in the cluster (its CRDs are cluster-scoped and must exist exactly once), OR opt in to the bundled subchart (see "Bundling cert-manager" below — single-tenant clusters only).

External install (recommended for shared clusters):

```bash
helm repo add jetstack https://charts.jetstack.io && helm repo update
helm install cert-manager jetstack/cert-manager \
  --set crds.enabled=true -n cert-manager --create-namespace
```

For single-tenant clusters where SIE is the only workload, the chart can also
install cert-manager as an opt-in subchart:

```yaml
certManagerBundle:
  certManager:
    install: true
```

Then enable cert-manager mode in your SIE values:

```yaml
ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tlsConfig:
    enabled: true
    mode: cert-manager
    certManager:
      email: ops@example.com
      # Use Let's Encrypt staging while iterating to avoid the 50 new-cert/registered-domain/week prod limit (duplicate-cert limit is 5/week):
      # server: https://acme-staging-v02.api.letsencrypt.org/directory
      kind: ClusterIssuer  # cluster-scoped; share across namespaces. Use "Issuer" for namespace-scoped.
      create: true         # chart renders the Issuer/ClusterIssuer
```

The chart renders a `{kind}` named `{release-fullname}-letsencrypt-prod` (release-scoped to avoid collisions when multiple SIE releases share a cluster) and adds the appropriate `cert-manager.io/cluster-issuer` (or `/issuer`) annotation to the main Ingress. cert-manager populates `ingress.tlsConfig.secretName` (default `sie-tls`); the same Secret is referenced by the oauth2-proxy Ingress when auth is enabled.

Note: Helm's standard `fullname` collapses when the release name already contains the chart name, so `helm install sie-cluster …` produces `sie-cluster-letsencrypt-prod` (not `sie-cluster-sie-cluster-letsencrypt-prod`). If you override `certManager.name`, set the full intended name explicitly rather than expecting a particular default.

Issuer kind tradeoff:

- `ClusterIssuer` — single ACME account / private key shared across all namespaces. Best for shared clusters.
- `Issuer` — namespace-scoped. Use for hard tenant isolation, or when you don't have permission to create cluster-scoped resources.

**Reusing an existing ClusterIssuer/Issuer.** In multi-tenant clusters where a platform team already manages a shared `ClusterIssuer`, set `create: false` and reference it by name:

```yaml
ingress:
  tlsConfig:
    enabled: true
    mode: cert-manager
    certManager:
      kind: ClusterIssuer
      create: false
      name: platform-letsencrypt-prod
```

The chart only adds the annotation — it does not render any Issuer resource.

**When to use this**: ACME / Let's Encrypt is reachable from your cluster (or you already have an internal Issuer/ClusterIssuer) and your platform team is OK with cert-manager being installed.

### `mode: self-signed` — self-signed CA (air-gapped / on-prem)

For clusters that cannot reach Let's Encrypt — typical for on-prem, regulated, or VPC-isolated environments — the chart can bootstrap a self-signed root CA and use it to issue the Ingress leaf cert. cert-manager is still required.

```yaml
certManagerBundle:
  certManager:
    install: true      # bundle cert-manager (SINGLE-TENANT clusters only — see warning above)

ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tls:
    enabled: true
    mode: self-signed
    secretName: sie-tls
    selfSigned:
      rootCA:
        commonName: "Acme Corp SIE Root CA"
        # 43800h = 5y, 720h = 30d renewBefore
      leaf:
        # 2160h = 90d, 360h = 15d renewBefore — match Let's Encrypt lifetimes
        dnsNames: []   # extra SANs in addition to ingress.host
        ipAddresses: []
```

Chain:

1. `SelfSigned` ClusterIssuer (bootstrap, name `{fullname}-selfsigned-bootstrap`).
2. Root CA `Certificate` (`{fullname}-root-ca`, isCA, 5y, RSA-4096) -> Secret `sie-root-ca-key-pair`.
3. CA `ClusterIssuer` (`sie-self-signed-ca`) backed by the root CA secret.
4. Ingress leaf `Certificate` (`{fullname}-ingress-leaf`, ECDSA-P256, 90d) -> Secret `sie-tls`, consumed by the Ingress.

Clients (browsers, curl) must trust the root CA. Export it with:

```bash
kubectl -n sie get secret sie-root-ca-key-pair \
  -o jsonpath='{.data.ca\.crt}' | base64 -d > sie-root-ca.crt
```

> **Root CA namespace constraint.** Two independent namespace-scoped lookups apply:
>
> 1. **cert-manager** only resolves Secrets referenced by a `ClusterIssuer` inside its `--cluster-resource-namespace` (defaults to its own Deployment's namespace). The chart writes the root CA to `ingress.tlsConfig.selfSigned.rootCA.namespace` (defaults to the release namespace, which is correct for the **bundled** subchart since cert-manager also runs in the release namespace).
> 2. **trust-manager** only resolves source Secrets for `Bundle` resources inside its `--trust-namespace` (defaults to `cert-manager`, regardless of where trust-manager itself runs).
>
> If you also enable `certManagerBundle.trustBundle.enabled: true` with the bundled trust-manager, override the trust namespace at install time so it matches where the root CA lives:
>
> ```bash
> helm install ... --set "trust-manager.app.trust.namespace=<release-namespace>"
> ```
>
> Otherwise the Bundle stays `Synced=False` with `SourceNotFound`. For external cert-manager (typically in `cert-manager` namespace), set `ingress.tlsConfig.selfSigned.rootCA.namespace: cert-manager` so the CA `ClusterIssuer` can find its Secret; the default trust-namespace then already matches.

**When to use this**: air-gapped / on-prem clusters where you can distribute the root CA to client machines (e.g. via MDM, internal trust store, or workload mount), and you want a single `helm install` to land a working HTTPS path.

### `mode: disabled` — no TLS resources

Use when TLS is terminated upstream of the Ingress (cloud LB, sidecar, service mesh):

```yaml
ingress:
  enabled: true
  className: nginx
  host: sie.example.com
  tls:
    enabled: false
    mode: disabled
```

### Trust distribution with trust-manager

When `mode: self-signed`, you can replicate the root CA into other namespaces as a ConfigMap so non-SIE workloads can trust SIE without out-of-band copying.

```yaml
certManagerBundle:
  trustManager:
    install: true      # required when not already installed externally
  trustBundle:
    enabled: true
    name: sie-root-ca-bundle
    target:
      configMapKey: ca.crt
    namespaceSelector:
      matchLabels:
        sie.io/trust: "true"  # label target namespaces to opt them in
```

Workloads mount the resulting ConfigMap and point their HTTP client at it as the CA bundle.

### Bundling cert-manager

The chart can install cert-manager and trust-manager as opt-in subchart dependencies (default off). **This is reserved for single-tenant clusters where SIE is the sole workload.** In any multi-tenant or shared cluster, install cert-manager once out-of-band and leave `certManagerBundle.certManager.install: false`.

Guards:

1. Both subcharts are gated by `*.install` flags that default to `false`; default chart behaviour is unchanged.
2. A template-time `lookup` aborts the install when `certManagerBundle.certManager.install: true` is combined with an existing `certificates.cert-manager.io` CRD.
3. A `pre-install` Job hook re-checks at apply time and aborts before any subchart resources are created (`lookup` returns empty during `helm template` / `--dry-run`, so the Job is the real safety net).
4. Jetstack's `crds.keep: true` default is left in place, so `helm uninstall` does not silently delete `Certificate` / `Issuer` resources belonging to other operators.

Override the conflict guard (DANGER):

```yaml
certManagerBundle:
  allowExistingCRDs: true   # bypass both guards; only if you accept the consequences
```

#### Vendored CRDs

cert-manager and trust-manager CRDs are vendored at `deploy/helm/sie-cluster/crds/`. Helm applies files in a chart's `crds/` directory **before** rendering templates, which is the only mechanism that lets bundled mode complete in a single `helm install` (the subcharts' own templated CRDs would land too late — Helm's RESTMapper discovery runs at install start and fails to resolve `Certificate` / `Bundle` references). Both subcharts have `crds.enabled: false` set in `values.yaml` so they don't try to re-install the same CRDs.

The vendored bundles are pinned to the same version as the subchart pins in `Chart.yaml`:

- `crds/cert-manager.crds.yaml` — fetched from the cert-manager release: `curl -fsSL -o crds/cert-manager.crds.yaml https://github.com/cert-manager/cert-manager/releases/download/v<X.Y.Z>/cert-manager.crds.yaml`
- `crds/trust-manager.crds.yaml` — extracted from the subchart tarball: `helm template trust-manager charts/trust-manager-v<X.Y.Z>.tgz --show-only templates/crd-trust.cert-manager.io_bundles.yaml > crds/trust-manager.crds.yaml`

When bumping the subchart version pin in `Chart.yaml`, re-vendor both files and re-run the golden-diff tests to surface CRD schema changes.

### Uninstall caveat

`crds.keep: true` is the Jetstack default. `helm uninstall sie-cluster` will **leave cert-manager CRDs behind on purpose**, so that `Certificate` / `Issuer` resources owned by other operators are not silently deleted. To remove them, run `kubectl delete crd <name>.cert-manager.io <name>.acme.cert-manager.io <name>.trust.cert-manager.io ...` explicitly. See the [cert-manager uninstall docs](https://cert-manager.io/docs/installation/helm/#uninstalling) for the full CRD list.

## Gated Models

Some HuggingFace models require authentication to download (gated models). Examples:

- `google/embeddinggemma-300m` - Manual gating (requires approval)
- `naver/splade-v3` - Auto gating (requires license acceptance)

### Prerequisites

1. Create a HuggingFace account and generate an access token at <https://huggingface.co/settings/tokens>
2. For manually gated models, request access on the model page (e.g., <https://huggingface.co/google/embeddinggemma-300m>)
3. For auto-gated models, accept the license agreement on the model page

### Kubernetes Setup

Create a secret with your HuggingFace token:

```bash
kubectl create secret generic hf-token \
  --namespace sie \
  --from-literal=token=hf_your_token_here
```

Configure the Helm chart to use the secret:

```yaml
workers:
  common:
    hfCache:
      tokenSecret: hf-token      # Secret name
      tokenSecretKey: token      # Key within the secret
```

The token is mounted as the `HF_TOKEN` environment variable, which HuggingFace libraries automatically detect.

### Local Development

For local development, set the `HF_TOKEN` environment variable:

```bash
# Option 1: Direct export
export HF_TOKEN=hf_your_token_here
mise run serve

# Option 2: From file
export HF_TOKEN=$(cat ~/.secrets/hf_token)
mise run serve
```

### Docker

Pass the token as an environment variable:

```bash
docker run -e HF_TOKEN=hf_your_token_here \
  -p 8080:8080 \
  sie-server:cuda12-default
```

## Telemetry

SIE collects anonymous usage telemetry (version, OS, architecture, GPU type) to help maintainers understand adoption and hardware distribution. Telemetry is on by default and sends a lightweight heartbeat once per hour.

**No IP addresses, hostnames, cluster names, API keys, or request data are collected.**

Disable telemetry:

```yaml
telemetry:
  enabled: false
```

Enterprise customers can route heartbeats through their own collector:

```yaml
telemetry:
  url: "https://telemetry.internal.example.com/api/telemetry"
```

Tag non-production deployments to filter them out of dashboards:

```yaml
telemetry:
  deploymentEnv: staging  # production (default) | staging | development | ci
```

> **Internal Superlinked clusters:** any cluster owned by Superlinked that is
> not a customer-facing production install MUST set `telemetry.deploymentEnv`
> to one of `staging | development | ci`. The chart default is `production`
> so that customer Helm installs are correctly tagged out of the box; internal
> stacks must opt out explicitly to keep them out of the production telemetry
> dashboards. See `deploy/terraform/{aws,gcp}/internal-examples/` for the
> per-cluster mapping.

## Observability

Observability components (Prometheus, Grafana, Loki, DCGM Exporter, Alloy, Event Exporter) are included as optional sub-chart dependencies. Enable them in your values overlay (e.g. `kube-prometheus-stack.install: true`, `observability.logs.install: true`, or `kubernetes-event-exporter.install: true`).

Pre-configured dashboards:

- Cluster overview (QPS, latency, GPU utilization)
- Per-model performance
- Worker health
