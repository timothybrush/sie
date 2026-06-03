//! Payload store — read-only side.
//!
//! Workers consume payload_refs written by the gateway
//! (`packages/sie_gateway/src/queue/payload_store.rs`). Only `get` is needed
//! on this side; the gateway writes/deletes.
//!
//! The ref format is:
//!   - local: filesystem path under the worker's configured `base_dir`
//!     (absolute paths are only accepted when they canonicalize inside
//!     `base_dir`; relative paths are resolved against it). When no
//!     `base_dir` is configured the reader is disabled for safety.
//!   - cloud: `s3://…` / `gs://…` URL (handled when the `cloud-storage`
//!     feature is enabled)
//!
//! Security: we canonicalize the resolved path and reject anything that
//! escapes `base_dir`. Without this, a compromised gateway could send a
//! `payload_ref` like `/etc/passwd` or `../../secrets/key.bin` and read
//! arbitrary files the worker process has access to.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use thiserror::Error;
use tracing::debug;

#[derive(Debug, Error)]
pub enum PayloadError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid payload ref: {0}")]
    InvalidRef(String),
    #[error("unsupported payload scheme: {0}")]
    Unsupported(String),
    #[error("object store: {0}")]
    ObjectStore(String),
}

#[async_trait]
pub trait PayloadStore: Send + Sync {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError>;
}

/// Local filesystem reader.
///
/// * When `base_dir` is `Some`, `payload_ref` is resolved against it and
///   the final (canonical) path MUST stay inside `base_dir`. Absolute refs
///   are accepted only when they canonicalize under `base_dir`.
/// * When `base_dir` is `None`, the store rejects all reads — this
///   matches the gateway's refusal to publish payload_refs when the
///   worker isn't configured with `SIE_PAYLOAD_STORE_URL`.
pub struct LocalPayloadStore {
    base_dir: Option<PathBuf>,
}

impl LocalPayloadStore {
    pub fn new(base_dir: Option<impl Into<PathBuf>>) -> Self {
        Self {
            base_dir: base_dir.map(Into::into),
        }
    }

    /// Resolve `payload_ref` to an absolute filesystem path that is
    /// guaranteed to sit under `base_dir` (when configured).
    ///
    /// The caller is expected to treat the returned path as the _only_
    /// safe target to read — do not re-interpret it or follow symlinks
    /// at the application layer.
    fn resolve(&self, payload_ref: &str) -> Result<PathBuf, PayloadError> {
        if payload_ref.is_empty() {
            return Err(PayloadError::InvalidRef(payload_ref.to_string()));
        }
        let Some(base) = &self.base_dir else {
            return Err(PayloadError::InvalidRef(format!(
                "payload store not configured; cannot resolve {payload_ref}"
            )));
        };

        let base_canon = base.canonicalize().map_err(|e| {
            PayloadError::InvalidRef(format!(
                "configured base_dir {} is not canonicalizable: {e}",
                base.display()
            ))
        })?;

        let raw = Path::new(payload_ref);
        let joined = if raw.is_absolute() {
            raw.to_path_buf()
        } else {
            base_canon.join(raw)
        };

        // We canonicalize the resolved path to eliminate `..`, symlinks,
        // and case-insensitive tricks, then check that it still sits
        // under the canonical base_dir. `canonicalize` requires the
        // target to exist; preserve missing-file semantics as an IO
        // NotFound rather than masking it as an invalid ref.
        let full = joined.canonicalize().map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                PayloadError::Io(e)
            } else {
                PayloadError::InvalidRef(format!(
                    "failed to canonicalize payload ref {payload_ref}: {e}"
                ))
            }
        })?;
        if !full.starts_with(&base_canon) {
            return Err(PayloadError::InvalidRef(format!(
                "payload ref {payload_ref} escapes base_dir {}",
                base_canon.display()
            )));
        }
        Ok(full)
    }
}

#[async_trait]
impl PayloadStore for LocalPayloadStore {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError> {
        let path = self.resolve(payload_ref)?;
        let bytes = tokio::fs::read(&path).await?;
        debug!(path = %path.display(), bytes = bytes.len(), "read local payload");
        Ok(bytes)
    }
}

#[cfg(feature = "cloud-storage")]
pub struct ObjectPayloadStore {
    store: Box<dyn object_store::ObjectStore>,
    prefix: object_store::path::Path,
    url_prefix: String,
    scheme: &'static str,
}

#[cfg(feature = "cloud-storage")]
impl ObjectPayloadStore {
    pub fn new_s3(path: &str) -> Result<Self, PayloadError> {
        use object_store::aws::AmazonS3Builder;

        let (bucket, prefix) = Self::parse_bucket_prefix(path)?;
        let store = AmazonS3Builder::from_env()
            .with_bucket_name(&bucket)
            .build()
            .map_err(|e| {
                PayloadError::Unsupported(format!("failed to build S3 payload store: {e}"))
            })?;
        Ok(Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix: format!("s3://{}", path.trim_end_matches('/')),
            scheme: "s3://",
        })
    }

    pub fn new_gcs(path: &str) -> Result<Self, PayloadError> {
        use object_store::gcp::GoogleCloudStorageBuilder;

        let (bucket, prefix) = Self::parse_bucket_prefix(path)?;
        let store = GoogleCloudStorageBuilder::from_env()
            .with_bucket_name(&bucket)
            .build()
            .map_err(|e| {
                PayloadError::Unsupported(format!("failed to build GCS payload store: {e}"))
            })?;
        Ok(Self {
            store: Box::new(store),
            prefix: object_store::path::Path::from(prefix),
            url_prefix: format!("gs://{}", path.trim_end_matches('/')),
            scheme: "gs://",
        })
    }

    fn parse_bucket_prefix(path: &str) -> Result<(String, String), PayloadError> {
        let path = path.trim_matches('/');
        let Some((bucket, prefix)) = path.split_once('/') else {
            if path.is_empty() {
                return Err(PayloadError::InvalidRef(
                    "payload store bucket is empty".into(),
                ));
            }
            return Ok((path.to_string(), String::new()));
        };
        if bucket.is_empty() {
            return Err(PayloadError::InvalidRef(
                "payload store bucket is empty".into(),
            ));
        }
        Ok((bucket.to_string(), prefix.trim_matches('/').to_string()))
    }

    fn relative_key<'a>(&self, payload_ref: &'a str) -> Result<&'a str, PayloadError> {
        if payload_ref.is_empty() {
            return Err(PayloadError::InvalidRef(payload_ref.to_string()));
        }

        let key = if payload_ref.starts_with(self.scheme) {
            let prefix = self.url_prefix.trim_end_matches('/');
            let Some(rest) = payload_ref.strip_prefix(prefix) else {
                return Err(PayloadError::InvalidRef(format!(
                    "payload ref {payload_ref} is outside configured store {prefix}"
                )));
            };
            let Some(rest) = rest.strip_prefix('/') else {
                return Err(PayloadError::InvalidRef(format!(
                    "payload ref {payload_ref} is outside configured store {prefix}"
                )));
            };
            rest
        } else if payload_ref.starts_with("s3://") || payload_ref.starts_with("gs://") {
            return Err(PayloadError::InvalidRef(format!(
                "payload ref {payload_ref} uses a different object-store scheme"
            )));
        } else {
            payload_ref
        };

        Self::validate_relative_key(key)?;
        Ok(key)
    }

    fn validate_relative_key(key: &str) -> Result<(), PayloadError> {
        if key.is_empty()
            || key.starts_with('/')
            || key.contains('\\')
            || key
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return Err(PayloadError::InvalidRef(key.to_string()));
        }
        Ok(())
    }

    fn object_key(&self, key: &str) -> object_store::path::Path {
        if self.prefix.as_ref().is_empty() {
            object_store::path::Path::from(key)
        } else {
            object_store::path::Path::from(format!("{}/{}", self.prefix, key))
        }
    }
}

#[cfg(feature = "cloud-storage")]
#[async_trait]
impl PayloadStore for ObjectPayloadStore {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError> {
        let key = self.relative_key(payload_ref)?;
        let object_key = self.object_key(key);
        let result = self.store.get(&object_key).await.map_err(|e| match e {
            object_store::Error::NotFound { .. } => {
                PayloadError::InvalidRef(format!("payload ref not found: {payload_ref}"))
            }
            other => PayloadError::ObjectStore(format!("get payload {object_key}: {other}")),
        })?;
        let bytes = result
            .bytes()
            .await
            .map_err(|e| PayloadError::ObjectStore(format!("read payload {object_key}: {e}")))?;
        debug!(key = %object_key, bytes = bytes.len(), "read object-store payload");
        Ok(bytes.to_vec())
    }
}

/// Metrics-observing wrapper — counts fetches, broken-out errors, and
/// records latency + payload size histograms. Kept separate from the
/// concrete stores so the metrics registry dependency doesn't leak into
/// their traits / tests.
pub struct MeteredPayloadStore {
    inner: std::sync::Arc<dyn PayloadStore>,
    metrics: std::sync::Arc<crate::metrics::MetricsRegistry>,
}

impl MeteredPayloadStore {
    pub fn new(
        inner: std::sync::Arc<dyn PayloadStore>,
        metrics: std::sync::Arc<crate::metrics::MetricsRegistry>,
    ) -> Self {
        Self { inner, metrics }
    }
}

#[async_trait]
impl PayloadStore for MeteredPayloadStore {
    async fn get(&self, payload_ref: &str) -> Result<Vec<u8>, PayloadError> {
        let start = std::time::Instant::now();
        self.metrics.payload_fetch_total.inc();
        let result = self.inner.get(payload_ref).await;
        let elapsed = start.elapsed().as_secs_f64();
        match &result {
            Ok(bytes) => {
                self.metrics
                    .payload_fetch_seconds
                    .with_label_values(&["ok"])
                    .observe(elapsed);
                self.metrics
                    .payload_bytes
                    .with_label_values(&["ok"])
                    .observe(bytes.len() as f64);
            }
            Err(e) => {
                let reason = match e {
                    PayloadError::Io(io) => match io.kind() {
                        std::io::ErrorKind::NotFound => "not_found",
                        std::io::ErrorKind::PermissionDenied => "permission_denied",
                        _ => "io",
                    },
                    PayloadError::InvalidRef(_) => "invalid_ref",
                    PayloadError::Unsupported(_) => "unsupported",
                    PayloadError::ObjectStore(_) => "object_store",
                };
                self.metrics
                    .payload_fetch_errors_total
                    .with_label_values(&[reason])
                    .inc();
                self.metrics
                    .payload_fetch_seconds
                    .with_label_values(&["error"])
                    .observe(elapsed);
            }
        }
        result
    }
}

/// Factory — mirrors the gateway's `create_payload_store`. Returns an
/// opaque trait object so call sites don't care which backend is in use.
///
/// URL handling:
/// * `None` or an empty string → local store with no base_dir (disabled,
///   rejects all reads). Pods without offload configured stay safe.
/// * `s3://…` / `gs://…` with the `cloud-storage` feature → object
///   storage reader.
/// * `s3://…` / `gs://…` without the feature → we can't read from
///   object storage, and we MUST NOT silently treat the URL string as a
///   filesystem path (`LocalPayloadStore::new(Some("s3://…"))` canonicalizes
///   to a broken, confusing error on every request). Return an explicit
///   `Unsupported` so the worker fails fast at startup instead of
///   NAKing every item at runtime.
/// * `file://path` → strip scheme, treat as local filesystem.
/// * Anything else → treated as a local filesystem path.
pub async fn create_payload_store(
    url: Option<&str>,
) -> Result<std::sync::Arc<dyn PayloadStore>, PayloadError> {
    let Some(url) = url.filter(|u| !u.is_empty()) else {
        return Ok(std::sync::Arc::new(LocalPayloadStore::new(None::<PathBuf>)));
    };

    if url.starts_with("s3://") || url.starts_with("gs://") {
        #[cfg(feature = "cloud-storage")]
        {
            if let Some(rest) = url.strip_prefix("s3://") {
                return Ok(std::sync::Arc::new(ObjectPayloadStore::new_s3(rest)?));
            }
            if let Some(rest) = url.strip_prefix("gs://") {
                return Ok(std::sync::Arc::new(ObjectPayloadStore::new_gcs(rest)?));
            }
        }
        #[cfg(not(feature = "cloud-storage"))]
        {
            return Err(PayloadError::Unsupported(format!(
                "payload store URL {url} requires the 'cloud-storage' feature; \
                 rebuild the worker with --features cloud-storage or point \
                 SIE_PAYLOAD_STORE_URL at a local path"
            )));
        }
    }

    let path = url.strip_prefix("file://").unwrap_or(url);
    Ok(std::sync::Arc::new(LocalPayloadStore::new(Some(path))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn local_get_absolute_path_inside_base_dir() {
        // Gateway writes absolute paths (e.g. /var/cache/sie/uuid.bin);
        // those are accepted as long as they sit under the configured
        // base_dir.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("req-1_0.bin");
        tokio::fs::write(&path, b"hello").await.unwrap();

        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let got = store.get(path.to_str().unwrap()).await.unwrap();
        assert_eq!(got, b"hello");
    }

    #[tokio::test]
    async fn local_get_relative_with_base() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("req-2_0.bin");
        tokio::fs::write(&path, b"world").await.unwrap();

        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let got = store.get("req-2_0.bin").await.unwrap();
        assert_eq!(got, b"world");
    }

    #[tokio::test]
    async fn missing_ref_surfaces_not_found_io() {
        let dir = TempDir::new().unwrap();
        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let err = store.get("missing.bin").await.unwrap_err();
        assert!(
            matches!(err, PayloadError::Io(ref e) if e.kind() == std::io::ErrorKind::NotFound),
            "expected NotFound IO for missing payload, got {err:?}"
        );
    }

    #[tokio::test]
    async fn local_get_without_base_dir_is_disabled() {
        // Safety: without a configured base_dir, the reader is disabled
        // so a compromised gateway can't feed us arbitrary paths.
        let store = LocalPayloadStore::new(None::<PathBuf>);
        let err = store.get("/etc/passwd").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[tokio::test]
    async fn absolute_ref_outside_base_dir_is_rejected() {
        // Classic path-traversal: base_dir is a tempdir, attacker sends
        // an absolute path pointing outside it (here: /tmp root, which
        // exists on every macOS/Linux box). The canonical check must
        // refuse.
        let base = TempDir::new().unwrap();
        let store = LocalPayloadStore::new(Some(base.path().to_path_buf()));
        let err = store.get("/etc/hosts").await.unwrap_err();
        assert!(
            matches!(err, PayloadError::InvalidRef(_)),
            "expected InvalidRef for out-of-base absolute, got {err:?}"
        );
    }

    #[tokio::test]
    async fn relative_dotdot_ref_cannot_escape_base_dir() {
        // Same thing as a relative ref that tries to climb out.
        let parent = TempDir::new().unwrap();
        let base = parent.path().join("sub");
        tokio::fs::create_dir(&base).await.unwrap();
        // Touch a file in the parent that an attacker might want to read.
        tokio::fs::write(parent.path().join("secret.bin"), b"secret")
            .await
            .unwrap();

        let store = LocalPayloadStore::new(Some(base.clone()));
        let err = store.get("../secret.bin").await.unwrap_err();
        assert!(
            matches!(err, PayloadError::InvalidRef(_)),
            "expected InvalidRef for dotdot escape, got {err:?}"
        );
    }

    #[tokio::test]
    async fn empty_ref_rejected() {
        let dir = TempDir::new().unwrap();
        let store = LocalPayloadStore::new(Some(dir.path().to_path_buf()));
        let err = store.get("").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[tokio::test]
    async fn factory_none_yields_disabled_store() {
        let store = create_payload_store(None).await.unwrap();
        let err = store.get("/etc/passwd").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[tokio::test]
    async fn factory_empty_url_yields_disabled_store() {
        // Treat `SIE_PAYLOAD_STORE_URL=` the same as unset, not as "the
        // current directory". The previous behaviour tried to use "" as a
        // filesystem path which canonicalized to CWD.
        let store = create_payload_store(Some("")).await.unwrap();
        let err = store.get("foo.bin").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[tokio::test]
    async fn factory_local_path_roundtrips() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pay.bin");
        tokio::fs::write(&path, b"ok").await.unwrap();

        let store = create_payload_store(Some(dir.path().to_str().unwrap()))
            .await
            .unwrap();
        let got = store.get("pay.bin").await.unwrap();
        assert_eq!(got, b"ok");
    }

    #[tokio::test]
    async fn factory_file_scheme_is_stripped() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("pay.bin");
        tokio::fs::write(&path, b"ok").await.unwrap();

        let url = format!("file://{}", dir.path().display());
        let store = create_payload_store(Some(&url)).await.unwrap();
        let got = store.get("pay.bin").await.unwrap();
        assert_eq!(got, b"ok");
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_accepts_full_ref_inside_prefix() {
        use object_store::path::Path;
        use object_store::{ObjectStore, PutPayload};

        let memory = object_store::memory::InMemory::new();
        memory
            .put(
                &Path::from("payloads/pay.bin"),
                PutPayload::from(b"cloud".to_vec()),
            )
            .await
            .unwrap();

        let store = ObjectPayloadStore {
            store: Box::new(memory),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };
        let got = store.get("s3://bucket/payloads/pay.bin").await.unwrap();
        assert_eq!(got, b"cloud");
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_rejects_ref_outside_prefix() {
        use object_store::path::Path;

        let store = ObjectPayloadStore {
            store: Box::new(object_store::memory::InMemory::new()),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };
        let err = store.get("s3://bucket/other/pay.bin").await.unwrap_err();
        assert!(matches!(err, PayloadError::InvalidRef(_)));
    }

    #[cfg(feature = "cloud-storage")]
    #[tokio::test]
    async fn object_store_get_accepts_plain_key_under_configured_prefix() {
        use object_store::path::Path;
        use object_store::{ObjectStore, PutPayload};

        let memory = object_store::memory::InMemory::new();
        memory
            .put(
                &Path::from("payloads/pay.bin"),
                PutPayload::from(b"cloud".to_vec()),
            )
            .await
            .unwrap();

        let store = ObjectPayloadStore {
            store: Box::new(memory),
            prefix: Path::from("payloads"),
            url_prefix: "s3://bucket/payloads".into(),
            scheme: "s3://",
        };
        let got = store.get("pay.bin").await.unwrap();
        assert_eq!(got, b"cloud");
    }

    #[cfg(not(feature = "cloud-storage"))]
    #[tokio::test]
    async fn factory_s3_without_feature_errors_loudly() {
        // Previously this would silently fall back to a LocalPayloadStore
        // whose base_dir was the literal string "s3://…", guaranteeing
        // every read produced a confusing `canonicalize` error. We now
        // refuse at construction time.
        match create_payload_store(Some("s3://bucket/prefix")).await {
            Ok(_) => panic!("expected Unsupported for s3:// without cloud-storage feature"),
            Err(e) => assert!(
                matches!(e, PayloadError::Unsupported(_)),
                "expected Unsupported for s3://, got {e:?}"
            ),
        }
    }
}
