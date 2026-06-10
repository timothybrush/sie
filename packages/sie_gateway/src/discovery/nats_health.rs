use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use tracing::{info, warn};

use crate::state::worker_registry::WorkerRegistry;
use crate::types::WorkerStatusMessage;

pub struct NatsHealthManager {
    registry: Arc<WorkerRegistry>,
    cancel_tx: tokio::sync::watch::Sender<()>,
    cancel_rx: tokio::sync::watch::Receiver<()>,
}

impl NatsHealthManager {
    pub fn new(registry: Arc<WorkerRegistry>) -> Self {
        let (cancel_tx, cancel_rx) = tokio::sync::watch::channel(());
        Self {
            registry,
            cancel_tx,
            cancel_rx,
        }
    }

    /// Subscribe to NATS health messages using a shared client (no separate connection).
    pub async fn start(
        &self,
        client: &async_nats::Client,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let sub = client.subscribe("sie.health.>").await?;
        info!("subscribed to sie.health.>");

        let registry = Arc::clone(&self.registry);
        let mut cancel_rx = self.cancel_rx.clone();

        tokio::spawn(async move {
            run_subscription(registry, sub, &mut cancel_rx).await;
        });

        Ok(())
    }

    pub async fn start_heartbeat_loop(&self) {
        let registry = Arc::clone(&self.registry);
        let mut cancel_rx = self.cancel_rx.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            loop {
                tokio::select! {
                    _ = interval.tick() => {
                        let sweep = registry.check_heartbeats().await;
                        for url in &sweep.unhealthy {
                            warn!(url = %url, "worker missed heartbeat (NATS)");
                        }
                        for url in &sweep.evicted {
                            info!(url = %url, "evicted stale worker after missed heartbeats (NATS)");
                        }
                    }
                    _ = cancel_rx.changed() => {
                        return;
                    }
                }
            }
        });
    }

    pub async fn stop(&self) {
        info!("stopping NATS health manager");
        let _ = self.cancel_tx.send(());
    }
}

async fn run_subscription(
    registry: Arc<WorkerRegistry>,
    mut sub: async_nats::Subscriber,
    cancel_rx: &mut tokio::sync::watch::Receiver<()>,
) {
    info!("NATS health subscription handler started");

    loop {
        tokio::select! {
            msg = sub.next() => {
                match msg {
                    Some(msg) => {
                        handle_nats_message(&registry, msg).await;
                    }
                    None => {
                        warn!("NATS subscription stream ended");
                        break;
                    }
                }
            }
            _ = cancel_rx.changed() => {
                info!("NATS health subscription cancelled");
                break;
            }
        }
    }
}

async fn handle_nats_message(registry: &WorkerRegistry, msg: async_nats::Message) {
    let subject = msg.subject.as_str();

    // Try msgpack first (compact binary format), then fall back to JSON
    let status: WorkerStatusMessage = if let Ok(s) = rmp_serde::from_slice(&msg.payload) {
        s
    } else if let Ok(s) = serde_json::from_slice(&msg.payload) {
        s
    } else {
        warn!(
            subject = %subject,
            "failed to parse NATS health message (tried msgpack and JSON)"
        );
        return;
    };

    handle_status_message(registry, subject, status).await;
}

async fn handle_status_message(
    registry: &WorkerRegistry,
    subject: &str,
    status: WorkerStatusMessage,
) {
    // Extract worker URL from the message or subject
    // The subject format is sie.health.<worker_identifier>
    // The worker URL should be in the message payload or derivable from the worker name
    let worker_url = if let Some(url) = extract_worker_url_from_status(&status, subject) {
        url
    } else {
        warn!(
            subject = %subject,
            "could not determine worker URL from NATS health message"
        );
        return;
    };

    if status.terminated {
        if registry.remove_worker(&worker_url).await {
            info!(url = %worker_url, subject = %subject, "worker removed via NATS tombstone");
        } else {
            info!(url = %worker_url, subject = %subject, "NATS tombstone for unknown worker ignored");
        }
        return;
    }

    let became_healthy = registry.update_worker(&worker_url, status).await;
    if became_healthy {
        info!(url = %worker_url, subject = %subject, "worker became healthy via NATS");
    }
}

fn extract_worker_url_from_status(status: &WorkerStatusMessage, subject: &str) -> Option<String> {
    // First, try the name field as a URL if it looks like one
    if status.name.starts_with("http://") || status.name.starts_with("https://") {
        return Some(status.name.clone());
    }

    // Extract the worker identifier from the NATS subject
    // Format: sie.health.<worker_id> where worker_id might be an IP:port or hostname
    let parts: Vec<&str> = subject.splitn(3, '.').collect();
    if parts.len() >= 3 {
        let worker_id = parts[2];
        // If it looks like a hostname:port or IP:port, construct URL
        if worker_id.contains(':') || worker_id.contains('.') {
            let url = format!("http://{worker_id}");
            return Some(url);
        }
        // Use the worker name from the status message with the subject as fallback
        if !status.name.is_empty() {
            return Some(format!("http://{}", status.name));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn status(ready: bool) -> WorkerStatusMessage {
        WorkerStatusMessage {
            name: "worker-1".into(),
            ready,
            gpu_count: 1,
            machine_profile: "rtx6000".into(),
            pool_name: "default".into(),
            bundle: "sglang".into(),
            bundle_config_hash: "hash".into(),
            loaded_models: vec![],
            models: vec![],
            gpus: vec![],
            queue_depth: None,
            memory_used_bytes: None,
            memory_total_bytes: None,
            saturated: false,
            terminated: false,
        }
    }

    #[tokio::test]
    async fn tombstone_removes_registered_worker() {
        let registry = WorkerRegistry::new(Duration::from_secs(15), None);
        handle_status_message(&registry, "sie.health.worker-1", status(true)).await;
        assert_eq!(registry.workers().await.len(), 1);

        let mut tombstone = status(false);
        tombstone.terminated = true;
        handle_status_message(&registry, "sie.health.worker-1", tombstone).await;

        assert!(registry.workers().await.is_empty());
        assert!(registry.healthy_workers().await.is_empty());
    }
}
