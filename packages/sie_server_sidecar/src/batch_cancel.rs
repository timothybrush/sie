//! Worker-local cancellation state for abandoned direct-dispatch batch work.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::subject::normalize_model_id;

const BATCH_CANCEL_TTL: Duration = Duration::from_secs(120);

#[derive(Clone, Default)]
pub struct BatchCancelState {
    inner: Arc<Mutex<HashMap<String, Instant>>>,
}

impl BatchCancelState {
    pub(crate) fn cancel(&self, request_id: String) {
        let now = Instant::now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        prune_expired(&mut guard, now);
        guard.insert(request_id, now + BATCH_CANCEL_TTL);
    }

    pub(crate) fn is_cancelled(&self, request_id: &str) -> bool {
        let now = Instant::now();
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        prune_expired(&mut guard, now);
        guard.contains_key(request_id)
    }
}

fn prune_expired(entries: &mut HashMap<String, Instant>, now: Instant) {
    entries.retain(|_, expires_at| *expires_at > now);
}

pub(crate) fn request_id_from_batch_cancel_subject(
    subject: &str,
    local_worker_id: &str,
) -> Option<String> {
    let mut parts = subject.splitn(4, '.');
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some("batch_cancel"), Some(_router_id), Some(worker_id), Some(request_id))
            if worker_id == normalize_worker_id(local_worker_id) && !request_id.is_empty() =>
        {
            Some(request_id.to_string())
        }
        _ => None,
    }
}

fn normalize_worker_id(worker_id: &str) -> String {
    normalize_model_id(worker_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_cancel_subject_parser_filters_worker_id() {
        assert_eq!(
            request_id_from_batch_cancel_subject("batch_cancel.gw.worker-1.req-1", "worker-1"),
            Some("req-1".to_string())
        );
        assert_eq!(
            request_id_from_batch_cancel_subject(
                "batch_cancel.gw.worker-1_dot_svc.req.with.dots",
                "worker-1.svc"
            ),
            Some("req.with.dots".to_string())
        );
        assert_eq!(
            request_id_from_batch_cancel_subject("batch_cancel.gw.worker-2.req-1", "worker-1"),
            None
        );
        assert_eq!(
            request_id_from_batch_cancel_subject("batch_cancel.gw.worker-1", "worker-1"),
            None
        );
    }

    #[test]
    fn batch_cancel_state_marks_request_ids() {
        let state = BatchCancelState::default();
        assert!(!state.is_cancelled("req-1"));
        state.cancel("req-1".to_string());
        assert!(state.is_cancelled("req-1"));
        assert!(!state.is_cancelled("req-2"));
    }
}
