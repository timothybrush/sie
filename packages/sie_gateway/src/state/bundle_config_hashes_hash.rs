//! Last-known fingerprint of `sie-config`'s per-bundle config hashes.
//!
//! This is a sibling of [`crate::state::bundles_hash::BundlesHash`]. The
//! bundle-set hash answers "which bundles/adapters exist?", while this value
//! answers "which per-bundle config hashes has the gateway installed from the
//! control plane?". Both can drift while the model-write epoch stays at 0 in
//! no-store or filesystem-baseline deployments.

use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default)]
pub struct BundleConfigHashesHash {
    inner: Arc<Mutex<String>>,
}

impl BundleConfigHashesHash {
    pub fn new() -> Self {
        Self::default()
    }

    /// Current best-known hash. Empty means "never installed successfully" or
    /// "sie-config reported no registry state"; any non-empty remote value is
    /// treated as drift by the poller.
    pub fn get(&self) -> String {
        self.inner
            .lock()
            .expect("BundleConfigHashesHash mutex poisoned")
            .clone()
    }

    /// Replace the stored hash. Returns `true` if the value changed.
    pub fn store(&self, value: String) -> bool {
        let mut guard = self
            .inner
            .lock()
            .expect("BundleConfigHashesHash mutex poisoned");
        if *guard == value {
            return false;
        }
        *guard = value;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_empty() {
        let h = BundleConfigHashesHash::new();
        assert_eq!(h.get(), "");
    }

    #[test]
    fn store_returns_true_on_change_and_false_on_noop() {
        let h = BundleConfigHashesHash::new();
        assert!(h.store("abc".to_string()));
        assert_eq!(h.get(), "abc");
        assert!(!h.store("abc".to_string()));
        assert!(h.store("def".to_string()));
        assert_eq!(h.get(), "def");
    }

    #[test]
    fn clones_share_state() {
        let a = BundleConfigHashesHash::new();
        let b = a.clone();
        a.store("xyz".to_string());
        assert_eq!(b.get(), "xyz");
    }
}
