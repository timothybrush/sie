//! Rolling trackers used by the adaptive controller:
//!
//! * [`BatchEfficiencyTracker`] — mean fill ratio
//!   `actual_cost / max_cost` over the last N batches.
//!
//! The latency tracker is *not* re-ported here; the controller shares
//! [`crate::latency::LatencyTracker`] since both ports use the same
//! fixed-window-plus-nearest-rank percentile implementation. See the
//! `scheduler::adaptive` module for how the two wire together.

use std::collections::VecDeque;

/// Rolling mean-fill-ratio tracker.
///
/// Ported from `sie_server/core/adaptive_batching.py::BatchEfficiencyTracker`.
///
/// A fill ratio near 1.0 means the GPU is fully saturated — batches
/// are consistently hitting the cost cap. Near 0 means the batcher
/// is flushing tiny batches; the adaptive cost knob should stay
/// conservative.
#[derive(Debug, Clone)]
pub struct BatchEfficiencyTracker {
    window_size: usize,
    ratios: VecDeque<f64>,
}

impl BatchEfficiencyTracker {
    /// Construct with an explicit window size. Python's default is 50;
    /// callers that want parity with the Python defaults should use
    /// [`Self::default`].
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            ratios: VecDeque::with_capacity(window_size),
        }
    }

    /// Record a batch fill ratio. `actual_cost` may be any cost unit;
    /// the ratio is taken against `max_cost`. Zero or negative
    /// `max_cost` is silently dropped (matches Python's guard) —
    /// adapters that haven't populated the cap yet shouldn't poison
    /// the mean.
    pub fn record(&mut self, actual_cost: u64, max_cost: u64) {
        if max_cost == 0 {
            return;
        }
        if self.ratios.len() == self.window_size {
            self.ratios.pop_front();
        }
        let ratio = actual_cost as f64 / max_cost as f64;
        self.ratios.push_back(ratio);
    }

    /// Mean of the current window, or `None` when nothing's been
    /// recorded yet. Matches Python's `mean_fill_ratio`.
    #[must_use]
    pub fn mean_fill_ratio(&self) -> Option<f64> {
        if self.ratios.is_empty() {
            return None;
        }
        let sum: f64 = self.ratios.iter().sum();
        Some(sum / self.ratios.len() as f64)
    }

    /// Number of samples currently in the window.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.ratios.len()
    }

    /// Clear all samples. Used by the controller on starvation
    /// recovery and on `reset()`.
    pub fn reset(&mut self) {
        self.ratios.clear();
    }
}

impl Default for BatchEfficiencyTracker {
    /// Default window of 50, matching Python.
    fn default() -> Self {
        Self::new(50)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tracker_reports_none() {
        let t = BatchEfficiencyTracker::default();
        assert_eq!(t.mean_fill_ratio(), None);
        assert_eq!(t.sample_count(), 0);
    }

    #[test]
    fn zero_max_cost_is_ignored() {
        let mut t = BatchEfficiencyTracker::default();
        t.record(10, 0);
        assert_eq!(t.sample_count(), 0);
    }

    #[test]
    fn mean_is_arithmetic_average() {
        let mut t = BatchEfficiencyTracker::default();
        t.record(1, 4); // 0.25
        t.record(3, 4); // 0.75
        let m = t.mean_fill_ratio().unwrap();
        assert!((m - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn window_drops_oldest_past_capacity() {
        let mut t = BatchEfficiencyTracker::new(2);
        t.record(0, 10); // 0.0
        t.record(10, 10); // 1.0
        t.record(5, 10); // 0.5 — evicts 0.0
        assert_eq!(t.sample_count(), 2);
        let m = t.mean_fill_ratio().unwrap();
        // Mean of (1.0 + 0.5) / 2 = 0.75
        assert!((m - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn reset_clears_window() {
        let mut t = BatchEfficiencyTracker::default();
        t.record(1, 2);
        t.reset();
        assert_eq!(t.sample_count(), 0);
        assert_eq!(t.mean_fill_ratio(), None);
    }
}
