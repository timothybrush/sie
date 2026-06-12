//! Graceful shutdown utilities.
//!
//! One `Shutdown` broadcast channel signals every long-running task to
//! stop. `install_signal_handlers` registers SIGTERM (pod preemption) and
//! SIGINT (Ctrl-C) to trip it. Consumers either call `wait()` or use it
//! directly in `tokio::select!`.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;
use tracing::info;

/// Latched broadcast signal. Tripped by `fire()` exactly once (idempotent);
/// every subsequent `wait()` — including those registered *after* `fire()` —
/// returns immediately.
///
/// Previous implementation used a bare `tokio::sync::Notify` whose
/// `notify_waiters()` only wakes *currently parked* futures. A late waiter
/// (e.g. a task that finishes its current `select!` branch after SIGTERM and
/// only then hits the `shutdown.wait()` branch) would block forever. The
/// `AtomicBool` latch closes that hole: the fast path checks `fired` first
/// and bails out synchronously; the slow path subscribes to the `Notify`
/// *then* re-checks the flag to close the register-between-fire race.
pub struct Shutdown {
    fired: AtomicBool,
    notify: Notify,
}

impl Shutdown {
    pub fn new() -> Self {
        Self {
            fired: AtomicBool::new(false),
            notify: Notify::new(),
        }
    }

    /// Trip the shutdown signal. Idempotent — once tripped, subsequent
    /// `wait()` callers return immediately. Safe to call from any thread.
    pub fn fire(&self) {
        // Release ordering pairs with the Acquire load in `wait`/`is_fired`
        // so a waiter that observes `fired=true` is guaranteed to see every
        // memory write ordered-before the `fire()` call.
        self.fired.store(true, Ordering::Release);
        self.notify.notify_waiters();
    }

    /// Returns true if `fire()` has already been called.
    pub fn is_fired(&self) -> bool {
        self.fired.load(Ordering::Acquire)
    }

    /// Wait until `fire()` has been called. Multiple callers may wait
    /// concurrently; each wakes when `fire()` runs. Callers that register
    /// *after* `fire()` return immediately.
    pub async fn wait(&self) {
        // Fast path: already fired.
        if self.is_fired() {
            return;
        }
        // Slow path: subscribe, then re-check. The re-check closes the window
        // where `fire()` runs between our first load and the `notified()`
        // registration (Notify would lose that notification).
        let notified = self.notify.notified();
        tokio::pin!(notified);
        if self.is_fired() {
            return;
        }
        notified.await;
    }
}

impl Default for Shutdown {
    fn default() -> Self {
        Self::new()
    }
}

/// Install SIGTERM + SIGINT handlers that trip the given `Shutdown`.
///
/// On non-unix targets (only here to keep `cargo check --all-targets`
/// happy on CI images lacking `libc::SIGTERM`), this does nothing.
pub fn install_signal_handlers(shutdown: Arc<Shutdown>) {
    #[cfg(unix)]
    {
        tokio::spawn(async move {
            use tokio::signal::unix::{signal, SignalKind};
            let mut term = match signal(SignalKind::terminate()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to install SIGTERM handler");
                    return;
                }
            };
            let mut int_ = match signal(SignalKind::interrupt()) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!(error = %e, "failed to install SIGINT handler");
                    return;
                }
            };
            tokio::select! {
                _ = term.recv() => info!("received SIGTERM"),
                _ = int_.recv() => info!("received SIGINT"),
            }
            shutdown.fire();
        });
    }
    #[cfg(not(unix))]
    {
        let _ = shutdown;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn fire_wakes_all_waiters() {
        let shutdown = Arc::new(Shutdown::new());
        let a = Arc::clone(&shutdown);
        let b = Arc::clone(&shutdown);
        let task_a = tokio::spawn(async move { a.wait().await });
        let task_b = tokio::spawn(async move { b.wait().await });

        tokio::time::sleep(Duration::from_millis(10)).await;
        shutdown.fire();

        tokio::time::timeout(Duration::from_millis(100), task_a)
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(Duration::from_millis(100), task_b)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn wait_after_fire_returns_immediately() {
        // Regression: the old implementation used bare `Notify::notify_waiters`
        // which is edge-triggered — a waiter that registered *after* fire()
        // would block forever until a second fire(). With the latched
        // AtomicBool + Notify design, `wait()` is level-triggered: once fired,
        // every subsequent waiter returns immediately.
        let shutdown = Arc::new(Shutdown::new());
        shutdown.fire();

        tokio::time::timeout(Duration::from_millis(100), shutdown.wait())
            .await
            .expect("wait() must return immediately after fire()");
    }

    #[tokio::test]
    async fn race_fire_between_subscribe_and_await() {
        // The classic missed-wakeup: we call `wait()` on a thread that is
        // currently in the middle of `Notify::notified()` registration when
        // `fire()` runs on another thread. The re-check after subscribe
        // closes that window.
        for _ in 0..200 {
            let shutdown = Arc::new(Shutdown::new());
            let waiter = {
                let s = Arc::clone(&shutdown);
                tokio::spawn(async move { s.wait().await })
            };
            // Yield to give the waiter a chance to park.
            tokio::task::yield_now().await;
            shutdown.fire();
            tokio::time::timeout(Duration::from_millis(50), waiter)
                .await
                .expect("waiter must not block after fire()")
                .unwrap();
        }
    }

    #[tokio::test]
    async fn is_fired_reflects_state() {
        let shutdown = Shutdown::new();
        assert!(!shutdown.is_fired());
        shutdown.fire();
        assert!(shutdown.is_fired());
        shutdown.fire();
        assert!(shutdown.is_fired());
    }
}
