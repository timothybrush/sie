//! Small helpers for structured logging.
//!
//! [`ErrChain`] renders an error together with its full `#[source]`
//! chain, so `warn!(error = %ErrChain(&e), ...)` shows everything from
//! the outermost user-facing message down to the root cause in one
//! structured field. Without this, the default `Display` for
//! `thiserror`-derived errors only shows the outermost layer — you lose
//! "caused by: connection refused" etc., which is exactly what you
//! need when diagnosing a problem from prod logs.

use std::error::Error;
use std::fmt;

/// Wrap an error so its `Display` walks the `#[source]` chain.
///
/// Each layer is joined with ` -> ` so the final rendering still fits
/// on a single log line (Loki / Cloud Logging both dislike newlines
/// inside structured fields). Example:
///
/// ```text
/// ipc call failed -> io error -> connection refused
/// ```
pub struct ErrChain<'a>(pub &'a dyn Error);

impl fmt::Display for ErrChain<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)?;
        let mut source = self.0.source();
        while let Some(cause) = source {
            write!(f, " -> {cause}")?;
            source = cause.source();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thiserror::Error;

    #[derive(Debug, Error)]
    enum Inner {
        #[error("inner boom")]
        Boom,
    }

    #[derive(Debug, Error)]
    enum Outer {
        #[error("outer wrap")]
        Wrap(#[source] Inner),
    }

    #[test]
    fn renders_full_chain() {
        let e = Outer::Wrap(Inner::Boom);
        let chain: &dyn Error = &e;
        let rendered = format!("{}", ErrChain(chain));
        assert_eq!(rendered, "outer wrap -> inner boom");
    }

    #[test]
    fn single_layer_renders_display_verbatim() {
        let e = Inner::Boom;
        let chain: &dyn Error = &e;
        assert_eq!(format!("{}", ErrChain(chain)), "inner boom");
    }

    #[test]
    fn deeper_chain_joined_with_arrow() {
        // Chain of three layers via std::io::Error wrapping.
        #[derive(Debug, Error)]
        enum Wrap {
            #[error("wrap")]
            Io(#[source] std::io::Error),
        }
        let io = std::io::Error::new(std::io::ErrorKind::BrokenPipe, "kernel said pipe");
        let outer = Wrap::Io(io);
        let chain: &dyn Error = &outer;
        assert_eq!(format!("{}", ErrChain(chain)), "wrap -> kernel said pipe");
    }
}
