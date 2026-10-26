//! Portfolio execution settings and their validation.
//!
//! Objective:
//! - Hold the settings of one portfolio run and reject values no machine can satisfy before
//!   any thread starts, so a bad value is an error and never a panic or an allocation failure.
//!
//! Technical approach:
//! - `Config` is a plain data record; `Config::validate` applies fixed upper bounds that match
//!   the Python layer (`RunLimits`): one year of seconds, which also keeps the value convertible
//!   to a `Duration`, a pool of at most one million entries, and a positive evaluation limit.
//!
//! Flow (pseudocode):
//! 1. The caller builds a `Config`.
//! 2. `run_portfolio` calls `validate` before touching the problem.
//!
//! Important criteria:
//! - Bounds are generous and exist to stop absurd or hostile values, not real runs.
//! - Validation never mutates the configuration.
//!
//! Inputs/Outputs:
//! - Input: a `Config`.
//! - Output: `Ok(())` or `EngineError::Invalid` naming the offending setting.

use crate::engine::EngineError;

/// Largest accepted time limit in seconds (one year).
pub const MAX_TIME_LIMIT: f64 = 31_536_000.0;

/// Largest accepted elite pool capacity.
pub const MAX_POOL_SIZE: usize = 1_000_000;

/// Portfolio execution settings.
#[derive(Clone, Debug)]
pub struct Config {
    /// Search seconds shared by all searches.
    pub time_limit: f64,
    /// Base seed from which per-search streams are derived.
    pub seed: u64,
    /// Elite pool capacity.
    pub pool_size: usize,
    /// Worker threads; `None` runs one thread per search.
    pub threads: Option<usize>,
    /// Per-search incumbent history cap; zero disables it.
    pub history_limit: usize,
    /// Local-search sweep strategy: 1 sequential acceptance, 2 fixed-base best.
    pub ls_strategy: u8,
    /// Fill the pool with `pool_size` random evaluated vectors before searches start.
    pub prefill: bool,
    /// Completed evaluations after which each search stops; `None` leaves only the time
    /// limit. It applies to every search separately and not to the prefill.
    pub max_evaluations: Option<u64>,
}

impl Config {
    /// Checks every setting against its domain and upper bound.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::Invalid` when the time limit is not in `(0, MAX_TIME_LIMIT]`, the
    /// local-search strategy is not 1 or 2, the pool size is not in `1..=MAX_POOL_SIZE` or the
    /// evaluation limit is zero.
    pub fn validate(&self) -> Result<(), EngineError> {
        let invalid = |message: &str| Err(EngineError::Invalid(message.to_owned()));
        if !(self.time_limit.is_finite()
            && self.time_limit > 0.0
            && self.time_limit <= MAX_TIME_LIMIT)
        {
            return invalid("time_limit must be positive, finite and at most 31536000 seconds");
        }
        if !(1..=2).contains(&self.ls_strategy) {
            return invalid("ls_strategy must be 1 or 2");
        }
        if !(1..=MAX_POOL_SIZE).contains(&self.pool_size) {
            return invalid("pool_size must be from 1 to 1000000");
        }
        if self.max_evaluations == Some(0) {
            return invalid("max_evaluations must be positive");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid() -> Config {
        Config {
            time_limit: 1.0,
            seed: 1,
            pool_size: 10,
            threads: None,
            history_limit: 0,
            ls_strategy: 1,
            prefill: false,
            max_evaluations: None,
        }
    }

    fn message(config: &Config) -> String {
        match config.validate() {
            Err(EngineError::Invalid(text)) => text,
            other => panic!("expected an invalid configuration, got {other:?}"),
        }
    }

    #[test]
    fn accepts_the_boundaries() {
        let mut config = valid();
        config.time_limit = MAX_TIME_LIMIT;
        config.pool_size = MAX_POOL_SIZE;
        config.ls_strategy = 2;
        config.max_evaluations = Some(u64::MAX);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn rejects_times_that_would_overflow_a_duration() {
        for value in [
            0.0,
            -1.0,
            f64::NAN,
            f64::INFINITY,
            1e300,
            MAX_TIME_LIMIT * 1.0001,
        ] {
            let mut config = valid();
            config.time_limit = value;
            assert!(message(&config).contains("time_limit"), "{value}");
        }
    }

    #[test]
    fn rejects_bad_strategy_pool_and_limit() {
        let mut config = valid();
        config.ls_strategy = 3;
        assert!(message(&config).contains("ls_strategy"));
        let mut config = valid();
        config.pool_size = 0;
        assert!(message(&config).contains("pool_size"));
        config.pool_size = MAX_POOL_SIZE + 1;
        assert!(message(&config).contains("pool_size"));
        let mut config = valid();
        config.max_evaluations = Some(0);
        assert!(message(&config).contains("max_evaluations"));
    }
}
