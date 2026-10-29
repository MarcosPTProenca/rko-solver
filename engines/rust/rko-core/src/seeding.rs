//! Startup of the shared elite pool: random prefill and caller-supplied keys.
//!
//! Objective:
//! - Fill the elite pool with evaluated vectors before any search starts, from the random
//!   prefill and from validated warm-start keys.
//!
//! Technical approach:
//! - `Seeding::validate` checks row dimension and the `[0, 1)` domain without touching the
//!   problem; `Seeding::start` runs the optional prefill and then evaluates every caller row
//!   once through one `Search`, inserting it under the pool's fixed-capacity, distinct-key
//!   admission rules.
//! - The startup `Search` has its own full-length deadline, so seeding never consumes the
//!   time budget of the searches that follow.
//!
//! Flow (pseudocode):
//! 1. Reject rows with a wrong length or a key outside `[0, 1)`.
//! 2. Optionally evaluate `pool_size` random vectors and report the work as `Initialization`.
//! 3. Evaluate each caller row through `Search::eval` inside the problem's thread context.
//! 4. Insert finite costs into the pool and map a stop into an `EngineError`.
//!
//! Important criteria:
//! - Every row is evaluated exactly once, including duplicates.
//! - A non-finite cost fails the run as `EngineError::InvalidCost` for `initial_keys`.
//!
//! Inputs/Outputs:
//! - Input: a `Problem`, a `Config`, the pool and the shared cancel flag.
//! - Output: the optional `Initialization` report or an `EngineError`.

use crate::engine::{Config, EngineError, Initialization};
use crate::pool::Pool;
use crate::problem::Problem;
use crate::rng::{Rng, derive_seed};
use crate::search::{Search, Stop};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

/// Validation and evaluation of caller-supplied warm-start keys.
pub struct Seeding;

impl Seeding {
    /// Checks every row for the problem dimension and finite values in `[0, 1)`.
    ///
    /// # Arguments
    ///
    /// * `keys` - Candidate key rows.
    /// * `dimension` - Number of random keys the problem expects.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::Invalid` when any row is malformed.
    pub fn validate(keys: &[Vec<f64>], dimension: usize) -> Result<(), EngineError> {
        let malformed = keys.iter().any(|row| {
            row.len() != dimension
                || row
                    .iter()
                    .any(|key| !key.is_finite() || !(0.0..1.0).contains(key))
        });
        if malformed {
            return Err(EngineError::Invalid(
                "'initial_keys' rows must match the problem dimension and contain finite values in [0, 1)."
                    .to_owned(),
            ));
        }
        Ok(())
    }

    /// Fills the pool from the random prefill and then from the caller-supplied rows.
    ///
    /// # Arguments
    ///
    /// * `problem` - Decoder/objective implementation.
    /// * `config` - Settings providing the prefill flag, seed, pool size, time limit and rows.
    /// * `searches` - Number of searches that will consume the pool.
    /// * `pool` - Shared elite pool receiving the evaluated vectors.
    /// * `cancel` - Shared cancel flag.
    ///
    /// # Returns
    ///
    /// The prefill report, or `None` when no prefill was requested.
    ///
    /// # Errors
    ///
    /// Returns `EngineError::InvalidCost` for a non-finite cost and `EngineError::Invalid` when
    /// the caller-row evaluation stops for another reason.
    pub fn start(
        problem: &dyn Problem,
        config: &Config,
        searches: usize,
        pool: &Pool,
        cancel: &AtomicBool,
    ) -> Result<Option<Initialization>, EngineError> {
        let initialization = if config.prefill {
            Some(Self::prefill(problem, config, searches, pool, cancel)?)
        } else {
            None
        };
        Self::offer(problem, config, searches, pool, cancel)?;
        Ok(initialization)
    }

    /// Evaluates `pool_size` random vectors into the pool and reports the work.
    fn prefill(
        problem: &dyn Problem,
        config: &Config,
        searches: usize,
        pool: &Pool,
        cancel: &AtomicBool,
    ) -> Result<Initialization, EngineError> {
        let seed = derive_seed(config.seed, u64::MAX);
        let mut rng = Rng::new(seed);
        let started = Instant::now();
        let deadline = started + Duration::from_secs_f64(config.time_limit);
        let mut search = Search::new(problem, cancel, started, deadline, 0);
        let mut keys = vec![0.0; problem.dimension()];
        let mut outcome = Ok(());
        problem.with_thread_context(&mut || {
            for _ in 0..config.pool_size {
                crate::operators::random_keys(&mut keys, &mut rng);
                match search.eval(&keys) {
                    Ok(cost) => {
                        pool.insert(cost, &keys, searches);
                    }
                    Err(stop) => {
                        outcome = Err(stop);
                        break;
                    }
                }
            }
        });
        if outcome == Err(Stop::InvalidCost) {
            return Err(EngineError::InvalidCost {
                algorithm: "initialization",
            });
        }
        Ok(Initialization {
            seed,
            evaluations: search.evaluations(),
            elapsed: search.elapsed(),
            pool_len: pool.len(),
        })
    }

    /// Evaluates each caller-supplied row once and offers it to the pool.
    fn offer(
        problem: &dyn Problem,
        config: &Config,
        searches: usize,
        pool: &Pool,
        cancel: &AtomicBool,
    ) -> Result<(), EngineError> {
        if config.initial_keys.is_empty() {
            return Ok(());
        }
        let started = Instant::now();
        let deadline = started + Duration::from_secs_f64(config.time_limit);
        let mut search = Search::new(problem, cancel, started, deadline, 0);
        let mut outcome = Ok(());
        problem.with_thread_context(&mut || {
            for row in &config.initial_keys {
                match search.eval(row) {
                    Ok(cost) => {
                        pool.insert(cost, row, searches);
                    }
                    Err(stop) => {
                        outcome = Err(stop);
                        break;
                    }
                }
            }
        });
        match outcome {
            Ok(()) => Ok(()),
            Err(Stop::InvalidCost) => Err(EngineError::InvalidCost {
                algorithm: "initial_keys",
            }),
            Err(stop) => Err(EngineError::Invalid(format!(
                "initial_keys evaluation stopped: {}",
                stop.label()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::run_portfolio;
    use crate::mh::{Algorithm, Spec};

    struct Bowl(f64);

    impl Problem for Bowl {
        fn dimension(&self) -> usize {
            4
        }
        fn evaluate(&self, keys: &[f64]) -> f64 {
            if self.0.is_nan() {
                return f64::NAN;
            }
            keys.iter().map(|key| (key - 0.25).powi(2)).sum()
        }
    }

    fn multistart() -> Vec<Spec> {
        vec![Spec {
            algorithm: Algorithm::MultiStart,
            offline: Vec::new(),
            states: Vec::new(),
            adaptive: false,
        }]
    }

    fn config(rows: Vec<Vec<f64>>) -> Config {
        Config {
            time_limit: 0.05,
            max_evaluations: Some(1),
            initial_keys: rows,
            ..Config::default()
        }
    }

    #[test]
    fn validate_rejects_wrong_length_and_out_of_domain_keys() {
        assert!(Seeding::validate(&[vec![0.1; 4]], 4).is_ok());
        for row in [
            vec![0.1; 3],
            vec![0.1, 0.1, 0.1, 1.0],
            vec![0.1, 0.1, 0.1, -0.5],
            vec![0.1, 0.1, 0.1, f64::NAN],
            vec![0.1, 0.1, 0.1, f64::INFINITY],
        ] {
            assert!(matches!(
                Seeding::validate(&[row], 4),
                Err(EngineError::Invalid(_))
            ));
        }
    }

    #[test]
    fn seeded_optimum_survives_a_minimal_search() {
        let result = run_portfolio(&Bowl(0.0), &multistart(), &config(vec![vec![0.25; 4]]))
            .expect("run succeeds");
        assert_eq!(result.best_cost, 0.0);
        assert_eq!(result.best_keys, vec![0.25; 4]);
    }

    #[test]
    fn duplicate_rows_are_each_evaluated_once() {
        let pool = Pool::new(4);
        let cancel = AtomicBool::new(false);
        let settings = config(vec![vec![0.5; 4], vec![0.5; 4], vec![0.1; 4]]);
        Seeding::start(&Bowl(0.0), &settings, 1, &pool, &cancel).expect("seeding succeeds");
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn nonfinite_seed_cost_is_rejected() {
        let error =
            run_portfolio(&Bowl(f64::NAN), &multistart(), &config(vec![vec![0.1; 4]])).unwrap_err();
        assert!(matches!(
            error,
            EngineError::InvalidCost {
                algorithm: "initial_keys"
            }
        ));
    }
}
