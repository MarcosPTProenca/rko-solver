//! Budget and isolation contracts for the repository-only temperature estimator.
//!
//! Objective:
//! - Ensure calibration samples use the normal search evaluator and a fixed start.
//!
//! Technical approach:
//! - Use scalar objectives and exact evaluation limits, without timing assertions.
//!
//! Flow (pseudocode):
//! 1. Score a fixed start and sample independent shakes.
//! 2. Assert incomplete sampling stops at the exact evaluation limit.
//! 3. Verify sampled improvements reach the incumbent without changing the start.
//!
//! Important criteria:
//! - The 32 samples consume budget; no detached evaluator or random stream is permitted.
//!
//! Inputs/Outputs:
//! - Input: deterministic small objectives, seeds and evaluation limits.
//! - Output: assertions on evaluations, stop reasons and incumbents.

use rko_core::mh::Context;
use rko_core::mh::sa_temperature::{SAMPLE_COUNT, SaTemperature};
use rko_core::pool::Pool;
use rko_core::problem::Problem;
use rko_core::rng::Rng;
use rko_core::search::{Search, Stop};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

struct Objective(f64);

impl Problem for Objective {
    fn dimension(&self) -> usize {
        8
    }

    fn evaluate(&self, keys: &[f64]) -> f64 {
        self.0 * keys.iter().map(|key| key * key).sum::<f64>()
    }
}

fn sample(limit: u64, scale: f64) -> (Result<Option<f64>, Stop>, u64, f64) {
    let problem = Objective(scale);
    let cancel = AtomicBool::new(false);
    let now = Instant::now();
    let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(60), 0)
        .with_max_evaluations(limit);
    let start = [0.0; 8];
    let original = start;
    let start_cost = search.eval(&start).unwrap();
    let pool = Pool::new(1);
    let mut rng = Rng::new(42);
    let mut scratch = [0.0; 8];
    let mut context = Context {
        search: &mut search,
        pool: &pool,
        rng: &mut rng,
        source: 0,
        strategy: 1,
        progress_schedule: false,
        sa_t0_auto: false,
    };
    let result = SaTemperature::new(0.8).calibrate(
        &start,
        start_cost,
        [0.1, 0.9],
        &mut context,
        &mut scratch,
    );
    assert_eq!(start, original);
    assert_eq!(start_cost, 0.0);
    (result, search.evaluations(), search.best_cost())
}

#[test]
fn incomplete_calibration_stops_at_the_evaluation_limit() {
    for limit in [1, 4, 32] {
        let (result, count, _) = sample(limit, 1.0);
        assert_eq!(result, Err(Stop::MaxEvaluations));
        assert_eq!(count, limit);
    }
}

#[test]
fn complete_sampling_counts_all_neighbors_and_preserves_start() {
    let (result, count, best) = sample(100, 1.0);
    assert!(result.unwrap().unwrap() > 0.0);
    assert_eq!(count, 1 + SAMPLE_COUNT as u64);
    assert_eq!(best, 0.0);
}

#[test]
fn improvements_reach_the_incumbent_even_when_no_uphill_delta_exists() {
    let (result, count, best) = sample(100, -1.0);
    assert_eq!(result, Ok(None));
    assert_eq!(count, 1 + SAMPLE_COUNT as u64);
    assert!(best < 0.0);
}

#[test]
fn flat_objective_falls_back_after_budgeted_sampling() {
    let (result, count, best) = sample(100, 0.0);
    assert_eq!(result, Ok(None));
    assert_eq!(count, 1 + SAMPLE_COUNT as u64);
    assert_eq!(best, 0.0);
}
