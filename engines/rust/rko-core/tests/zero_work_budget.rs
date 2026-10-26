//! Regression tests: annealing algorithms must honor their budget when no cooling step runs.
//!
//! Objective:
//! - Prove that an offline SA or LNS whose `t0` is below its cooling floor (0.0001 and 0.01)
//!   still ends at the time limit instead of spinning forever in its outer loop.
//!
//! Technical approach:
//! - Run the portfolio on a worker thread and wait on a channel with a generous timeout, so a
//!   regression fails the test instead of hanging the whole suite.
//!
//! Flow (pseudocode):
//! 1. Build a Knapsack and a spec whose `t0` is below the cooling floor.
//! 2. Run the portfolio for 0.3 seconds on a worker thread.
//! 3. Require a result within 20 seconds that reports the time limit.
//!
//! Important criteria:
//! - Python and the C++ reference check the budget in this loop; Rust did not.
//! - Other algorithms evaluate on every outer iteration, so only these two can do no work.
//!
//! Inputs/Outputs:
//! - Input: a small Knapsack and one SA or LNS spec.
//! - Output: a `Stop::TimeLimit` report.

use rko_core::config::Config;
use rko_core::engine::run_portfolio;
use rko_core::mh::{Algorithm, Spec};
use rko_core::problem::Knapsack;
use rko_core::search::Stop;
use std::sync::mpsc::channel;
use std::thread::spawn;
use std::time::Duration;

fn finishes_at_the_time_limit(spec: Spec, what: &str) {
    let (sender, receiver) = channel();
    spawn(move || {
        let profits: Vec<i64> = (0..30).map(|index| 10 + (index * 37) % 91).collect();
        let weights: Vec<i64> = (0..30).map(|index| 5 + (index * 53) % 47).collect();
        let problem = Knapsack::new(&profits, &weights, 400);
        let config = Config {
            time_limit: 0.3,
            seed: 5,
            pool_size: 10,
            threads: None,
            history_limit: 0,
            ls_strategy: 1,
            prefill: false,
            max_evaluations: None,
        };
        sender
            .send(run_portfolio(&problem, &[spec], &config).unwrap())
            .unwrap();
    });
    let result = receiver
        .recv_timeout(Duration::from_secs(20))
        .unwrap_or_else(|_| panic!("{what} never checked its budget"));
    assert_eq!(result.reports[0].stop, Stop::TimeLimit);
    assert!(result.reports[0].elapsed < 5.0);
}

#[test]
fn lns_with_a_temperature_below_the_cooling_floor_still_honors_the_budget() {
    let spec = Spec {
        algorithm: Algorithm::Lns,
        offline: vec![0.1, 0.3, 0.001, 0.95],
        states: Vec::new(),
        adaptive: false,
    };
    finishes_at_the_time_limit(spec, "LNS");
}

#[test]
fn sa_with_a_temperature_below_the_cooling_floor_still_honors_the_budget() {
    let spec = Spec {
        algorithm: Algorithm::Sa,
        offline: vec![100.0, 1e-300, 0.98, 0.01, 0.05],
        states: Vec::new(),
        adaptive: false,
    };
    finishes_at_the_time_limit(spec, "SA");
}
