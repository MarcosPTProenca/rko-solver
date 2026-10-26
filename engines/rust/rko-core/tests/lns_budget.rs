//! Regression test: LNS must honor its budget when its temperature never enters the cooling loop.
//!
//! Objective:
//! - Prove that an offline LNS whose `t0` is below the 0.01 cooling floor still ends at the time
//!   limit instead of spinning forever in its outer loop.
//!
//! Technical approach:
//! - Run the portfolio on a worker thread and wait on a channel with a generous timeout, so a
//!   regression fails the test instead of hanging the whole suite.
//!
//! Flow (pseudocode):
//! 1. Build a Knapsack and an LNS spec with `t0 = 0.001`.
//! 2. Run the portfolio for 0.3 seconds on a worker thread.
//! 3. Require a result within 20 seconds that reports the time limit.
//!
//! Important criteria:
//! - Python and the C++ reference check the budget in this loop; Rust did not.
//!
//! Inputs/Outputs:
//! - Input: a small Knapsack and one LNS spec.
//! - Output: a `Stop::TimeLimit` report.

use rko_core::config::Config;
use rko_core::engine::run_portfolio;
use rko_core::mh::{Algorithm, Spec};
use rko_core::problem::Knapsack;
use rko_core::search::Stop;
use std::sync::mpsc::channel;
use std::thread::spawn;
use std::time::Duration;

#[test]
fn lns_with_a_temperature_below_the_cooling_floor_still_honors_the_budget() {
    let spec = Spec {
        algorithm: Algorithm::Lns,
        offline: vec![0.1, 0.3, 0.001, 0.95],
        states: Vec::new(),
        adaptive: false,
    };
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
        .expect("LNS never checked its budget");
    assert_eq!(result.reports[0].stop, Stop::TimeLimit);
    assert!(result.reports[0].elapsed < 5.0);
}
