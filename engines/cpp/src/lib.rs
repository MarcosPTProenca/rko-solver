//! Optional engine that runs the RKO C++ v2.0 reference algorithms.
//!
//! Objective:
//! - Offer the published C++ implementation behind the same problem and result types as the
//!   Rust engine, so a solver can switch engines with one setting and results can be compared
//!   like for like.
//!
//! Technical approach:
//! - The reference headers are vendored unchanged (see `vendor/rko_cpp_v2`, MIT license). A
//!   driver translation unit supplies the globals and the evaluation hook they expect, plus
//!   a watchdog that raises the reference's own stop flag at the deadline, and runs one OpenMP
//!   thread per algorithm.
//! - Parameters are passed in the reference's axis order and its online state graph is built
//!   from the same filtered state list the other engines use.
//! - Evaluations cross a C function pointer; the only `unsafe` code of this crate is that
//!   foreign-function boundary.
//!
//! Flow (pseudocode):
//! 1. Validate the specs and map their parameters to the reference's order.
//! 2. Prefill the pool as the reference does, start the watchdog and one thread per algorithm.
//! 3. Collect per-algorithm counters, the pool incumbent and map them to `PortfolioResult`.
//!
//! Important criteria:
//! - The engine exists only when built with `RKO_BUILD_CPP=1` on Linux; otherwise every call
//!   returns an error and `available` is false.
//! - Budgets are at least 0.05 s (the reference counts whole seconds, so a fractional budget
//!   becomes a shortened last second), runs are serialized per process, and the pool prefill
//!   is part of the reference and always happens.
//! - A per-algorithm evaluation limit is enforced by the driver: the stop flag the headers poll
//!   also reports a thread-local limit, and the first objective call after the limit throws a
//!   private exception that ends that algorithm (the reference's IPR polls only its own timer),
//!   so the count is exact while the headers stay byte-identical.
//! - The reference draws the descent neighborhood with an unseeded `rand()`, so that part of
//!   its behavior does not depend on the seed.
//!
//! Inputs/Outputs:
//! - Input: a `Problem`, algorithm `Spec`s and an engine `Config`.
//! - Output: a `PortfolioResult` or an `EngineError`.

use rko_core::engine::{Config, EngineError, PortfolioResult};
use rko_core::mh::{Algorithm, Spec};
use rko_core::problem::Problem;

/// Whether this build contains the C++ engine.
#[must_use]
pub fn available() -> bool {
    cfg!(rko_cpp)
}

/// Maps a parameter vector from the Rust engine's axis order to the reference's.
///
/// Simulated annealing stores `[sa_max, t0, alpha, beta_min, beta_max]` and the reference
/// reads `[sa_max, alpha, beta_min, beta_max, t0]`; offline GRASP gains the unused greedy
/// rate in front, and offline BRKGA-CS drops the correlation threshold the reference fixes.
#[must_use]
pub fn reference_order(algorithm: Algorithm, values: &[f64]) -> Vec<f64> {
    match algorithm {
        Algorithm::Sa if values.len() == 5 => {
            vec![values[0], values[2], values[3], values[4], values[1]]
        }
        Algorithm::Grasp if values.len() == 2 => vec![0.1, values[0], values[1]],
        Algorithm::BrkgaCs if values.len() == 5 => values[..4].to_vec(),
        _ => values.to_vec(),
    }
}

/// Runs the portfolio on the C++ reference algorithms.
pub fn run_portfolio(
    problem: &dyn Problem,
    specs: &[Spec],
    config: &Config,
) -> Result<PortfolioResult, EngineError> {
    if specs.is_empty() {
        return Err(EngineError::Invalid(
            "at least one algorithm is required".to_owned(),
        ));
    }
    for spec in specs {
        spec.validate().map_err(EngineError::Invalid)?;
    }
    config.validate()?;
    if !(config.time_limit.is_finite() && config.time_limit >= 0.05) {
        return Err(EngineError::Invalid(
            "the C++ engine needs a time_limit of at least 0.05 seconds".to_owned(),
        ));
    }
    if config.threads.is_some() {
        return Err(EngineError::Invalid(
            "the C++ engine runs one thread per algorithm and does not take a worker cap"
                .to_owned(),
        ));
    }
    if config.pool_size < 2 {
        return Err(EngineError::Invalid(
            "the C++ engine needs pool_size of at least 2".to_owned(),
        ));
    }
    if problem.dimension() < 10 && specs.iter().any(|spec| spec.algorithm == Algorithm::Ipr) {
        return Err(EngineError::Invalid(
            "the C++ reference's IPR needs at least 10 keys: its block size is dimension / 10 and \
             it divides by zero below that"
                .to_owned(),
        ));
    }
    imp::run(problem, specs, config)
}

#[cfg(not(rko_cpp))]
mod imp {
    use super::{Config, EngineError, PortfolioResult, Problem, Spec};

    pub(super) fn run(
        _problem: &dyn Problem,
        _specs: &[Spec],
        _config: &Config,
    ) -> Result<PortfolioResult, EngineError> {
        Err(EngineError::Invalid(
            "this build has no C++ engine; rebuild with RKO_BUILD_CPP=1 on Linux".to_owned(),
        ))
    }
}

#[cfg(rko_cpp)]
#[path = "ffi.rs"]
mod imp;

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(rko_cpp)]
    use rko_core::search::Stop;

    #[test]
    fn simulated_annealing_order_matches_the_reference_reader() {
        let values = [1000.0, 1_000_000.0, 0.99, 0.01, 0.05];
        assert_eq!(
            reference_order(Algorithm::Sa, &values),
            vec![1000.0, 0.99, 0.01, 0.05, 1_000_000.0]
        );
    }

    #[test]
    fn grasp_and_brkga_cs_offline_vectors_are_adapted() {
        assert_eq!(
            reference_order(Algorithm::Grasp, &[0.125, 0.001]),
            vec![0.1, 0.125, 0.001]
        );
        assert_eq!(
            reference_order(Algorithm::BrkgaCs, &[50.0, 0.15, 0.1, 0.7, 0.6]),
            vec![50.0, 0.15, 0.1, 0.7]
        );
        assert_eq!(reference_order(Algorithm::Ils, &[0.1, 0.2]), vec![0.1, 0.2]);
    }

    #[cfg(not(rko_cpp))]
    #[test]
    fn stub_reports_the_missing_engine() {
        use rko_core::problem::Knapsack;
        let problem = Knapsack::new(&[1, 2], &[1, 1], 2);
        let spec = Spec {
            algorithm: Algorithm::Ipr,
            offline: Vec::new(),
            states: Vec::new(),
            adaptive: false,
        };
        let config = Config {
            time_limit: 1.0,
            seed: 1,
            pool_size: 4,
            threads: None,
            history_limit: 0,
            ls_strategy: 1,
            prefill: false,
            max_evaluations: None,
        };
        assert!(!available());
        assert!(matches!(
            run_portfolio(&problem, &[spec], &config),
            Err(EngineError::Invalid(message)) if message.contains("RKO_BUILD_CPP")
        ));
    }

    #[cfg(rko_cpp)]
    #[test]
    fn history_is_recorded_bounded_and_ends_at_the_final_incumbent() {
        let problem = knapsack();
        let mut settings = config(1.0);
        settings.history_limit = 4;
        settings.max_evaluations = Some(300);
        let result = run_portfolio(&problem, &[spec(Algorithm::Sa, false)], &settings).unwrap();
        let report = &result.reports[0];
        assert!(report.history.len() >= 2 && report.history.len() <= 4);
        assert!(report.history_observations >= report.history.len() as u64);
        assert_eq!(report.history[0].evaluations, 1);
        let last = report.history.last().unwrap();
        assert_eq!(last.cost, report.best_cost);
        assert!(report.history.windows(2).all(|pair| {
            pair[0].cost >= pair[1].cost && pair[0].evaluations <= pair[1].evaluations
        }));
        settings.history_limit = 0;
        let silent = run_portfolio(&problem, &[spec(Algorithm::Sa, false)], &settings).unwrap();
        assert!(silent.reports[0].history.is_empty());
        assert_eq!(silent.reports[0].history_observations, 0);
    }

    fn config(time_limit: f64) -> Config {
        Config {
            time_limit,
            seed: 7,
            pool_size: 10,
            threads: None,
            history_limit: 0,
            ls_strategy: 1,
            prefill: false,
            max_evaluations: None,
        }
    }

    fn knapsack() -> rko_core::problem::Knapsack {
        let profits: Vec<i64> = (0..80).map(|index| 10 + (index * 37) % 91).collect();
        let weights: Vec<i64> = (0..80).map(|index| 5 + (index * 53) % 47).collect();
        rko_core::problem::Knapsack::new(&profits, &weights, 900)
    }

    fn spec(algorithm: Algorithm, adaptive: bool) -> Spec {
        use rko_core::qlearning::QLearning;
        let (offline, axes): (Vec<f64>, Vec<Vec<f64>>) = match algorithm {
            Algorithm::Sa => (
                vec![100.0, 1e5, 0.98, 0.01, 0.05],
                vec![
                    vec![50.0, 100.0],
                    vec![1e5],
                    vec![0.97, 0.99],
                    vec![0.01],
                    vec![0.05],
                ],
            ),
            Algorithm::Ils => (vec![0.1, 0.2], vec![vec![0.05, 0.1], vec![0.1, 0.2]]),
            Algorithm::Vns => (vec![3.0, 0.05], vec![vec![2.0, 3.0], vec![0.05, 0.1]]),
            Algorithm::Lns => (
                vec![0.1, 0.3, 1000.0, 0.95],
                vec![
                    vec![0.05, 0.1],
                    vec![0.15, 0.2],
                    vec![1000.0],
                    vec![0.95, 0.96],
                ],
            ),
            Algorithm::Brkga => (
                vec![30.0, 0.15, 0.10, 0.7],
                vec![
                    vec![20.0, 30.0],
                    vec![0.1, 0.2],
                    vec![0.05, 0.1],
                    vec![0.6, 0.7],
                ],
            ),
            Algorithm::BrkgaCs => (
                vec![30.0, 0.2, 0.05, 0.7, 0.6],
                vec![
                    vec![20.0, 30.0],
                    vec![0.1, 0.2],
                    vec![0.05, 0.1],
                    vec![0.6, 0.7],
                ],
            ),
            Algorithm::Pso => (
                vec![15.0, 2.0, 2.0, 0.7],
                vec![
                    vec![10.0, 15.0],
                    vec![1.5, 2.0],
                    vec![1.5, 2.0],
                    vec![0.4, 0.7],
                ],
            ),
            Algorithm::Ga => (
                vec![20.0, 0.8, 0.02],
                vec![vec![10.0, 20.0], vec![0.6, 0.8], vec![0.01, 0.02]],
            ),
            Algorithm::Grasp => (
                vec![0.125, 0.01],
                vec![vec![0.1, 0.5], vec![0.125, 0.0625], vec![0.01, 0.005]],
            ),
            Algorithm::Ipr | Algorithm::MultiStart => (Vec::new(), Vec::new()),
        };
        Spec {
            algorithm,
            offline,
            states: QLearning::grid_states(&axes, |_| true),
            adaptive: adaptive && !axes.is_empty(),
        }
    }

    #[test]
    fn ipr_on_fewer_than_ten_keys_is_rejected_instead_of_crashing_the_process() {
        let problem = rko_core::problem::Knapsack::new(&[1, 2, 3], &[1, 1, 1], 2);
        let specs = [spec(Algorithm::Ipr, false)];
        assert!(matches!(
            run_portfolio(&problem, &specs, &config(1.0)),
            Err(EngineError::Invalid(message)) if message.contains("at least 10 keys")
        ));
    }

    #[test]
    fn budgets_below_the_minimum_and_worker_caps_are_rejected() {
        let problem = knapsack();
        let specs = [spec(Algorithm::Sa, false)];
        assert!(matches!(
            run_portfolio(&problem, &specs, &config(0.01)),
            Err(EngineError::Invalid(message)) if message.contains("0.05")
        ));
        let mut capped = config(1.0);
        capped.threads = Some(2);
        assert!(run_portfolio(&problem, &specs, &capped).is_err());
        assert!(run_portfolio(&problem, &[], &config(1.0)).is_err());
    }

    #[cfg(rko_cpp)]
    #[test]
    fn every_algorithm_runs_fixed_and_adaptive_with_consistent_results() {
        use rko_core::problem::Problem;
        let problem = knapsack();
        assert!(available());
        for adaptive in [false, true] {
            let specs: Vec<Spec> = Algorithm::all()
                .into_iter()
                .map(|algorithm| spec(algorithm, adaptive))
                .collect();
            let result = run_portfolio(&problem, &specs, &config(1.0)).unwrap();
            assert_eq!(result.reports.len(), 11);
            assert_eq!(problem.evaluate(&result.best_keys), result.best_cost);
            assert!(
                result.best_cost < -300.0,
                "portfolio cost {}",
                result.best_cost
            );
            assert!(result.elapsed < 3.0, "elapsed {}", result.elapsed);
            let init = result.initialization.unwrap();
            assert_eq!(init.evaluations, 10);
            for report in &result.reports {
                assert!(
                    report.evaluations >= 1,
                    "{} never evaluated",
                    report.algorithm
                );
                assert!(
                    report.elapsed < 2.5,
                    "{} ran {}",
                    report.algorithm,
                    report.elapsed
                );
                assert!(report.best_keys.iter().all(|key| (0.0..=1.0).contains(key)));
                assert_eq!(problem.evaluate(&report.best_keys), report.best_cost);
            }
        }
    }

    #[cfg(rko_cpp)]
    #[test]
    fn the_evaluation_limit_is_exact_per_algorithm_and_leaves_peers_running() {
        let problem = knapsack();
        let specs: Vec<Spec> = Algorithm::all()
            .into_iter()
            .map(|algorithm| spec(algorithm, false))
            .collect();
        let mut settings = config(10.0);
        settings.max_evaluations = Some(400);
        let result = run_portfolio(&problem, &specs, &settings).unwrap();
        assert!(result.elapsed < 8.0, "elapsed {}", result.elapsed);
        for report in &result.reports {
            assert_eq!(report.evaluations, 400, "{}", report.algorithm);
            assert_eq!(report.stop, Stop::MaxEvaluations, "{}", report.algorithm);
            assert_eq!(
                problem.evaluate(&report.best_keys),
                report.best_cost,
                "{}",
                report.algorithm
            );
        }
        assert_eq!(result.initialization.unwrap().evaluations, 10);
        let mut zero = config(1.0);
        zero.max_evaluations = Some(0);
        assert!(run_portfolio(&problem, &specs, &zero).is_err());
    }

    #[cfg(rko_cpp)]
    #[test]
    fn fractional_budgets_end_near_their_deadline() {
        let problem = knapsack();
        let specs = [spec(Algorithm::Sa, false), spec(Algorithm::Brkga, false)];
        let result = run_portfolio(&problem, &specs, &config(0.3)).unwrap();
        assert!(result.elapsed < 0.9, "elapsed {}", result.elapsed);
        assert!(result.reports.iter().all(|report| report.evaluations >= 1));
    }

    #[cfg(rko_cpp)]
    #[test]
    fn nonfinite_objective_is_reported_and_runs_can_repeat() {
        struct Broken;
        impl Problem for Broken {
            fn dimension(&self) -> usize {
                10
            }
            fn evaluate(&self, _keys: &[f64]) -> f64 {
                f64::NAN
            }
        }
        let specs = [spec(Algorithm::Sa, false)];
        assert!(matches!(
            run_portfolio(&Broken, &specs, &config(1.0)),
            Err(EngineError::InvalidCost { .. })
        ));
        let problem = knapsack();
        for _ in 0..2 {
            assert!(run_portfolio(&problem, &specs, &config(1.0)).is_ok());
        }
    }
}
