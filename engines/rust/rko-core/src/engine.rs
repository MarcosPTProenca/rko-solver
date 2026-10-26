//! Cooperative portfolio scheduler.
//!
//! Objective:
//! - Run several metaheuristics concurrently against one problem and one elite pool,
//!   and report each search's outcome and the global incumbent.
//!
//! Technical approach:
//! - `std::thread::scope` workers; searches are assigned round-robin. A worker with
//!   several searches gives each an equal share of its remaining time.
//! - Each search owns a `Search`, an `Rng` derived from `(seed, index)` and a
//!   `Context`; only the lock-light `Pool` and a cancel flag are shared.
//! - A failing search cancels its peers; panics are converted into errors.
//!
//! Flow (pseudocode):
//! 1. Derive per-search seeds and distribute searches over workers.
//! 2. Each worker runs its searches until `Stop`, recording counters and history.
//! 3. Join, pick the best incumbent across searches and the pool.
//!
//! Important criteria:
//! - Every search ends no later than the common deadline (plus one evaluation).
//! - Results are deterministic only for a fixed seed on a single worker; concurrent
//!   pool exchange makes multi-worker trajectories timing dependent.
//!
//! Inputs/Outputs:
//! - Input: a `Problem`, algorithm `Spec`s and a `Config`.
//! - Output: `PortfolioResult` with best keys/cost and per-search reports.

pub use crate::config::Config;
use crate::mh::{Context, Spec};
use crate::pool::Pool;
use crate::problem::Problem;
use crate::rng::{Rng, derive_seed};
use crate::search::{HistoryPoint, Search, Stop};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Work done filling the shared pool before the searches.
#[derive(Clone, Copy, Debug)]
pub struct Initialization {
    /// Seed of the initialization stream.
    pub seed: u64,
    /// Completed finite evaluations.
    pub evaluations: u64,
    /// Seconds spent.
    pub elapsed: f64,
    /// Pool occupancy afterwards.
    pub pool_len: usize,
}

/// Outcome of one search.
#[derive(Clone, Debug)]
pub struct SearchReport {
    /// Position in the submitted specs.
    pub index: usize,
    /// Algorithm label.
    pub algorithm: &'static str,
    /// Seed used by this search.
    pub seed: u64,
    /// Completed finite evaluations.
    pub evaluations: u64,
    /// Seconds spent in this search.
    pub elapsed: f64,
    /// Seconds this search was allotted.
    pub budget: f64,
    /// Why the search stopped.
    pub stop: Stop,
    /// Best cost found by this search.
    pub best_cost: f64,
    /// Best keys found by this search.
    pub best_keys: Vec<f64>,
    /// Retained incumbent history.
    pub history: Vec<HistoryPoint>,
    /// Incumbent observations offered to the history.
    pub history_observations: u64,
}

/// Aggregated portfolio outcome.
#[derive(Clone, Debug)]
pub struct PortfolioResult {
    /// Best keys across searches.
    pub best_keys: Vec<f64>,
    /// Best cost across searches.
    pub best_cost: f64,
    /// Per-search reports in submission order.
    pub reports: Vec<SearchReport>,
    /// Wall-clock seconds for the whole run.
    pub elapsed: f64,
    /// Worker threads used.
    pub workers: usize,
    /// Pool prefill report, when requested.
    pub initialization: Option<Initialization>,
}

/// Failure of a portfolio run.
#[derive(Debug)]
pub enum EngineError {
    /// A search produced a non-finite objective.
    InvalidCost {
        /// Algorithm that observed it.
        algorithm: &'static str,
    },
    /// A worker thread panicked.
    Panic,
    /// The configuration was rejected.
    Invalid(String),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCost { algorithm } => {
                write!(
                    f,
                    "{algorithm}: objective callbacks must return finite costs"
                )
            }
            Self::Panic => write!(f, "a search worker panicked"),
            Self::Invalid(message) => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for EngineError {}

/// Runs the portfolio to completion.
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
    config.validate()?;
    for spec in specs {
        spec.validate().map_err(EngineError::Invalid)?;
    }
    let workers = config.threads.unwrap_or(specs.len()).clamp(1, specs.len());
    let started = Instant::now();
    let deadline = started + Duration::from_secs_f64(config.time_limit);
    let pool = Pool::new(config.pool_size);
    let cancel = AtomicBool::new(false);
    let failure: std::sync::Mutex<Option<&'static str>> = std::sync::Mutex::new(None);
    let mut initialization = None;
    if config.prefill {
        let seed = derive_seed(config.seed, u64::MAX);
        let mut rng = Rng::new(seed);
        let mut search = Search::new(problem, &cancel, started, deadline, 0);
        let mut keys = vec![0.0; problem.dimension()];
        let mut outcome = Ok(());
        problem.with_thread_context(&mut || {
            for _ in 0..config.pool_size {
                crate::operators::random_keys(&mut keys, &mut rng);
                match search.eval(&keys) {
                    Ok(cost) => {
                        pool.insert(cost, &keys, specs.len());
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
        initialization = Some(Initialization {
            seed,
            evaluations: search.evaluations(),
            elapsed: search.elapsed(),
            pool_len: pool.len(),
        });
    }

    let outcome = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|worker| {
                let assigned: Vec<usize> = (worker..specs.len()).step_by(workers).collect();
                let (pool, cancel, failure) = (&pool, &cancel, &failure);
                scope.spawn(move || {
                    let mut reports = Vec::with_capacity(assigned.len());
                    problem.with_thread_context(&mut || {
                        for (position, &index) in assigned.iter().enumerate() {
                            let remaining_searches = (assigned.len() - position) as u32;
                            let now = Instant::now();
                            let remaining = deadline.saturating_duration_since(now);
                            let share = remaining / remaining_searches;
                            let search_started = now;
                            let search_deadline = now + share;
                            let seed = derive_seed(config.seed, index as u64);
                            let mut rng = Rng::new(seed);
                            let mut search = Search::new(
                                problem,
                                cancel,
                                search_started,
                                search_deadline,
                                config.history_limit,
                            )
                            .with_max_evaluations(config.max_evaluations.unwrap_or(u64::MAX));
                            let spec = &specs[index];
                            let mut context = Context {
                                search: &mut search,
                                pool,
                                rng: &mut rng,
                                source: index,
                                strategy: config.ls_strategy,
                            };
                            let stop = match spec.run(&mut context) {
                                Err(stop) => stop,
                                Ok(()) => Stop::TimeLimit,
                            };
                            if stop == Stop::InvalidCost {
                                failure
                                    .lock()
                                    .expect("failure lock poisoned")
                                    .get_or_insert(spec.algorithm.name());
                                cancel.store(true, Ordering::Relaxed);
                            }
                            search.finish_history();
                            reports.push(SearchReport {
                                index,
                                algorithm: spec.algorithm.name(),
                                seed,
                                evaluations: search.evaluations(),
                                elapsed: search.elapsed(),
                                budget: search.budget(),
                                stop,
                                best_cost: search.best_cost(),
                                best_keys: search.best_keys().to_vec(),
                                history: search.history().to_vec(),
                                history_observations: search.history_observations(),
                            });
                        }
                    });
                    reports
                })
            })
            .collect();
        let mut collected = Vec::new();
        let mut panicked = false;
        for handle in handles {
            match handle.join() {
                Ok(reports) => collected.extend(reports),
                Err(_) => {
                    panicked = true;
                    cancel.store(true, Ordering::Relaxed);
                }
            }
        }
        (collected, panicked)
    });

    let (mut reports, panicked) = outcome;
    if panicked {
        return Err(EngineError::Panic);
    }
    if let Some(algorithm) = *failure.lock().expect("failure lock poisoned") {
        return Err(EngineError::InvalidCost { algorithm });
    }
    reports.sort_by_key(|report| report.index);
    let mut best_cost = f64::INFINITY;
    let mut best_keys = Vec::new();
    for report in &reports {
        if report.best_cost < best_cost {
            best_cost = report.best_cost;
            best_keys.clone_from(&report.best_keys);
        }
    }
    if let Some(entry) = pool.best()
        && entry.cost < best_cost
    {
        best_cost = entry.cost;
        best_keys = entry.keys;
    }
    Ok(PortfolioResult {
        best_keys,
        best_cost,
        reports,
        elapsed: started.elapsed().as_secs_f64(),
        workers,
        initialization,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mh::Algorithm;
    use crate::problem::Knapsack;
    use crate::qlearning::QLearning;
    use std::sync::{Mutex, MutexGuard};

    static SERIAL: Mutex<()> = Mutex::new(());

    fn serial() -> MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn instance() -> Knapsack {
        let profits: Vec<i64> = (0..80).map(|index| 10 + (index * 37) % 91).collect();
        let weights: Vec<i64> = (0..80).map(|index| 5 + (index * 53) % 47).collect();
        Knapsack::new(&profits, &weights, 900)
    }

    fn specs(adaptive: bool) -> Vec<Spec> {
        let brkga_axes = vec![
            vec![30.0, 50.0],
            vec![0.10, 0.20],
            vec![0.05, 0.10],
            vec![0.6, 0.7],
        ];
        let sa_axes = vec![
            vec![100.0, 250.0],
            vec![1e5],
            vec![0.97, 0.99],
            vec![0.01],
            vec![0.05],
        ];
        vec![
            Spec {
                algorithm: Algorithm::Brkga,
                offline: vec![50.0, 0.15, 0.10, 0.70],
                states: QLearning::grid_states(&brkga_axes, |_| true),
                adaptive,
            },
            Spec {
                algorithm: Algorithm::Sa,
                offline: vec![250.0, 1e6, 0.99, 0.01, 0.05],
                states: QLearning::grid_states(&sa_axes, |_| true),
                adaptive,
            },
        ]
    }

    fn config(threads: Option<usize>) -> Config {
        Config {
            time_limit: 0.4,
            seed: 5,
            pool_size: 10,
            threads,
            history_limit: 16,
            ls_strategy: 1,
            prefill: false,
            max_evaluations: None,
        }
    }

    #[test]
    fn portfolio_improves_over_random_and_reports_everything() {
        let _guard = serial();
        let problem = instance();
        for adaptive in [false, true] {
            let result = run_portfolio(&problem, &specs(adaptive), &config(None)).unwrap();
            assert_eq!(result.reports.len(), 2);
            assert_eq!(result.workers, 2);
            assert!(result.best_cost < -300.0, "cost {}", result.best_cost);
            assert_eq!(problem.evaluate(&result.best_keys), result.best_cost);
            for report in &result.reports {
                assert!(report.evaluations >= 1);
                assert_eq!(report.stop, Stop::TimeLimit);
                assert!(report.elapsed < 3.0);
                assert!(!report.history.is_empty());
                assert_eq!(problem.evaluate(&report.best_keys), report.best_cost);
            }
        }
    }

    fn small_spec(algorithm: Algorithm, adaptive: bool) -> Spec {
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
    fn every_algorithm_runs_fixed_and_adaptive_with_consistent_results() {
        let _guard = serial();
        let problem = instance();
        for adaptive in [false, true] {
            for strategy in [1, 2] {
                let specs: Vec<Spec> = Algorithm::all()
                    .into_iter()
                    .map(|algorithm| small_spec(algorithm, adaptive))
                    .collect();
                let mut settings = config(None);
                settings.time_limit = 0.5;
                settings.ls_strategy = strategy;
                let result = run_portfolio(&problem, &specs, &settings).unwrap();
                assert_eq!(result.reports.len(), 11);
                assert_eq!(problem.evaluate(&result.best_keys), result.best_cost);
                for report in &result.reports {
                    assert!(
                        report.evaluations >= 1,
                        "{} never evaluated",
                        report.algorithm
                    );
                    assert_eq!(report.stop, Stop::TimeLimit, "{}", report.algorithm);
                    assert!(
                        report.elapsed < 4.0,
                        "{} overran: {}",
                        report.algorithm,
                        report.elapsed
                    );
                    assert_eq!(problem.evaluate(&report.best_keys), report.best_cost);
                    assert!(report.best_keys.iter().all(|key| (0.0..1.0).contains(key)));
                }
                let best_known = result
                    .reports
                    .iter()
                    .map(|report| report.best_cost)
                    .fold(f64::INFINITY, f64::min);
                assert!(best_known < -400.0, "portfolio cost {best_known}");
            }
        }
    }

    #[test]
    fn evaluation_limit_stops_every_algorithm_exactly() {
        use crate::problem::test_support::BatchKnapsack;
        let _guard = serial();
        let scalar = instance();
        let blocks = BatchKnapsack::new(instance(), 16);
        let problems: [(&str, &dyn Problem); 2] = [("scalar", &scalar), ("blocks", &blocks)];
        for (label, problem) in problems {
            for adaptive in [false, true] {
                let specs: Vec<Spec> = Algorithm::all()
                    .into_iter()
                    .map(|algorithm| small_spec(algorithm, adaptive))
                    .collect();
                let mut settings = config(None);
                settings.time_limit = 20.0;
                settings.max_evaluations = Some(300);
                let result = run_portfolio(problem, &specs, &settings).unwrap();
                for report in &result.reports {
                    assert!(
                        report.evaluations <= 300,
                        "{label} {} used {}",
                        report.algorithm,
                        report.evaluations
                    );
                    assert_eq!(
                        report.stop,
                        Stop::MaxEvaluations,
                        "{label} {}",
                        report.algorithm
                    );
                    assert_eq!(report.evaluations, 300, "{label} {}", report.algorithm);
                    assert!(report.elapsed < 10.0);
                }
            }
        }
    }

    #[test]
    fn a_single_worker_with_a_limit_is_reproducible() {
        let _guard = serial();
        let problem = instance();
        let specs: Vec<Spec> = Algorithm::all()
            .into_iter()
            .map(|algorithm| small_spec(algorithm, false))
            .collect();
        let mut settings = config(Some(1));
        settings.time_limit = 30.0;
        settings.max_evaluations = Some(250);
        let first = run_portfolio(&problem, &specs, &settings).unwrap();
        let second = run_portfolio(&problem, &specs, &settings).unwrap();
        assert_eq!(first.best_cost, second.best_cost);
        assert_eq!(first.best_keys, second.best_keys);
        for (left, right) in first.reports.iter().zip(&second.reports) {
            assert_eq!(left.evaluations, right.evaluations, "{}", left.algorithm);
            assert_eq!(left.best_cost, right.best_cost, "{}", left.algorithm);
        }
    }

    #[test]
    fn a_zero_limit_is_rejected_and_prefill_ignores_the_limit() {
        let _guard = serial();
        let problem = instance();
        let mut settings = config(None);
        settings.max_evaluations = Some(0);
        assert!(matches!(
            run_portfolio(&problem, &specs(false), &settings),
            Err(EngineError::Invalid(message)) if message.contains("max_evaluations")
        ));
        settings.max_evaluations = Some(40);
        settings.prefill = true;
        settings.pool_size = 25;
        let result = run_portfolio(&problem, &specs(false), &settings).unwrap();
        assert_eq!(result.initialization.unwrap().evaluations, 25);
        assert!(result.reports.iter().all(|report| report.evaluations <= 40));
    }

    #[test]
    fn every_algorithm_runs_on_a_batch_scored_problem() {
        use crate::problem::test_support::BatchKnapsack;
        use std::sync::atomic::Ordering;
        let _guard = serial();
        let problem = BatchKnapsack::new(instance(), 16);
        for adaptive in [false, true] {
            let specs: Vec<Spec> = Algorithm::all()
                .into_iter()
                .map(|algorithm| small_spec(algorithm, adaptive))
                .collect();
            let mut settings = config(None);
            settings.time_limit = 0.5;
            let result = run_portfolio(&problem, &specs, &settings).unwrap();
            assert_eq!(result.reports.len(), 11);
            assert_eq!(problem.inner.evaluate(&result.best_keys), result.best_cost);
            for report in &result.reports {
                assert!(
                    report.evaluations >= 1,
                    "{} never evaluated",
                    report.algorithm
                );
                assert_eq!(report.stop, Stop::TimeLimit, "{}", report.algorithm);
                assert_eq!(
                    problem.inner.evaluate(&report.best_keys),
                    report.best_cost,
                    "{}",
                    report.algorithm
                );
            }
            assert!(
                problem.rows.load(Ordering::Relaxed) > 4 * problem.calls.load(Ordering::Relaxed),
                "blocks should carry several rows on average"
            );
        }
    }

    #[test]
    fn single_worker_shares_time_between_searches() {
        let _guard = serial();
        let problem = instance();
        let result = run_portfolio(&problem, &specs(true), &config(Some(1))).unwrap();
        assert_eq!(result.workers, 1);
        let elapsed: f64 = result.reports.iter().map(|report| report.elapsed).sum();
        assert!(elapsed < 2.0, "elapsed {elapsed}");
        assert!(result.reports.iter().all(|report| report.evaluations >= 1));
    }

    #[test]
    fn rejects_invalid_parameter_vectors_before_running() {
        let _guard = serial();
        let problem = instance();
        let mut spec = small_spec(Algorithm::Brkga, false);
        spec.offline = vec![1.0, 0.15, 0.10, 0.7];
        assert!(matches!(
            run_portfolio(&problem, &[spec.clone()], &config(None)),
            Err(EngineError::Invalid(message)) if message.contains("BRKGA")
        ));
        spec.offline = vec![20.0, 0.5, 0.5, 0.7];
        assert!(run_portfolio(&problem, &[spec.clone()], &config(None)).is_err());
        spec.offline = vec![20.0, 0.1];
        assert!(run_portfolio(&problem, &[spec], &config(None)).is_err());
        let mut sa = small_spec(Algorithm::Sa, true);
        sa.states = vec![vec![10.0, 1e5, 0.9, 0.5, 0.1]];
        assert!(run_portfolio(&problem, &[sa.clone()], &config(None)).is_err());
        sa.states.clear();
        assert!(run_portfolio(&problem, &[sa], &config(None)).is_err());
        for algorithm in Algorithm::all() {
            assert!(small_spec(algorithm, false).validate().is_ok());
            assert!(small_spec(algorithm, true).validate().is_ok());
        }
    }

    #[test]
    fn rejects_invalid_configuration() {
        let _guard = serial();
        let problem = instance();
        assert!(run_portfolio(&problem, &[], &config(None)).is_err());
        let mut bad = config(None);
        bad.time_limit = 0.0;
        assert!(run_portfolio(&problem, &specs(false), &bad).is_err());
    }

    #[test]
    fn nonfinite_cost_cancels_and_errors() {
        let _guard = serial();
        struct Broken;
        impl Problem for Broken {
            fn dimension(&self) -> usize {
                10
            }
            fn evaluate(&self, _keys: &[f64]) -> f64 {
                f64::NAN
            }
        }
        let error = run_portfolio(&Broken, &specs(false), &config(None)).unwrap_err();
        assert!(matches!(error, EngineError::InvalidCost { .. }));
    }
}
