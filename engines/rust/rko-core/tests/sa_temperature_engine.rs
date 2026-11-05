//! Engine-wiring contracts for the repository-only offline temperature experiment.
//!
//! Objective:
//! - Verify actual SA dispatch samples independent neighbors and rejects adaptive use.
//!
//! Technical approach:
//! - Capture the evaluated vectors and reconstruct the worker's deterministic random stream.
//!
//! Flow (pseudocode):
//! 1. Run offline SA for one start plus 32 calibration evaluations.
//! 2. Reproduce independent shakes from the unchanged start and compare every vector.
//! 3. Refuse the experimental flag for adaptive SA before evaluation.
//!
//! Important criteria:
//! - No Python option or adaptive-parameter behavior is introduced by this experiment.
//!
//! Inputs/Outputs:
//! - Input: a flat traced objective, fixed seed and explicit experiment configuration.
//! - Output: exact trajectory and validation assertions.

use rko_core::config::Config;
use rko_core::engine::{EngineError, run_portfolio};
use rko_core::mh::{Algorithm, Spec};
use rko_core::operators::{random_keys, shake};
use rko_core::problem::Problem;
use rko_core::rng::{Rng, derive_seed};
use rko_core::search::Stop;
use std::sync::Mutex;

struct Traced(Mutex<Vec<Vec<f64>>>);

impl Problem for Traced {
    fn dimension(&self) -> usize {
        8
    }

    fn evaluate(&self, keys: &[f64]) -> f64 {
        self.0.lock().unwrap().push(keys.to_vec());
        0.0
    }
}

fn spec(adaptive: bool) -> Spec {
    let parameters = vec![1000.0, 1e6, 0.99, 0.01, 0.05];
    Spec {
        algorithm: Algorithm::Sa,
        offline: parameters.clone(),
        states: vec![parameters],
        adaptive,
    }
}

fn config() -> Config {
    Config {
        time_limit: 60.0,
        seed: 7,
        threads: Some(1),
        max_evaluations: Some(33),
        sa_t0_auto: true,
        ..Config::default()
    }
}

#[test]
fn dispatch_samples_from_the_fixed_start_using_the_worker_stream() {
    let problem = Traced(Mutex::new(Vec::new()));
    let result = run_portfolio(&problem, &[spec(false)], &config()).unwrap();
    assert_eq!(result.reports[0].stop, Stop::MaxEvaluations);
    assert_eq!(result.reports[0].evaluations, 33);
    let mut rng = Rng::new(derive_seed(7, 0));
    let mut start = vec![0.0; 8];
    random_keys(&mut start, &mut rng);
    let mut expected = vec![start.clone()];
    for _ in 0..32 {
        let mut neighbor = start.clone();
        shake(&mut neighbor, 0.01, 0.05, &mut rng);
        expected.push(neighbor);
    }
    assert_eq!(*problem.0.lock().unwrap(), expected);
}

#[test]
fn adaptive_temperature_experiment_is_refused_before_evaluation() {
    let problem = Traced(Mutex::new(Vec::new()));
    assert!(matches!(
        run_portfolio(&problem, &[spec(true)], &config()),
        Err(EngineError::Invalid(message)) if message.contains("offline")
    ));
    assert!(problem.0.lock().unwrap().is_empty());
}
