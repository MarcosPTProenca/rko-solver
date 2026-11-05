//! End-to-end production-parameter wiring in the repository-only native benchmark.
//!
//! Objective:
//! - Prove an exported offline specification drives the actual kernel, not just its metadata.
//!
//! Technical approach:
//! - Read packaged parameter values and compare the executable with direct core execution.
//! - Use a tiny synthetic function and 512 evaluations, never a scientific study input.
//!
//! Flow (pseudocode):
//! 1. Export the five packaged values in native order.
//! 2. Run the benchmark with fixed control, with and without calibration.
//! 3. Match returned keys/objective exactly and reject adaptive parameter injection.
//!
//! Important criteria:
//! - No public Python signature, existing default or study dataset changes.
//!
//! Inputs/Outputs:
//! - Input: packaged parameters and one synthetic eight-coordinate function.
//! - Output: exact wiring and refusal assertions.

use rko_core::config::Config;
use rko_core::engine::run_portfolio;
use rko_core::mh::{Algorithm, Spec};
use rko_core::problem::Rastrigin;
use std::process::Command;

const INSTANCE: &str = "8\n0.12\n-0.71\n0.34\n1.2\n-1.8\n0.63\n-0.2\n1.1\n";

fn packaged_values() -> Vec<f64> {
    let text = include_str!("../../../../src/rko_solver/parameters/sa.toml");
    ["sa_max", "t0", "alpha", "beta_min", "beta_max"]
        .iter()
        .map(|name| {
            text.lines()
                .find_map(|line| {
                    let (key, value) = line.split_once('=')?;
                    (key.trim() == *name).then(|| {
                        value
                            .split('#')
                            .next()
                            .unwrap()
                            .trim()
                            .parse::<f64>()
                            .unwrap()
                    })
                })
                .unwrap()
        })
        .collect()
}

#[test]
fn exported_values_drive_the_actual_kernel_and_parameter_echo() {
    let values = packaged_values();
    let encoded = values
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let path = std::env::temp_dir().join(format!("rko-specification-{}.txt", std::process::id()));
    std::fs::write(&path, INSTANCE).unwrap();
    let problem = Rastrigin::parse(INSTANCE).unwrap();
    let spec = Spec {
        algorithm: Algorithm::Sa,
        offline: values.clone(),
        states: Vec::new(),
        adaptive: false,
    };
    for calibrated in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_rko_bench"));
        command.arg(&path).args([
            "60",
            "--problem",
            "rastrigin",
            "--algo",
            "SA",
            "--fixed",
            "--threads",
            "1",
            "--seed",
            "42",
            "--evaluations",
            "512",
            "--sa-parameters",
            &encoded,
        ]);
        if calibrated {
            command.arg("--sa-t0-auto");
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let expected = run_portfolio(
            &problem,
            std::slice::from_ref(&spec),
            &Config {
                time_limit: 60.0,
                seed: 42,
                threads: Some(1),
                max_evaluations: Some(512),
                sa_t0_auto: calibrated,
                ..Config::default()
            },
        )
        .unwrap();
        let keys = expected
            .best_keys
            .iter()
            .map(|value| format!("{value:?}"))
            .collect::<Vec<_>>()
            .join(",");
        let expected_line = format!(
            "RKO_BENCH_RESULT {{\"keys\":[{keys}],\"objective\":{:?}}}",
            expected.best_cost
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.lines().any(|line| line == expected_line));
        assert!(stdout.contains(&format!("\"sa_parameters\":{values:?}")));
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn exported_offline_values_are_refused_in_adaptive_mode() {
    let path = std::env::temp_dir().join(format!(
        "rko-adaptive-specification-{}.txt",
        std::process::id()
    ));
    std::fs::write(&path, INSTANCE).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_rko_bench"))
        .arg(&path)
        .args([
            "60",
            "--problem",
            "rastrigin",
            "--algo",
            "SA",
            "--sa-parameters",
            "1027,340.3863933617792,0.8601360548373771,0.0035204893377555175,0.008242788398191462",
        ])
        .output()
        .unwrap();
    std::fs::remove_file(path).unwrap();
    assert_eq!(output.status.code(), Some(2));
}
