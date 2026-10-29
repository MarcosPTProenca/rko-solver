//! Command-line benchmark driver shared by the Rust and C++ engine binaries.
//!
//! Objective:
//! - Run a portfolio on a Knapsack or TSP instance file and print one JSON record,
//!   so the engine can be compared with the C++ reference without a Python host.
//!
//! Technical approach:
//! - Parses the RKO instance format, builds each requested `Spec` from a selectable
//!   parameter grid (`toml` mirrors the packaged Python files, `cpp` mirrors the C++
//!   `ParametersOnline.txt`) and calls the engine function it is given, so the same driver
//!   serves `rko_bench` and the C++ engine's `rko_bench_cpp`.
//!
//! Flow (pseudocode):
//! 1. Parse arguments and the instance.
//! 2. Build specs, filter infeasible online states and run the portfolio.
//! 3. Print profit, weight, feasibility, evaluation counts and timing as JSON.
//!
//! Important criteria:
//! - Feasibility and profit are recomputed independently from the returned keys.
//!
//! Inputs/Outputs:
//! - Input: `<instance> <seconds> [--problem kp|tsp|rastrigin] [--algo A,B] [--grid toml|cpp] [--seed N]
//!   [--threads N] [--fixed] [--prefill] [--strategy 1|2] [--targets gap:profit,...]`.
//! - Output: an `RKO_BENCH_RESULT` line with full-precision keys, then one summary JSON object;
//!   first feasible target hits are reported on standard error.

use crate::engine::{Config, EngineError, PortfolioResult};
use crate::mh::{Algorithm, Spec};
use crate::problem::{Knapsack, Problem, Rastrigin, Tsp};
use crate::qlearning::QLearning;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Function that runs a portfolio on some engine.
pub type Engine = fn(&dyn Problem, &[Spec], &Config) -> Result<PortfolioResult, EngineError>;

struct Tracked<'a> {
    inner: &'a Knapsack,
    targets: Vec<(u32, f64)>,
    floor: f64,
    reported: Vec<AtomicBool>,
    origin: Instant,
}

impl Problem for Tracked<'_> {
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    #[inline]
    fn evaluate(&self, keys: &[f64]) -> f64 {
        let (profit, weight) = self.inner.totals(keys);
        if profit >= self.floor && weight <= self.inner.capacity() {
            for ((gap, target), flag) in self.targets.iter().zip(&self.reported) {
                if profit >= *target && !flag.swap(true, Ordering::Relaxed) {
                    eprintln!(
                        "RKO_BENCH_TARGET gap={gap} elapsed_seconds={:.9}",
                        self.origin.elapsed().as_secs_f64()
                    );
                }
            }
        }
        self.inner.cost_from_totals(profit, weight)
    }
}

fn grid(algorithm: Algorithm, flavor: &str) -> (Vec<f64>, Vec<Vec<f64>>) {
    let cpp = flavor == "cpp";
    let axes =
        |values: &[&[f64]]| -> Vec<Vec<f64>> { values.iter().map(|axis| axis.to_vec()).collect() };
    match algorithm {
        Algorithm::Brkga if cpp => (
            vec![987.0, 0.20, 0.15, 0.65],
            axes(&[
                &[300.0, 400.0, 500.0],
                &[0.10, 0.15, 0.20],
                &[0.10, 0.15, 0.20],
                &[0.60, 0.65, 0.70],
            ]),
        ),
        Algorithm::Brkga => (
            vec![50.0, 0.15, 0.10, 0.70],
            axes(&[
                &[30.0, 50.0, 100.0],
                &[0.10, 0.15, 0.20],
                &[0.05, 0.10, 0.15],
                &[0.60, 0.70, 0.80],
            ]),
        ),
        Algorithm::BrkgaCs if cpp => (
            vec![987.0, 0.25, 0.03, 0.65, 0.6],
            axes(&[
                &[377.0, 610.0, 987.0],
                &[0.15, 0.20, 0.25],
                &[0.01, 0.02, 0.03],
                &[0.60, 0.65, 0.70],
            ]),
        ),
        Algorithm::BrkgaCs => (
            vec![50.0, 0.15, 0.10, 0.70, 0.6],
            axes(&[
                &[30.0, 50.0, 100.0],
                &[0.10, 0.15, 0.20],
                &[0.05, 0.10, 0.15],
                &[0.60, 0.70, 0.80],
            ]),
        ),
        Algorithm::Sa if cpp => (
            vec![1000.0, 1_000_000.0, 0.99, 0.01, 0.05],
            axes(&[
                &[250.0, 500.0, 750.0],
                &[100_000.0],
                &[0.97, 0.99, 0.99],
                &[0.03, 0.04, 0.05],
                &[0.08, 0.09, 0.10],
            ]),
        ),
        Algorithm::Sa => (
            vec![1000.0, 1_000_000.0, 0.99, 0.01, 0.05],
            axes(&[
                &[250.0, 500.0, 750.0, 1000.0],
                &[100_000.0, 500_000.0, 1_000_000.0],
                &[0.97, 0.98, 0.99],
                &[0.01, 0.03, 0.05],
                &[0.05, 0.08, 0.10],
            ]),
        ),
        Algorithm::Ils if cpp => (
            vec![0.10, 0.20],
            axes(&[&[0.05, 0.10, 0.15], &[0.15, 0.20, 0.25]]),
        ),
        Algorithm::Ils => (
            vec![0.10, 0.20],
            axes(&[&[0.05, 0.10, 0.20, 0.30], &[0.10, 0.20, 0.30, 0.40]]),
        ),
        Algorithm::Vns if cpp => (
            vec![5.0, 0.05],
            axes(&[&[5.0, 8.0, 10.0], &[0.02, 0.04, 0.05]]),
        ),
        Algorithm::Vns => (
            vec![5.0, 0.05],
            axes(&[&[3.0, 5.0, 7.0], &[0.05, 0.10, 0.20]]),
        ),
        Algorithm::Lns => (
            vec![0.10, 0.30, 10_000.0, 0.95],
            axes(&[
                &[0.05, 0.10, 0.15],
                &[0.15, 0.20, 0.25],
                &[1000.0, 10_000.0],
                &[0.95, 0.96],
            ]),
        ),
        Algorithm::Pso if cpp => (
            vec![200.0, 2.05, 2.05, 0.73],
            axes(&[
                &[100.0, 200.0, 300.0],
                &[1.95, 2.0, 2.05],
                &[1.95, 2.0, 2.05],
                &[0.73, 0.74, 0.75],
            ]),
        ),
        Algorithm::Pso => (
            vec![30.0, 2.0, 2.0, 0.70],
            axes(&[
                &[20.0, 30.0, 50.0],
                &[1.5, 2.0, 2.5],
                &[1.5, 2.0, 2.5],
                &[0.4, 0.7, 0.9],
            ]),
        ),
        Algorithm::Ga if cpp => (
            vec![400.0, 0.98, 0.005],
            axes(&[
                &[200.0, 400.0, 600.0],
                &[0.90, 0.95, 0.98],
                &[0.02, 0.03, 0.05],
            ]),
        ),
        Algorithm::Ga => (
            vec![50.0, 0.80, 0.02],
            axes(&[
                &[30.0, 50.0, 100.0],
                &[0.60, 0.80, 0.90],
                &[0.01, 0.02, 0.05],
            ]),
        ),
        Algorithm::Grasp => (
            vec![0.125, 0.00098],
            axes(&[
                &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                &[0.125, 0.0625],
                &[0.00098, 0.00049],
            ]),
        ),
        Algorithm::Ipr | Algorithm::MultiStart => (Vec::new(), Vec::new()),
    }
}

fn feasible(algorithm: Algorithm, state: &[f64]) -> bool {
    match algorithm {
        Algorithm::Brkga => {
            let population = state[0] as usize;
            let elite = ((population as f64 * state[1]) as usize).max(1);
            let mutants = (population as f64 * state[2]) as usize;
            elite + mutants < population
        }
        Algorithm::BrkgaCs => {
            let population = state[0] as usize;
            ((population as f64 * state[1]) as usize).max(1) < population
        }
        Algorithm::Sa => state[3] <= state[4],
        Algorithm::Ils | Algorithm::Lns => state[0] <= state[1],
        Algorithm::Grasp => state[2] <= state[1],
        _ => true,
    }
}

/// Parses `std::env::args`, runs the requested portfolio on `engine` and prints the result.
pub fn run(name: &str, engine: Engine) {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    if arguments.len() < 2 {
        eprintln!(
            "usage: {name} <instance> <seconds> [--problem kp|tsp|rastrigin] [--algo A,B] [--grid toml|cpp] [--seed N] [--threads N] [--fixed] [--prefill]"
        );
        std::process::exit(2);
    }
    let text = std::fs::read_to_string(&arguments[0]).expect("instance unreadable");
    let seconds: f64 = arguments[1].parse().expect("seconds must be a number");
    let mut kind = "kp".to_owned();
    let mut algos = "BRKGA".to_owned();
    let mut flavor = "toml".to_owned();
    let mut seed = 42_u64;
    let mut threads = None;
    let mut adaptive = true;
    let mut strategy = 1_u8;
    let mut prefill = false;
    let mut targets: Vec<(u32, f64)> = Vec::new();
    let mut index = 2;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--problem" => {
                kind.clone_from(&arguments[index + 1]);
                index += 1;
            }
            "--algo" => {
                algos.clone_from(&arguments[index + 1]);
                index += 1;
            }
            "--grid" => {
                flavor.clone_from(&arguments[index + 1]);
                index += 1;
            }
            "--seed" => {
                seed = arguments[index + 1].parse().expect("seed");
                index += 1;
            }
            "--threads" => {
                threads = Some(arguments[index + 1].parse().expect("threads"));
                index += 1;
            }
            "--fixed" => adaptive = false,
            "--prefill" => prefill = true,
            "--strategy" => {
                strategy = arguments[index + 1].parse().expect("strategy");
                index += 1;
            }
            "--targets" => {
                targets = arguments[index + 1]
                    .split(',')
                    .map(|pair| {
                        let (gap, profit) = pair.split_once(':').expect("gap:profit");
                        (gap.parse().expect("gap"), profit.parse().expect("profit"))
                    })
                    .collect();
                index += 1;
            }
            other => panic!("unknown argument {other}"),
        }
        index += 1;
    }
    let algos = if algos == "all" {
        "SA,ILS,VNS,BRKGA,BRKGA_CS,PSO,GA,LNS,GRASP,IPR".to_owned()
    } else {
        algos
    };
    let specs: Vec<Spec> = algos
        .split(',')
        .map(|name| {
            let algorithm = Algorithm::parse(name).expect("unknown algorithm");
            let (offline, axes) = grid(algorithm, &flavor);
            Spec {
                algorithm,
                offline,
                states: QLearning::grid_states(&axes, |state| feasible(algorithm, state)),
                adaptive,
            }
        })
        .collect();
    let config = Config {
        time_limit: seconds,
        seed,
        pool_size: 10,
        threads,
        ls_strategy: strategy,
        prefill,
        ..Config::default()
    };
    let (result, extra) = if kind == "rastrigin" {
        let function = Rastrigin::parse(&text).expect("instance malformed");
        let result = engine(&function, &specs, &config).expect("run failed");
        (result, String::new())
    } else if kind == "tsp" {
        let tsp = Tsp::parse(&text).expect("instance malformed");
        let result = engine(&tsp, &specs, &config).expect("run failed");
        (result, String::new())
    } else {
        let problem = Knapsack::parse(&text).expect("instance malformed");
        let floor = targets
            .iter()
            .map(|(_, profit)| *profit)
            .fold(f64::INFINITY, f64::min);
        let tracked = Tracked {
            inner: &problem,
            reported: targets.iter().map(|_| AtomicBool::new(false)).collect(),
            targets,
            floor,
            origin: Instant::now(),
        };
        let result = engine(&tracked, &specs, &config).expect("run failed");
        let (profit, weight, ok) = problem.solution(&result.best_keys);
        (
            result,
            format!("\"profit\":{profit},\"weight\":{weight},\"feasible\":{ok},"),
        )
    };
    let evaluations: u64 = result.reports.iter().map(|report| report.evaluations).sum();
    let keys: Vec<String> = result
        .best_keys
        .iter()
        .map(|key| format!("{key:?}"))
        .collect();
    println!(
        "RKO_BENCH_RESULT {{\"keys\":[{}],\"objective\":{:?}}}",
        keys.join(","),
        result.best_cost
    );
    println!(
        "{{\"algorithms\":\"{algos}\",\"seed\":{seed},{extra}\"cost\":{},\"evaluations\":{evaluations},\"elapsed\":{:.6},\"workers\":{}}}",
        result.best_cost, result.elapsed, result.workers
    );
}
