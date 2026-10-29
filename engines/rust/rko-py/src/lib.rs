//! Python bindings for the native RKO engine.
//!
//! Objective:
//! - Expose `rko_core` to Python as `rko_solver._native` with two problem kinds: a built-in
//!   Knapsack evaluated entirely in Rust and a callback problem wrapping a Python
//!   decoder/objective pair.
//!
//! Technical approach:
//! - `run` releases the GIL for the whole portfolio; callback problems re-attach on
//!   each evaluation, so scalar callbacks serialize on the GIL while native problems
//!   run fully parallel.
//! - Python exceptions raised inside callbacks are stored, converted into a
//!   non-finite cost to unwind the search, and re-raised unchanged afterwards.
//! - Results cross the boundary once, as plain dictionaries and lists.
//!
//! Flow (pseudocode):
//! 1. Python builds a problem object and algorithm specs.
//! 2. `run` converts the specs, detaches from the GIL and calls the selected engine.
//! 3. Reports and the incumbent are returned as Python containers.
//!
//! Important criteria:
//! - A callback receives a fresh array each call, never a view of engine memory.
//! - Panics inside the engine become Python `RuntimeError`s, never abort the process.
//! - A non-finite cost raises the same `ValueError` text as the Python engine.
//!
//! Inputs/Outputs:
//! - Input: a problem object, specs, a time limit, a seed and execution settings.
//! - Output: a dictionary with best keys/cost and per-search reports.

mod compiled;

use compiled::{CompiledAdapter, CompiledProblem};
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rko_core::engine::{Config, EngineError, PortfolioResult};
use rko_core::mh::{Algorithm, Spec};
use rko_core::problem::{Knapsack, Problem};
use std::sync::Mutex;

/// Built-in 0/1 Knapsack evaluated without touching Python.
#[pyclass(frozen, module = "rko_solver._native")]
struct KnapsackProblem {
    inner: Knapsack,
}

#[pymethods]
impl KnapsackProblem {
    #[new]
    fn new(profits: Vec<i64>, weights: Vec<i64>, capacity: i64) -> PyResult<Self> {
        if profits.len() != weights.len() || profits.is_empty() {
            return Err(PyValueError::new_err(
                "profits and weights must be nonempty and of equal length",
            ));
        }
        Ok(Self {
            inner: Knapsack::new(&profits, &weights, capacity),
        })
    }

    fn evaluate(&self, keys: Vec<f64>) -> PyResult<f64> {
        if keys.len() != self.inner.dimension() {
            return Err(PyValueError::new_err(
                "keys length must match the item count",
            ));
        }
        Ok(self.inner.evaluate(&keys))
    }

    fn solution(&self, keys: Vec<f64>) -> PyResult<(i64, i64, bool)> {
        if keys.len() != self.inner.dimension() {
            return Err(PyValueError::new_err(
                "keys length must match the item count",
            ));
        }
        Ok(self.inner.solution(&keys))
    }

    #[getter]
    fn dimension(&self) -> usize {
        self.inner.dimension()
    }
}

/// Python decoder/objective pair evaluated through the interpreter.
#[pyclass(frozen, module = "rko_solver._native")]
struct CallbackProblem {
    decoder: Py<PyAny>,
    cost: Py<PyAny>,
    dimension: usize,
    failure: Mutex<Option<PyErr>>,
}

#[pymethods]
impl CallbackProblem {
    #[new]
    fn new(decoder: Py<PyAny>, cost: Py<PyAny>, dimension: usize) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("dimension must be positive"));
        }
        Ok(Self {
            decoder,
            cost,
            dimension,
            failure: Mutex::new(None),
        })
    }
}

impl CallbackProblem {
    fn call(&self, keys: &[f64]) -> Result<f64, PyErr> {
        Python::attach(|py| {
            let array = PyArray1::from_slice(py, keys);
            let decoded = self.decoder.bind(py).call1((array,))?;
            self.cost.bind(py).call1((decoded,))?.extract::<f64>()
        })
    }
}

/// Python objective scoring a `(rows, dimension)` array into `rows` costs per call.
#[pyclass(frozen, module = "rko_solver._native")]
struct BatchProblem {
    objective: Py<PyAny>,
    dimension: usize,
    block: usize,
    failure: Mutex<Option<PyErr>>,
}

#[pymethods]
impl BatchProblem {
    #[new]
    fn new(objective: Py<PyAny>, dimension: usize, block: usize) -> PyResult<Self> {
        if dimension == 0 || block == 0 {
            return Err(PyValueError::new_err(
                "dimension and batch size must be positive",
            ));
        }
        Ok(Self {
            objective,
            dimension,
            block,
            failure: Mutex::new(None),
        })
    }
}

impl BatchProblem {
    fn call(&self, keys: &[f64], out: &mut [f64]) -> Result<(), PyErr> {
        Python::attach(|py| {
            let rows = out.len();
            let matrix = PyArray1::from_slice(py, keys).reshape([rows, self.dimension])?;
            let reply = self.objective.bind(py).call1((matrix,))?;
            if let Ok(array) = reply.extract::<PyReadonlyArray1<f64>>() {
                let costs = array.as_array();
                if costs.len() != rows {
                    return Err(PyValueError::new_err(format!(
                        "batch objective returned {} costs for {rows} rows",
                        costs.len()
                    )));
                }
                for (slot, &value) in out.iter_mut().zip(costs.iter()) {
                    *slot = value;
                }
                return Ok(());
            }
            let costs = reply.extract::<Vec<f64>>()?;
            if costs.len() != rows {
                return Err(PyValueError::new_err(format!(
                    "batch objective returned {} costs for {rows} rows",
                    costs.len()
                )));
            }
            out.copy_from_slice(&costs);
            Ok(())
        })
    }
}

struct BatchAdapter<'a>(&'a BatchProblem);

impl Problem for BatchAdapter<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension
    }

    fn evaluate(&self, keys: &[f64]) -> f64 {
        let mut cost = [f64::NAN];
        self.evaluate_batch(keys, &mut cost);
        cost[0]
    }

    fn evaluate_batch(&self, keys: &[f64], out: &mut [f64]) {
        if let Err(error) = self.0.call(keys, out) {
            self.0
                .failure
                .lock()
                .expect("failure lock poisoned")
                .get_or_insert(error);
            out.fill(f64::NAN);
        }
    }

    fn batch_size(&self) -> usize {
        self.0.block
    }

    fn check_stride(&self) -> u32 {
        1
    }

    fn with_thread_context(&self, body: &mut dyn FnMut()) {
        Python::attach(|_| body());
    }
}

struct CallbackAdapter<'a>(&'a CallbackProblem);

impl Problem for CallbackAdapter<'_> {
    fn dimension(&self) -> usize {
        self.0.dimension
    }

    fn evaluate(&self, keys: &[f64]) -> f64 {
        match self.0.call(keys) {
            Ok(value) => value,
            Err(error) => {
                self.0
                    .failure
                    .lock()
                    .expect("failure lock poisoned")
                    .get_or_insert(error);
                f64::NAN
            }
        }
    }

    fn check_stride(&self) -> u32 {
        1
    }

    fn with_thread_context(&self, body: &mut dyn FnMut()) {
        Python::attach(|_| body());
    }
}

type SpecTuple = (String, Vec<f64>, Vec<Vec<f64>>, bool);

fn parse_specs(specs: Vec<SpecTuple>) -> PyResult<Vec<Spec>> {
    specs
        .into_iter()
        .map(|(name, offline, states, adaptive)| {
            let algorithm = Algorithm::parse(&name)
                .ok_or_else(|| PyValueError::new_err(format!("Unknown algorithm: {name}")))?;
            Ok(Spec {
                algorithm,
                offline,
                states,
                adaptive,
            })
        })
        .collect()
}

fn result_to_dict<'py>(py: Python<'py>, result: &PortfolioResult) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("keys", result.best_keys.clone())?;
    output.set_item("cost", result.best_cost)?;
    output.set_item("elapsed", result.elapsed)?;
    output.set_item("workers", result.workers)?;
    if let Some(init) = &result.initialization {
        let entry = PyDict::new(py);
        entry.set_item("seed", init.seed)?;
        entry.set_item("evaluations", init.evaluations)?;
        entry.set_item("elapsed", init.elapsed)?;
        entry.set_item("pool_len", init.pool_len)?;
        output.set_item("initialization", entry)?;
    }
    let reports = PyList::empty(py);
    for report in &result.reports {
        let entry = PyDict::new(py);
        entry.set_item("algorithm", report.algorithm)?;
        entry.set_item("seed", report.seed)?;
        entry.set_item("evaluations", report.evaluations)?;
        entry.set_item("elapsed", report.elapsed)?;
        entry.set_item("budget", report.budget)?;
        entry.set_item("stop", report.stop.label())?;
        entry.set_item("best_cost", report.best_cost)?;
        entry.set_item("best_keys", report.best_keys.clone())?;
        let history: Vec<(f64, u64, f64)> = report
            .history
            .iter()
            .map(|point| (point.elapsed, point.evaluations, point.cost))
            .collect();
        entry.set_item("history", history)?;
        entry.set_item("history_observations", report.history_observations)?;
        reports.append(entry)?;
    }
    output.set_item("reports", reports)?;
    Ok(output)
}

fn engine_error(error: &EngineError) -> PyErr {
    match error {
        EngineError::Invalid(message) => PyValueError::new_err(message.clone()),
        EngineError::InvalidCost { .. } => PyValueError::new_err(error.to_string()),
        EngineError::Panic => PyRuntimeError::new_err(error.to_string()),
    }
}

/// Runs a portfolio on the selected engine: the Rust engine or the C++ reference engine.
fn execute(
    engine: &str,
    problem: &dyn Problem,
    specs: &[Spec],
    config: &Config,
) -> Result<PortfolioResult, EngineError> {
    match engine {
        "cpp" => rko_cpp::run_portfolio(problem, specs, config),
        _ => rko_core::engine::run_portfolio(problem, specs, config),
    }
}

/// Whether this build includes the C++ reference engine.
#[pyfunction]
fn cpp_available() -> bool {
    rko_cpp::available()
}

/// Runs a native portfolio and returns a result dictionary.
#[pyfunction]
#[pyo3(signature = (problem, specs, time_limit, seed, pool_size, threads=None, history_limit=0, ls_strategy=1, prefill=false, engine="rust", max_evaluations=None, initial_keys=None))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    problem: &Bound<'py, PyAny>,
    specs: Vec<SpecTuple>,
    time_limit: f64,
    seed: u64,
    pool_size: usize,
    threads: Option<usize>,
    history_limit: usize,
    ls_strategy: u8,
    prefill: bool,
    engine: &str,
    max_evaluations: Option<u64>,
    initial_keys: Option<Vec<Vec<f64>>>,
) -> PyResult<Bound<'py, PyDict>> {
    let py = problem.py();
    let specs = parse_specs(specs)?;
    let config = Config {
        time_limit,
        seed,
        pool_size,
        threads,
        history_limit,
        ls_strategy,
        prefill,
        max_evaluations,
        initial_keys: initial_keys.unwrap_or_default(),
    };
    if let Ok(knapsack) = problem.cast::<KnapsackProblem>() {
        let knapsack = knapsack.get();
        let outcome = py.detach(|| execute(engine, &knapsack.inner, &specs, &config));
        return match outcome {
            Ok(result) => result_to_dict(py, &result),
            Err(error) => Err(engine_error(&error)),
        };
    }
    if let Ok(callback) = problem.cast::<CallbackProblem>() {
        let callback = callback.get();
        let adapter = CallbackAdapter(callback);
        let outcome = py.detach(|| execute(engine, &adapter, &specs, &config));
        let stored = callback
            .failure
            .lock()
            .expect("failure lock poisoned")
            .take();
        if let Some(error) = stored {
            return Err(error);
        }
        return match outcome {
            Ok(result) => result_to_dict(py, &result),
            Err(error) => Err(engine_error(&error)),
        };
    }
    if let Ok(batch) = problem.cast::<BatchProblem>() {
        let batch = batch.get();
        let adapter = BatchAdapter(batch);
        let outcome = py.detach(|| execute(engine, &adapter, &specs, &config));
        let stored = batch.failure.lock().expect("failure lock poisoned").take();
        if let Some(error) = stored {
            return Err(error);
        }
        return match outcome {
            Ok(result) => result_to_dict(py, &result),
            Err(error) => Err(engine_error(&error)),
        };
    }
    if let Ok(compiled) = problem.cast::<CompiledProblem>() {
        let adapter = CompiledAdapter(compiled.get());
        let outcome = py.detach(|| execute(engine, &adapter, &specs, &config));
        return match outcome {
            Ok(result) => result_to_dict(py, &result),
            Err(error) => Err(engine_error(&error)),
        };
    }
    Err(PyValueError::new_err(
        "problem must be a KnapsackProblem, CallbackProblem, BatchProblem or CompiledProblem",
    ))
}

/// Native engine module.
#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<KnapsackProblem>()?;
    module.add_class::<CallbackProblem>()?;
    module.add_class::<BatchProblem>()?;
    module.add_class::<CompiledProblem>()?;
    module.add_function(wrap_pyfunction!(run, module)?)?;
    module.add_function(wrap_pyfunction!(cpp_available, module)?)?;
    Ok(())
}
