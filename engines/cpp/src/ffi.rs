//! Foreign-function bridge to the C++ reference driver.
//!
//! Objective:
//! - Marshal specs and settings into the driver's C layout, run it, and map its counters,
//!   incumbents and histories back to `PortfolioResult`.
//!
//! Technical approach:
//! - Parameters are reordered to the reference's axis order and passed as raw slices; every
//!   output is a caller-owned buffer written through one `RawOutputs` struct.
//! - The objective crosses a C function pointer whose trampoline catches panics and maps them
//!   to a non-finite cost, which the driver reports as an invalid-cost failure.
//!
//! Flow (pseudocode):
//! 1. Build the raw specs, configuration and output buffers.
//! 2. Hold the process-wide run lock and call `rko_cpp_run`.
//! 3. Translate the status code and read the per-search reports and histories.
//!
//! Important criteria:
//! - Runs are serialized because the reference keeps its elite pool in a global.
//! - The history buffer holds `history_limit + 1` samples per search, the most the driver emits.
//! - This module holds the only `unsafe` code of the crate.
//!
//! Inputs/Outputs:
//! - Input: a `Problem`, algorithm `Spec`s and an engine `Config`.
//! - Output: a `PortfolioResult` or an `EngineError`.

#![allow(unsafe_code)]

use super::{Config, EngineError, PortfolioResult, Problem, Spec, reference_order};
use rko_core::engine::{Initialization, SearchReport};
use rko_core::search::{HistoryPoint, Stop};
use std::ffi::c_void;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::time::Instant;

static RUN_LOCK: Mutex<()> = Mutex::new(());

#[repr(C)]
struct RawSpec {
    algorithm: i32,
    adaptive: i32,
    offline: *const f64,
    offline_len: usize,
    states: *const f64,
    state_count: usize,
    state_len: usize,
}

#[repr(C)]
struct RawConfig {
    time_limit: f64,
    seed: u64,
    pool_size: i32,
    strategy: i32,
    max_evaluations: u64,
    history_limit: u64,
    dimension: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct RawReport {
    evaluations: u64,
    elapsed: f64,
    best_cost: f64,
    hit_limit: i32,
    history_len: u64,
    history_observations: u64,
}

#[repr(C)]
struct RawOutputs {
    reports: *mut RawReport,
    report_keys: *mut f64,
    initialization: *mut RawReport,
    best_cost: *mut f64,
    best_keys: *mut f64,
    history_out: *mut f64,
    error: *mut u8,
    error_len: usize,
}

type EvalFn = extern "C" fn(*const f64, usize, *mut c_void) -> f64;

unsafe extern "C" {
    fn rko_cpp_run(
        config: *const RawConfig,
        specs: *const RawSpec,
        spec_count: usize,
        eval: EvalFn,
        user: *mut c_void,
        initial_keys: *const f64,
        initial_key_count: usize,
        outputs: *const RawOutputs,
    ) -> i32;
}

struct Callback<'a> {
    problem: &'a dyn Problem,
}

extern "C" fn trampoline(keys: *const f64, size: usize, user: *mut c_void) -> f64 {
    let callback = unsafe { &*user.cast::<Callback<'_>>() };
    let row = unsafe { std::slice::from_raw_parts(keys, size) };
    catch_unwind(AssertUnwindSafe(|| callback.problem.evaluate(row))).unwrap_or(f64::NAN)
}

fn algorithm_code(spec: &Spec) -> i32 {
    rko_core::mh::Algorithm::all()
        .iter()
        .position(|candidate| *candidate == spec.algorithm)
        .map_or(0, |index| index as i32)
}

fn reference_inputs(specs: &[Spec]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let offline = specs
        .iter()
        .map(|spec| reference_order(spec.algorithm, &spec.offline))
        .collect();
    let states = specs
        .iter()
        .map(|spec| {
            spec.states
                .iter()
                .flat_map(|state| reference_order(spec.algorithm, state))
                .collect()
        })
        .collect();
    (offline, states)
}

fn raw_specs<'a>(
    specs: &'a [Spec],
    offline: &'a [Vec<f64>],
    states: &'a [Vec<f64>],
) -> Vec<RawSpec> {
    specs
        .iter()
        .enumerate()
        .map(|(index, spec)| RawSpec {
            algorithm: algorithm_code(spec),
            adaptive: i32::from(spec.adaptive),
            offline: offline[index].as_ptr(),
            offline_len: offline[index].len(),
            states: states[index].as_ptr(),
            state_count: spec.states.len(),
            state_len: spec
                .states
                .first()
                .map_or(0, |state| reference_order(spec.algorithm, state).len()),
        })
        .collect()
}

fn history_points(buffer: &[f64], count: u64) -> Vec<HistoryPoint> {
    buffer[..count as usize * 3]
        .as_chunks::<3>()
        .0
        .iter()
        .map(|point| HistoryPoint {
            elapsed: point[0],
            evaluations: point[1] as u64,
            cost: point[2],
        })
        .collect()
}

fn search_report(
    index: usize,
    spec: &Spec,
    config: &Config,
    raw: &RawReport,
    keys: &[f64],
    history: Vec<HistoryPoint>,
) -> SearchReport {
    SearchReport {
        index,
        algorithm: spec.algorithm.name(),
        seed: config.seed,
        evaluations: raw.evaluations,
        elapsed: raw.elapsed,
        budget: config.time_limit,
        stop: if raw.hit_limit != 0 {
            Stop::MaxEvaluations
        } else {
            Stop::TimeLimit
        },
        best_cost: raw.best_cost,
        best_keys: keys.to_vec(),
        history,
        history_observations: raw.history_observations,
    }
}

pub(super) fn run(
    problem: &dyn Problem,
    specs: &[Spec],
    config: &Config,
    initial_keys: &[Vec<f64>],
) -> Result<PortfolioResult, EngineError> {
    let dimension = problem.dimension();
    let (offline, states) = reference_inputs(specs);
    let raw_specs = raw_specs(specs, &offline, &states);
    let initial_key_values: Vec<f64> = initial_keys.iter().flatten().copied().collect();
    let raw_config = RawConfig {
        time_limit: config.time_limit,
        seed: config.seed,
        pool_size: i32::try_from(config.pool_size).unwrap_or(i32::MAX),
        strategy: i32::from(config.ls_strategy),
        max_evaluations: config.max_evaluations.unwrap_or(0),
        history_limit: config.history_limit as u64,
        dimension,
    };
    let mut reports = vec![RawReport::default(); specs.len()];
    let mut report_keys = vec![0.0_f64; specs.len() * dimension];
    let mut initialization = RawReport::default();
    let mut best_cost = f64::INFINITY;
    let mut best_keys = vec![0.0_f64; dimension];
    let history_stride = (config.history_limit + 1) * 3;
    let mut history_out = vec![0.0_f64; specs.len() * history_stride];
    let mut error = vec![0_u8; 256];
    let callback = Callback { problem };
    let _serialized = RUN_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let outputs = RawOutputs {
        reports: reports.as_mut_ptr(),
        report_keys: report_keys.as_mut_ptr(),
        initialization: &mut initialization,
        best_cost: &mut best_cost,
        best_keys: best_keys.as_mut_ptr(),
        history_out: history_out.as_mut_ptr(),
        error: error.as_mut_ptr(),
        error_len: error.len(),
    };
    let started = Instant::now();
    let status = unsafe {
        rko_cpp_run(
            &raw_config,
            raw_specs.as_ptr(),
            raw_specs.len(),
            trampoline,
            std::ptr::from_ref(&callback).cast_mut().cast::<c_void>(),
            initial_key_values.as_ptr(),
            initial_keys.len(),
            &outputs,
        )
    };
    let end = error
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(error.len());
    match status {
        0 => {}
        1 => {
            return Err(EngineError::InvalidCost {
                algorithm: "C++ reference",
            });
        }
        _ => {
            return Err(EngineError::Invalid(
                String::from_utf8_lossy(&error[..end]).into_owned(),
            ));
        }
    }
    let search_reports = specs
        .iter()
        .enumerate()
        .map(|(index, spec)| {
            let history = history_points(
                &history_out[index * history_stride..],
                reports[index].history_len,
            );
            let keys = &report_keys[index * dimension..(index + 1) * dimension];
            search_report(index, spec, config, &reports[index], keys, history)
        })
        .collect();
    Ok(PortfolioResult {
        best_keys,
        best_cost,
        reports: search_reports,
        elapsed: started.elapsed().as_secs_f64(),
        workers: specs.len(),
        initialization: Some(Initialization {
            seed: config.seed,
            evaluations: initialization.evaluations,
            elapsed: initialization.elapsed,
            pool_len: config.pool_size,
        }),
    })
}
