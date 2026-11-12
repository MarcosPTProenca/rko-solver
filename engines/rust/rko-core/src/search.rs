//! Budgeted evaluation wrapper shared by every metaheuristic.
//!
//! Objective:
//! - Apply deadline, evaluation limit, cancellation, finite-cost validation and incumbent
//!   retention at the single boundary every operator uses to score a candidate.
//!
//! Technical approach:
//! - `Search::eval` counts completed evaluations, retains the best completed
//!   candidate and aborts with `Stop` once the budget is exhausted.
//! - An optional evaluation limit is exact: no evaluation beyond it is ever performed or
//!   counted, and a block call scores only the rows that fit before reporting the stop.
//! - Wall-clock reads are amortized over `check_stride` evaluations; the cancel flag
//!   is an atomic relaxed load on every call.
//! - Scalar incumbent samples are optionally retained in a bounded, deterministically
//!   compacted history for later reporting.
//!
//! Flow (pseudocode):
//! 1. Reject the call if cancelled, at the evaluation limit or, after the first valid result,
//!    past deadline.
//! 2. Score the keys and reject non-finite costs.
//! 3. Update the evaluation counter, incumbent and history.
//!
//! Important criteria:
//! - The first evaluation always completes even if the deadline already passed.
//! - A search is owned by one thread and is never shared.
//!
//! Inputs/Outputs:
//! - Input: a `Problem`, a duration or deadline and a shared cancel flag.
//! - Output: costs, a best-so-far incumbent and bounded history samples.

use crate::problem::{Incremental, Problem};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Reason a search stopped before completing naturally.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The wall-clock budget elapsed.
    TimeLimit,
    /// The configured number of evaluations was reached.
    MaxEvaluations,
    /// A peer failed or the caller requested cancellation.
    Cancelled,
    /// The objective returned a non-finite value.
    InvalidCost,
}

impl Stop {
    /// Stable machine-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::TimeLimit => "time_limit",
            Self::MaxEvaluations => "max_evaluations",
            Self::Cancelled => "cancelled",
            Self::InvalidCost => "invalid_cost",
        }
    }
}

/// One retained incumbent observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HistoryPoint {
    /// Seconds since the search started.
    pub elapsed: f64,
    /// Completed evaluations at the observation.
    pub evaluations: u64,
    /// Best cost at the observation.
    pub cost: f64,
}

/// Temperature schedule facts observed by one simulated-annealing search.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnnealingTrace {
    /// Initial temperature used by the first level, after calibration or the fixed fallback.
    pub t0: f64,
    /// Temperature at which the run stops cooling.
    pub cutoff: f64,
    /// Temperature levels entered, across restarts.
    pub levels: u64,
    /// Whether the start temperature came from the sampled estimator.
    pub calibrated: bool,
    /// Largest and median sampled uphill cost difference behind the estimator, when sampled.
    pub calibration: Option<(f64, f64)>,
}

/// Budget-aware evaluator and incumbent holder for one search.
pub struct Search<'a> {
    problem: &'a dyn Problem,
    cancel: &'a AtomicBool,
    deadline: Instant,
    started: Instant,
    stride: u32,
    tick: u32,
    evaluations: u64,
    max_evaluations: u64,
    best_cost: f64,
    best_keys: Vec<f64>,
    history_limit: usize,
    history: Vec<HistoryPoint>,
    history_observations: u64,
    incremental: Option<Box<dyn Incremental + 'a>>,
    annealing: Option<AnnealingTrace>,
    calibration_sample: Option<(f64, f64)>,
}

impl<'a> Search<'a> {
    /// Creates a search that ends at `deadline`.
    #[must_use]
    pub fn new(
        problem: &'a dyn Problem,
        cancel: &'a AtomicBool,
        started: Instant,
        deadline: Instant,
        history_limit: usize,
    ) -> Self {
        Self {
            problem,
            cancel,
            deadline,
            started,
            stride: problem.check_stride().max(1),
            tick: 0,
            evaluations: 0,
            max_evaluations: u64::MAX,
            best_cost: f64::INFINITY,
            best_keys: Vec::new(),
            history_limit,
            history: Vec::new(),
            history_observations: 0,
            incremental: None,
            annealing: None,
            calibration_sample: None,
        }
    }

    /// Caps the completed evaluations of this search at `limit` (at least one).
    #[must_use]
    pub fn with_max_evaluations(mut self, limit: u64) -> Self {
        self.max_evaluations = limit.max(1);
        self
    }

    /// Number of keys in the underlying problem.
    #[inline]
    #[must_use]
    pub fn dimension(&self) -> usize {
        self.problem.dimension()
    }

    /// Seconds elapsed since the search started.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> f64 {
        self.started.elapsed().as_secs_f64()
    }

    /// Total seconds between start and deadline.
    #[must_use]
    pub fn budget(&self) -> f64 {
        self.deadline
            .saturating_duration_since(self.started)
            .as_secs_f64()
    }

    /// Share of the time budget or evaluation limit already used, in `[0, 1]`.
    pub fn progress(&self) -> f64 {
        let used = self.elapsed() / self.budget().max(f64::MIN_POSITIVE);
        let share = self.evaluations as f64 / self.max_evaluations as f64;
        used.max(share).min(1.0)
    }

    /// Completed finite evaluations so far.
    #[must_use]
    pub fn evaluations(&self) -> u64 {
        self.evaluations
    }

    /// Best completed cost, or infinity before the first evaluation.
    #[must_use]
    pub fn best_cost(&self) -> f64 {
        self.best_cost
    }

    /// Best completed key vector (empty before the first evaluation).
    #[must_use]
    pub fn best_keys(&self) -> &[f64] {
        &self.best_keys
    }

    /// Checks cancellation and, after a first result, the deadline.
    #[inline]
    pub fn check(&mut self) -> Result<(), Stop> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Stop::Cancelled);
        }
        if self.evaluations >= self.max_evaluations {
            return Err(Stop::MaxEvaluations);
        }
        if self.best_cost.is_finite() && Instant::now() >= self.deadline {
            return Err(Stop::TimeLimit);
        }
        Ok(())
    }

    /// Scores `keys`, enforcing the budget and retaining the incumbent.
    #[inline]
    pub fn eval(&mut self, keys: &[f64]) -> Result<f64, Stop> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Stop::Cancelled);
        }
        if self.evaluations >= self.max_evaluations {
            return Err(Stop::MaxEvaluations);
        }
        self.tick += 1;
        if self.tick >= self.stride {
            self.tick = 0;
            if self.best_cost.is_finite() && Instant::now() >= self.deadline {
                return Err(Stop::TimeLimit);
            }
        }
        let value = self.problem.evaluate(keys);
        if !value.is_finite() {
            return Err(Stop::InvalidCost);
        }
        self.evaluations += 1;
        if value < self.best_cost {
            self.best_cost = value;
            self.best_keys.clear();
            self.best_keys.extend_from_slice(keys);
            if self.history_limit > 0 {
                self.record_history();
            }
        }
        Ok(value)
    }

    /// Largest block the problem scores efficiently; one means scalar evaluation.
    #[inline]
    #[must_use]
    pub fn batch_size(&self) -> usize {
        self.problem.batch_size()
    }

    /// Scores consecutive row-major vectors in one problem call.
    ///
    /// The deadline and cancellation are checked once before the block, so a block is
    /// never interrupted; callers keep blocks within `batch_size`. Every row counts as
    /// one completed evaluation and the best row updates the incumbent. When fewer rows
    /// remain under the evaluation limit than the block holds, only that prefix is scored
    /// (and counted) before `Stop::MaxEvaluations` is returned.
    pub fn eval_batch(&mut self, keys: &[f64], out: &mut [f64]) -> Result<(), Stop> {
        let width = self.problem.dimension();
        debug_assert_eq!(keys.len(), out.len() * width);
        if out.is_empty() {
            return Ok(());
        }
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Stop::Cancelled);
        }
        let remaining = self.max_evaluations.saturating_sub(self.evaluations);
        if remaining == 0 {
            return Err(Stop::MaxEvaluations);
        }
        if self.best_cost.is_finite() && Instant::now() >= self.deadline {
            return Err(Stop::TimeLimit);
        }
        let truncated = remaining < out.len() as u64;
        let rows = if truncated {
            remaining as usize
        } else {
            out.len()
        };
        let (keys, out) = (&keys[..rows * width], &mut out[..rows]);
        self.problem.evaluate_batch(keys, out);
        if out.iter().any(|value| !value.is_finite()) {
            return Err(Stop::InvalidCost);
        }
        self.evaluations += out.len() as u64;
        let mut best_row = 0;
        for (row, &value) in out.iter().enumerate() {
            if value < out[best_row] {
                best_row = row;
            }
        }
        if out[best_row] < self.best_cost {
            self.best_cost = out[best_row];
            self.best_keys.clear();
            self.best_keys
                .extend_from_slice(&keys[best_row * width..(best_row + 1) * width]);
            if self.history_limit > 0 {
                self.record_history();
            }
        }
        if truncated {
            return Err(Stop::MaxEvaluations);
        }
        Ok(())
    }

    /// Scores independent row-major candidates, grouped by `batch_size`.
    ///
    /// Scalar problems are scored row by row with a deadline check per row; batch-capable
    /// problems receive blocks of at most `batch_size` rows. On `Err`, rows scored before the
    /// stop keep their costs in `out` and count toward the incumbent.
    pub fn eval_rows(&mut self, rows: &[f64], out: &mut [f64]) -> Result<(), Stop> {
        let width = self.problem.dimension();
        let capacity = self.problem.batch_size();
        if capacity <= 1 {
            for (row, slot) in rows.chunks_exact(width).zip(out.iter_mut()) {
                *slot = self.eval(row)?;
            }
            return Ok(());
        }
        let mut start = 0;
        while start < out.len() {
            let end = (start + capacity).min(out.len());
            self.eval_batch(&rows[start * width..end * width], &mut out[start..end])?;
            start = end;
        }
        Ok(())
    }

    /// Binds incremental evaluation to `keys` when the problem supports it.
    ///
    /// Must be called before a run of `probe`/`commit` on the same vector.
    pub fn sync(&mut self, keys: &[f64]) {
        match self.incremental.as_mut() {
            Some(state) => state.reset(keys),
            None => self.incremental = self.problem.incremental(keys),
        }
    }

    /// Whether `probe` currently uses constant-time incremental evaluation.
    #[must_use]
    pub fn is_incremental(&self) -> bool {
        self.incremental.is_some()
    }

    /// Scores `keys` with `changes` applied; `keys` is left unchanged on return.
    ///
    /// Counts as one evaluation and updates the incumbent, exactly like `eval`.
    #[inline]
    pub fn probe(&mut self, keys: &mut [f64], changes: &[(usize, f64)]) -> Result<f64, Stop> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Stop::Cancelled);
        }
        if self.evaluations >= self.max_evaluations {
            return Err(Stop::MaxEvaluations);
        }
        self.tick += 1;
        if self.tick >= self.stride {
            self.tick = 0;
            if self.best_cost.is_finite() && Instant::now() >= self.deadline {
                return Err(Stop::TimeLimit);
            }
        }
        let value = if let Some(state) = self.incremental.as_ref() {
            state.probe(keys, changes)
        } else {
            let mut previous = [0.0_f64; 4];
            let spill: Vec<f64>;
            let saved: &[f64] = if changes.len() <= previous.len() {
                for (slot, &(index, _)) in previous.iter_mut().zip(changes) {
                    *slot = keys[index];
                }
                &previous[..changes.len()]
            } else {
                spill = changes.iter().map(|&(index, _)| keys[index]).collect();
                &spill
            };
            for &(index, value) in changes {
                keys[index] = value;
            }
            let value = self.problem.evaluate(keys);
            for (&(index, _), &old) in changes.iter().zip(saved) {
                keys[index] = old;
            }
            value
        };
        if !value.is_finite() {
            return Err(Stop::InvalidCost);
        }
        self.evaluations += 1;
        if value < self.best_cost {
            self.best_cost = value;
            self.best_keys.clear();
            self.best_keys.extend_from_slice(keys);
            for &(index, new) in changes {
                self.best_keys[index] = new;
            }
            if self.history_limit > 0 {
                self.record_history();
            }
        }
        Ok(value)
    }

    /// Applies `changes` to `keys` and to the incremental state.
    pub fn commit(&mut self, keys: &mut [f64], changes: &[(usize, f64)]) {
        if let Some(state) = self.incremental.as_mut() {
            state.commit(keys, changes);
        }
        for &(index, value) in changes {
            keys[index] = value;
        }
    }

    fn record_history(&mut self) {
        self.history_observations += 1;
        self.history.push(HistoryPoint {
            elapsed: self.elapsed(),
            evaluations: self.evaluations,
            cost: self.best_cost,
        });
        if self.history.len() > self.history_limit {
            let last = self.history[self.history.len() - 1];
            let mut compacted = Vec::with_capacity(self.history_limit);
            compacted.push(self.history[0]);
            compacted.extend(
                self.history[2..self.history.len() - 1]
                    .iter()
                    .step_by(2)
                    .copied(),
            );
            compacted.push(last);
            self.history = compacted;
        }
    }

    /// Appends a closing sample so the curve ends at the final incumbent.
    pub fn finish_history(&mut self) {
        if self.history_limit > 0 && self.best_cost.is_finite() {
            self.record_history();
        }
    }

    /// Retained history samples.
    #[must_use]
    pub fn history(&self) -> &[HistoryPoint] {
        &self.history
    }

    /// Number of incumbent observations offered to the history.
    #[must_use]
    pub fn history_observations(&self) -> u64 {
        self.history_observations
    }

    /// Retain the largest and median sampled uphill cost difference of the start estimator.
    pub fn record_calibration(&mut self, maximum: f64, median: f64) {
        self.calibration_sample = Some((maximum, median));
    }

    /// Start recording the annealing schedule once the start temperature and cutoff are known.
    pub fn begin_annealing(&mut self, t0: f64, cutoff: f64, calibrated: bool) {
        self.annealing = Some(AnnealingTrace {
            t0,
            cutoff,
            levels: 0,
            calibrated,
            calibration: None,
        });
    }

    /// Count one entered temperature level; ignored when no schedule was begun.
    pub fn count_annealing_level(&mut self) {
        if let Some(trace) = self.annealing.as_mut() {
            trace.levels += 1;
        }
    }

    /// Return the annealing schedule recorded so far, if the algorithm began one.
    #[must_use]
    pub fn annealing(&self) -> Option<AnnealingTrace> {
        self.annealing.map(|trace| AnnealingTrace {
            calibration: self.calibration_sample,
            ..trace
        })
    }

    /// Extends the deadline by `extra`; used by tests of deadline behavior.
    pub fn extend_deadline(&mut self, extra: Duration) {
        self.deadline += extra;
    }
}

#[cfg(test)]
mod tests;
