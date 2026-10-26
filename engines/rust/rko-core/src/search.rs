//! Budgeted evaluation wrapper shared by every metaheuristic.
//!
//! Objective:
//! - Apply deadline, cancellation, finite-cost validation and incumbent retention at
//!   the single boundary every operator uses to score a candidate.
//!
//! Technical approach:
//! - `Search::eval` counts completed evaluations, retains the best completed
//!   candidate and aborts with `Stop` once the budget is exhausted.
//! - Wall-clock reads are amortized over `check_stride` evaluations; the cancel flag
//!   is an atomic relaxed load on every call.
//! - Scalar incumbent samples are optionally retained in a bounded, deterministically
//!   compacted history for later reporting.
//!
//! Flow (pseudocode):
//! 1. Reject the call if cancelled or, after the first valid result, past deadline.
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

/// Budget-aware evaluator and incumbent holder for one search.
pub struct Search<'a> {
    problem: &'a dyn Problem,
    cancel: &'a AtomicBool,
    deadline: Instant,
    started: Instant,
    stride: u32,
    tick: u32,
    evaluations: u64,
    best_cost: f64,
    best_keys: Vec<f64>,
    history_limit: usize,
    history: Vec<HistoryPoint>,
    history_observations: u64,
    incremental: Option<Box<dyn Incremental + 'a>>,
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
            best_cost: f64::INFINITY,
            best_keys: Vec::new(),
            history_limit,
            history: Vec::new(),
            history_observations: 0,
            incremental: None,
        }
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
    /// one completed evaluation and the best row updates the incumbent.
    pub fn eval_batch(&mut self, keys: &[f64], out: &mut [f64]) -> Result<(), Stop> {
        let width = self.problem.dimension();
        debug_assert_eq!(keys.len(), out.len() * width);
        if out.is_empty() {
            return Ok(());
        }
        if self.cancel.load(Ordering::Relaxed) {
            return Err(Stop::Cancelled);
        }
        if self.best_cost.is_finite() && Instant::now() >= self.deadline {
            return Err(Stop::TimeLimit);
        }
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

    /// Extends the deadline by `extra`; used by tests of deadline behavior.
    pub fn extend_deadline(&mut self, extra: Duration) {
        self.deadline += extra;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::Knapsack;

    fn fixture() -> Knapsack {
        Knapsack::new(&[10, 20, 30], &[5, 6, 7], 12)
    }

    #[test]
    fn tracks_best_and_counts() {
        let problem = fixture();
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0);
        assert_eq!(search.eval(&[0.9, 0.1, 0.1]).unwrap(), -10.0);
        assert_eq!(search.eval(&[0.9, 0.9, 0.1]).unwrap(), -30.0);
        assert_eq!(search.eval(&[0.1, 0.1, 0.1]).unwrap(), 0.0);
        assert_eq!(search.evaluations(), 3);
        assert_eq!(search.best_cost(), -30.0);
        assert_eq!(search.best_keys(), &[0.9, 0.9, 0.1]);
    }

    #[test]
    fn first_evaluation_survives_expired_deadline() {
        let problem = fixture();
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now, 0);
        assert!(search.eval(&[0.9, 0.1, 0.1]).is_ok());
        let mut stopped = false;
        for _ in 0..16 {
            if search.eval(&[0.9, 0.1, 0.1]) == Err(Stop::TimeLimit) {
                stopped = true;
                break;
            }
        }
        assert!(stopped);
    }

    #[test]
    fn cancellation_is_immediate() {
        let problem = fixture();
        let cancel = AtomicBool::new(true);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0);
        assert_eq!(search.eval(&[0.9, 0.1, 0.1]), Err(Stop::Cancelled));
    }

    #[test]
    fn probe_matches_eval_with_and_without_incremental_state() {
        use crate::problem::test_support::DeltaKnapsack;
        let knapsack = Knapsack::new(&[10, 20, 30, 15], &[5, 6, 7, 4], 12);
        let fast = DeltaKnapsack(knapsack.clone());
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        for incremental in [true, false] {
            let problem: &dyn Problem = if incremental { &fast } else { &knapsack };
            let mut search = Search::new(problem, &cancel, now, now + Duration::from_secs(5), 0);
            let mut keys = vec![0.9, 0.1, 0.9, 0.1];
            search.eval(&keys).unwrap();
            search.sync(&keys);
            assert_eq!(search.is_incremental(), incremental);
            let snapshot = keys.clone();
            let changes = [(1, 0.8), (3, 0.7)];
            let moved = search.probe(&mut keys, &changes).unwrap();
            assert_eq!(keys, snapshot, "probe must restore keys");
            let mut expected = snapshot.clone();
            expected[1] = 0.8;
            expected[3] = 0.7;
            assert_eq!(moved, knapsack.evaluate(&expected));
            search.commit(&mut keys, &changes);
            assert_eq!(keys, expected);
            assert_eq!(
                search.probe(&mut keys, &[]).unwrap(),
                knapsack.evaluate(&expected)
            );
            assert_eq!(search.evaluations(), 3);
        }
    }

    #[test]
    fn history_compaction_keeps_cap_and_endpoints() {
        struct Descending;
        impl Problem for Descending {
            fn dimension(&self) -> usize {
                1
            }
            fn evaluate(&self, keys: &[f64]) -> f64 {
                -keys[0]
            }
        }
        let problem = Descending;
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 8);
        for step in 0..100 {
            search.eval(&[f64::from(step) / 100.0]).unwrap();
        }
        search.finish_history();
        assert!(search.history().len() <= 8);
        assert_eq!(search.history().first().unwrap().evaluations, 1);
        assert_eq!(search.history().last().unwrap().cost, -0.99);
        assert_eq!(search.history_observations(), 101);
    }
}
