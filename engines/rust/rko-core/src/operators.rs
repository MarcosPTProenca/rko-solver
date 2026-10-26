//! Allocation-free random-key operators.
//!
//! Objective:
//! - Implement blending crossover, shaking and Nelder-Mead local search on caller
//!   owned buffers so hot loops never allocate.
//!
//! Technical approach:
//! - Blending draws one 64-bit word per gene: the high 53 bits drive the 2% mutation
//!   test and bit zero drives the 50% parent choice, halving RNG calls while keeping
//!   both decisions independent and uniform.
//! - Shaking applies `floor(n * U(beta_min, beta_max)) + 1` elementary moves chosen
//!   uniformly among random-reset, mirror, swap and neighbor-swap.
//! - Nelder-Mead treats three keys vectors as a simplex and uses blending as its
//!   centroid/reflection/expansion/contraction/shrink moves, with all points held in
//!   a fixed arena reused across calls.
//!
//! Flow (pseudocode):
//! 1. Build the simplex from the incumbent and two pool elites (random fallback).
//! 2. Sort, blend the centroid, then reflect/expand/contract/shrink until
//!    `max_iter` consecutive non-improving iterations.
//! 3. Return the best point only if it beats the incumbent.
//!
//! Important criteria:
//! - Keys stay in `[0, 1)`; complement moves are clamped to 0.9999999.
//! - Pool elites reuse their stored cost and are not re-evaluated.
//!
//! Inputs/Outputs:
//! - Input: key slices, cost, a `Search`, the shared `Pool` and an `Rng`.
//! - Output: perturbed or improved keys and their cost.

use crate::pool::Pool;
use crate::rng::Rng;
use crate::search::{Search, Stop};

/// Writes a uniform-crossover child with 2% mutation into `out`.
///
/// When `complement` is true, genes inherited from `parent_b` are replaced by
/// `clamp(1 - gene, 0, 0.9999999)`.
#[inline]
pub fn blend(parent_a: &[f64], parent_b: &[f64], complement: bool, out: &mut [f64], rng: &mut Rng) {
    debug_assert_eq!(parent_a.len(), parent_b.len());
    debug_assert_eq!(parent_a.len(), out.len());
    for ((slot, &gene_a), &gene_b) in out.iter_mut().zip(parent_a).zip(parent_b) {
        let word = rng.next_u64();
        if Rng::unit_from_word(word) < 0.02 {
            *slot = rng.f64();
        } else {
            let from_b = if complement {
                (1.0 - gene_b).clamp(0.0, 0.999_999_9)
            } else {
                gene_b
            };
            *slot = if word & 1 == 0 { gene_a } else { from_b };
        }
    }
}

/// Fills `out` with independent uniform keys.
#[inline]
pub fn random_keys(out: &mut [f64], rng: &mut Rng) {
    for slot in out {
        *slot = rng.f64();
    }
}

/// Perturbs `keys` in place with randomly chosen elementary moves.
pub fn shake(keys: &mut [f64], beta_min: f64, beta_max: f64, rng: &mut Rng) {
    let n = keys.len();
    if n == 0 {
        return;
    }
    let intensity = (n as f64 * rng.uniform(beta_min, beta_max)) as usize + 1;
    for _ in 0..intensity {
        match rng.range_inclusive(1, 4) {
            1 => {
                let index = rng.below(n);
                keys[index] = rng.f64();
            }
            2 => {
                let index = rng.below(n);
                keys[index] = if keys[index] > 0.0001 {
                    1.0 - keys[index]
                } else {
                    0.9999
                };
            }
            3 if n >= 2 => {
                let first = rng.below(n);
                let second = rng.below(n);
                keys.swap(first, second);
            }
            4 if n >= 2 => {
                let index = rng.below(n - 1);
                keys.swap(index, index + 1);
            }
            _ => {}
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Point {
    keys: Vec<f64>,
    cost: f64,
}

impl Point {
    fn copy_from(&mut self, keys: &[f64], cost: f64) {
        self.keys.clear();
        self.keys.extend_from_slice(keys);
        self.cost = cost;
    }
}

const X1: usize = 0;
const X2: usize = 1;
const X3: usize = 2;
const X0: usize = 3;
const XR: usize = 4;
const XE: usize = 5;
const XC: usize = 6;
const BEST: usize = 7;
const XD: usize = 8;

/// How Nelder-Mead picks its two pool vertices.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PoolPolicy {
    /// Two distinct random entries; only points derived from the incumbent can win.
    #[default]
    Random,
    /// The two best entries, which may themselves be returned when they beat the incumbent.
    Elite,
}

/// Reusable Nelder-Mead local search over random keys.
#[derive(Clone, Debug)]
pub struct NelderMead {
    points: [Point; 9],
    policy: PoolPolicy,
    rows: Vec<f64>,
    costs: Vec<f64>,
}

impl Default for NelderMead {
    fn default() -> Self {
        Self::new(PoolPolicy::Random)
    }
}

impl NelderMead {
    /// Creates a searcher with an empty arena that grows on first use.
    #[must_use]
    pub fn new(policy: PoolPolicy) -> Self {
        Self {
            points: Default::default(),
            policy,
            rows: Vec::new(),
            costs: Vec::new(),
        }
    }

    fn sort_simplex(&mut self) {
        if self.points[X1].cost > self.points[X2].cost {
            self.points.swap(X1, X2);
        }
        if self.points[X1].cost > self.points[X3].cost {
            self.points.swap(X1, X3);
        }
        if self.points[X2].cost > self.points[X3].cost {
            self.points.swap(X2, X3);
        }
    }

    fn make(
        &mut self,
        first: usize,
        second: usize,
        destination: usize,
        complement: bool,
        rng: &mut Rng,
    ) {
        let mut out = std::mem::take(&mut self.points[destination].keys);
        out.resize(self.points[first].keys.len(), 0.0);
        blend(
            &self.points[first].keys,
            &self.points[second].keys,
            complement,
            &mut out,
            rng,
        );
        self.points[destination].keys = out;
    }

    fn consider(&mut self, slot: usize, improved: &mut bool, improved_origin: &mut bool) {
        let cost = self.points[slot].cost;
        if cost < self.points[BEST].cost {
            let keys = std::mem::take(&mut self.points[slot].keys);
            self.points[BEST].copy_from(&keys, cost);
            self.points[slot].keys = keys;
            *improved = true;
            *improved_origin = true;
        }
    }

    fn evaluate(
        &mut self,
        destination: usize,
        search: &mut Search<'_>,
        improved: &mut bool,
        improved_origin: &mut bool,
    ) -> Result<f64, Stop> {
        let cost = search.eval(&self.points[destination].keys)?;
        self.points[destination].cost = cost;
        self.consider(destination, improved, improved_origin);
        Ok(cost)
    }

    fn evaluate_slots(&mut self, slots: &[usize], search: &mut Search<'_>) -> Result<(), Stop> {
        self.rows.clear();
        for &slot in slots {
            self.rows.extend_from_slice(&self.points[slot].keys);
        }
        self.costs.resize(slots.len(), 0.0);
        search.eval_rows(&self.rows, &mut self.costs)?;
        for (&slot, &cost) in slots.iter().zip(&self.costs) {
            self.points[slot].cost = cost;
        }
        Ok(())
    }

    fn step_scalar(
        &mut self,
        search: &mut Search<'_>,
        rng: &mut Rng,
        improved: &mut bool,
        improved_origin: &mut bool,
    ) -> Result<(), Stop> {
        let mut shrink = false;
        self.make(X0, X3, XR, true, rng);
        let reflected = self.evaluate(XR, search, improved, improved_origin)?;
        if reflected < self.points[X1].cost {
            self.make(XR, X0, XE, true, rng);
            let expanded = self.evaluate(XE, search, improved, improved_origin)?;
            if expanded < reflected {
                self.points.swap(X3, XE);
            } else {
                self.points.swap(X3, XR);
            }
        } else if reflected < self.points[X2].cost {
            self.points.swap(X3, XR);
        } else if reflected < self.points[X3].cost {
            self.make(XR, X0, XC, false, rng);
            let contracted = self.evaluate(XC, search, improved, improved_origin)?;
            if contracted < reflected {
                self.points.swap(X3, XC);
            } else {
                shrink = true;
            }
        } else {
            self.make(X0, X3, XC, false, rng);
            let contracted = self.evaluate(XC, search, improved, improved_origin)?;
            if contracted < self.points[X3].cost {
                self.points.swap(X3, XC);
            } else {
                shrink = true;
            }
        }
        if shrink {
            self.make(X1, X2, XC, false, rng);
            self.evaluate(XC, search, improved, improved_origin)?;
            self.points.swap(X2, XC);
            self.make(X1, X3, XC, false, rng);
            self.evaluate(XC, search, improved, improved_origin)?;
            self.points.swap(X3, XC);
        }
        Ok(())
    }

    fn step_batched(
        &mut self,
        search: &mut Search<'_>,
        rng: &mut Rng,
        improved: &mut bool,
        improved_origin: &mut bool,
    ) -> Result<(), Stop> {
        self.make(X0, X3, XR, true, rng);
        self.make(XR, X0, XE, true, rng);
        self.make(XR, X0, XC, false, rng);
        self.make(X0, X3, XD, false, rng);
        self.evaluate_slots(&[XR, XE, XC, XD], search)?;
        self.consider(XR, improved, improved_origin);
        let reflected = self.points[XR].cost;
        let mut shrink = false;
        if reflected < self.points[X1].cost {
            self.consider(XE, improved, improved_origin);
            if self.points[XE].cost < reflected {
                self.points.swap(X3, XE);
            } else {
                self.points.swap(X3, XR);
            }
        } else if reflected < self.points[X2].cost {
            self.points.swap(X3, XR);
        } else if reflected < self.points[X3].cost {
            self.consider(XC, improved, improved_origin);
            if self.points[XC].cost < reflected {
                self.points.swap(X3, XC);
            } else {
                shrink = true;
            }
        } else {
            self.consider(XD, improved, improved_origin);
            if self.points[XD].cost < self.points[X3].cost {
                self.points.swap(X3, XD);
            } else {
                shrink = true;
            }
        }
        if shrink {
            self.make(X1, X2, XC, false, rng);
            self.make(X1, X3, XD, false, rng);
            self.evaluate_slots(&[XC, XD], search)?;
            self.consider(XC, improved, improved_origin);
            self.consider(XD, improved, improved_origin);
            self.points.swap(X2, XC);
            self.points.swap(X3, XD);
        }
        Ok(())
    }

    /// Refines `keys` in place and returns the resulting cost.
    ///
    /// `keys` is overwritten only when a strictly better point is found; otherwise
    /// it and `cost` are returned unchanged.
    pub fn run(
        &mut self,
        keys: &mut [f64],
        cost: f64,
        search: &mut Search<'_>,
        pool: &Pool,
        rng: &mut Rng,
    ) -> Result<f64, Stop> {
        let n = keys.len();
        let max_iter = ((n as f64 * (-2.0_f64).exp()) as usize).max(1);
        let origin_cost = cost;
        self.points[X1].copy_from(keys, cost);
        self.points[BEST].copy_from(keys, cost);
        let (head, tail) = self.points.split_at_mut(X3);
        let (second, third) = (&mut head[X2], &mut tail[0]);
        let sampled = if self.policy == PoolPolicy::Elite {
            pool.best_pair(&mut second.keys, &mut third.keys)
        } else {
            pool.sample_pair(rng, &mut second.keys, &mut third.keys)
        };
        match sampled {
            Some((second_cost, third_cost)) => {
                second.cost = second_cost;
                third.cost = third_cost;
            }
            None => {
                for point in [&mut *second, &mut *third] {
                    point.keys.resize(n, 0.0);
                    random_keys(&mut point.keys, rng);
                }
                self.evaluate_slots(&[X2, X3], search)?;
            }
        }
        if self.policy == PoolPolicy::Elite {
            for slot in [X2, X3] {
                if self.points[slot].cost < self.points[BEST].cost {
                    let (arena, best) = self.points.split_at_mut(BEST);
                    best[0].copy_from(&arena[slot].keys, arena[slot].cost);
                }
            }
        }
        self.sort_simplex();

        let mut improved = false;
        let mut improved_origin = self.points[BEST].cost < origin_cost;
        self.make(X1, X2, X0, false, rng);
        self.evaluate(X0, search, &mut improved, &mut improved_origin)?;

        let batched = search.batch_size() > 1;
        let mut iterations = 1;
        while iterations <= max_iter {
            if batched {
                self.step_batched(search, rng, &mut improved, &mut improved_origin)?;
            } else {
                self.step_scalar(search, rng, &mut improved, &mut improved_origin)?;
            }

            self.sort_simplex();
            self.make(X1, X2, X0, false, rng);
            self.evaluate(X0, search, &mut improved, &mut improved_origin)?;

            if improved {
                improved = false;
                iterations = 0;
            } else {
                iterations += 1;
            }
        }

        if improved_origin && self.points[BEST].cost < origin_cost {
            keys.copy_from_slice(&self.points[BEST].keys);
            Ok(self.points[BEST].cost)
        } else {
            Ok(origin_cost)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{Knapsack, Problem};
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn blend_statistics_match_specification() {
        let mut rng = Rng::new(11);
        let n = 20_000;
        let a = vec![0.25; n];
        let b = vec![0.75; n];
        let mut out = vec![0.0; n];
        blend(&a, &b, false, &mut out, &mut rng);
        let from_a = out.iter().filter(|&&value| value == 0.25).count() as f64;
        let from_b = out.iter().filter(|&&value| value == 0.75).count() as f64;
        let mutated = n as f64 - from_a - from_b;
        assert!(
            (mutated / n as f64 - 0.02).abs() < 0.006,
            "mutated {mutated}"
        );
        assert!((from_a / (from_a + from_b) - 0.5).abs() < 0.02);
        assert!(out.iter().all(|value| (0.0..1.0).contains(value)));
    }

    #[test]
    fn blend_complement_clamps() {
        let mut rng = Rng::new(2);
        let a = vec![0.5; 5_000];
        let b = vec![0.0; 5_000];
        let mut out = vec![0.0; 5_000];
        blend(&a, &b, true, &mut out, &mut rng);
        assert!(out.iter().all(|&value| value < 1.0));
        assert!(out.contains(&0.999_999_9));
    }

    #[test]
    fn shake_changes_bounded_genes_and_keeps_range() {
        let mut rng = Rng::new(4);
        let original: Vec<f64> = (0..200).map(|index| f64::from(index) / 200.0).collect();
        let mut keys = original.clone();
        shake(&mut keys, 0.01, 0.05, &mut rng);
        let changed = keys.iter().zip(&original).filter(|(x, y)| x != y).count();
        assert!(changed >= 1 && changed <= 2 * (200.0 * 0.05) as usize + 2);
        assert!(keys.iter().all(|value| (0.0..1.0).contains(value)));
    }

    #[test]
    fn nelder_mead_batches_candidates_and_keeps_its_guarantees() {
        use crate::problem::test_support::BatchKnapsack;
        use std::sync::atomic::Ordering;
        let profits: Vec<i64> = (1..=60).collect();
        let weights: Vec<i64> = (1..=60).map(|value| value * 2).collect();
        let problem = BatchKnapsack::new(Knapsack::new(&profits, &weights, 900), 8);
        let mut rng = Rng::new(17);
        let cancel = AtomicBool::new(false);
        let pool = Pool::new(5);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0);
        let mut keys = vec![0.0; 60];
        random_keys(&mut keys, &mut rng);
        search.eval(&keys).unwrap();
        let mut searcher = NelderMead::default();
        for _ in 0..30 {
            let before = problem.inner.evaluate(&keys);
            let after = searcher
                .run(&mut keys, before, &mut search, &pool, &mut rng)
                .unwrap();
            assert!(after <= before);
            assert_eq!(after, problem.inner.evaluate(&keys));
            pool.insert(after, &keys, 0);
        }
        let calls = problem.calls.load(Ordering::Relaxed);
        let rows = problem.rows.load(Ordering::Relaxed);
        assert!(
            rows > calls,
            "candidates should share calls: {rows} rows in {calls} calls"
        );
    }

    #[test]
    fn nelder_mead_never_worsens_and_respects_budget() {
        let mut rng = Rng::new(9);
        let profits: Vec<i64> = (1..=60).collect();
        let weights: Vec<i64> = (1..=60).map(|value| value * 2).collect();
        let problem = Knapsack::new(&profits, &weights, 900);
        let cancel = AtomicBool::new(false);
        let pool = Pool::new(5);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0);
        let mut keys = vec![0.0; 60];
        random_keys(&mut keys, &mut rng);
        let start = search.eval(&keys).unwrap();
        let mut searcher = NelderMead::default();
        for _ in 0..20 {
            let before = problem.evaluate(&keys);
            let after = searcher
                .run(&mut keys, before, &mut search, &pool, &mut rng)
                .unwrap();
            assert!(after <= before);
            assert_eq!(after, problem.evaluate(&keys));
            pool.insert(after, &keys, 0);
        }
        assert!(search.best_cost() <= start);
    }
}
