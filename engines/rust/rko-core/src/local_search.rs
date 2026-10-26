//! Neighborhood local searches and Random Variable Neighborhood Descent.
//!
//! Objective:
//! - Provide swap, mirror and Farey-interval sweeps plus the RVND driver that
//!   combines them with Nelder-Mead, all on reusable scratch buffers.
//!
//! Technical approach:
//! - Every sweep visits a freshly shuffled coordinate order. Strategy 1 accepts an
//!   improving move immediately and continues from it (reverting rejected moves in
//!   place); strategy 2 scores every move against a fixed base and applies only the
//!   best one at the end.
//! - The descent tries the four neighborhoods in the fixed order mirror, Farey,
//!   Nelder-Mead, swap, returning to the first after every improvement and finishing when
//!   all four fail in a row. The reference picks the next neighborhood at random; the
//!   fixed order puts the `O(n^2)` swap sweep last and reached better optima with fewer
//!   evaluations in the study that chose it.
//!
//! Flow (pseudocode):
//! 1. Shuffle the visiting order.
//! 2. Evaluate each neighbor through `Search::eval`, keeping or reverting it.
//! 3. Descent: run the first untried neighborhood on a copy, adopt it only if it improves.
//!
//! Important criteria:
//! - Cost per sweep is `O(n^2)` (swap), `O(n)` (mirror) and `O(18 n)` (Farey) evaluations.
//! - The incumbent cost is supplied by the caller and never re-evaluated.
//!
//! Inputs/Outputs:
//! - Input: keys, their cost, a strategy (1 or 2), `Search`, `Pool` and `Rng`.
//! - Output: improved keys in place and their cost.

use crate::operators::{NelderMead, PoolPolicy};
use crate::pool::Pool;
use crate::rng::Rng;
use crate::search::{Search, Stop};

/// Interval boundaries of the Farey discretization.
pub const FAREY: [f64; 19] = [
    0.00, 0.142_857, 0.166_667, 0.20, 0.25, 0.285_714, 0.333_333, 0.40, 0.428_571, 0.50, 0.571_429,
    0.60, 0.666_667, 0.714_286, 0.75, 0.80, 0.833_333, 0.857_143, 1.0,
];

const SWAP: u8 = 0;
const MIRROR: u8 = 1;
const NELDER_MEAD: u8 = 2;
const FAREY_SWEEP: u8 = 3;
const NEIGHBORHOOD_ORDER: [u8; 4] = [MIRROR, FAREY_SWEEP, NELDER_MEAD, SWAP];

/// Flips one key about 0.5, guarding values at the lower boundary.
#[inline]
fn mirrored(value: f64) -> f64 {
    if value > 0.000_01 {
        1.0 - value
    } else {
        0.999_99
    }
}

/// One neighborhood move, resolved against the current vector when it is scored.
#[derive(Clone, Copy, Debug)]
enum Move {
    /// Exchange the keys at two positions.
    Swap(usize, usize),
    /// Mirror the key at a position about 0.5.
    Mirror(usize),
    /// Replace the key at a position with a value.
    Set(usize, f64),
}

/// Up to two `(index, new value)` changes that realize a move.
#[derive(Clone, Copy, Debug)]
struct Changes {
    items: [(usize, f64); 2],
    length: usize,
}

impl Changes {
    fn as_slice(&self) -> &[(usize, f64)] {
        &self.items[..self.length]
    }
}

impl Move {
    fn changes(self, keys: &[f64]) -> Changes {
        match self {
            Self::Swap(a, b) => Changes {
                items: [(a, keys[b]), (b, keys[a])],
                length: 2,
            },
            Self::Mirror(index) => Changes {
                items: [(index, mirrored(keys[index])), (0, 0.0)],
                length: 1,
            },
            Self::Set(index, value) => Changes {
                items: [(index, value), (0, 0.0)],
                length: 1,
            },
        }
    }
}

/// Reusable local-search engine with scratch buffers.
#[derive(Clone, Debug, Default)]
pub struct LocalSearch {
    order: Vec<usize>,
    candidate: Vec<f64>,
    rows: Vec<f64>,
    costs: Vec<f64>,
    changes: Vec<Changes>,
    nelder_mead: NelderMead,
}

impl LocalSearch {
    /// Creates an engine with empty buffers and random pool vertices.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an engine whose Nelder-Mead uses the given pool vertex policy.
    #[must_use]
    pub fn with_policy(policy: PoolPolicy) -> Self {
        Self {
            nelder_mead: NelderMead::new(policy),
            ..Self::default()
        }
    }

    /// Runs Nelder-Mead from `keys`.
    pub fn nelder_mead(
        &mut self,
        keys: &mut [f64],
        cost: f64,
        search: &mut Search<'_>,
        pool: &Pool,
        rng: &mut Rng,
    ) -> Result<f64, Stop> {
        self.nelder_mead.run(keys, cost, search, pool, rng)
    }

    fn shuffled_order(&mut self, n: usize, rng: &mut Rng) {
        self.order.clear();
        self.order.extend(0..n);
        rng.shuffle(&mut self.order);
    }

    /// Scores `moves` against `keys`, accepting per `strategy` and returning the final cost.
    ///
    /// Scalar problems follow the exact sequential path. Batch-capable problems score
    /// blocks of upcoming moves together: strategy 2 uses a fixed base so blocks are
    /// exact; strategy 1 speculates, and when a move is accepted the remaining
    /// moves of the block are queued again and scored against the new base, so the
    /// accepted trajectory equals the sequential one and only stale scores are wasted.
    fn sweep(
        &mut self,
        keys: &mut [f64],
        cost: f64,
        strategy: u8,
        search: &mut Search<'_>,
        moves: &mut dyn Iterator<Item = Move>,
    ) -> Result<f64, Stop> {
        search.sync(keys);
        let capacity = search.batch_size();
        let mut best = cost;
        if capacity <= 1 {
            let mut chosen = None;
            for step in moves {
                let changes = step.changes(keys);
                let value = search.probe(keys, changes.as_slice())?;
                if value < best {
                    best = value;
                    if strategy == 1 {
                        search.commit(keys, changes.as_slice());
                    } else {
                        chosen = Some(changes);
                    }
                }
            }
            if let Some(changes) = chosen {
                search.commit(keys, changes.as_slice());
            }
            return Ok(best);
        }

        let width = keys.len();
        let mut pending: std::collections::VecDeque<Move> = std::collections::VecDeque::new();
        let mut exhausted = false;
        let mut block = if strategy == 1 {
            capacity.min(8)
        } else {
            capacity
        };
        let mut chosen: Option<Changes> = None;
        loop {
            while pending.len() < block && !exhausted {
                match moves.next() {
                    Some(step) => pending.push_back(step),
                    None => exhausted = true,
                }
            }
            if pending.is_empty() {
                break;
            }
            let count = pending.len().min(block);
            self.rows.clear();
            self.changes.clear();
            for step in pending.iter().take(count) {
                let changes = step.changes(keys);
                let start = self.rows.len();
                self.rows.extend_from_slice(keys);
                for &(index, value) in changes.as_slice() {
                    self.rows[start + index] = value;
                }
                self.changes.push(changes);
            }
            self.costs.resize(count, 0.0);
            search.eval_batch(&self.rows, &mut self.costs)?;
            debug_assert_eq!(self.rows.len(), count * width);
            let mut accepted = None;
            for (row, &value) in self.costs.iter().enumerate() {
                if value < best {
                    best = value;
                    if strategy == 1 {
                        accepted = Some(row);
                        break;
                    }
                    chosen = Some(self.changes[row]);
                }
            }
            match accepted {
                Some(row) => {
                    let changes = self.changes[row];
                    search.commit(keys, changes.as_slice());
                    for _ in 0..=row {
                        pending.pop_front();
                    }
                    block = capacity.min(8);
                }
                None => {
                    for _ in 0..count {
                        pending.pop_front();
                    }
                    block = (block * 2).min(capacity);
                }
            }
        }
        if let Some(changes) = chosen {
            search.commit(keys, changes.as_slice());
        }
        Ok(best)
    }

    /// Pairwise-swap sweep over all coordinate pairs.
    pub fn swap(
        &mut self,
        keys: &mut [f64],
        cost: f64,
        strategy: u8,
        search: &mut Search<'_>,
        rng: &mut Rng,
    ) -> Result<f64, Stop> {
        let n = keys.len();
        if n < 2 {
            return Ok(cost);
        }
        self.shuffled_order(n, rng);
        let order = std::mem::take(&mut self.order);
        let mut pairs = (0..n - 1).flat_map(|i| (i + 1..n).map(move |j| (i, j)));
        let mut moves =
            std::iter::from_fn(|| pairs.next().map(|(i, j)| Move::Swap(order[i], order[j])));
        let result = self.sweep(keys, cost, strategy, search, &mut moves);
        self.order = order;
        result
    }

    /// Single-coordinate mirror sweep.
    pub fn mirror(
        &mut self,
        keys: &mut [f64],
        cost: f64,
        strategy: u8,
        search: &mut Search<'_>,
        rng: &mut Rng,
    ) -> Result<f64, Stop> {
        let n = keys.len();
        if n == 0 {
            return Ok(cost);
        }
        self.shuffled_order(n, rng);
        let order = std::mem::take(&mut self.order);
        let mut moves = order.iter().map(|&index| Move::Mirror(index));
        let result = self.sweep(keys, cost, strategy, search, &mut moves);
        self.order = order;
        result
    }

    /// Farey-interval resampling sweep over each coordinate.
    pub fn farey(
        &mut self,
        keys: &mut [f64],
        cost: f64,
        strategy: u8,
        search: &mut Search<'_>,
        rng: &mut Rng,
    ) -> Result<f64, Stop> {
        let n = keys.len();
        if n == 0 {
            return Ok(cost);
        }
        self.shuffled_order(n, rng);
        let order = std::mem::take(&mut self.order);
        let mut cells = order
            .iter()
            .flat_map(|&index| (0..FAREY.len() - 1).map(move |interval| (index, interval)));
        let mut moves = std::iter::from_fn(|| {
            cells.next().map(|(index, interval)| {
                Move::Set(index, rng.uniform(FAREY[interval], FAREY[interval + 1]))
            })
        });
        let result = self.sweep(keys, cost, strategy, search, &mut moves);
        self.order = order;
        result
    }

    /// Variable neighborhood descent over mirror, Farey, Nelder-Mead and swap, in that order.
    ///
    /// The reference draws the next neighborhood at random. A fixed order that leaves the
    /// quadratic swap sweep last reached better local optima with fewer evaluations on
    /// knapsack, tour and continuous problems (see the native study), so the cheap
    /// neighborhoods run first and swap only when everything else has failed.
    pub fn rvnd(
        &mut self,
        keys: &mut [f64],
        cost: f64,
        strategy: u8,
        search: &mut Search<'_>,
        pool: &Pool,
        rng: &mut Rng,
    ) -> Result<f64, Stop> {
        let mut current_cost = cost;
        let mut failed = [false; 4];
        let mut candidate = std::mem::take(&mut self.candidate);
        let result = (|| {
            while let Some(kind) = NEIGHBORHOOD_ORDER
                .iter()
                .copied()
                .find(|kind| !failed[usize::from(*kind)])
            {
                candidate.clear();
                candidate.extend_from_slice(keys);
                let value = match kind {
                    SWAP => self.swap(&mut candidate, current_cost, strategy, search, rng)?,
                    MIRROR => self.mirror(&mut candidate, current_cost, strategy, search, rng)?,
                    NELDER_MEAD => {
                        self.nelder_mead
                            .run(&mut candidate, current_cost, search, pool, rng)?
                    }
                    _ => self.farey(&mut candidate, current_cost, strategy, search, rng)?,
                };
                if value < current_cost {
                    keys.copy_from_slice(&candidate);
                    current_cost = value;
                    failed = [false; 4];
                } else {
                    failed[usize::from(kind)] = true;
                }
            }
            Ok(current_cost)
        })();
        self.candidate = candidate;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operators::random_keys;
    use crate::problem::{Knapsack, Problem};
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    fn problem() -> Knapsack {
        let profits: Vec<i64> = (0..30).map(|index| 5 + (index * 17) % 23).collect();
        let weights: Vec<i64> = (0..30).map(|index| 3 + (index * 11) % 19).collect();
        Knapsack::new(&profits, &weights, 120)
    }

    fn run_sweep(
        strategy: u8,
        sweep: impl Fn(
            &mut LocalSearch,
            &mut [f64],
            f64,
            u8,
            &mut Search<'_>,
            &mut Rng,
        ) -> Result<f64, Stop>,
    ) {
        let problem = problem();
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(10), 0);
        let mut rng = Rng::new(3);
        let mut local = LocalSearch::new();
        for _ in 0..5 {
            let mut keys = vec![0.0; 30];
            random_keys(&mut keys, &mut rng);
            let before = problem.evaluate(&keys);
            let after = sweep(
                &mut local,
                &mut keys,
                before,
                strategy,
                &mut search,
                &mut rng,
            )
            .unwrap();
            assert!(after <= before);
            assert_eq!(
                after,
                problem.evaluate(&keys),
                "returned cost must match keys"
            );
            assert!(keys.iter().all(|key| (0.0..1.0).contains(key)));
        }
    }

    #[test]
    fn sweeps_return_consistent_costs_for_both_strategies() {
        for strategy in [1, 2] {
            run_sweep(strategy, |local, keys, cost, strategy, search, rng| {
                local.swap(keys, cost, strategy, search, rng)
            });
            run_sweep(strategy, |local, keys, cost, strategy, search, rng| {
                local.mirror(keys, cost, strategy, search, rng)
            });
            run_sweep(strategy, |local, keys, cost, strategy, search, rng| {
                local.farey(keys, cost, strategy, search, rng)
            });
        }
    }

    #[test]
    fn incremental_and_plain_evaluation_follow_identical_trajectories() {
        use crate::problem::test_support::DeltaKnapsack;
        let slow = problem();
        let fast = DeltaKnapsack(slow.clone());
        let cancel = AtomicBool::new(false);
        for strategy in [1, 2] {
            let mut outcomes = Vec::new();
            for incremental in [true, false] {
                let target: &dyn Problem = if incremental { &fast } else { &slow };
                let now = Instant::now();
                let mut search =
                    Search::new(target, &cancel, now, now + Duration::from_secs(30), 0);
                let pool = Pool::new(5);
                let mut rng = Rng::new(21);
                let mut local = LocalSearch::new();
                let mut keys = vec![0.0; 30];
                random_keys(&mut keys, &mut rng);
                let start = search.eval(&keys).unwrap();
                let cost = local
                    .rvnd(&mut keys, start, strategy, &mut search, &pool, &mut rng)
                    .unwrap();
                assert_eq!(search.is_incremental(), incremental);
                outcomes.push((keys, cost, search.evaluations()));
            }
            assert_eq!(outcomes[0], outcomes[1], "strategy {strategy}");
        }
    }

    #[test]
    fn batched_sweeps_reproduce_the_scalar_trajectory() {
        use crate::problem::test_support::BatchKnapsack;
        use std::sync::atomic::Ordering;
        let scalar = problem();
        let cancel = AtomicBool::new(false);
        for strategy in [1, 2] {
            for size in [2, 7, 64] {
                for kind in 0..3 {
                    let batched = BatchKnapsack::new(scalar.clone(), size);
                    let mut outcomes = Vec::new();
                    for use_batch in [false, true] {
                        let target: &dyn Problem = if use_batch { &batched } else { &scalar };
                        let now = Instant::now();
                        let mut search =
                            Search::new(target, &cancel, now, now + Duration::from_secs(30), 0);
                        let mut rng = Rng::new(99 + kind);
                        let mut local = LocalSearch::new();
                        let mut keys = vec![0.0; 30];
                        random_keys(&mut keys, &mut rng);
                        let start = search.eval(&keys).unwrap();
                        let cost = match kind {
                            0 => local.swap(&mut keys, start, strategy, &mut search, &mut rng),
                            1 => local.mirror(&mut keys, start, strategy, &mut search, &mut rng),
                            _ => local.farey(&mut keys, start, strategy, &mut search, &mut rng),
                        }
                        .unwrap();
                        outcomes.push((keys, cost, search.evaluations()));
                    }
                    assert_eq!(
                        (&outcomes[0].0, outcomes[0].1),
                        (&outcomes[1].0, outcomes[1].1),
                        "kind {kind} strategy {strategy} block {size}"
                    );
                    assert!(outcomes[1].2 >= outcomes[0].2);
                    if strategy == 2 {
                        assert_eq!(outcomes[1].2, outcomes[0].2, "fixed base wastes nothing");
                    }
                    assert!(batched.calls.load(Ordering::Relaxed) > 0);
                    assert!(
                        batched.rows.load(Ordering::Relaxed)
                            > batched.calls.load(Ordering::Relaxed)
                    );
                }
            }
        }
    }

    #[test]
    fn descent_tries_cheap_neighborhoods_first_and_swap_last() {
        use std::sync::Mutex;
        struct Recorder {
            seen: Mutex<Vec<Vec<f64>>>,
        }
        impl Problem for Recorder {
            fn dimension(&self) -> usize {
                6
            }
            fn evaluate(&self, keys: &[f64]) -> f64 {
                self.seen.lock().unwrap().push(keys.to_vec());
                1.0
            }
        }
        let problem = Recorder {
            seen: Mutex::new(Vec::new()),
        };
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(30), 0);
        let pool = Pool::new(3);
        let mut rng = Rng::new(5);
        let mut local = LocalSearch::new();
        let start = [0.2, 0.9, 0.4, 0.6, 0.1, 0.8];
        let mut keys = start.to_vec();
        let cost = local
            .rvnd(&mut keys, 1.0, 1, &mut search, &pool, &mut rng)
            .unwrap();
        assert_eq!(cost, 1.0, "a flat objective never improves");
        let seen = problem.seen.lock().unwrap();
        let differs = |row: &Vec<f64>| row.iter().zip(&start).filter(|(a, b)| a != b).count();
        for row in &seen[..6] {
            assert_eq!(differs(row), 1, "mirror sweep first");
            let changed = row.iter().zip(&start).find(|(a, b)| a != b).unwrap();
            assert!((changed.0 - (1.0 - changed.1)).abs() < 1e-12);
        }
        assert!(
            seen[6..6 + 6 * 18].iter().all(|row| differs(row) == 1),
            "Farey sweep second"
        );
        let swap_rows = &seen[seen.len() - 15..];
        assert!(
            swap_rows.iter().all(|row| differs(row) == 2),
            "swap sweep last"
        );
    }

    #[test]
    fn rvnd_reaches_local_optimum_for_all_neighborhoods() {
        let problem = problem();
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(30), 0);
        let pool = Pool::new(5);
        let mut rng = Rng::new(8);
        let mut local = LocalSearch::new();
        let mut keys = vec![0.0; 30];
        random_keys(&mut keys, &mut rng);
        let before = problem.evaluate(&keys);
        let after = local
            .rvnd(&mut keys, before, 1, &mut search, &pool, &mut rng)
            .unwrap();
        assert!(after < before);
        assert_eq!(after, problem.evaluate(&keys));
        let mut mirrored_keys = keys.clone();
        let again = local
            .mirror(&mut mirrored_keys, after, 2, &mut search, &mut rng)
            .unwrap();
        assert_eq!(again, after, "RVND result must be mirror-optimal");
    }

    #[test]
    fn deadline_interrupts_a_long_sweep() {
        let problem = Knapsack::new(&vec![5; 3000], &vec![3; 3000], 600);
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_millis(5), 0);
        let mut rng = Rng::new(1);
        let mut keys = vec![0.2; 3000];
        let first = search.eval(&keys).unwrap();
        let mut local = LocalSearch::new();
        let outcome = local.swap(&mut keys, first, 2, &mut search, &mut rng);
        assert_eq!(outcome, Err(Stop::TimeLimit));
    }
}
