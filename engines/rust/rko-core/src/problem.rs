//! Problem definition contract and built-in problems.
//!
//! Objective:
//! - Describe what the engine needs from a random-key problem and ship a reference
//!   0/1 Knapsack used by tests and benchmarks.
//!
//! Technical approach:
//! - A `Problem` maps a key vector in `[0, 1)^n` directly to a finite minimization
//!   cost, fusing decoder and objective so no intermediate solution is allocated.
//! - Knapsack accumulates profit and weight in `f64`, which is exact for integers
//!   below 2^53 and lets the compiler vectorize the threshold test.
//!
//! Flow (pseudocode):
//! 1. For every key above 0.5 add the item's profit and weight.
//! 2. Penalize each unit of capacity excess by 100000.
//! 3. Return the negated penalized profit so the engine minimizes.
//!
//! Important criteria:
//! - `evaluate` must be deterministic and thread-safe (`Sync`).
//! - Non-finite results are rejected by the search wrapper, never propagated.
//!
//! Inputs/Outputs:
//! - Input: a key slice whose length equals `dimension()`.
//! - Output: a finite cost to minimize.

/// Incremental evaluator that scores single moves in time proportional to the move size.
///
/// The `keys` slice passed to every method is the vector the state currently
/// represents, before any proposed change.
pub trait Incremental {
    /// Rebuilds the state from scratch for `keys`.
    fn reset(&mut self, keys: &[f64]);

    /// Cost the vector would have after applying `changes` as `(index, new value)` pairs.
    fn probe(&self, keys: &[f64], changes: &[(usize, f64)]) -> f64;

    /// Applies `changes` to the internal state; the caller updates `keys` itself.
    fn commit(&mut self, keys: &[f64], changes: &[(usize, f64)]);
}

/// Random-key problem evaluated by the engine.
pub trait Problem: Sync {
    /// Number of random keys expected by `evaluate`.
    fn dimension(&self) -> usize;

    /// Decodes and scores one key vector; lower is better.
    fn evaluate(&self, keys: &[f64]) -> f64;

    /// Scores `keys.len() / dimension()` consecutive row-major vectors into `out`.
    ///
    /// The default evaluates row by row; override it when scoring a block together is
    /// cheaper than scoring each row alone (vectorized or foreign-function evaluators).
    fn evaluate_batch(&self, keys: &[f64], out: &mut [f64]) {
        let width = self.dimension();
        for (row, slot) in keys.chunks_exact(width).zip(out.iter_mut()) {
            *slot = self.evaluate(row);
        }
    }

    /// Largest block `evaluate_batch` handles efficiently; one means scoring is scalar.
    ///
    /// Searches only group independent candidates, or speculate on sequential ones,
    /// when this exceeds one, so scalar problems keep the exact sequential code path.
    fn batch_size(&self) -> usize {
        1
    }

    /// Evaluations between wall-clock checks; use 1 for expensive callbacks.
    fn check_stride(&self) -> u32 {
        8
    }

    /// Optional incremental evaluator used by single-coordinate local searches.
    fn incremental<'a>(&'a self, _keys: &[f64]) -> Option<Box<dyn Incremental + 'a>> {
        None
    }

    /// Runs `body` inside any per-thread context the problem needs; the engine wraps
    /// each worker's whole sequence of searches in one call.
    fn with_thread_context(&self, body: &mut dyn FnMut()) {
        body();
    }
}

/// Error raised while parsing a Knapsack instance.
#[derive(Debug)]
pub enum InstanceError {
    /// The file contains fewer numbers than the declared item count requires.
    Truncated,
    /// A token could not be parsed as an integer.
    Malformed(String),
}

impl std::fmt::Display for InstanceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(f, "instance ended before all items were read"),
            Self::Malformed(token) => write!(f, "malformed integer token: {token}"),
        }
    }
}

impl std::error::Error for InstanceError {}

thread_local! {
    static TOUR_ORDER: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Symmetric Euclidean TSP decoded by sorting keys into a visiting order.
///
/// Distances are rounded to the nearest integer (TSPLIB `EUC_2D`), so tour lengths
/// are exact integers in `f64`. Keys with equal values are ordered by index.
#[derive(Clone, Debug)]
pub struct Tsp {
    size: usize,
    distances: Vec<f64>,
}

impl Tsp {
    /// Builds an instance from city coordinates.
    #[must_use]
    pub fn new(coordinates: &[(f64, f64)]) -> Self {
        let size = coordinates.len();
        let mut distances = vec![0.0; size * size];
        for (row, &(x1, y1)) in coordinates.iter().enumerate() {
            for (column, &(x2, y2)) in coordinates.iter().enumerate() {
                distances[row * size + column] = ((x1 - x2).hypot(y1 - y2) + 0.5).floor();
            }
        }
        Self { size, distances }
    }

    /// Parses the city count followed by one `x y` pair per city.
    pub fn parse(text: &str) -> Result<Self, InstanceError> {
        let mut tokens = text.split_whitespace();
        let count: usize = tokens
            .next()
            .ok_or(InstanceError::Truncated)?
            .parse()
            .map_err(|_| InstanceError::Malformed("city count".to_owned()))?;
        let mut coordinates = Vec::with_capacity(count);
        for _ in 0..count {
            let mut coordinate = [0.0_f64; 2];
            for slot in &mut coordinate {
                let token = tokens.next().ok_or(InstanceError::Truncated)?;
                *slot = token
                    .parse()
                    .map_err(|_| InstanceError::Malformed(token.to_owned()))?;
            }
            coordinates.push((coordinate[0], coordinate[1]));
        }
        Ok(Self::new(&coordinates))
    }

    /// Length of the tour that visits cities in `order`, closing back to the start.
    #[must_use]
    pub fn tour_length(&self, order: &[u32]) -> f64 {
        let mut previous = *order.last().unwrap_or(&0) as usize;
        let mut total = 0.0;
        for &city in order {
            total += self.distances[previous * self.size + city as usize];
            previous = city as usize;
        }
        total
    }
}

impl Problem for Tsp {
    fn dimension(&self) -> usize {
        self.size
    }

    fn evaluate(&self, keys: &[f64]) -> f64 {
        TOUR_ORDER.with_borrow_mut(|order| {
            order.clear();
            order.extend(0..self.size as u32);
            order.sort_unstable_by(|&left, &right| {
                keys[left as usize]
                    .total_cmp(&keys[right as usize])
                    .then(left.cmp(&right))
            });
            self.tour_length(order)
        })
    }
}

/// Shifted Rastrigin function over `[-5.12, 5.12)^n` decoded linearly from the keys.
///
/// The global minimum is 0 at the shift vector; the landscape is highly multimodal.
#[derive(Clone, Debug)]
pub struct Rastrigin {
    shifts: Vec<f64>,
}

impl Rastrigin {
    /// Builds the function with one optimum coordinate per key.
    #[must_use]
    pub fn new(shifts: Vec<f64>) -> Self {
        Self { shifts }
    }

    /// Parses the dimension followed by one shift per coordinate.
    pub fn parse(text: &str) -> Result<Self, InstanceError> {
        let mut tokens = text.split_whitespace();
        let count: usize = tokens
            .next()
            .ok_or(InstanceError::Truncated)?
            .parse()
            .map_err(|_| InstanceError::Malformed("dimension".to_owned()))?;
        let mut shifts = Vec::with_capacity(count);
        for _ in 0..count {
            let token = tokens.next().ok_or(InstanceError::Truncated)?;
            shifts.push(
                token
                    .parse()
                    .map_err(|_| InstanceError::Malformed(token.to_owned()))?,
            );
        }
        Ok(Self::new(shifts))
    }
}

impl Problem for Rastrigin {
    fn dimension(&self) -> usize {
        self.shifts.len()
    }

    fn evaluate(&self, keys: &[f64]) -> f64 {
        let mut total = 10.0 * self.shifts.len() as f64;
        for (&key, &shift) in keys.iter().zip(&self.shifts) {
            let z = -5.12 + 10.24 * key - shift;
            total += z * z - 10.0 * (std::f64::consts::TAU * z).cos();
        }
        total
    }
}

/// 0/1 Knapsack with a linear capacity-excess penalty.
#[derive(Clone, Debug)]
pub struct Knapsack {
    profits: Vec<f64>,
    weights: Vec<f64>,
    capacity: f64,
}

impl Knapsack {
    /// Penalty applied per unit of capacity excess.
    pub const PENALTY: f64 = 100_000.0;

    /// Builds a Knapsack from parallel profit and weight vectors.
    #[must_use]
    pub fn new(profits: &[i64], weights: &[i64], capacity: i64) -> Self {
        assert_eq!(
            profits.len(),
            weights.len(),
            "profits and weights must align"
        );
        Self {
            profits: profits.iter().map(|&value| value as f64).collect(),
            weights: weights.iter().map(|&value| value as f64).collect(),
            capacity: capacity as f64,
        }
    }

    /// Parses the RKO text format: item count, capacity, then profit/weight pairs.
    pub fn parse(text: &str) -> Result<Self, InstanceError> {
        let mut numbers = Vec::new();
        for token in text.split_whitespace() {
            numbers.push(
                token
                    .parse::<i64>()
                    .map_err(|_| InstanceError::Malformed(token.to_owned()))?,
            );
        }
        if numbers.len() < 2 {
            return Err(InstanceError::Truncated);
        }
        let count = usize::try_from(numbers[0]).map_err(|_| InstanceError::Truncated)?;
        let capacity = numbers[1];
        if numbers.len() < 2 + 2 * count {
            return Err(InstanceError::Truncated);
        }
        let mut profits = Vec::with_capacity(count);
        let mut weights = Vec::with_capacity(count);
        for item in 0..count {
            profits.push(numbers[2 + 2 * item]);
            weights.push(numbers[3 + 2 * item]);
        }
        Ok(Self::new(&profits, &weights, capacity))
    }

    /// Returns profit, weight and feasibility of the selection encoded by `keys`.
    #[must_use]
    pub fn solution(&self, keys: &[f64]) -> (i64, i64, bool) {
        let (profit, weight) = self.totals(keys);
        (profit as i64, weight as i64, weight <= self.capacity)
    }

    /// Capacity of the knapsack.
    #[must_use]
    pub fn capacity(&self) -> f64 {
        self.capacity
    }

    /// Penalized cost for given profit and weight totals.
    #[inline]
    #[must_use]
    pub fn cost_from_totals(&self, profit: f64, weight: f64) -> f64 {
        Self::PENALTY * (weight - self.capacity).max(0.0) - profit
    }

    /// Profit and weight of the selection encoded by `keys` (key above 0.5 selects).
    #[inline]
    #[must_use]
    pub fn totals(&self, keys: &[f64]) -> (f64, f64) {
        let mut profit = 0.0;
        let mut weight = 0.0;
        for ((&key, &item_profit), &item_weight) in
            keys.iter().zip(&self.profits).zip(&self.weights)
        {
            let selected = key > 0.5;
            profit += if selected { item_profit } else { 0.0 };
            weight += if selected { item_weight } else { 0.0 };
        }
        (profit, weight)
    }
}

impl Problem for Knapsack {
    fn dimension(&self) -> usize {
        self.profits.len()
    }

    #[inline]
    fn evaluate(&self, keys: &[f64]) -> f64 {
        let (profit, weight) = self.totals(keys);
        self.cost_from_totals(profit, weight)
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    /// Knapsack scored in blocks that counts its batch calls, for tests only.
    pub(crate) struct BatchKnapsack {
        pub(crate) inner: Knapsack,
        pub(crate) size: usize,
        pub(crate) calls: std::sync::atomic::AtomicUsize,
        pub(crate) rows: std::sync::atomic::AtomicUsize,
    }

    impl BatchKnapsack {
        pub(crate) fn new(inner: Knapsack, size: usize) -> Self {
            Self {
                inner,
                size,
                calls: std::sync::atomic::AtomicUsize::new(0),
                rows: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl Problem for BatchKnapsack {
        fn dimension(&self) -> usize {
            self.inner.dimension()
        }

        fn evaluate(&self, keys: &[f64]) -> f64 {
            self.inner.evaluate(keys)
        }

        fn evaluate_batch(&self, keys: &[f64], out: &mut [f64]) {
            use std::sync::atomic::Ordering;
            assert!(out.len() <= self.size, "block larger than batch_size");
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.rows.fetch_add(out.len(), Ordering::Relaxed);
            for (row, slot) in keys.chunks_exact(self.dimension()).zip(out.iter_mut()) {
                *slot = self.inner.evaluate(row);
            }
        }

        fn batch_size(&self) -> usize {
            self.size
        }
    }

    /// Knapsack wrapper exposing constant-time move evaluation, for tests only.
    #[derive(Clone)]
    pub(crate) struct DeltaKnapsack(pub(crate) Knapsack);

    struct State<'a> {
        problem: &'a Knapsack,
        profit: f64,
        weight: f64,
    }

    impl State<'_> {
        fn shift(&self, keys: &[f64], changes: &[(usize, f64)]) -> (f64, f64) {
            let (mut profit, mut weight) = (self.profit, self.weight);
            for &(index, value) in changes {
                let before = keys[index] > 0.5;
                let after = value > 0.5;
                if before != after {
                    let sign = if after { 1.0 } else { -1.0 };
                    profit += sign * self.problem.profits[index];
                    weight += sign * self.problem.weights[index];
                }
            }
            (profit, weight)
        }
    }

    impl Incremental for State<'_> {
        fn reset(&mut self, keys: &[f64]) {
            (self.profit, self.weight) = self.problem.totals(keys);
        }

        fn probe(&self, keys: &[f64], changes: &[(usize, f64)]) -> f64 {
            let (profit, weight) = self.shift(keys, changes);
            self.problem.cost_from_totals(profit, weight)
        }

        fn commit(&mut self, keys: &[f64], changes: &[(usize, f64)]) {
            (self.profit, self.weight) = self.shift(keys, changes);
        }
    }

    impl Problem for DeltaKnapsack {
        fn dimension(&self) -> usize {
            self.0.dimension()
        }

        fn evaluate(&self, keys: &[f64]) -> f64 {
            self.0.evaluate(keys)
        }

        fn incremental<'a>(&'a self, keys: &[f64]) -> Option<Box<dyn Incremental + 'a>> {
            let mut state = State {
                problem: &self.0,
                profit: 0.0,
                weight: 0.0,
            };
            state.reset(keys);
            Some(Box::new(state))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::DeltaKnapsack;
    use super::*;

    #[test]
    fn evaluates_selection_and_penalty() {
        let problem = Knapsack::new(&[10, 20, 30], &[5, 6, 7], 12);
        assert_eq!(problem.evaluate(&[0.9, 0.9, 0.1]), -30.0);
        assert_eq!(problem.evaluate(&[0.9, 0.9, 0.9]), 100_000.0 * 6.0 - 60.0);
        assert_eq!(problem.solution(&[0.9, 0.9, 0.1]), (30, 11, true));
        assert_eq!(problem.solution(&[0.9, 0.9, 0.9]), (60, 18, false));
    }

    #[test]
    fn incremental_probe_and_commit_match_full_evaluation() {
        let profits: Vec<i64> = (0..40).map(|index| 5 + (index * 17) % 23).collect();
        let weights: Vec<i64> = (0..40).map(|index| 3 + (index * 11) % 19).collect();
        let problem = DeltaKnapsack(Knapsack::new(&profits, &weights, 150));
        let mut rng = crate::rng::Rng::new(12);
        let mut keys: Vec<f64> = (0..40).map(|_| rng.f64()).collect();
        let mut state = problem.incremental(&keys).unwrap();
        for _ in 0..2_000 {
            let count = 1 + rng.below(3);
            let mut changes: Vec<(usize, f64)> = Vec::new();
            for _ in 0..count {
                let index = rng.below(40);
                if changes.iter().all(|change| change.0 != index) {
                    changes.push((index, rng.f64()));
                }
            }
            let mut moved = keys.clone();
            for &(index, value) in &changes {
                moved[index] = value;
            }
            let expected = problem.evaluate(&moved);
            assert_eq!(state.probe(&keys, &changes), expected);
            if rng.f64() < 0.5 {
                state.commit(&keys, &changes);
                keys = moved;
                let mut fresh = problem.incremental(&keys).unwrap();
                fresh.reset(&keys);
                assert_eq!(fresh.probe(&keys, &[]), problem.evaluate(&keys));
                assert_eq!(state.probe(&keys, &[]), problem.evaluate(&keys));
            }
        }
    }

    #[test]
    fn tsp_decodes_by_key_order_and_rounds_distances() {
        let problem = Tsp::new(&[(0.0, 0.0), (3.0, 4.0), (6.0, 0.0), (3.0, -4.0)]);
        assert_eq!(problem.dimension(), 4);
        assert_eq!(problem.evaluate(&[0.1, 0.2, 0.3, 0.4]), 20.0);
        assert_eq!(problem.evaluate(&[0.1, 0.3, 0.2, 0.4]), 24.0);
    }

    #[test]
    fn tsp_parses_and_breaks_key_ties_by_index() {
        let problem = Tsp::parse("3\n0 0\n0 10\n10 0\n").unwrap();
        assert_eq!(
            problem.evaluate(&[0.5, 0.5, 0.5]),
            problem.tour_length(&[0, 1, 2])
        );
        assert!(Tsp::parse("2\n0 0\n1").is_err());
        assert!(Tsp::parse("").is_err());
    }

    #[test]
    fn rastrigin_is_zero_at_its_shifted_optimum() {
        let problem = Rastrigin::parse("3\n1.0 -2.0 0.5\n").unwrap();
        let keys: Vec<f64> = [1.0, -2.0, 0.5]
            .iter()
            .map(|s| (s + 5.12) / 10.24)
            .collect();
        assert!(problem.evaluate(&keys).abs() < 1e-9);
        assert!(problem.evaluate(&[0.9, 0.1, 0.4]) > 1.0);
        assert!(Rastrigin::parse("2\n1.0").is_err());
    }

    #[test]
    fn threshold_is_strict() {
        let problem = Knapsack::new(&[10], &[1], 5);
        assert_eq!(problem.evaluate(&[0.5]), 0.0);
        assert_eq!(problem.evaluate(&[0.5000001]), -10.0);
    }

    #[test]
    fn parses_reference_format() {
        let problem = Knapsack::parse("2\n10\n5 3\n7 4\n").unwrap();
        assert_eq!(problem.dimension(), 2);
        assert_eq!(problem.evaluate(&[1.0, 1.0]), -12.0);
        assert!(Knapsack::parse("3\n10\n1 2").is_err());
        assert!(Knapsack::parse("x").is_err());
    }
}
