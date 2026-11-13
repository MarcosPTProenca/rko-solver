//! Symmetric Euclidean TSP over random keys, with an exact incremental swap score.
//!
//! Objective:
//! - Score a visiting order decoded from random keys by its closed tour length.
//!
//! Technical approach:
//! - Keys are sorted by value, ties broken by index; distances are rounded to integers (TSPLIB
//!   `EUC_2D`), so every tour length is an exact integer in `f64`.
//! - The incremental state keeps the decoded order and each city's position. Exchanging the key
//!   values of two cities exchanges their positions, so the score changes only on the edges
//!   touching those positions. That exact case is O(1); ties and other changes are re-evaluated.
//!
//! Flow (pseudocode):
//! 1. `evaluate`: sort city indices by (key, index) into a thread-local buffer; sum the tour.
//! 2. `rebuild`: store the order and positions for the current keys and the tour length.
//! 3. `probe` of a two-key exchange without ties: add the change of the affected edges.
//! 4. `commit` of the same exchange: swap the two positions and update the cached length.
//!
//! Important criteria:
//! - An exchange is exact only when no other city has either key value; otherwise the probe
//!   falls back to a full evaluation, so results never depend on the shortcut.
//!
//! Inputs/Outputs:
//! - Input: a key slice whose length equals the city count.
//! - Output: the tour length as a finite `f64`.

use crate::problem::{Incremental, InstanceError, Problem};

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

    fn incremental<'a>(&'a self, keys: &[f64]) -> Option<Box<dyn Incremental + 'a>> {
        let mut state = TourState {
            problem: self,
            order: Vec::with_capacity(self.size),
            position: vec![0; self.size],
            cost: 0.0,
            scratch: std::cell::RefCell::new(Vec::with_capacity(self.size)),
        };
        state.rebuild(keys);
        Some(Box::new(state))
    }
}

/// Incremental tour state that scores a swap of two keys by its four or three affected edges.
///
/// Keeps the decoded order and each city's position. An exact swap exchanges the positions of
/// the two cities, so it is scored in constant time when no other city shares either key value.
/// Any other change is evaluated from scratch.
struct TourState<'a> {
    problem: &'a Tsp,
    order: Vec<u32>,
    position: Vec<usize>,
    cost: f64,
    scratch: std::cell::RefCell<Vec<f64>>,
}

impl TourState<'_> {
    fn rebuild(&mut self, keys: &[f64]) {
        self.order.clear();
        self.order.extend(0..self.problem.size as u32);
        self.order.sort_unstable_by(|&left, &right| {
            keys[left as usize]
                .total_cmp(&keys[right as usize])
                .then(left.cmp(&right))
        });
        for (place, &city) in self.order.iter().enumerate() {
            self.position[city as usize] = place;
        }
        self.cost = self.problem.tour_length(&self.order);
    }

    fn exact_swap_positions(&self, keys: &[f64], i: usize, j: usize) -> Option<(usize, usize)> {
        let n = self.order.len();
        let (pi, pj) = (self.position[i], self.position[j]);
        let tied = |place: usize, key: f64| {
            let before = (place + n - 1) % n;
            let after = (place + 1) % n;
            keys[self.order[before] as usize] == key || keys[self.order[after] as usize] == key
        };
        if tied(pi, keys[i]) || tied(pj, keys[j]) {
            None
        } else {
            Some((pi.min(pj), pi.max(pj)))
        }
    }

    fn swap_delta(&self, p: usize, q: usize) -> f64 {
        let n = self.order.len();
        let mut edges = [p, (p + n - 1) % n, q, (q + n - 1) % n];
        edges.sort_unstable();
        let swapped = |place: usize| match place {
            x if x == p => self.order[q],
            x if x == q => self.order[p],
            x => self.order[x],
        };
        let mut delta = 0.0;
        let mut previous = usize::MAX;
        for &edge in &edges {
            if edge == previous {
                continue;
            }
            previous = edge;
            let (a, b) = (self.order[edge], self.order[(edge + 1) % n]);
            let (c, d) = (swapped(edge), swapped((edge + 1) % n));
            delta += self.problem.distances[c as usize * n + d as usize]
                - self.problem.distances[a as usize * n + b as usize];
        }
        delta
    }

    fn evaluate_changes(&self, keys: &[f64], changes: &[(usize, f64)]) -> f64 {
        let mut moved = self.scratch.borrow_mut();
        moved.clear();
        moved.extend_from_slice(keys);
        for &(index, value) in changes {
            moved[index] = value;
        }
        self.problem.evaluate(&moved)
    }
}

impl Incremental for TourState<'_> {
    fn reset(&mut self, keys: &[f64]) {
        self.rebuild(keys);
    }

    fn is_delta(&self, keys: &[f64], changes: &[(usize, f64)]) -> bool {
        match changes {
            [(i, vi), (j, vj)] if i != j && *vi == keys[*j] && *vj == keys[*i] => {
                keys[*i] == keys[*j] || self.exact_swap_positions(keys, *i, *j).is_some()
            }
            _ => false,
        }
    }

    fn probe(&self, keys: &[f64], changes: &[(usize, f64)]) -> f64 {
        match changes {
            [(i, vi), (j, vj)] if i != j && *vi == keys[*j] && *vj == keys[*i] => {
                if keys[*i] == keys[*j] {
                    return self.cost;
                }
                match self.exact_swap_positions(keys, *i, *j) {
                    Some((p, q)) => self.cost + self.swap_delta(p, q),
                    None => self.evaluate_changes(keys, changes),
                }
            }
            _ => self.evaluate_changes(keys, changes),
        }
    }

    fn commit(&mut self, keys: &[f64], changes: &[(usize, f64)]) {
        match changes {
            [(i, vi), (j, vj)]
                if i != j
                    && *vi == keys[*j]
                    && *vj == keys[*i]
                    && keys[*i] != keys[*j]
                    && self.exact_swap_positions(keys, *i, *j).is_some() =>
            {
                let (p, q) = self.exact_swap_positions(keys, *i, *j).unwrap_or((0, 0));
                self.cost += self.swap_delta(p, q);
                self.order.swap(p, q);
                self.position[self.order[p] as usize] = p;
                self.position[self.order[q] as usize] = q;
            }
            _ => {
                let mut moved = keys.to_vec();
                for &(index, value) in changes {
                    moved[index] = value;
                }
                self.rebuild(&moved);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::local_search::LocalSearch;
    use crate::operators::random_keys;
    use crate::rng::Rng;
    use crate::search::Search;
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn tsp_swap_probe_matches_full_evaluation_for_every_pair() {
        let mut rng = crate::rng::Rng::new(157);
        let coordinates: Vec<(f64, f64)> = (0..30)
            .map(|_| (rng.f64() * 100.0, rng.f64() * 100.0))
            .collect();
        let problem = Tsp::new(&coordinates);
        let mut keys: Vec<f64> = (0..30).map(|_| rng.f64()).collect();
        let mut state = problem.incremental(&keys).unwrap();
        state.reset(&keys);
        for _ in 0..3_000 {
            let i = rng.below(30);
            let j = rng.below(30);
            if i == j {
                continue;
            }
            let changes = [(i, keys[j]), (j, keys[i])];
            let mut swapped = keys.clone();
            swapped.swap(i, j);
            assert_eq!(state.probe(&keys, &changes), problem.evaluate(&swapped));
            if rng.f64() < 0.5 {
                state.commit(&keys, &changes);
                keys = swapped;
            }
        }
    }

    #[test]
    fn tsp_swap_probe_is_exact_with_tied_keys() {
        let mut rng = crate::rng::Rng::new(158);
        let coordinates: Vec<(f64, f64)> = (0..12)
            .map(|_| (rng.f64() * 50.0, rng.f64() * 50.0))
            .collect();
        let problem = Tsp::new(&coordinates);
        let keys: Vec<f64> = (0..12).map(|index| f64::from(index % 3) / 3.0).collect();
        let state = problem.incremental(&keys).unwrap();
        for i in 0..12 {
            for j in 0..12 {
                if i == j {
                    continue;
                }
                let mut swapped = keys.clone();
                swapped.swap(i, j);
                assert_eq!(
                    state.probe(&keys, &[(i, keys[j]), (j, keys[i])]),
                    problem.evaluate(&swapped)
                );
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

    fn swap_sweep(target: &dyn Problem, strategy: u8) -> (Vec<f64>, f64, u64, u64) {
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(target, &cancel, now, now + Duration::from_secs(30), 0);
        let mut rng = Rng::new(21);
        let mut keys = vec![0.0; 25];
        random_keys(&mut keys, &mut rng);
        let start = search.eval(&keys).unwrap();
        let cost = LocalSearch::new()
            .swap(&mut keys, start, strategy, &mut search, &mut rng)
            .unwrap();
        (keys, cost, search.evaluations(), search.delta_evaluations())
    }

    #[test]
    fn tsp_swap_delta_follows_the_plain_trajectory_and_counts_deltas() {
        struct Plain(Tsp);
        impl Problem for Plain {
            fn dimension(&self) -> usize {
                self.0.dimension()
            }
            fn evaluate(&self, keys: &[f64]) -> f64 {
                self.0.evaluate(keys)
            }
        }
        let mut geometry = Rng::new(7);
        let coordinates: Vec<(f64, f64)> = (0..25)
            .map(|_| (geometry.f64() * 100.0, geometry.f64() * 100.0))
            .collect();
        let fast = Tsp::new(&coordinates);
        let slow = Plain(Tsp::new(&coordinates));
        for strategy in [1, 2] {
            let (keys_fast, cost_fast, evals_fast, deltas_fast) = swap_sweep(&fast, strategy);
            let (keys_slow, cost_slow, evals_slow, deltas_slow) = swap_sweep(&slow, strategy);
            assert_eq!(
                (keys_fast, cost_fast, evals_fast),
                (keys_slow, cost_slow, evals_slow)
            );
            assert!(deltas_fast > 0, "strategy {strategy}");
            assert_eq!(deltas_slow, 0, "strategy {strategy}");
        }
    }
}
