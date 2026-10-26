//! Shared elite solution pool.
//!
//! Objective:
//! - Let concurrent searches exchange their best distinct chromosomes cheaply.
//!
//! Technical approach:
//! - A bounded, cost-sorted vector behind an `RwLock`; searches mostly read two
//!   random entries (Nelder-Mead) and rarely write.
//! - An atomic holds the cost an entry must beat to be admitted, so non-improving
//!   inserts return without touching the lock.
//! - Identical chromosomes are deduplicated; a strictly better observation replaces
//!   the retained one.
//!
//! Flow (pseudocode):
//! 1. Reject a candidate that cannot beat the admission threshold.
//! 2. Under the write lock, drop or replace an identical chromosome.
//! 3. Insert in cost order (after equal costs) and truncate to capacity.
//!
//! Important criteria:
//! - Entries are always sorted by ascending cost and bounded by `capacity`.
//! - Costs and keys must be finite; callers validate through `Search`.
//!
//! Inputs/Outputs:
//! - Input: cost, key slice and a producing-algorithm index.
//! - Output: admission flag, sampled pairs and the current best entry.

use crate::rng::Rng;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// One retained elite chromosome.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Objective value.
    pub cost: f64,
    /// Random keys.
    pub keys: Vec<f64>,
    /// Index of the algorithm that produced the entry.
    pub source: usize,
}

/// Thread-safe fixed-capacity pool of distinct elite chromosomes.
#[derive(Debug)]
pub struct Pool {
    capacity: usize,
    entries: RwLock<Vec<Entry>>,
    threshold_bits: AtomicU64,
}

impl Pool {
    /// Creates an empty pool; `capacity` must be at least one.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 1, "pool capacity must be positive");
        Self {
            capacity,
            entries: RwLock::new(Vec::with_capacity(capacity + 1)),
            threshold_bits: AtomicU64::new(f64::INFINITY.to_bits()),
        }
    }

    /// Configured capacity.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Current occupancy.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.read().expect("pool lock poisoned").len()
    }

    /// Whether the pool holds no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Offers a candidate; returns whether it was admitted and whether it is a new best.
    pub fn insert(&self, cost: f64, keys: &[f64], source: usize) -> (bool, bool) {
        if cost >= f64::from_bits(self.threshold_bits.load(Ordering::Relaxed)) {
            return (false, false);
        }
        let mut entries = self.entries.write().expect("pool lock poisoned");
        if let Some(position) = entries.iter().position(|entry| entry.keys == keys) {
            if cost >= entries[position].cost {
                return (false, false);
            }
            entries.remove(position);
        }
        if entries.len() >= self.capacity && cost >= entries[entries.len() - 1].cost {
            return (false, false);
        }
        let new_best = entries.first().is_none_or(|entry| cost < entry.cost);
        let slot = entries.partition_point(|entry| entry.cost <= cost);
        entries.insert(
            slot,
            Entry {
                cost,
                keys: keys.to_vec(),
                source,
            },
        );
        if entries.len() > self.capacity {
            entries.pop();
        }
        let threshold = if entries.len() >= self.capacity {
            entries[entries.len() - 1].cost
        } else {
            f64::INFINITY
        };
        self.threshold_bits
            .store(threshold.to_bits(), Ordering::Relaxed);
        (true, new_best)
    }

    /// Copies two distinct random entries into the supplied buffers.
    ///
    /// Returns their costs, or `None` when fewer than two entries exist.
    pub fn sample_pair(
        &self,
        rng: &mut Rng,
        first: &mut Vec<f64>,
        second: &mut Vec<f64>,
    ) -> Option<(f64, f64)> {
        let entries = self.entries.read().expect("pool lock poisoned");
        if entries.len() < 2 {
            return None;
        }
        let a = rng.below(entries.len());
        let mut b = rng.below(entries.len() - 1);
        if b >= a {
            b += 1;
        }
        first.clear();
        first.extend_from_slice(&entries[a].keys);
        second.clear();
        second.extend_from_slice(&entries[b].keys);
        Some((entries[a].cost, entries[b].cost))
    }

    /// Copies the two best entries into the supplied buffers.
    ///
    /// Returns their costs, or `None` when fewer than two entries exist.
    pub fn best_pair(&self, first: &mut Vec<f64>, second: &mut Vec<f64>) -> Option<(f64, f64)> {
        let entries = self.entries.read().expect("pool lock poisoned");
        if entries.len() < 2 {
            return None;
        }
        first.clear();
        first.extend_from_slice(&entries[0].keys);
        second.clear();
        second.extend_from_slice(&entries[1].keys);
        Some((entries[0].cost, entries[1].cost))
    }

    /// Returns clones of the top `limit` entries.
    #[must_use]
    pub fn top(&self, limit: usize) -> Vec<Entry> {
        let entries = self.entries.read().expect("pool lock poisoned");
        entries.iter().take(limit).cloned().collect()
    }

    /// Returns a clone of the best entry.
    #[must_use]
    pub fn best(&self) -> Option<Entry> {
        self.entries
            .read()
            .expect("pool lock poisoned")
            .first()
            .cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_sorted_bounded_distinct() {
        let pool = Pool::new(3);
        assert_eq!(pool.insert(5.0, &[0.1], 0), (true, true));
        assert_eq!(pool.insert(3.0, &[0.2], 0), (true, true));
        assert_eq!(pool.insert(4.0, &[0.3], 1), (true, false));
        assert_eq!(pool.insert(6.0, &[0.4], 1), (false, false));
        assert_eq!(pool.insert(3.5, &[0.5], 1), (true, false));
        let costs: Vec<f64> = pool.top(10).iter().map(|entry| entry.cost).collect();
        assert_eq!(costs, vec![3.0, 3.5, 4.0]);
        assert_eq!(pool.len(), 3);
    }

    #[test]
    fn identical_chromosome_only_improves() {
        let pool = Pool::new(3);
        pool.insert(5.0, &[0.1], 0);
        assert_eq!(pool.insert(5.0, &[0.1], 1), (false, false));
        assert_eq!(pool.insert(4.0, &[0.1], 1), (true, true));
        assert_eq!(pool.len(), 1);
        assert_eq!(pool.best().unwrap().cost, 4.0);
    }

    #[test]
    fn equal_cost_distinct_chromosomes_coexist() {
        let pool = Pool::new(3);
        pool.insert(5.0, &[0.1], 0);
        assert_eq!(pool.insert(5.0, &[0.2], 0), (true, false));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn sample_pair_needs_two_and_returns_distinct() {
        let pool = Pool::new(4);
        let mut rng = Rng::new(1);
        let (mut a, mut b) = (Vec::new(), Vec::new());
        pool.insert(1.0, &[0.1], 0);
        assert!(pool.sample_pair(&mut rng, &mut a, &mut b).is_none());
        pool.insert(2.0, &[0.2], 0);
        for _ in 0..50 {
            let (cost_a, cost_b) = pool.sample_pair(&mut rng, &mut a, &mut b).unwrap();
            assert_ne!(cost_a, cost_b);
            assert_ne!(a, b);
        }
    }

    #[test]
    fn concurrent_inserts_preserve_invariants() {
        let pool = Pool::new(10);
        std::thread::scope(|scope| {
            for thread in 0..4_u32 {
                let pool = &pool;
                scope.spawn(move || {
                    let mut rng = Rng::new(u64::from(thread));
                    for _ in 0..2_000 {
                        let key = rng.f64();
                        pool.insert(key, &[key], thread as usize);
                    }
                });
            }
        });
        let entries = pool.top(100);
        assert_eq!(entries.len(), 10);
        assert!(entries.windows(2).all(|pair| pair[0].cost <= pair[1].cost));
    }
}
