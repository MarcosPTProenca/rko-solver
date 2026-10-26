//! Flat population storage with rank indirection.
//!
//! Objective:
//! - Hold a population of key vectors contiguously and rank it without moving keys.
//!
//! Technical approach:
//! - Keys and costs live in flat vectors addressed by slot; `order` maps rank to slot.
//! - Compaction gathers the best `size` ranked entries when a population shrinks.
//!
//! Flow (pseudocode):
//! 1. Evaluate chromosomes into slots, then `rank` sorts the slot permutation.
//! 2. Read parents through `ranked`; copy only elites between generations.
//!
//! Important criteria:
//! - Ranking is a stable sort by `total_cmp`, so equal costs keep slot order.
//!
//! Inputs/Outputs:
//! - Input: key dimension and population size.
//! - Output: slot/rank accessors over keys and costs.

pub(super) struct Population {
    pub(super) n: usize,
    pub(super) keys: Vec<f64>,
    pub(super) costs: Vec<f64>,
    pub(super) order: Vec<usize>,
}

impl Population {
    pub(super) fn new(n: usize, size: usize) -> Self {
        Self {
            n,
            keys: vec![0.0; n * size],
            costs: vec![0.0; size],
            order: (0..size).collect(),
        }
    }

    pub(super) fn resize(&mut self, size: usize) {
        self.keys.resize(self.n * size, 0.0);
        self.costs.resize(size, 0.0);
    }

    #[inline]
    pub(super) fn slot(&self, slot: usize) -> &[f64] {
        &self.keys[slot * self.n..(slot + 1) * self.n]
    }

    #[inline]
    pub(super) fn slot_mut(&mut self, slot: usize) -> &mut [f64] {
        &mut self.keys[slot * self.n..(slot + 1) * self.n]
    }

    #[inline]
    pub(super) fn ranked(&self, rank: usize) -> &[f64] {
        self.slot(self.order[rank])
    }

    pub(super) fn rank(&mut self, size: usize) {
        self.order.clear();
        self.order.extend(0..size);
        let costs = &self.costs;
        self.order
            .sort_by(|&left, &right| costs[left].total_cmp(&costs[right]));
    }

    pub(super) fn compact(&mut self, size: usize, scratch: &mut Vec<f64>) {
        scratch.clear();
        scratch.reserve(self.n * size);
        let mut costs = Vec::with_capacity(size);
        for rank in 0..size.min(self.order.len()) {
            scratch.extend_from_slice(self.ranked(rank));
            costs.push(self.costs[self.order[rank]]);
        }
        std::mem::swap(&mut self.keys, scratch);
        self.costs = costs;
        self.order = (0..self.costs.len()).collect();
    }
}
