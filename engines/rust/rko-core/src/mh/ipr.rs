//! Implicit Path Relinking between pool elites.
//!
//! Objective:
//! - Generate new solutions along greedy block-wise paths between two elites.
//!
//! Technical approach:
//! - Two distinct pool entries are sampled; the keys are split into blocks of
//!   `max(1, floor(0.1 n))`. Each step copies one unfixed guide block into the
//!   current solution, keeps the best neighbor and fixes its block.
//! - A block already equal to the guide costs no evaluation (its cost is the current
//!   cost) but still takes part in first-tie selection.
//! - While the pool holds fewer than two entries the search seeds it up to its capacity
//!   with random evaluated vectors, so relinking has diverse endpoints even when no peer
//!   algorithm fills the pool; it then waits briefly if the pool is still too small.
//!
//! Flow (pseudocode):
//! 1. Seed the pool if needed.
//! 2. Loop: sample two elites, relink, publish the best path solution.
//!
//! Important criteria:
//! - A relink costs at most `blocks * (blocks - 1) / 2` evaluations.
//!
//! Inputs/Outputs:
//! - Input: a search `Context` (no tunable parameters).
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Context, Spec};
use crate::operators::random_keys;
use crate::search::Stop;
use std::time::Duration;

/// Runs implicit path relinking until the search budget ends.
pub fn run(_spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let mut keys = vec![0.0; n];
    random_keys(&mut keys, context.rng);
    let cost = context.search.eval(&keys)?;
    if context.pool.len() < 2 {
        context.pool.insert(cost, &keys, context.source);
        let wanted = context.pool.capacity();
        while context.pool.len() < wanted {
            context.search.check()?;
            random_keys(&mut keys, context.rng);
            let seed_cost = context.search.eval(&keys)?;
            context.pool.insert(seed_cost, &keys, context.source);
        }
    }

    let block_size = ((n as f64 * 0.10) as usize).max(1);
    let blocks = n.div_ceil(block_size);
    let mut fixed = vec![false; blocks];
    let mut current = vec![0.0; n];
    let mut guide = vec![0.0; n];
    let mut path_best = vec![0.0; n];
    let mut iteration_best = vec![0.0; n];
    let mut rows: Vec<f64> = Vec::new();
    let mut row_costs: Vec<f64> = Vec::new();
    let mut scored: Vec<usize> = Vec::new();

    loop {
        if context
            .pool
            .sample_pair(context.rng, &mut current, &mut guide)
            .is_none()
        {
            context.search.check()?;
            std::thread::sleep(Duration::from_millis(1));
            continue;
        }
        fixed.fill(false);
        path_best.copy_from_slice(&current);
        let mut current_cost = context.search.eval(&current)?;
        let mut path_cost = current_cost;
        for _ in 0..(blocks.saturating_sub(1)).max(1) {
            rows.clear();
            scored.clear();
            for (block, &is_fixed) in fixed.iter().enumerate() {
                if is_fixed {
                    continue;
                }
                let start = block * block_size;
                let end = ((block + 1) * block_size).min(n);
                if current[start..end] != guide[start..end] {
                    let offset = rows.len();
                    rows.extend_from_slice(&current);
                    rows[offset + start..offset + end].copy_from_slice(&guide[start..end]);
                    scored.push(block);
                }
            }
            row_costs.resize(scored.len(), 0.0);
            context.search.eval_rows(&rows, &mut row_costs)?;
            let mut iteration_cost = f64::INFINITY;
            let mut chosen_block = None;
            let mut chosen_row = None;
            let mut next_row = 0;
            for (block, &is_fixed) in fixed.iter().enumerate() {
                if is_fixed {
                    continue;
                }
                let (cost, row) = if scored.get(next_row) == Some(&block) {
                    next_row += 1;
                    (row_costs[next_row - 1], Some(next_row - 1))
                } else {
                    (current_cost, None)
                };
                if cost < iteration_cost {
                    iteration_cost = cost;
                    chosen_block = Some(block);
                    chosen_row = row;
                }
            }
            match chosen_row {
                Some(row) => {
                    iteration_best.copy_from_slice(&rows[row * n..(row + 1) * n]);
                }
                None => iteration_best.copy_from_slice(&current),
            }
            if iteration_cost < path_cost {
                path_cost = iteration_cost;
                path_best.copy_from_slice(&iteration_best);
            }
            if let Some(block) = chosen_block {
                current.copy_from_slice(&iteration_best);
                current_cost = iteration_cost;
                fixed[block] = true;
            }
        }
        context.pool.insert(path_cost, &path_best, context.source);
    }
}
