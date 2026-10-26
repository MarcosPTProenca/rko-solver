//! Large Neighborhood Search with Farey repair.
//!
//! Objective:
//! - Destroy a random subset of keys, greedily repair them over Farey values and
//!   accept the result with a simulated-annealing criterion.
//!
//! Technical approach:
//! - Destroy sets between `beta_min * n` and `beta_max * n` random keys to zero.
//! - Repair visits each destroyed key and keeps its best Farey candidate, reusing
//!   the running cost so no evaluation is wasted on an unchanged vector.
//! - A Nelder-Mead pass follows; Metropolis acceptance cools geometrically and
//!   restarts at `0.3 * t0` after reaching 0.01.
//!
//! Flow (pseudocode):
//! 1. Start from a random vector refined by Nelder-Mead; publish it.
//! 2. Destroy, repair, refine, accept by Metropolis at the current temperature.
//! 3. Cool and reward by the relative quality against the best before the move.
//!
//! Important criteria:
//! - Per iteration cost is `O(destroyed * 18)` evaluations plus Nelder-Mead.
//!
//! Inputs/Outputs:
//! - Input: `Spec` with axes `[beta_min, beta_max, t0, alpha_lns]` and a `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Adapter, Context, Spec};
use crate::local_search::{FAREY, LocalSearch};
use crate::operators::{PoolPolicy, random_keys};
use crate::qlearning::QLearning;
use crate::search::{Search, Stop};

/// Runs LNS until the search budget ends.
pub fn run(spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let mut adapter = Adapter::new(spec, context)?;
    let mut local = LocalSearch::with_policy(PoolPolicy::Elite);
    let mut current = vec![0.0; n];
    random_keys(&mut current, context.rng);
    let first = context.search.eval(&current)?;
    let mut current_cost = local.nelder_mead(
        &mut current,
        first,
        context.search,
        context.pool,
        context.rng,
    )?;
    let mut best_cost = current_cost;
    context.pool.insert(best_cost, &current, context.source);
    let mut candidate = vec![0.0; n];
    let mut positions: Vec<usize> = (0..n).collect();
    let mut rows: Vec<f64> = Vec::new();
    let mut row_costs: Vec<f64> = Vec::new();
    let mut first_pass = true;
    loop {
        context.search.check()?;
        let mut temperature = if first_pass {
            adapter.params()[2]
        } else {
            adapter.params()[2] * 0.3
        };
        first_pass = false;
        while temperature > 0.01 {
            adapter.begin(context);
            let params = adapter.params();
            let (beta_min, beta_max, alpha) = (params[0], params[1], params[3]);
            let previous_best = best_cost;

            let low = ((beta_min * n as f64) as usize).max(1);
            let high = ((beta_max * n as f64) as usize).max(1);
            let intensity = context
                .rng
                .range_inclusive(low.min(high), high.max(low))
                .min(n);
            candidate.copy_from_slice(&current);
            positions.clear();
            positions.extend(0..n);
            for index in 0..intensity {
                let other = index + context.rng.below(n - index);
                positions.swap(index, other);
                candidate[positions[index]] = 0.0;
            }
            let mut candidate_cost = context.search.eval(&candidate)?;
            candidate_cost = repair(
                &mut candidate,
                candidate_cost,
                &positions[..intensity],
                &mut rows,
                &mut row_costs,
                context.search,
            )?;
            candidate_cost = local.nelder_mead(
                &mut candidate,
                candidate_cost,
                context.search,
                context.pool,
                context.rng,
            )?;

            let delta = candidate_cost - current_cost;
            let accepted = delta <= 0.0
                || (temperature > 0.0 && context.rng.f64() < (-delta / temperature).exp());
            if accepted {
                current.copy_from_slice(&candidate);
                current_cost = candidate_cost;
                if current_cost < best_cost {
                    best_cost = current_cost;
                    context.pool.insert(best_cost, &current, context.source);
                }
            }
            temperature *= alpha;
            adapter.finish(QLearning::relative_reward(previous_best, candidate_cost));
        }
    }
}

/// Re-sets each destroyed position to the best of the Farey values, one position at a time.
fn repair(
    candidate: &mut [f64],
    mut candidate_cost: f64,
    destroyed: &[usize],
    rows: &mut Vec<f64>,
    row_costs: &mut Vec<f64>,
    search: &mut Search<'_>,
) -> Result<f64, Stop> {
    let steps = FAREY.len() - 1;
    for &position in destroyed {
        rows.clear();
        for &value in &FAREY[..steps] {
            let offset = rows.len();
            rows.extend_from_slice(candidate);
            rows[offset + position] = value;
        }
        row_costs.resize(steps, 0.0);
        search.eval_rows(rows, row_costs)?;
        let mut best_value = candidate[position];
        for (&value, &cost) in FAREY[..steps].iter().zip(row_costs.iter()) {
            if cost < candidate_cost {
                candidate_cost = cost;
                best_value = value;
            }
        }
        candidate[position] = best_value;
    }
    Ok(candidate_cost)
}
