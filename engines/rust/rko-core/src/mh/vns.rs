//! Variable Neighborhood Search.
//!
//! Objective:
//! - Systematically enlarge the shaking intensity until RVND finds an improvement.
//!
//! Technical approach:
//! - For `k = 1..k_max` shake the current solution with intensity drawn from
//!   `[k * beta_min, (k + 1) * beta_min)`, refine with RVND and move on strict
//!   improvement; a new global best resets `k` to one.
//! - Optional Q-learning adapts `(k_max, beta_min)` per outer iteration.
//!
//! Flow (pseudocode):
//! 1. Start from a random vector; publish it.
//! 2. Loop over neighborhoods `k`, shake, descend, accept improvements.
//! 3. Reward 1 when the global best improved during the pass, else a relative reward.
//!
//! Important criteria:
//! - Accepting a non-best improvement keeps the same `k`; failures increment it.
//!
//! Inputs/Outputs:
//! - Input: `Spec` with axes `[k_max, beta_min]` and a search `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Adapter, Context, Spec};
use crate::local_search::LocalSearch;
use crate::operators::{PoolPolicy, random_keys, shake};
use crate::qlearning::QLearning;
use crate::search::Stop;

/// Runs VNS until the search budget ends.
pub fn run(spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let mut adapter = Adapter::new(spec, context)?;
    let mut local = LocalSearch::with_policy(PoolPolicy::Elite);
    let mut current = vec![0.0; n];
    random_keys(&mut current, context.rng);
    let mut current_cost = context.search.eval(&current)?;
    let mut best_cost = current_cost;
    context.pool.insert(best_cost, &current, context.source);
    let mut candidate = vec![0.0; n];
    loop {
        adapter.begin(context);
        let k_max = adapter.params()[0] as usize;
        let beta_min = adapter.params()[1];
        let mut improved = false;
        let mut k = 1_usize;
        while k <= k_max {
            let beta = context
                .rng
                .uniform(k as f64 * beta_min, (k + 1) as f64 * beta_min);
            candidate.copy_from_slice(&current);
            shake(&mut candidate, beta, beta, context.rng);
            let start = context.search.eval(&candidate)?;
            let cost = local.rvnd(
                &mut candidate,
                start,
                context.strategy,
                context.search,
                context.pool,
                context.rng,
            )?;
            if cost < current_cost {
                current.copy_from_slice(&candidate);
                current_cost = cost;
                if current_cost < best_cost {
                    best_cost = current_cost;
                    k = 1;
                    improved = true;
                    context.pool.insert(best_cost, &current, context.source);
                }
            } else {
                k += 1;
            }
        }
        adapter.finish(if improved {
            1.0
        } else {
            QLearning::relative_reward(best_cost, current_cost)
        });
    }
}
