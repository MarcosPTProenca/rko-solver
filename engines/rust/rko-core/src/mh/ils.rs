//! Iterated Local Search.
//!
//! Objective:
//! - Escape local optima by perturbing the incumbent and re-descending with RVND.
//!
//! Technical approach:
//! - Shake a copy of the best solution with intensity in `[beta_min, beta_max]`, refine
//!   it with RVND and keep it only when it strictly improves the incumbent.
//! - Optional Q-learning adapts `(beta_min, beta_max)` per iteration.
//!
//! Flow (pseudocode):
//! 1. Start from a random vector refined by RVND; publish it.
//! 2. Loop: shake the best, run RVND, accept strict improvements, publish them.
//! 3. Reward 1 for an improvement, else the relative reward of the local optimum.
//!
//! Important criteria:
//! - Each iteration is dominated by RVND sweeps (`O(n^2)` evaluations for swap).
//!
//! Inputs/Outputs:
//! - Input: `Spec` with axes `[beta_min, beta_max]` and a search `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Adapter, Context, Spec};
use crate::local_search::LocalSearch;
use crate::operators::{random_keys, shake};
use crate::qlearning::QLearning;
use crate::search::Stop;

/// Runs ILS until the search budget ends.
pub fn run(spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let mut adapter = Adapter::new(spec, context)?;
    let mut local = LocalSearch::new();
    let mut best = vec![0.0; n];
    random_keys(&mut best, context.rng);
    let first = context.search.eval(&best)?;
    let mut best_cost = local.rvnd(
        &mut best,
        first,
        context.strategy,
        context.search,
        context.pool,
        context.rng,
    )?;
    context.pool.insert(best_cost, &best, context.source);
    let mut candidate = vec![0.0; n];
    loop {
        adapter.begin(context);
        let (beta_min, beta_max) = (adapter.params()[0], adapter.params()[1]);
        candidate.copy_from_slice(&best);
        shake(&mut candidate, beta_min, beta_max, context.rng);
        let start = context.search.eval(&candidate)?;
        let cost = local.rvnd(
            &mut candidate,
            start,
            context.strategy,
            context.search,
            context.pool,
            context.rng,
        )?;
        let improved = cost < best_cost;
        if improved {
            best_cost = cost;
            best.copy_from_slice(&candidate);
            context.pool.insert(best_cost, &best, context.source);
        }
        adapter.finish(if improved {
            1.0
        } else {
            QLearning::relative_reward(best_cost, cost)
        });
    }
}
