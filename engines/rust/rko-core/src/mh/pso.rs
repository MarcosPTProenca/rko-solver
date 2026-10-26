//! Particle Swarm Optimization over random keys.
//!
//! Objective:
//! - Move a swarm toward personal and global bests while keeping keys in `[0, 1)`.
//!
//! Technical approach:
//! - Velocity update `w * (v + c1 r1 (pbest - x) + c2 r2 (gbest - x))` with per-gene
//!   uniform `r1, r2`; a proposed position outside `[0, 1)` is rejected and the
//!   velocity is zeroed.
//! - Each generation also refines one random personal best with Nelder-Mead.
//! - Optional Q-learning adapts `(p_size, c1, c2, w)`; the swarm grows with random
//!   particles or is truncated.
//!
//! Flow (pseudocode):
//! 1. Initialize positions and velocities uniformly and score the swarm.
//! 2. Per generation move/evaluate every particle, then refine one personal best.
//! 3. Publish a better global best and apply the Q-learning reward.
//!
//! Important criteria:
//! - Complexity per generation is `O(p * n)` plus the local search.
//!
//! Inputs/Outputs:
//! - Input: `Spec` with axes `[p_size, c1, c2, w]` and a search `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Adapter, Context, Spec};
use crate::local_search::LocalSearch;
use crate::operators::{PoolPolicy, random_keys};
use crate::qlearning::QLearning;
use crate::search::Stop;

/// Runs PSO until the search budget ends.
pub fn run(spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let mut adapter = Adapter::new(spec, context)?;
    let mut local = LocalSearch::with_policy(PoolPolicy::Elite);
    let size = adapter.params()[0] as usize;
    let mut positions = vec![0.0; n * size];
    let mut velocities = vec![0.0; n * size];
    let mut personal = vec![0.0; n * size];
    let mut costs = vec![0.0; size];
    let mut personal_costs = vec![0.0; size];
    random_keys(&mut positions, context.rng);
    random_keys(&mut velocities, context.rng);
    for particle in 0..size {
        costs[particle] = context
            .search
            .eval(&positions[particle * n..(particle + 1) * n])?;
    }
    personal.copy_from_slice(&positions);
    personal_costs.copy_from_slice(&costs);
    let mut leader = 0;
    for particle in 1..size {
        if costs[particle] < costs[leader] {
            leader = particle;
        }
    }
    let mut global = positions[leader * n..(leader + 1) * n].to_vec();
    let mut global_cost = costs[leader];
    context.pool.insert(global_cost, &global, context.source);
    let mut refined = vec![0.0; n];

    loop {
        adapter.begin(context);
        let (swarm, c1, c2, inertia) = {
            let params = adapter.params();
            (params[0] as usize, params[1], params[2], params[3])
        };
        if adapter.adaptive() {
            let current = costs.len();
            if current < swarm {
                positions.resize(n * swarm, 0.0);
                velocities.resize(n * swarm, 0.0);
                personal.resize(n * swarm, 0.0);
                costs.resize(swarm, 0.0);
                personal_costs.resize(swarm, 0.0);
                for particle in current..swarm {
                    let range = particle * n..(particle + 1) * n;
                    random_keys(&mut positions[range.clone()], context.rng);
                    random_keys(&mut velocities[range.clone()], context.rng);
                    let cost = context.search.eval(&positions[range.clone()])?;
                    costs[particle] = cost;
                    personal_costs[particle] = cost;
                    personal[range.clone()].copy_from_slice(&positions[range]);
                }
            } else if current > swarm {
                positions.truncate(n * swarm);
                velocities.truncate(n * swarm);
                personal.truncate(n * swarm);
                costs.truncate(swarm);
                personal_costs.truncate(swarm);
            }
        }

        let mut best_of_generation = f64::INFINITY;
        let mut improved = false;
        let previous_global = global_cost;
        for particle in 0..costs.len() {
            let range = particle * n..(particle + 1) * n;
            let position = &mut positions[range.clone()];
            let velocity = &mut velocities[range.clone()];
            let own_best = &personal[range.clone()];
            for gene in 0..n {
                let r1 = context.rng.f64();
                let r2 = context.rng.f64();
                let updated = inertia
                    * (velocity[gene]
                        + c1 * r1 * (own_best[gene] - position[gene])
                        + c2 * r2 * (global[gene] - position[gene]));
                let proposed = position[gene] + updated;
                if (0.0..1.0).contains(&proposed) {
                    position[gene] = proposed;
                    velocity[gene] = updated;
                } else {
                    velocity[gene] = 0.0;
                }
            }
            let cost = context.search.eval(&positions[range.clone()])?;
            costs[particle] = cost;
            best_of_generation = best_of_generation.min(cost);
            if cost < personal_costs[particle] {
                personal[range.clone()].copy_from_slice(&positions[range.clone()]);
                personal_costs[particle] = cost;
            }
            if cost < global_cost {
                global.copy_from_slice(&positions[range]);
                global_cost = cost;
                improved = true;
            }
        }

        let chosen = context.rng.below(costs.len());
        let range = chosen * n..(chosen + 1) * n;
        refined.copy_from_slice(&personal[range.clone()]);
        let refined_cost = local.nelder_mead(
            &mut refined,
            personal_costs[chosen],
            context.search,
            context.pool,
            context.rng,
        )?;
        if refined_cost < personal_costs[chosen] {
            personal[range].copy_from_slice(&refined);
            personal_costs[chosen] = refined_cost;
            if refined_cost < global_cost {
                global.copy_from_slice(&refined);
                global_cost = refined_cost;
                improved = true;
            }
        }
        if global_cost < previous_global {
            context.pool.insert(global_cost, &global, context.source);
        }
        adapter.finish(if improved {
            1.0
        } else {
            QLearning::relative_reward(global_cost, best_of_generation)
        });
    }
}
