//! Genetic Algorithm with tournament selection and uniform crossover.
//!
//! Objective:
//! - Evolve a population by tournament selection, pairwise uniform crossover with
//!   per-gene mutation and a Nelder-Mead refinement of one individual.
//!
//! Technical approach:
//! - Three distinct random individuals compete per selection; selected parents are
//!   duplicated into a working generation.
//! - Each consecutive pair crosses with probability `prob_cros`, swapping genes with
//!   probability 0.5 and mutating each child gene with probability `prob_mut`;
//!   only crossed children are re-evaluated.
//! - Optional Q-learning adapts `(size_pop, prob_cros, prob_mut)`.
//!
//! Flow (pseudocode):
//! 1. Score a random population and publish the best.
//! 2. Select, cross/mutate, refine one individual, replace the generation.
//! 3. Publish a better best and apply the Q-learning reward.
//!
//! Important criteria:
//! - Population size must be at least three for tournament selection.
//!
//! Inputs/Outputs:
//! - Input: `Spec` with axes `[size_pop, prob_cros, prob_mut]` and a search `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Adapter, Context, Spec};
use crate::local_search::LocalSearch;
use crate::operators::{PoolPolicy, random_keys};
use crate::qlearning::QLearning;
use crate::search::Stop;

/// Runs the genetic algorithm until the search budget ends.
pub fn run(spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let mut adapter = Adapter::new(spec, context)?;
    let mut local = LocalSearch::with_policy(PoolPolicy::Elite);
    let size = adapter.params()[0] as usize;
    let mut population = vec![0.0; n * size];
    let mut costs = vec![0.0; size];
    random_keys(&mut population, context.rng);
    context.search.eval_rows(&population, &mut costs)?;
    let mut leader = 0;
    for index in 1..size {
        if costs[index] < costs[leader] {
            leader = index;
        }
    }
    let mut best_cost = costs[leader];
    context.pool.insert(
        best_cost,
        &population[leader * n..(leader + 1) * n],
        context.source,
    );
    let mut selected = vec![0.0; n * size];
    let mut selected_costs = vec![0.0; size];
    let mut offspring = vec![0.0; n * size];
    let mut offspring_costs = vec![0.0; size];
    let mut refined = vec![0.0; n];
    let mut crossed: Vec<usize> = Vec::new();
    let mut gathered: Vec<f64> = Vec::new();
    let mut gathered_costs: Vec<f64> = Vec::new();

    loop {
        adapter.begin(context);
        let (size_pop, crossover, mutation) = {
            let params = adapter.params();
            (params[0] as usize, params[1], params[2])
        };
        if adapter.adaptive() {
            let current = costs.len();
            if current < size_pop {
                population.resize(n * size_pop, 0.0);
                costs.resize(size_pop, 0.0);
                random_keys(&mut population[current * n..size_pop * n], context.rng);
                context.search.eval_rows(
                    &population[current * n..size_pop * n],
                    &mut costs[current..size_pop],
                )?;
            } else if current > size_pop {
                population.truncate(n * size_pop);
                costs.truncate(size_pop);
            }
        }
        let size = costs.len();
        selected.resize(n * size, 0.0);
        selected_costs.resize(size, 0.0);
        offspring.resize(n * size, 0.0);
        offspring_costs.resize(size, 0.0);

        for slot in 0..size {
            let first = context.rng.below(size);
            let mut second = context.rng.below(size - 1);
            if second >= first {
                second += 1;
            }
            let mut third = context.rng.below(size - 2);
            let (low, high) = if first < second {
                (first, second)
            } else {
                (second, first)
            };
            if third >= low {
                third += 1;
            }
            if third >= high {
                third += 1;
            }
            let mut winner = first;
            for rival in [second, third] {
                if costs[rival] < costs[winner] {
                    winner = rival;
                }
            }
            selected[slot * n..(slot + 1) * n]
                .copy_from_slice(&population[winner * n..(winner + 1) * n]);
            selected_costs[slot] = costs[winner];
        }
        offspring.copy_from_slice(&selected);
        offspring_costs.copy_from_slice(&selected_costs);

        crossed.clear();
        let mut pair = 0;
        while pair + 1 < size {
            if context.rng.f64() < crossover {
                let (head, tail) = offspring.split_at_mut((pair + 1) * n);
                let first = &mut head[pair * n..];
                let second = &mut tail[..n];
                let parent_a = &selected[pair * n..(pair + 1) * n];
                let parent_b = &selected[(pair + 1) * n..(pair + 2) * n];
                for gene in 0..n {
                    if context.rng.f64() < 0.5 {
                        first[gene] = parent_b[gene];
                    }
                    if context.rng.f64() < 0.5 {
                        second[gene] = parent_a[gene];
                    }
                    if context.rng.f64() <= mutation {
                        first[gene] = context.rng.f64();
                    }
                    if context.rng.f64() <= mutation {
                        second[gene] = context.rng.f64();
                    }
                }
                crossed.push(pair);
            }
            pair += 2;
        }
        gathered.clear();
        for &pair in &crossed {
            gathered.extend_from_slice(&offspring[pair * n..(pair + 2) * n]);
        }
        gathered_costs.resize(crossed.len() * 2, 0.0);
        context.search.eval_rows(&gathered, &mut gathered_costs)?;
        for (slot, &pair) in crossed.iter().enumerate() {
            offspring_costs[pair] = gathered_costs[2 * slot];
            offspring_costs[pair + 1] = gathered_costs[2 * slot + 1];
        }
        let mut best_offspring = f64::INFINITY;
        let mut pair = 0;
        while pair + 1 < size {
            best_offspring = best_offspring
                .min(offspring_costs[pair])
                .min(offspring_costs[pair + 1]);
            pair += 2;
        }

        let chosen = context.rng.below(size);
        refined.copy_from_slice(&offspring[chosen * n..(chosen + 1) * n]);
        let refined_cost = local.nelder_mead(
            &mut refined,
            offspring_costs[chosen],
            context.search,
            context.pool,
            context.rng,
        )?;
        offspring[chosen * n..(chosen + 1) * n].copy_from_slice(&refined);
        offspring_costs[chosen] = refined_cost;
        best_offspring = best_offspring.min(refined_cost);

        std::mem::swap(&mut population, &mut offspring);
        std::mem::swap(&mut costs, &mut offspring_costs);
        let mut leader = 0;
        for index in 1..size {
            if costs[index] < costs[leader] {
                leader = index;
            }
        }
        let improved = costs[leader] < best_cost;
        if improved {
            best_cost = costs[leader];
            context.pool.insert(
                best_cost,
                &population[leader * n..(leader + 1) * n],
                context.source,
            );
        }
        adapter.finish(if improved {
            1.0
        } else {
            QLearning::relative_reward(best_cost, best_offspring)
        });
    }
}
