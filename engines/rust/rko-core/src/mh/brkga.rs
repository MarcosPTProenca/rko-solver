//! Biased Random-Key Genetic Algorithm.
//!
//! Objective:
//! - Evolve a population of random-key chromosomes with elite-biased crossover.
//!
//! Technical approach:
//! - The population lives in two flat key buffers that swap each generation; ranks
//!   are an index permutation, so parents are read in place and only elites are
//!   copied between generations.
//! - Offspring combine one elite and one non-elite parent gene by gene (parametric
//!   uniform crossover); mutants are fresh random chromosomes.
//! - One elite per generation is refined with Nelder-Mead and its result may enter
//!   the shared pool; optional Q-learning adapts `(p, pe, pm, rhoe)` per generation.
//!
//! Flow (pseudocode):
//! 1. Initialize and rank `p` random chromosomes; publish the best.
//! 2. Each generation: copy elites, cross `p - elite - mutants` offspring, add mutants.
//! 3. Refine one random elite, rerank, publish improvements, update Q-values.
//!
//! Important criteria:
//! - Complexity per generation is `O(p * n)` plus the local search.
//! - Online population changes grow with random chromosomes or truncate the worst.
//!
//! Inputs/Outputs:
//! - Input: `Spec` with axes `[p, pe, pm, rhoe]` and a search `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::population::Population;
use super::{Context, Spec};
use crate::operators::{NelderMead, random_keys};
use crate::qlearning::QLearning;
use crate::rng::Rng;
use crate::search::Stop;

#[derive(Clone, Copy)]
struct Parameters {
    p: usize,
    pe: f64,
    pm: f64,
    rhoe: f64,
}

impl Parameters {
    fn from(values: &[f64]) -> Self {
        Self {
            p: values[0] as usize,
            pe: values[1],
            pm: values[2],
            rhoe: values[3],
        }
    }
}

#[inline]
fn crossover(elite: &[f64], other: &[f64], rhoe: f64, out: &mut [f64], rng: &mut Rng) {
    for ((slot, &from_elite), &from_other) in out.iter_mut().zip(elite).zip(other) {
        *slot = if rng.f64() < rhoe {
            from_elite
        } else {
            from_other
        };
    }
}

/// Runs BRKGA until the search budget ends.
pub fn run(spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let mut controller = if spec.adaptive {
        Some(
            QLearning::new(spec.states.clone(), context.search.budget(), context.rng)
                .map_err(|_| Stop::InvalidCost)?,
        )
    } else {
        None
    };
    let mut state = 0;
    let mut params = Parameters::from(&spec.offline);
    if let Some(controller) = &controller {
        state = context.rng.below(controller.len());
        params = Parameters::from(&controller.state(state).parameters);
    }

    let mut current = Population::new(n, params.p);
    let mut next = Population::new(n, params.p);
    let mut scratch: Vec<f64> = Vec::new();
    random_keys(&mut current.keys[..params.p * n], context.rng);
    context.search.eval_rows(
        &current.keys[..params.p * n],
        &mut current.costs[..params.p],
    )?;
    current.rank(params.p);
    let mut best_cost = current.costs[current.order[0]];
    context
        .pool
        .insert(best_cost, current.ranked(0), context.source);

    let mut searcher = NelderMead::default();
    let mut action = 0;
    let mut schedule = None;
    loop {
        if let Some(controller) = controller.as_mut() {
            let timing = controller.schedule(context.search.elapsed());
            action = controller.choose_action(state, timing.epsilon, context.rng);
            let target = controller.state(state).actions[action];
            params = Parameters::from(&controller.state(target).parameters);
            schedule = Some(timing);
            let size = current.order.len();
            if size != params.p {
                if size > params.p {
                    current.compact(params.p, &mut scratch);
                } else {
                    current.resize(params.p);
                    random_keys(&mut current.keys[size * n..params.p * n], context.rng);
                    context.search.eval_rows(
                        &current.keys[size * n..params.p * n],
                        &mut current.costs[size..params.p],
                    )?;
                    current.rank(params.p);
                }
                next.resize(params.p);
            }
        }

        let p = params.p;
        let elite = ((p as f64 * params.pe) as usize).max(1);
        let mutants = (p as f64 * params.pm) as usize;
        let crossovers = p.saturating_sub(elite + mutants);

        for rank in 0..elite {
            let source = current.order[rank];
            let (head, tail) = (
                &current.keys[source * n..(source + 1) * n],
                &mut next.keys[rank * n..(rank + 1) * n],
            );
            tail.copy_from_slice(head);
            next.costs[rank] = current.costs[source];
        }

        for index in 0..crossovers {
            let elite_parent = context.rng.below(elite);
            let other_parent = context.rng.range_inclusive(elite, p - 1);
            let slot = elite + index;
            let child = &mut next.keys[slot * n..(slot + 1) * n];
            crossover(
                current.ranked(elite_parent),
                current.ranked(other_parent),
                params.rhoe,
                child,
                context.rng,
            );
        }
        random_keys(
            &mut next.keys[(elite + crossovers) * n..(elite + crossovers + mutants) * n],
            context.rng,
        );
        let fresh = elite + crossovers + mutants;
        context.search.eval_rows(
            &next.keys[elite * n..fresh * n],
            &mut next.costs[elite..fresh],
        )?;
        let best_offspring = next.costs[elite..elite + crossovers]
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        let refined = context.rng.below(elite);
        let refined_cost = next.costs[refined];
        let refined_keys = &mut next.keys[refined * n..(refined + 1) * n];
        next.costs[refined] = searcher.run(
            refined_keys,
            refined_cost,
            context.search,
            context.pool,
            context.rng,
        )?;

        std::mem::swap(&mut current, &mut next);
        current.rank(p);
        let leader = current.order[0];
        let leader_cost = current.costs[leader];
        let improved = leader_cost < best_cost;
        if improved {
            best_cost = leader_cost;
            context
                .pool
                .insert(best_cost, current.slot(leader), context.source);
        }

        if let (Some(controller), Some(timing)) = (controller.as_mut(), schedule) {
            let reward = if improved {
                1.0
            } else {
                QLearning::relative_reward(best_cost, best_offspring)
            };
            let following = controller.state(state).actions[action];
            controller.update(state, action, reward, timing);
            state = following;
        }
    }
}
