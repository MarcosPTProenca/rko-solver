//! Simulated Annealing over random keys.
//!
//! Objective:
//! - Minimize by accepting worse neighbors with Metropolis probability while the
//!   temperature cools, restarting at a lower temperature after each schedule.
//!
//! Technical approach:
//! - Neighbors are shaken copies of the current keys scored through `Search::eval`.
//! - After each temperature level a Nelder-Mead pass may produce a new incumbent.
//! - Optional Q-learning chooses `(sa_max, t0, alpha, beta_min, beta_max)` per level
//!   and is rewarded by incumbent improvement or relative level quality.
//!
//! Flow (pseudocode):
//! 1. Score a random start and publish it.
//! 2. For each temperature: `sa_max` shake/evaluate/accept steps, then cool.
//! 3. Refine with Nelder-Mead, update Q-values, reanneal from `0.3 * t0`.
//!
//! Important criteria:
//! - A schedule ends when `T <= 0.0001`; the next one starts at `0.3 * t0`.
//! - A selected `t0` applies from the next schedule.
//!
//! Inputs/Outputs:
//! - Input: `Spec` with axes `[sa_max, t0, alpha, beta_min, beta_max]` and a `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Context, Spec};
use crate::operators::{NelderMead, PoolPolicy, random_keys, shake};
use crate::qlearning::QLearning;
use crate::search::Stop;

/// Temperature multiplier of the first annealing pass and of every later reannealing pass.
const RESTART_SCALE: [f64; 2] = [1.0, 0.3];

#[derive(Clone, Copy)]
struct Parameters {
    sa_max: usize,
    t0: f64,
    alpha: f64,
    beta_min: f64,
    beta_max: f64,
}

impl Parameters {
    fn from(values: &[f64]) -> Self {
        Self {
            sa_max: values[0] as usize,
            t0: values[1],
            alpha: values[2],
            beta_min: values[3],
            beta_max: values[4],
        }
    }
}

/// Runs simulated annealing until the search budget ends.
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

    let mut keys = vec![0.0; n];
    let mut neighbor = vec![0.0; n];
    let mut refined = vec![0.0; n];
    let capacity = context.search.batch_size();
    let mut block = capacity.min(4);
    let mut rows: Vec<f64> = Vec::new();
    let mut row_costs: Vec<f64> = Vec::new();
    random_keys(&mut keys, context.rng);
    let mut current_cost = context.search.eval(&keys)?;
    let mut best_cost = current_cost;
    context.pool.insert(best_cost, &keys, context.source);

    let mut searcher = NelderMead::new(PoolPolicy::Elite);
    let mut reannealing = false;
    loop {
        context.search.check()?;
        let mut temperature = params.t0 * RESTART_SCALE[usize::from(reannealing)];
        while temperature > 0.0001 {
            let previous_best = best_cost;
            let mut action = 0;
            let mut next_state = state;
            let mut schedule = None;
            if let Some(controller) = controller.as_mut() {
                let timing = controller.schedule(context.search.elapsed());
                action = controller.choose_action(state, timing.epsilon, context.rng);
                next_state = controller.state(state).actions[action];
                params = Parameters::from(&controller.state(next_state).parameters);
                schedule = Some(timing);
            }
            let mut level_best = f64::INFINITY;
            let mut steps = 0;
            if capacity <= 1 {
                while steps < params.sa_max {
                    steps += 1;
                    neighbor.copy_from_slice(&keys);
                    shake(&mut neighbor, params.beta_min, params.beta_max, context.rng);
                    let cost = context.search.eval(&neighbor)?;
                    level_best = level_best.min(cost);
                    let delta = cost - current_cost;
                    if delta < 0.0 {
                        keys.copy_from_slice(&neighbor);
                        current_cost = cost;
                        if current_cost < best_cost {
                            best_cost = current_cost;
                            context.pool.insert(best_cost, &keys, context.source);
                        }
                    } else if context.rng.f64() < (-delta / temperature).exp() {
                        keys.copy_from_slice(&neighbor);
                        current_cost = cost;
                    }
                }
            } else {
                while steps < params.sa_max {
                    let count = block.min(params.sa_max - steps).min(capacity);
                    rows.clear();
                    rows.resize(count * n, 0.0);
                    for row in rows.chunks_exact_mut(n) {
                        row.copy_from_slice(&keys);
                        shake(row, params.beta_min, params.beta_max, context.rng);
                    }
                    row_costs.resize(count, 0.0);
                    context.search.eval_rows(&rows, &mut row_costs)?;
                    let mut accepted = None;
                    for (row, &cost) in row_costs.iter().enumerate() {
                        steps += 1;
                        level_best = level_best.min(cost);
                        let delta = cost - current_cost;
                        if delta < 0.0 || context.rng.f64() < (-delta / temperature).exp() {
                            accepted = Some(row);
                            break;
                        }
                    }
                    match accepted {
                        Some(row) => {
                            keys.copy_from_slice(&rows[row * n..(row + 1) * n]);
                            current_cost = row_costs[row];
                            if current_cost < best_cost {
                                best_cost = current_cost;
                                context.pool.insert(best_cost, &keys, context.source);
                            }
                            block = (block / 2).max(2);
                        }
                        None => block = (block * 2).min(capacity),
                    }
                }
            }
            temperature *= params.alpha;
            refined.copy_from_slice(&keys);
            let refined_cost = searcher.run(
                &mut refined,
                current_cost,
                context.search,
                context.pool,
                context.rng,
            )?;
            if refined_cost < best_cost {
                best_cost = refined_cost;
                keys.copy_from_slice(&refined);
                current_cost = refined_cost;
                context.pool.insert(best_cost, &keys, context.source);
            }
            if let (Some(controller), Some(timing)) = (controller.as_mut(), schedule)
                && steps > 0
            {
                let reward = if best_cost < previous_best {
                    1.0
                } else {
                    QLearning::relative_reward(best_cost, level_best)
                };
                controller.update(state, action, reward, timing);
                state = next_state;
            }
        }
        reannealing = true;
    }
}
