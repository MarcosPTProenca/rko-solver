//! Greedy Randomized Adaptive Search Procedure with lattice line search.
//!
//! Objective:
//! - Alternate a constructive phase that commits coordinates one at a time with an
//!   RVND improvement phase, shrinking the line-search step after failures.
//!
//! Technical approach:
//! - Construction fixes `ceil(U(0.3, 0.7) * n)` coordinates; each pass samples
//!   `floor(0.1 * remaining)` candidates (all when fewer than two) and scores each by
//!   a line search over the lattice `z + tau * h` inside `[0, 1)`.
//! - A line search draws `bit_length(floor(1/h) - 1) + 1` lattice points without
//!   replacement using sparse swap-removal, so cost is logarithmic in `1/h`.
//! - The step `h` starts at `hs` and halves after a construction that does not improve
//!   the global best, until it drops below `he`; acceptance of worse solutions follows
//!   a time-decaying Metropolis rule.
//!
//! Flow (pseudocode):
//! 1. Score a random start and publish it.
//! 2. For `h = hs` down to `he`: construct, run RVND, publish and accept.
//! 3. Reward by global-best improvement over the whole `h` sweep.
//!
//! Important criteria:
//! - Lattice arithmetic uses `f64`; boundary decisions may differ from exact
//!   rational arithmetic in the last ulp, which does not affect search statistics.
//!
//! Inputs/Outputs:
//! - Input: `Spec` (offline `[hs, he]`, online `[alpha, hs, he]`) and a `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::{Adapter, Context, Spec};
use crate::local_search::LocalSearch;
use crate::operators::random_keys;
use crate::qlearning::QLearning;
use crate::rng::Rng;
use crate::search::{Search, Stop};

fn bit_length(value: u64) -> u32 {
    u64::BITS - value.leading_zeros()
}

#[derive(Default)]
struct Scratch {
    values: Vec<f64>,
    rows: Vec<f64>,
    costs: Vec<f64>,
}

fn line_search(
    position: usize,
    step: f64,
    keys: &mut [f64],
    search: &mut Search<'_>,
    rng: &mut Rng,
    scratch: &mut Scratch,
) -> Result<(f64, f64), Stop> {
    let origin = keys[position];
    let divisions = ((1.0 / step) as u64).max(1);
    let max_shift = divisions / 2 + 1;
    let positive = ((((1.0 - origin) / step).ceil() as i64) - 1)
        .max(0)
        .min(max_shift as i64) as u64;
    let negative = ((origin / step).floor() as i64)
        .max(0)
        .min(max_shift as i64) as u64;
    let paired = positive.min(negative);
    let mut remaining = 1 + positive + negative;
    let samples = remaining.min(u64::from(bit_length(divisions - 1)) + 1);
    let mut swaps: Vec<(u64, u64)> = Vec::with_capacity(samples as usize + 1);
    let (mut best_value, mut best_cost) = (origin, f64::INFINITY);
    let ceiling = 1.0_f64.next_down();
    scratch.values.clear();
    for _ in 0..samples {
        let index = rng.below(remaining as usize) as u64;
        let rank = swaps
            .iter()
            .find(|entry| entry.0 == index)
            .map_or(index, |entry| entry.1);
        remaining -= 1;
        let tail = swaps
            .iter()
            .find(|entry| entry.0 == remaining)
            .map_or(remaining, |entry| entry.1);
        match swaps.iter_mut().find(|entry| entry.0 == index) {
            Some(entry) => entry.1 = tail,
            None => swaps.push((index, tail)),
        }
        let mut value = origin;
        if rank != 0 {
            let (tau, direction) = if rank <= 2 * paired {
                (rank.div_ceil(2), if rank % 2 == 1 { 1.0 } else { -1.0 })
            } else {
                (rank - paired, if positive > negative { 1.0 } else { -1.0 })
            };
            value = (origin + direction * tau as f64 * step).clamp(0.0, ceiling);
        }
        scratch.values.push(value);
    }
    if search.batch_size() > 1 {
        scratch.rows.clear();
        for &value in &scratch.values {
            let offset = scratch.rows.len();
            scratch.rows.extend_from_slice(keys);
            scratch.rows[offset + position] = value;
        }
        scratch.costs.resize(scratch.values.len(), 0.0);
        search.eval_rows(&scratch.rows, &mut scratch.costs)?;
        for (&value, &cost) in scratch.values.iter().zip(&scratch.costs) {
            if cost < best_cost {
                best_cost = cost;
                best_value = value;
            }
        }
    } else {
        for &value in &scratch.values {
            let cost = search.probe(keys, &[(position, value)])?;
            if cost < best_cost {
                best_cost = cost;
                best_value = value;
            }
        }
    }
    Ok((best_value, best_cost))
}

/// Runs GRASP until the search budget ends.
pub fn run(spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let mut adapter = Adapter::new(spec, context)?;
    let mut local = LocalSearch::new();
    let mut current = vec![0.0; n];
    random_keys(&mut current, context.rng);
    let mut current_cost = context.search.eval(&current)?;
    let mut best_cost = current_cost;
    context.pool.insert(best_cost, &current, context.source);
    let mut candidate = vec![0.0; n];
    let mut unfixed: Vec<usize> = Vec::with_capacity(n);
    let mut scratch = Scratch::default();
    let budget = context.search.budget();

    loop {
        let previous_best = best_cost;
        adapter.begin(context);
        let (start, end) = if adapter.adaptive() {
            (adapter.params()[1], adapter.params()[2])
        } else {
            (adapter.params()[0], adapter.params()[1])
        };
        let mut step = start;
        while step >= end {
            let mut improved = false;
            candidate.copy_from_slice(&current);
            unfixed.clear();
            unfixed.extend(0..n);
            context.search.sync(&candidate);
            let intensity = (context.rng.uniform(0.3, 0.7) * n as f64).ceil() as usize;
            for _ in 0..intensity.min(n) {
                let mut subset = (unfixed.len() as f64 * 0.1) as usize;
                if subset < 2 {
                    subset = unfixed.len();
                }
                for index in 0..subset {
                    let other = index + context.rng.below(unfixed.len() - index);
                    unfixed.swap(index, other);
                }
                let mut chosen = unfixed[0];
                let mut chosen_cost = f64::INFINITY;
                let mut chosen_value = candidate[chosen];
                for &position in &unfixed[..subset] {
                    let (value, cost) = line_search(
                        position,
                        step,
                        &mut candidate,
                        context.search,
                        context.rng,
                        &mut scratch,
                    )?;
                    if cost < chosen_cost {
                        chosen_cost = cost;
                        chosen = position;
                        chosen_value = value;
                    }
                }
                context
                    .search
                    .commit(&mut candidate, &[(chosen, chosen_value)]);
                let index = unfixed
                    .iter()
                    .position(|&value| value == chosen)
                    .unwrap_or(0);
                unfixed.swap_remove(index);
            }
            let constructed = context.search.eval(&candidate)?;
            let refined_cost = local.rvnd(
                &mut candidate,
                constructed,
                context.strategy,
                context.search,
                context.pool,
                context.rng,
            )?;
            if refined_cost < best_cost {
                best_cost = refined_cost;
                improved = true;
                context.pool.insert(best_cost, &candidate, context.source);
            }
            if refined_cost < current_cost {
                current.copy_from_slice(&candidate);
                current_cost = refined_cost;
            } else {
                let temperature = (100.0 - 100.0 * (context.search.elapsed() / budget)).max(1e-9);
                let delta = refined_cost - current_cost;
                if context.rng.f64() < (-delta / temperature).exp() {
                    current.copy_from_slice(&candidate);
                    current_cost = refined_cost;
                }
            }
            if !improved {
                step /= 2.0;
            }
        }
        adapter.finish(if best_cost < previous_best {
            1.0
        } else {
            QLearning::relative_reward(best_cost, current_cost)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::{Knapsack, Problem};
    use std::sync::atomic::AtomicBool;
    use std::time::{Duration, Instant};

    #[test]
    fn line_search_samples_lattice_inside_unit_interval_and_restores_keys() {
        let problem = Knapsack::new(&[3, 5, 7, 2], &[2, 3, 4, 1], 6);
        let cancel = AtomicBool::new(false);
        let now = Instant::now();
        let mut search = Search::new(&problem, &cancel, now, now + Duration::from_secs(5), 0);
        let mut rng = Rng::new(6);
        for origin in [0.0, 0.3, 0.999, 0.5] {
            for step in [0.5, 0.125, 0.0625, 0.001] {
                let mut keys = vec![origin, 0.2, 0.9, 0.1];
                let snapshot = keys.clone();
                let (value, cost) = line_search(
                    0,
                    step,
                    &mut keys,
                    &mut search,
                    &mut rng,
                    &mut Scratch::default(),
                )
                .unwrap();
                assert_eq!(keys, snapshot, "line search must restore the coordinate");
                assert!((0.0..1.0).contains(&value), "value {value}");
                let mut check = snapshot.clone();
                check[0] = value;
                assert_eq!(problem.evaluate(&check), cost);
            }
        }
    }

    #[test]
    fn bit_length_matches_integer_logarithm() {
        assert_eq!(bit_length(0), 0);
        assert_eq!(bit_length(1), 1);
        assert_eq!(bit_length(7), 3);
        assert_eq!(bit_length(1019), 10);
    }
}
