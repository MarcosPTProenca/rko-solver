//! BRKGA with community-structure elite refinement (BRKGA-CS).
//!
//! Objective:
//! - Extend BRKGA by clustering elite chromosomes into correlation communities and
//!   intensifying only one representative per community.
//!
//! Technical approach:
//! - Offspring use a crossover with per-gene mutation probability `pm`, otherwise
//!   inheriting from the elite parent with probability `rhoe`.
//! - Every tenth stagnant generation, elites form a graph whose edges are Pearson
//!   correlations at least `sigma` in magnitude; weighted label propagation yields
//!   communities, one random representative each. The first representative is refined
//!   with RVND, the others with Nelder-Mead; non-representatives are randomized
//!   with probability `1 - rhoe` per gene.
//! - Optional Q-learning adapts `(p, pe, pm, rhoe)`.
//!
//! Flow (pseudocode):
//! 1. Score and rank a random population; publish the best.
//! 2. Copy elites, cross offspring, add mutants, refine communities when due.
//! 3. Rerank, publish improvements, update Q-values.
//!
//! Important criteria:
//! - Constant chromosomes and undefined correlations contribute no edge.
//!
//! Inputs/Outputs:
//! - Input: `Spec` with axes `[p, pe, pm, rhoe]` (offline slot 4 may hold `sigma`,
//!   default 0.6) and a search `Context`.
//! - Output: `Result<(), Stop>`; the incumbent and pool carry the result.

use super::population::Population;
use super::{Adapter, Context, Spec};
use crate::local_search::LocalSearch;
use crate::operators::random_keys;
use crate::qlearning::QLearning;
use crate::rng::Rng;
use crate::search::Stop;

fn pearson(left: &[f64], right: &[f64]) -> Option<f64> {
    let count = left.len() as f64;
    let mean_left = left.iter().sum::<f64>() / count;
    let mean_right = right.iter().sum::<f64>() / count;
    let (mut covariance, mut variance_left, mut variance_right) = (0.0, 0.0, 0.0);
    for (&a, &b) in left.iter().zip(right) {
        let (da, db) = (a - mean_left, b - mean_right);
        covariance += da * db;
        variance_left += da * da;
        variance_right += db * db;
    }
    let denominator = (variance_left * variance_right).sqrt();
    let value = covariance / denominator;
    value.is_finite().then_some(value)
}

fn label_propagation(adjacency: &[Vec<(usize, f64)>], rng: &mut Rng) -> Vec<usize> {
    let nodes = adjacency.len();
    let mut labels: Vec<usize> = (0..nodes).collect();
    let mut order: Vec<usize> = (0..nodes).collect();
    for _ in 0..nodes * 10 {
        order.clear();
        order.extend(0..nodes);
        rng.shuffle(&mut order);
        let mut moved = false;
        for &node in &order {
            if adjacency[node].is_empty() {
                continue;
            }
            let mut weights: Vec<(usize, f64)> = Vec::new();
            for &(neighbor, weight) in &adjacency[node] {
                let label = labels[neighbor];
                match weights.iter_mut().find(|entry| entry.0 == label) {
                    Some(entry) => entry.1 += weight,
                    None => weights.push((label, weight)),
                }
            }
            let mut best = weights[0];
            for &entry in &weights[1..] {
                if entry.1 > best.1 {
                    best = entry;
                }
            }
            if best.0 != labels[node] {
                labels[node] = best.0;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    labels
}

fn representatives(next: &Population, elite: usize, sigma: f64, rng: &mut Rng) -> Vec<usize> {
    let correlatable: Vec<bool> = (0..elite)
        .map(|slot| {
            let keys = next.slot(slot);
            keys.len() > 1 && keys.iter().any(|&value| value != keys[0])
        })
        .collect();
    let mut adjacency: Vec<Vec<(usize, f64)>> = vec![Vec::new(); elite];
    for i in 0..elite {
        if !correlatable[i] {
            continue;
        }
        for j in i + 1..elite {
            if !correlatable[j] {
                continue;
            }
            if let Some(correlation) = pearson(next.slot(i), next.slot(j))
                && correlation.abs() >= sigma
            {
                adjacency[i].push((j, correlation.abs()));
                adjacency[j].push((i, correlation.abs()));
            }
        }
    }
    let labels = label_propagation(&adjacency, rng);
    let mut communities: Vec<(usize, Vec<usize>)> = Vec::new();
    for (index, &label) in labels.iter().enumerate() {
        match communities.iter_mut().find(|entry| entry.0 == label) {
            Some(entry) => entry.1.push(index),
            None => communities.push((label, vec![index])),
        }
    }
    communities
        .iter()
        .map(|(_, members)| members[rng.below(members.len())])
        .collect()
}

/// Runs BRKGA-CS until the search budget ends.
pub fn run(spec: &Spec, context: &mut Context<'_, '_>) -> Result<(), Stop> {
    let n = context.search.dimension();
    let sigma = spec.offline.get(4).copied().unwrap_or(0.6);
    let mut adapter = Adapter::new(spec, context)?;
    let mut local = LocalSearch::new();
    let mut p = adapter.params()[0] as usize;
    let mut current = Population::new(n, p);
    let mut next = Population::new(n, p);
    let mut scratch: Vec<f64> = Vec::new();
    random_keys(&mut current.keys[..p * n], context.rng);
    context
        .search
        .eval_rows(&current.keys[..p * n], &mut current.costs[..p])?;
    current.rank(p);
    let mut best_cost = current.costs[current.order[0]];
    context
        .pool
        .insert(best_cost, current.ranked(0), context.source);

    let mut stagnation = 0_usize;
    let mut refined = vec![0.0; n];
    loop {
        adapter.begin(context);
        let (pe, pm, rhoe) = {
            let params = adapter.params();
            p = {
                let wanted = params[0] as usize;
                let size = current.order.len();
                if size != wanted {
                    if size > wanted {
                        current.compact(wanted, &mut scratch);
                    } else {
                        current.resize(wanted);
                        random_keys(&mut current.keys[size * n..wanted * n], context.rng);
                        context.search.eval_rows(
                            &current.keys[size * n..wanted * n],
                            &mut current.costs[size..wanted],
                        )?;
                        current.rank(wanted);
                    }
                    next.resize(wanted);
                }
                wanted
            };
            (params[1], params[2], params[3])
        };

        let elite = ((p as f64 * pe) as usize).max(1);
        let mutants = (p as f64 * (1.0 - pe) * 0.1) as usize;
        let crossovers = p.saturating_sub(elite + mutants);

        for rank in 0..elite {
            let source = current.order[rank];
            next.keys[rank * n..(rank + 1) * n]
                .copy_from_slice(&current.keys[source * n..(source + 1) * n]);
            next.costs[rank] = current.costs[source];
        }
        for index in 0..crossovers {
            let elite_parent = context.rng.below(elite);
            let other_parent = context.rng.range_inclusive(elite, p - 1);
            let slot = elite + index;
            let child = &mut next.keys[slot * n..(slot + 1) * n];
            let (parent_a, parent_b) = (current.ranked(elite_parent), current.ranked(other_parent));
            for gene in 0..n {
                child[gene] = if context.rng.f64() < pm {
                    context.rng.f64()
                } else if context.rng.f64() < rhoe {
                    parent_a[gene]
                } else {
                    parent_b[gene]
                };
            }
        }
        let fresh = elite + crossovers + mutants;
        random_keys(
            &mut next.keys[(elite + crossovers) * n..fresh * n],
            context.rng,
        );
        context.search.eval_rows(
            &next.keys[elite * n..fresh * n],
            &mut next.costs[elite..fresh],
        )?;

        if elite > 1 && (stagnation == 0 || stagnation.is_multiple_of(10)) {
            let promising = representatives(&next, elite, sigma, context.rng);
            for (rank, &slot) in promising.iter().enumerate() {
                refined.copy_from_slice(next.slot(slot));
                let cost = if rank == 0 {
                    local.rvnd(
                        &mut refined,
                        next.costs[slot],
                        context.strategy,
                        context.search,
                        context.pool,
                        context.rng,
                    )?
                } else {
                    local.nelder_mead(
                        &mut refined,
                        next.costs[slot],
                        context.search,
                        context.pool,
                        context.rng,
                    )?
                };
                next.slot_mut(slot).copy_from_slice(&refined);
                next.costs[slot] = cost;
            }
            for slot in 0..elite {
                if promising.contains(&slot) {
                    continue;
                }
                for gene in 0..n {
                    if context.rng.f64() > rhoe {
                        next.keys[slot * n + gene] = context.rng.f64();
                    }
                }
                next.costs[slot] = context.search.eval(next.slot(slot))?;
            }
        }

        std::mem::swap(&mut current, &mut next);
        current.rank(p);
        let leader = current.order[0];
        let leader_cost = current.costs[leader];
        let improved = leader_cost < best_cost;
        if improved {
            best_cost = leader_cost;
            stagnation = 0;
            context
                .pool
                .insert(best_cost, current.slot(leader), context.source);
        } else {
            stagnation += 1;
        }
        let offspring_best = if p > elite {
            current.costs[elite..p]
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min)
        } else {
            best_cost
        };
        adapter.finish(if improved {
            1.0
        } else {
            QLearning::relative_reward(best_cost, offspring_best)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_detects_perfect_and_undefined_correlation() {
        let left = [0.1, 0.2, 0.3, 0.4];
        assert!((pearson(&left, &[0.2, 0.4, 0.6, 0.8]).unwrap() - 1.0).abs() < 1e-12);
        assert!((pearson(&left, &[0.8, 0.6, 0.4, 0.2]).unwrap() + 1.0).abs() < 1e-12);
        assert!(pearson(&left, &[0.5; 4]).is_none());
    }

    #[test]
    fn label_propagation_merges_connected_components() {
        let mut rng = Rng::new(1);
        let adjacency = vec![
            vec![(1, 1.0)],
            vec![(0, 1.0), (2, 1.0)],
            vec![(1, 1.0)],
            vec![(4, 1.0)],
            vec![(3, 1.0)],
            Vec::new(),
        ];
        let labels = label_propagation(&adjacency, &mut rng);
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_ne!(labels[0], labels[3]);
        assert_eq!(labels[5], 5);
    }
}
