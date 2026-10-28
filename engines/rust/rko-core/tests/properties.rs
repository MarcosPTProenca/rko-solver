//! Seeded randomized property tests for the random-key operators and the elite pool.
//!
//! Objective:
//! - Check invariants that must hold for every input, using many generated cases instead of a few
//!   hand-written ones, without adding a dependency to this dependency-free crate.
//!
//! Technical approach:
//! - The crate's own `Rng` generates dimensions, keys, costs and operator parameters from a fixed
//!   list of seeds, so every failure is reproducible from the seed in the assertion message.
//!
//! Flow (pseudocode):
//! 1. For each seed build random inputs of random size.
//! 2. Apply an operator or insert into a pool.
//! 3. Assert the invariant on the output.
//!
//! Important criteria:
//! - Operators keep every key finite and inside `[0, 1)`.
//! - The pool stays sorted by cost, holds distinct key vectors and never exceeds its capacity.
//!
//! Inputs/Outputs:
//! - Input: seeds from 0 to 399 and sizes up to 64.
//! - Output: assertions naming the seed that broke an invariant.

use rko_core::mh::Algorithm;
use rko_core::operators::{blend, random_keys, shake};
use rko_core::pool::Pool;
use rko_core::rng::Rng;

const SEEDS: u64 = 400;

fn unit(keys: &[f64]) -> bool {
    keys.iter()
        .all(|key| key.is_finite() && (0.0..1.0).contains(key))
}

#[test]
fn random_keys_are_finite_and_in_the_unit_interval() {
    for seed in 0..SEEDS {
        let mut rng = Rng::new(seed);
        let size = 1 + rng.below(64);
        let mut keys = vec![0.0; size];
        random_keys(&mut keys, &mut rng);
        assert!(unit(&keys), "seed {seed}");
    }
}

#[test]
fn shake_preserves_length_and_the_unit_interval_for_any_intensity() {
    for seed in 0..SEEDS {
        let mut rng = Rng::new(seed);
        let size = 1 + rng.below(64);
        let mut keys = vec![0.0; size];
        random_keys(&mut keys, &mut rng);
        let low = rng.f64();
        let high = low + (1.0 - low) * rng.f64();
        shake(&mut keys, low, high, &mut rng);
        assert_eq!(keys.len(), size, "seed {seed}");
        assert!(unit(&keys), "seed {seed}");
    }
}

#[test]
fn blend_children_stay_in_the_unit_interval_with_and_without_complement() {
    for seed in 0..SEEDS {
        let mut rng = Rng::new(seed);
        let size = 1 + rng.below(64);
        let (mut first, mut second, mut child) =
            (vec![0.0; size], vec![0.0; size], vec![0.0; size]);
        random_keys(&mut first, &mut rng);
        random_keys(&mut second, &mut rng);
        for complement in [false, true] {
            blend(&first, &second, complement, &mut child, &mut rng);
            assert!(unit(&child), "seed {seed} complement {complement}");
        }
    }
}

#[test]
fn rng_below_stays_in_range_and_shuffle_is_a_permutation() {
    for seed in 0..SEEDS {
        let mut rng = Rng::new(seed);
        let bound = 1 + rng.below(1000);
        for _ in 0..50 {
            assert!(rng.below(bound) < bound, "seed {seed}");
        }
        let mut items: Vec<usize> = (0..1 + rng.below(40)).collect();
        let expected = items.clone();
        rng.shuffle(&mut items);
        items.sort_unstable();
        assert_eq!(items, expected, "seed {seed}");
    }
}

#[test]
fn pool_stays_sorted_distinct_and_within_capacity_for_any_insert_sequence() {
    for seed in 0..SEEDS {
        let mut rng = Rng::new(seed);
        let capacity = 1 + rng.below(12);
        let pool = Pool::new(capacity);
        let dimension = 1 + rng.below(4);
        let mut keys = vec![0.0; dimension];
        let mut offered = Vec::new();
        for _ in 0..60 {
            random_keys(&mut keys, &mut rng);
            if rng.below(4) == 0 && !offered.is_empty() {
                keys.clone_from(&offered[rng.below(offered.len())]);
            }
            let cost = f64::from(rng.below(20) as u32) - 10.0;
            pool.insert(cost, &keys, 0);
            offered.push(keys.clone());
            let entries = pool.top(capacity + 5);
            assert!(entries.len() <= capacity, "seed {seed}");
            assert!(
                entries.windows(2).all(|pair| pair[0].cost <= pair[1].cost),
                "seed {seed}"
            );
            for (index, entry) in entries.iter().enumerate() {
                assert!(
                    entries[index + 1..]
                        .iter()
                        .all(|other| other.keys != entry.keys),
                    "seed {seed}: duplicate keys"
                );
            }
            assert_eq!(pool.len(), entries.len(), "seed {seed}");
        }
        let best = pool.best().expect("pool holds something after 60 inserts");
        assert!(pool.top(1)[0].cost == best.cost, "seed {seed}");
    }
}

#[test]
fn pool_best_never_increases_and_equals_the_lowest_admitted_cost() {
    for seed in 0..SEEDS {
        let mut rng = Rng::new(seed);
        let pool = Pool::new(1 + rng.below(8));
        let mut keys = vec![0.0; 3];
        let mut lowest = f64::INFINITY;
        let mut previous = f64::INFINITY;
        for _ in 0..80 {
            random_keys(&mut keys, &mut rng);
            let cost = rng.uniform(-100.0, 100.0);
            pool.insert(cost, &keys, 0);
            lowest = lowest.min(cost);
            let best = pool.best().expect("nonempty").cost;
            assert!(best <= previous, "seed {seed}: best increased");
            assert!(
                (best - lowest).abs() < 1e-12,
                "seed {seed}: best is not the lowest offered"
            );
            previous = best;
        }
    }
}

#[test]
fn brkga_validation_accepts_a_population_only_when_it_holds_a_mutant_and_an_offspring() {
    for seed in 0..SEEDS {
        let mut rng = Rng::new(seed);
        let population = 2.0 + rng.below(200) as f64;
        let elite_fraction = 0.05 + 0.5 * rng.below(100) as f64 / 100.0;
        let mutant_fraction = rng.below(40) as f64 / 100.0;
        let values = [population, elite_fraction, mutant_fraction, 0.7];
        let elite = (population * elite_fraction).floor().max(1.0);
        let mutants = (population * mutant_fraction).floor();
        let expected = mutants >= 1.0 && elite + mutants < population;
        assert_eq!(
            Algorithm::Brkga.validate_values(&values, false).is_ok(),
            expected,
            "seed {seed}: {values:?}"
        );
    }
}
