"""Survival bound and almost-sure discovery for BRKGA's protected exploration.

Objective:
- Check, on a target with known uniform-key mass, that BRKGA finds it at least as often as its
  fresh uniform keys alone guarantee, and that every engine finds it given enough evaluations.

Technical approach:
- The decoder thresholds five keys at one half and the objective is zero only when all five
  bits are one, so a uniform key vector hits the target with mass `mu = 1/32`. Crossover and the
  elite local step can only add hits, never remove the mutants' ones.
- The Python algorithm runs with an injected local-search step that costs exactly one
  evaluation, so a budget of `p + G * (p - elite + 1)` evaluations holds exactly `G` full
  generations and the number of uniform draws is known: `p` initial keys plus `m` per generation.
- An exact one-sided binomial tail turns "the observed miss rate exceeds `(1 - mu)^(p + m*G)`"
  into a significance test over thousands of seeded runs.

Flow (pseudocode):
1. For each generation count `G`, run BRKGA for 2000 seeds with the exact evaluation budget.
2. Count the runs that never reached cost zero and compare with the guaranteed miss bound.
3. Run every engine through the public API with a large budget and require every seed to hit.

Important criteria:
- Seeds are fixed, so each assertion is deterministic; the binomial margin keeps it from failing
  when an unrelated change reorders the random draws.
- The bound is one-sided: observing fewer misses than the bound is expected, because crossover
  and the elite step also reach the target.
- Rust and C++ spend a different, unreported number of evaluations in their local step, so they
  are checked only for discovery within 6000 evaluations. Even if that step cost 30 evaluations
  per generation, the guaranteed draws would make a miss less likely than 1e-10 per seed.

Inputs/Outputs:
- Input: seeded BRKGA runs on a five-bit threshold target.
- Output: assertions on miss counts and on the final cost of every seed.
"""

import math
import numpy as np
import pytest
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.brkga import BRKGA
from rko_solver.orchestration.search import SearchSession

native = pytest.importorskip("rko_solver._native")
BITS = 5
MASS = 0.5**BITS
POPULATION = 10
ELITE_FRACTION = 0.1
MUTANT_FRACTION = 0.5
MUTANTS = int(POPULATION * MUTANT_FRACTION)
PER_GENERATION = POPULATION - max(1, int(POPULATION * ELITE_FRACTION)) + 1
SEEDS = 2000
ALPHA = 1e-6
SETTINGS = {"python": {"execution": "sequential"}, "rust": {"n_jobs": 1}, "cpp": {}}
needs_cpp = pytest.mark.skipif(not native.cpp_available(), reason="built without the C++ engine")
ENGINES = ["python", "rust", pytest.param("cpp", marks=needs_cpp)]


def decode(keys: np.ndarray) -> np.ndarray:
    """Threshold every key at one half.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: Boolean vector, true where the key is at least one half.
    """
    return np.asarray(keys) >= 0.5


def cost(bits: np.ndarray) -> float:
    """Return zero only for the all-ones target.

    Args:
        bits (np.ndarray): Decoded boolean vector.

    Returns:
        float: Zero at the target, one elsewhere.
    """
    return 0.0 if bool(np.all(bits)) else 1.0


def count_evaluation_step(
    keys: np.ndarray, decoder: object, objective: object, *, pool: object = None
) -> tuple[np.ndarray, float]:
    """Replace the elite local step with exactly one objective evaluation.

    Args:
        keys (np.ndarray): Elite key vector chosen for refinement.
        decoder (object): Decoder callable.
        objective (object): Objective callable.
        pool (object): Unused shared pool.

    Returns:
        tuple[np.ndarray, float]: The unchanged keys and their cost.
    """
    return keys, objective(decoder(keys))


def hits_target(seed: int, generations: int) -> bool:
    """Run BRKGA for an exact number of generations and report whether it found the target.

    Args:
        seed (int): Seed of the private random streams.
        generations (int): Full generations covered by the evaluation budget.

    Returns:
        bool: Whether the best cost reached zero.
    """
    algorithm = BRKGA(
        SolutionPool(capacity=4),
        decode,
        cost,
        BITS,
        p=POPULATION,
        pe=ELITE_FRACTION,
        pm=MUTANT_FRACTION,
        rhoe=0.7,
        local_search_operator=count_evaluation_step,
    )
    session = SearchSession(algorithm.decoder, algorithm.cost)
    with RandomStreams(seed).activate():
        result = session.solve(
            algorithm,
            60.0,
            random_state=seed,
            max_evaluations=POPULATION + PER_GENERATION * generations,
        )
    return result[2] == 0.0


def binomial_upper_tail(trials: int, successes: int, probability: float) -> float:
    """Return `P(X >= successes)` for `X ~ Binomial(trials, probability)`.

    Args:
        trials (int): Number of independent trials.
        successes (int): Observed success count.
        probability (float): Success probability under the null hypothesis.

    Returns:
        float: Exact upper-tail probability.
    """
    if successes <= 0:
        return 1.0
    log_p, log_q = math.log(probability), math.log1p(-probability)
    total = 0.0
    for k in range(successes, trials + 1):
        log_term = (
            math.lgamma(trials + 1)
            - math.lgamma(k + 1)
            - math.lgamma(trials - k + 1)
            + k * log_p
            + (trials - k) * log_q
        )
        total += math.exp(log_term)
    return min(1.0, total)


@pytest.mark.parametrize("generations", [0, 2, 6, 10])
def test_the_miss_rate_stays_under_the_protected_exploration_bound(generations: int) -> None:
    """Misses are no more frequent than the initial keys and the mutants alone predict.

    Args:
        generations (int): Full generations covered by the evaluation budget.
    """
    misses = sum(not hits_target(seed, generations) for seed in range(SEEDS))
    guaranteed_draws = POPULATION + MUTANTS * generations
    bound = (1.0 - MASS) ** guaranteed_draws
    assert binomial_upper_tail(SEEDS, misses, bound) > ALPHA, (
        f"{misses}/{SEEDS} misses against a bound of {bound:.4f}"
    )


def test_the_bound_is_informative_not_vacuous() -> None:
    """The observed miss rate is close enough to the bound for a broken engine to show."""
    generations = 6
    misses = sum(not hits_target(seed, generations) for seed in range(SEEDS))
    bound = (1.0 - MASS) ** (POPULATION + MUTANTS * generations)
    assert 0.5 * bound < misses / SEEDS <= bound + 0.05


@pytest.mark.parametrize("engine", ENGINES)
def test_every_engine_finds_a_positive_mass_target_given_enough_evaluations(engine: str) -> None:
    """A miss would need the local step to cost over 30 evaluations per generation to be likely.

    Args:
        engine (str): Engine under test.
    """
    misses = [
        seed
        for seed in range(60)
        if RKO(
            Callbacks(decode, cost, BITS),
            engine=engine,
            algorithms=["BRKGA"],
            adaptation="fixed",
            hyperparameters={
                "BRKGA": {
                    "p": POPULATION,
                    "pe": ELITE_FRACTION,
                    "pm": MUTANT_FRACTION,
                    "rhoe": 0.7,
                }
            },
            max_evaluations=6000,
            random_state=seed,
            **SETTINGS[engine],
        )
        .solve(60.0)
        .cost
        != 0.0
    ]
    assert misses == []
