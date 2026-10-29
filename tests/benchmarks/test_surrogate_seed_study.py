"""Surrogate seed study contract tests.

Objective:
- Check the problem families, the surrogate fit and the decision rule without running a search.

Technical approach:
- Use small instances, synthetic costs with known structure and hand-built trial records.

Flow (pseudocode):
1. Build instances and decode keys.
2. Fit the surrogate on structured and structureless costs.
3. Feed synthetic records to the verdict and normalization.

Important criteria:
- Instances are reproducible, pickleable and decode to valid values.
- A structureless cost yields no seeds; a structured one yields seeds that beat random keys.
- The verdict needs three clear wins and no significant loss.

Inputs/Outputs:
- Input: generated problems and synthetic trial records.
- Output: assertions on decoded values, seeds and verdicts.
"""

import numpy as np
import pickle
import pytest
from benchmarks.engines.surrogate_seed_study import AMPLITUDES
from benchmarks.engines.surrogate_seed_study import ARMS
from benchmarks.engines.surrogate_seed_study import BEST_RANDOM
from benchmarks.engines.surrogate_seed_study import FAMILIES
from benchmarks.engines.surrogate_seed_study import IdentityDecoder
from benchmarks.engines.surrogate_seed_study import Problem
from benchmarks.engines.surrogate_seed_study import ProblemFamily
from benchmarks.engines.surrogate_seed_study import SeedingStudy
from benchmarks.engines.surrogate_seed_study import SphereCost
from benchmarks.engines.surrogate_seed_study import SurrogateSeeds
from typing import Any


@pytest.mark.parametrize("family", FAMILIES)
def test_families_are_reproducible_pickleable_and_finite(family: str) -> None:
    """Each family decodes random keys to a finite objective and survives pickling.

    Args:
        family (str): Problem family.
    """
    problem = ProblemFamily.make(family, 30, 5)
    again = ProblemFamily.make(family, 30, 5)
    keys = np.random.default_rng(0).random(30)
    value = problem.cost(problem.decoder(keys))
    assert np.isfinite(value) and value == again.cost(again.decoder(keys))
    clone = pickle.loads(pickle.dumps((problem.decoder, problem.cost)))
    assert clone[1](clone[0](keys)) == value


def test_unknown_family_is_rejected() -> None:
    """Only the documented families are accepted."""
    with pytest.raises(ValueError, match="family"):
        ProblemFamily.make("other", 10, 1)


def test_tsp_decoder_returns_a_permutation() -> None:
    """The sort decoder visits every city exactly once."""
    problem = ProblemFamily.make("tsp_sort", 25, 2)
    order = problem.decoder(np.random.default_rng(3).random(25))
    assert sorted(order.tolist()) == list(range(25))


def test_surrogate_finds_structure_and_its_seeds_beat_random_keys() -> None:
    """On a sphere the proposed keys are far better than the median random key."""
    problem = Problem("sphere-40-1", IdentityDecoder(), SphereCost(np.linspace(0.1, 0.9, 40)), 40)
    sampler = SurrogateSeeds(problem, 1)
    seeds = sampler.surrogate()
    assert len(seeds) == len(AMPLITUDES) and sampler.correlation >= 0.2
    best = min(problem.cost(np.asarray(seed)) for seed in seeds)
    assert best < float(np.median(sampler.costs))
    assert all(0.0 <= key < 1.0 for seed in seeds for key in seed)


def test_structureless_cost_yields_no_seeds() -> None:
    """A pure-noise objective has no held-out signal, so the surrogate stays silent."""
    noise = np.random.default_rng(7)
    problem = Problem("noise-40-1", IdentityDecoder(), lambda _: float(noise.random()), 40)
    sampler = SurrogateSeeds(problem, 2)
    assert sampler.surrogate() == [] and sampler.correlation < 0.2


def test_best_random_returns_the_lowest_sampled_costs() -> None:
    """The control arm offers the cheapest sampled vectors."""
    problem = Problem("sphere-20-1", IdentityDecoder(), SphereCost(np.full(20, 0.5)), 20)
    sampler = SurrogateSeeds(problem, 4)
    seeds = sampler.best_random()
    assert len(seeds) == BEST_RANDOM
    offered = sorted(problem.cost(np.asarray(seed)) for seed in seeds)
    assert offered == sorted(sampler.costs.tolist())[:BEST_RANDOM]


def records(gaps: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    """Build synthetic trial records with small deterministic jitter.

    Args:
        gaps (dict[str, dict[str, float]]): Mean gap per family and arm.

    Returns:
        list[dict[str, Any]]: Eight instances per family and arm.
    """
    rows = []
    for family, arms in gaps.items():
        for arm in ARMS:
            for index in range(8):
                rows.append({"family": family, "arm": arm, "gap": arms[arm] + 0.01 * (index % 3)})
    return rows


def level(plain: float, surrogate: float, random_arm: float) -> dict[str, float]:
    """Return the per-arm mean gaps of one synthetic family.

    Args:
        plain (float): Plain gap.
        surrogate (float): Surrogate gap.
        random_arm (float): Best-random gap.

    Returns:
        dict[str, float]: Mean gap per arm.
    """
    return {
        "plain": plain,
        "plain_repeat": plain,
        "surrogate": surrogate,
        "best_random": random_arm,
    }


def test_verdict_requires_three_clear_wins_and_no_significant_loss() -> None:
    """Three wins promote the surrogate; one significant loss vetoes it."""
    study = SeedingStudy(10, 1.0, 8, 500)
    win = level(1.0, 0.2, 0.8)
    tie = level(1.0, 1.0, 1.0)
    promoted = study.verdict(records(dict(zip(FAMILIES, [win, win, win, tie, tie], strict=True))))
    assert promoted["promising"] and len(promoted["wins"]) == 3
    vetoed = study.verdict(
        records(dict(zip(FAMILIES, [win, win, win, tie, level(1.0, 1.5, 1.0)], strict=True)))
    )
    assert not vetoed["promising"] and vetoed["significant_losses"] == ["rastrigin"]
    too_few = study.verdict(records(dict(zip(FAMILIES, [win, win, tie, tie, tie], strict=True))))
    assert not too_few["promising"]


def test_a_win_that_only_matches_best_random_does_not_count() -> None:
    """Beating plain while losing to the sampling control is not evidence for the surrogate."""
    study = SeedingStudy(10, 1.0, 8, 500)
    sampled = level(1.0, 0.3, 0.1)
    verdict = study.verdict(records(dict.fromkeys(FAMILIES, sampled)))
    assert verdict["wins"] == [] and not verdict["promising"]


def test_normalization_puts_the_best_known_cost_at_zero() -> None:
    """Gaps are nonnegative and the best arm of an instance scores zero."""
    rows = [
        {"instance": "sphere-10-3000", "family": "sphere", "arm": arm, "cost": cost}
        for arm, cost in zip(ARMS, [0.5, 0.4, 0.3, 0.2], strict=True)
    ]
    done = SeedingStudy.normalize(rows)
    gaps = {row["arm"]: row["gap"] for row in done}
    assert min(gaps.values()) >= 0.0 and gaps["surrogate"] == pytest.approx(0.0)
    assert gaps["plain"] > gaps["best_random"] > gaps["surrogate"]
