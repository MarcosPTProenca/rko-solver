"""Measure peak memory for a generic GA population transition.

Objective:
- Quantify transient memory used while the genetic algorithm creates a generation.

Technical approach:
- Run one seeded generation on an identity-decoded sphere objective.
- Keep the population size and key dimension fixed and measure Python/NumPy
  allocations with tracemalloc.
- Stub local search only to isolate the GA population transition from Nelder-Mead.

Flow (pseudocode):
1. Create a GA with 100 chromosomes of 5,000 keys.
2. Run initialization and exactly one complete generation.
3. Report tracemalloc's peak allocation and restore the patched operator.

Important criteria:
- The benchmark is domain-neutral and measures an internal GA memory path.
- The local-search stub means this is not whole-solver peak RSS or runtime evidence.
- Identical seeds make the generation transition deterministic for comparison.

Inputs/Outputs:
- Input: fixed population and key dimensions; no external data.
- Output: peak traced allocation in MiB.
"""

import gc
import numpy as np
import random
import rko_solver.metaheuristics.ga as ga_module
import tracemalloc
from collections.abc import Callable
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ga import GeneticAlgorithm
from typing import Any


class _SingleGenerationGA(GeneticAlgorithm):
    """Stop after one full generation for repeatable allocation measurement.

    Responsibilities:
    - Exercise initialization and one complete population transition.
    - Avoid wall-clock variability in the benchmark's generation count.

    Attributes:
        _budget_checks (int): Number of loop-boundary checks made by solve.
    """

    def __init__(
        self,
        solution_pool: SolutionPool,
        decoder: Callable[[np.ndarray], Any],
        cost: Callable[[Any], float],
        tam_solution: int,
        *,
        size_pop: int,
        prob_cros: float,
        prob_mut: float,
    ) -> None:
        """Initialize a GA and its deterministic generation counter.

        Args:
            solution_pool (SolutionPool): Pool used by the metaheuristic.
            decoder (Callable[[np.ndarray], Any]): Random-key decoder.
            cost (Callable[[Any], float]): Objective for decoded candidates.
            tam_solution (int): Random-key vector length.
            size_pop (int): Number of chromosomes in the population.
            prob_cros (float): Probability of applying pair crossover.
            prob_mut (float): Per-key mutation probability.
        """
        super().__init__(
            solution_pool,
            decoder,
            cost,
            tam_solution,
            size_pop=size_pop,
            prob_cros=prob_cros,
            prob_mut=prob_mut,
        )
        self._budget_checks = 0

    def time_limit_reached(self, limit_time: float) -> bool:
        """Permit one generation, then stop at the next loop boundary.

        Args:
            limit_time (float): Ignored time limit retained for the base contract.

        Returns:
            bool: True after one generation has completed.
        """
        self._budget_checks += 1
        return self._budget_checks > 1


def _decode(keys: np.ndarray) -> np.ndarray:
    """Return the generic random-key vector unchanged.

    Args:
        keys (np.ndarray): Candidate chromosome.

    Returns:
        np.ndarray: The supplied chromosome.
    """
    return keys


def _cost(keys: np.ndarray) -> float:
    """Evaluate a generic sphere objective.

    Args:
        keys (np.ndarray): Candidate vector.

    Returns:
        float: Sum of squared coordinates.
    """
    return float(np.dot(keys, keys))


def _local_search_stub(
    keys: np.ndarray,
    decoder: Callable[[np.ndarray], Any],
    cost: Callable[[Any], float],
    pool: object = None,
) -> tuple[np.ndarray, float]:
    """Evaluate one candidate without allocating a Nelder-Mead simplex.

    Args:
        keys (np.ndarray): Candidate chromosome selected for local search.
        decoder (Callable[[np.ndarray], Any]): Decoder callback supplied by the GA.
        cost (Callable[[Any], float]): Objective callback supplied by the GA.
        pool (object): Unused optional elite-pool interface.

    Returns:
        tuple[np.ndarray, float]: Original keys and their objective value.
    """
    return keys, float(cost(decoder(keys)))


def main() -> None:
    """Run one deterministic GA generation and report peak traced memory."""
    random.seed(42)
    np.random.seed(42)
    algorithm = _SingleGenerationGA(
        SolutionPool(capacity=1),
        _decode,
        _cost,
        5000,
        size_pop=100,
        prob_cros=0.8,
        prob_mut=0.02,
    )
    previous_local_search = ga_module.nelder_mead_search
    try:
        ga_module.nelder_mead_search = _local_search_stub
        gc.collect()
        tracemalloc.start()
        result = algorithm.solve(1.0)
        _, peak = tracemalloc.get_traced_memory()
        print(f"peak_traced_memory_mib={peak / 1024**2:.2f}")
        print(f"best_cost={result[2]:.8g}")
    finally:
        tracemalloc.stop()
        ga_module.nelder_mead_search = previous_local_search


if __name__ == "__main__":
    main()
