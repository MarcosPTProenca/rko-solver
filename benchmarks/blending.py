"""Compare legacy and batched blending throughput inside serial SA.

Objective:
- Measure whether the blending change improves real search evaluation throughput.

Technical approach:
- Recreate the previous Python loop as a benchmark-only reference.
- Run warmed NumPy, 1,000-key serial SA with the same five seeds and time budget.
- Count objective evaluations and report median evaluations per second.

Flow (pseudocode):
1. Replace Nelder-Mead's shared blending function with each measured variant.
2. Run five seeded searches and count completed objective evaluations.
3. Restore the production operator and report per-seed and median throughput.

Important criteria:
- This is one synthetic cheap-objective workload, not a universal performance claim.
- No production algorithm, random stream, or configured dependency is changed.

Inputs/Outputs:
- Input: no external data; the synthetic problem has 1,000 variables.
- Output: per-seed objective counts and median evaluations per second.
"""

import numpy as np
import random
import rko_solver.components.local_search.nelder_mead_ls as nelder_mead
import statistics
import time
from rko_solver import RKO
from rko_solver.components.blending.blending import blending


class _Objective:
    """Count objective calls while evaluating a smooth synthetic minimization task."""

    def __init__(self) -> None:
        """Initialize the per-run evaluation counter."""
        self.evaluations = 0

    def __call__(self, keys: np.ndarray) -> float:
        """Count and evaluate one candidate.

        Args:
            keys (np.ndarray): Random-key vector.

        Returns:
            float: Sum of squared coordinates.
        """
        self.evaluations += 1
        return float(np.dot(keys, keys))


def _reference_blending(
    parent_a: np.ndarray, parent_b: np.ndarray, factor: float = 1.0
) -> np.ndarray:
    """Run the pre-optimization gene loop for benchmark comparison.

    Args:
        parent_a (np.ndarray): First parent key vector.
        parent_b (np.ndarray): Second parent key vector.
        factor (float): Complementary mode factor when equal to -1.

    Returns:
        np.ndarray: Recombined child vector.
    """
    child = np.empty(len(parent_a), dtype=float)
    for index in range(len(parent_a)):
        if random.random() < 0.02:
            child[index] = random.random()
        elif random.random() < 0.5:
            child[index] = parent_a[index]
        elif factor == -1:
            child[index] = min(max(1.0 - float(parent_b[index]), 0.0), 0.9999999)
        else:
            child[index] = parent_b[index]
    return child


def _decode(keys: np.ndarray) -> np.ndarray:
    """Pass the synthetic candidate through unchanged.

    Args:
        keys (np.ndarray): Random-key vector.

    Returns:
        np.ndarray: The same candidate vector.
    """
    return keys


def main() -> None:
    """Run paired fixed-wall-time searches and report median throughput."""
    measurements: dict[str, list[float]] = {"reference": [], "optimized": []}
    original_blending = nelder_mead._blend
    try:
        for label, operator in (
            ("reference", _reference_blending),
            ("optimized", blending),
        ):
            nelder_mead._blend = operator
            for seed in range(200, 205):
                objective = _Objective()
                started = time.perf_counter()
                result = RKO(
                    _decode,
                    objective,
                    1000,
                    algorithms=["SA"],
                    control=0,
                    execution="sequential",
                    random_state=seed,
                ).solve(2.0)
                elapsed = time.perf_counter() - started
                throughput = objective.evaluations / elapsed
                measurements[label].append(throughput)
                print(
                    f"{label} seed={seed} evaluations={objective.evaluations} "
                    f"evaluations_per_second={throughput:.1f} cost={result.cost:.6f}",
                    flush=True,
                )
    finally:
        nelder_mead._blend = original_blending
    reference = statistics.median(measurements["reference"])
    optimized = statistics.median(measurements["optimized"])
    print(f"reference median={reference:.1f} evaluations/s")
    print(f"optimized median={optimized:.1f} evaluations/s")
    print(f"throughput improvement={(optimized / reference - 1) * 100:.1f}%")


if __name__ == "__main__":
    main()
