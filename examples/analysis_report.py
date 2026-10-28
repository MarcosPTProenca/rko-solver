"""Standalone simulated-annealing solve with offline interactive analysis export.

Objective:
- Demonstrate bounded history capture and post-run HTML/JSON reports for a custom decoder.

Technical approach:
- Decode unit random keys into coordinates in [-5, 5] and minimize squared norm.
- Run one managed sequential search with at most 1000 retained scalar observations.
- Generate a self-contained report only after the timed solve has completed.

Flow (pseudocode):
1. Configure a seeded standalone simulated-annealing search and bounded history.
2. Solve for a short illustrative budget, retaining immutable result metadata.
3. Create the output directory and export HTML plus versioned data-only JSON.

Important criteria:
- The example requires the optional reports extra and never launches a browser.
- The demonstration budget is not competitive-performance evidence.
- Existing example report files are atomically replaced; no candidate keys are exported.

Inputs/Outputs:
- Input: fixed example settings and the caller's working directory.
- Output: results/analysis-report.html and results/analysis-report.json.
"""

import numpy as np
from pathlib import Path
from rko_solver.metaheuristics import SA
from rko_solver.reporting import OptimizationReport


def decode(keys: np.ndarray) -> np.ndarray:
    """Map random keys into bounded numerical coordinates.

    Args:
        keys (np.ndarray): Unit-interval random-key vector.

    Returns:
        np.ndarray: Candidate coordinates in [-5, 5].
    """
    return 10.0 * keys - 5.0


def objective(solution: np.ndarray) -> float:
    """Compute the finite squared norm for minimization.

    Args:
        solution (np.ndarray): Decoded coordinates.

    Returns:
        float: Squared Euclidean norm.
    """
    return float(np.dot(solution, solution))


def main() -> None:
    """Run the example and save post-run artifacts without opening a browser.

    Raises:
        ImportError: If the optional reports extra is unavailable.
        OSError: If the output directory or files cannot be written.
        ValueError: If example settings or objective values fail validation.
    """
    solver = SA(
        decode,
        objective,
        10,
        adaptation="fixed",
        execution="sequential",
        random_state=42,
        history_limit=1000,
    )
    result = solver.solve(time_limit=2.0)
    directory = Path("results")
    directory.mkdir(exist_ok=True)
    report = OptimizationReport(result, title="Standalone simulated annealing")
    print(report.write_html(directory / "analysis-report.html"))
    print(report.write_json(directory / "analysis-report.json"))


if __name__ == "__main__":
    main()
