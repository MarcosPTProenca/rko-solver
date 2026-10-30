"""Test the decoder-form result on a permutation sort decoder, the traveling salesman problem.

Objective:
- Decide whether a prior tour inside the residual-priority decoder beats the same tour seeded
  through `initial_keys`, on a problem class that is not greedy-fill packing.

Technical approach:
- Two Euclidean families (uniform cities and clustered cities) and two prior tours: nearest
  neighbour, and that tour improved by 2-opt, so the study sees a rough prior and a near-local-
  optimal one.
- The eight arms of the packing study: plain RKO; each prior seeded with one key vector
  (`seed1`) and with 32 perturbed orders (`seed32`); each prior inside the residual decoder
  (`resid`); and a rerun of the strong residual arm, which measures run-to-run noise.
- Instances have 100 cities. Spreads and seeding noises are fractions of the tour, so 0.01
  moves a city about one position; a throwaway smoke test at 250 cities showed that a spread
  of 0.05 there decodes random keys to near-random tours and that the key-space search cannot
  improve a 2-opt tour in the allowance, so the size and the grids were set before the study.
- The residual spread and the seeding noise are each chosen on validation instances at the
  shorter budget by tour length over the Held-Karp lower bound, so the seeding control is tuned
  as much as the decoder.
- Quality is the gap to the shortest tour any arm found and the gap to the lower bound.

Flow (pseudocode):
1. Choose the residual spread and seeding noise per family and prior on validation instances.
2. Solve every test instance with every arm at every budget; charge the prior tour's cost.
3. Re-decode each returned key vector independently and compare with the solver's cost.
4. Compare arms in pairs over instances and apply the decision rule.

Important criteria:
- Decision rule, fixed before any result: advance only if, for both priors, on both families
  and at both budgets, the residual decoder beats 32-order seeding and beats plain RKO, each
  with a paired 95 percent bootstrap interval entirely below zero. If only the strong prior
  holds, that is reported as a partial result and does not advance.
- Test seeds 8000 onward are disjoint from the validation seeds from 6500 onward.

Inputs/Outputs:
- Input: design settings and an output path on the command line.
- Output: a JSON file with chosen parameters, every trial, pair statistics and the decision.
"""

from __future__ import annotations

import numpy as np
from benchmarks.engines.form_study import DecoderFormDesign
from benchmarks.engines.form_study import FormStudy
from benchmarks.engines.solver_run import SolverRun
from benchmarks.engines.tsp_problems import BOUND_STEPS
from benchmarks.engines.tsp_problems import KINDS
from benchmarks.engines.tsp_problems import TspCost
from benchmarks.engines.tsp_problems import TspDecoder
from benchmarks.engines.tsp_problems import TspFamily
from benchmarks.engines.tsp_problems import TspInstance
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

PRIORS = ("nn", "twoopt")
ARMS = (
    "plain",
    "nn_seed1",
    "nn_seed32",
    "nn_resid",
    "twoopt_seed1",
    "twoopt_seed32",
    "twoopt_resid",
    "twoopt_resid_rerun",
)
PAIRS = (
    ("twoopt_resid", "twoopt_seed32"),
    ("twoopt_resid", "twoopt_seed1"),
    ("twoopt_resid", "plain"),
    ("nn_resid", "nn_seed32"),
    ("nn_resid", "nn_seed1"),
    ("nn_resid", "plain"),
    ("twoopt_resid_rerun", "twoopt_resid"),
)
SPREADS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
NOISES = (0.005, 0.01, 0.03, 0.1, 0.3)
CITIES = 100
VALIDATION = 4


class TspDecoderFormStudy(FormStudy[TspInstance]):
    """Run the decoder-form arms on traveling-salesman instances.

    Responsibilities:
    - Build tours as priors, tour-position decoders and perturbed seed keys.
    - Report tour length against the Held-Karp bound, computed once per instance.

    Attributes:
        bounds (dict[str, float]): Cached lower bound per instance name.
    """

    kinds = KINDS
    priors = PRIORS
    arms = ARMS
    pairs = PAIRS
    decision = ("twoopt", "nn", True)
    spread_grid = SPREADS
    noises = NOISES

    def __init__(self, design: DecoderFormDesign) -> None:
        """Store the design and start an empty bound cache.

        Args:
            design (DecoderFormDesign): Study design; `n_items` is the number of cities.
        """
        super().__init__(design)
        self.bounds: dict[str, float] = {}

    def instance(self, kind: str, seed: int) -> TspInstance:
        """Generate an instance at the design size.

        Args:
            kind (str): Instance family.
            seed (int): Instance seed.

        Returns:
            TspInstance: Generated instance.
        """
        return TspFamily.make(kind, self.design.n_items, seed)

    @staticmethod
    def scores(prior: str, instance: TspInstance) -> np.ndarray:
        """Build one prior tour.

        Args:
            prior (str): `nn` or `twoopt`.
            instance (TspInstance): Traveling-salesman instance.

        Returns:
            np.ndarray: Tour as a permutation of the cities.

        Raises:
            ValueError: If the prior is unknown.
        """
        if prior not in PRIORS:
            raise ValueError(f"Unknown prior {prior!r}; expected one of {PRIORS}.")
        tour = TspFamily.nearest_neighbour(instance)
        return TspFamily.two_opt(instance, tour) if prior == "twoopt" else tour

    @staticmethod
    def assemble(
        instance: TspInstance, scores: np.ndarray | None, spread: float, seed: int
    ) -> Callable[[np.ndarray], Any]:
        """Build the sort decoder, plain when there is no prior tour.

        Args:
            instance (TspInstance): Traveling-salesman instance.
            scores (np.ndarray | None): Prior tour, or None for the plain decoder.
            spread (float): Residual scale.
            seed (int): Unused by the decoder.

        Returns:
            Callable[[np.ndarray], Any]: Decoder returning a visiting order.
        """
        return TspDecoder(instance, scores, spread)

    @staticmethod
    def seeds(
        instance: TspInstance, scores: np.ndarray, count: int, noise: float, seed: int
    ) -> list[list[float]]:
        """Encode the prior tour as one key vector and noisy variants of it.

        Args:
            instance (TspInstance): Traveling-salesman instance.
            scores (np.ndarray): Prior tour.
            count (int): Number of key vectors.
            noise (float): Position noise as a fraction of the tour.
            seed (int): Perturbation seed.

        Returns:
            list[list[float]]: Key vectors, the exact tour first.
        """
        return TspFamily.perturbed_keys(scores, count, noise, seed)

    def measure(
        self,
        instance: TspInstance,
        decoder: Callable[[np.ndarray], Any],
        keys: list[list[float]] | None,
        timing: tuple[int, float],
    ) -> dict[str, Any]:
        """Solve once and report the verified tour length and the lower bound.

        Args:
            instance (TspInstance): Traveling-salesman instance.
            decoder (Callable[[np.ndarray], Any]): Decoder.
            keys (list[list[float]] | None): Seed keys or None.
            timing (tuple[int, float]): Solver seed and search seconds.

        Returns:
            dict[str, Any]: `length`, `bound` and `verified`.
        """
        length, verified = SolverRun.solve(
            decoder,
            TspCost(instance),
            instance.n_cities,
            (self.design.execution, self.design.algorithms),
            timing,
            keys,
        )
        if instance.name not in self.bounds:
            self.bounds[instance.name] = TspFamily.lower_bound(instance, BOUND_STEPS)
        return {"length": length, "bound": self.bounds[instance.name], "verified": verified}

    @staticmethod
    def badness(row: dict[str, Any]) -> float:
        """Score a validation trial, lower being better.

        Args:
            row (dict[str, Any]): Trial record.

        Returns:
            float: Tour length over the lower bound.
        """
        return row["length"] / row["bound"]

    @staticmethod
    def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add the gap to the shortest tour any arm found and the gap to the lower bound.

        Args:
            rows (list[dict[str, Any]]): Trial records with `length` and `bound`.

        Returns:
            list[dict[str, Any]]: Records with `gap` and `bound_gap` in percent.
        """
        best: dict[str, float] = {}
        for row in rows:
            best[row["instance"]] = min(best.get(row["instance"], np.inf), row["length"])
        for row in rows:
            row["gap"] = 100.0 * (row["length"] - best[row["instance"]]) / best[row["instance"]]
            row["bound_gap"] = 100.0 * (row["length"] - row["bound"]) / row["bound"]
        return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the traveling-salesman study and write its JSON report.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit code.
    """
    design = DecoderFormDesign(n_items=CITIES, validation_instances=VALIDATION)
    return TspDecoderFormStudy.execute(argv, Path("results/tsp-decoder-form-study.json"), design)


if __name__ == "__main__":
    raise SystemExit(main())
