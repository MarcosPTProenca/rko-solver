"""Test whether a prior inside the decoder beats seeding with that prior, under fair controls.

Objective:
- Decide whether the residual-priority decoder form is worth more than `initial_keys` seeding for
  the same item prior, after the first comparison used one weak seed and a narrow spread grid.

Technical approach:
- Two multi-constraint packing families (five constraints with correlated profits, ten with
  independent profits) and two priors: the capacity-normalized ratio rule and a dual
  pseudo-utility from a subgradient solution of the linear relaxation.
- Eight arms per instance share one charged allowance: plain RKO; each prior seeded with one key
  vector (`seed1`) and with 32 perturbed orders (`seed32`, the stronger seeding control); each
  prior inside a residual decoder (`resid`); and a rerun of the dual residual arm with another
  solver seed, which measures run-to-run noise.
- Each residual spread comes from a six-value grid reaching 0.05, chosen on validation instances
  at the shorter budget by profit over the dual bound; the dual bound also gives an absolute
  quality scale next to the gap to the best profit any arm found.

Flow (pseudocode):
1. Choose each residual spread per family and prior on validation instances.
2. Solve every test instance with every arm at every budget; charge score computation.
3. Re-decode each returned key vector independently and compare with the solver's cost.
4. Compare arms in pairs over instances and apply the decision rule.

Important criteria:
- Decision rule, fixed before any result: advance only if, for the dual prior on both families
  at both budgets, the residual decoder beats 32-order seeding and beats plain RKO, each with a
  paired 95 percent bootstrap interval entirely below zero.
- The ratio-prior comparison and the rerun noise floor are reported but are not part of the rule.
- Test seeds are not those of the earlier learned-priority study, which informed this design.

Inputs/Outputs:
- Input: design settings and an output path on the command line.
- Output: a JSON file with chosen spreads, every trial, pair statistics and the decision.
"""

from __future__ import annotations

import numpy as np
from benchmarks.engines.form_study import SPREADS
from benchmarks.engines.form_study import FormStudy
from benchmarks.engines.learned_priority_study import PriorityStudy
from benchmarks.engines.packing_problems import PackingDecoder
from benchmarks.engines.packing_problems import PackingFamily
from benchmarks.engines.packing_problems import PackingInstance
from benchmarks.engines.packing_problems import PackingRun
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

KINDS = ("mkp", "mkp10")
PRIORS = ("ratio", "dual")
ARMS = (
    "plain",
    "ratio_seed1",
    "ratio_seed32",
    "ratio_resid",
    "dual_seed1",
    "dual_seed32",
    "dual_resid",
    "dual_resid_rerun",
)
PAIRS = (
    ("dual_resid", "dual_seed32"),
    ("dual_resid", "dual_seed1"),
    ("dual_resid", "plain"),
    ("ratio_resid", "ratio_seed32"),
    ("ratio_resid", "plain"),
    ("dual_resid_rerun", "dual_resid"),
)


class DecoderFormStudy(FormStudy[PackingInstance]):
    """Run the decoder-form arms on multi-constraint packing instances.

    Responsibilities:
    - Build item scores as priors, residual packing decoders and perturbed seed keys.
    - Report profit against the dual bound.
    """

    kinds = KINDS
    priors = PRIORS
    arms = ARMS
    pairs = PAIRS
    decision = ("dual", "ratio", False)
    spread_grid = SPREADS
    noises = ()

    def instance(self, kind: str, seed: int) -> PackingInstance:
        """Generate an instance at the design size.

        Args:
            kind (str): Instance family.
            seed (int): Instance seed.

        Returns:
            Any: Generated instance.
        """
        return PackingFamily.make(kind, self.design.n_items, seed)

    @staticmethod
    def scores(prior: str, instance: PackingInstance) -> np.ndarray:
        """Compute one prior's item scores.

        Args:
            prior (str): `ratio` or `dual`.
            instance (PackingInstance): Packing instance.

        Returns:
            np.ndarray: Positive score per item.

        Raises:
            ValueError: If the prior is unknown.
        """
        if prior == "ratio":
            return PackingFamily.hand_scores(instance)
        if prior == "dual":
            return PackingFamily.dual_scores(instance)
        raise ValueError(f"Unknown prior {prior!r}; expected one of {PRIORS}.")

    @staticmethod
    def assemble(
        instance: PackingInstance, scores: np.ndarray | None, spread: float, seed: int
    ) -> Callable[[np.ndarray], Any]:
        """Build the decoder, plain when there is no prior.

        Args:
            instance (PackingInstance): Packing instance.
            scores (np.ndarray | None): Prior, or None for the plain decoder.
            spread (float): Residual scale.
            seed (int): Unused by the packing decoder.

        Returns:
            Callable[[np.ndarray], Any]: Decoder.
        """
        return PackingDecoder(instance, scores, spread)

    @staticmethod
    def seeds(
        instance: PackingInstance, scores: np.ndarray, count: int, noise: float, seed: int
    ) -> list[list[float]]:
        """Encode the prior as one key vector and noisy variants of it.

        Args:
            instance (PackingInstance): Packing instance.
            scores (np.ndarray): Prior.
            count (int): Number of key vectors.
            noise (float): Perturbation scale.
            seed (int): Perturbation seed.

        Returns:
            list[list[float]]: Key vectors, the exact prior first.
        """
        return PackingFamily.perturbed_keys(scores, count, noise, seed)

    def measure(
        self,
        instance: PackingInstance,
        decoder: Callable[[np.ndarray], Any],
        keys: list[list[float]] | None,
        timing: tuple[int, float],
    ) -> dict[str, Any]:
        """Solve once and report the verified objective and its reference value.

        Args:
            instance (PackingInstance): Packing instance.
            decoder (Callable[[np.ndarray], Any]): Decoder.
            keys (list[list[float]] | None): Seed keys or None.
            timing (tuple[int, float]): Solver seed and search seconds.

        Returns:
            dict[str, Any]: `profit`, `bound` and `verified`.
        """
        observed, verified = PackingRun.solve(
            instance, decoder, keys, timing, (self.design.execution, self.design.algorithms)
        )
        return {
            "profit": observed,
            "bound": PackingFamily.dual(instance)[1],
            "verified": verified,
        }

    @staticmethod
    def badness(row: dict[str, Any]) -> float:
        """Score a validation trial, lower being better.

        Args:
            row (dict[str, Any]): Trial record.

        Returns:
            float: Negative profit over the dual bound.
        """
        return -row["profit"] / row["bound"]

    @staticmethod
    def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add the gap to the best profit any arm found and the gap to the dual bound.

        Args:
            rows (list[dict[str, Any]]): Trial records with `profit` and `bound`.

        Returns:
            list[dict[str, Any]]: Records with `gap` and `bound_gap` in percent.
        """
        for row in PriorityStudy.normalize(rows):
            row["bound_gap"] = 100.0 * (row["bound"] - row["profit"]) / row["bound"]
        return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Run the packing study and write its JSON report.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit code.
    """
    return DecoderFormStudy.execute(argv, Path("results/decoder-form-study.json"))


if __name__ == "__main__":
    raise SystemExit(main())
