"""Test the sequential quantile-codec decoder against seeding, on the traveling-salesman problem.

Objective:
- Decide whether drawing the visiting order sequentially from a prior tour, one inverse-CDF draw
  per city, beats the mechanisms already measured for the same prior: 32-order seeding through
  `initial_keys` and the residual decoder of the decoder-form study.

Technical approach:
- The two prior tours and two Euclidean families of the decoder-form study (`tsp_problems.py`):
  nearest neighbour and that tour improved by 2-opt, on uniform and clustered instances.
- Instances have 100 cities, as in the decoder-form study. The temperature grid starts at 0.003:
  `QuantileCodec` turns the softmax into `exp((scores - scores.max()) / temperature)`, and below
  about 0.0013 every weight but the maximum underflows to zero in float64, which degenerates the
  draw into a non-permutation. A smoke check before the study confirmed this boundary and that the
  chosen temperature sits at 0.003, the grid's floor, on every family and prior.
- Four arms per instance at one charged allowance: plain RKO; each prior seeded with 32 perturbed
  orders (`seed32`, as in the decoder-form study); each prior through `TspQuantileDecoder`
  (`quantile`); and a rerun of the 2-opt quantile arm, which measures run-to-run noise.
- `TspQuantileDecoder` charges one softmax and one cumulative sum per decoded candidate, not one
  key-vector preparation like the other arms, so its preparation cost is zero and its per-call
  cost is paid inside the charged allowance like the plain decoder's.
- The softmax temperature is chosen on validation instances at the shorter budget by tour length
  over the Held-Karp lower bound, the same selection the decoder-form study uses for its spread.

Flow (pseudocode):
1. Choose the temperature per family and prior on validation instances.
2. Solve every test instance with every arm at every budget; verify every returned key vector.
3. Compare arm pairs over instances and apply the decision rule.

Important criteria:
- Decision rule, fixed before any result: advance only if, for both priors, on both families and
  at both budgets, the quantile decoder beats 32-order seeding and beats plain RKO, each with a
  paired 95 percent bootstrap interval entirely below zero. If only the 2-opt prior holds, that is
  reported as a partial result and does not advance.
- Test seeds are disjoint from validation seeds, reusing the decoder-form study's ranges.

Inputs/Outputs:
- Input: design settings and an output path on the command line.
- Output: a JSON file with the chosen temperatures, every trial, pair statistics and the verdict.
"""

from __future__ import annotations

import argparse
import numpy as np
import sys
import time
from benchmarks.engines.form_study import SEED_COUNT
from benchmarks.engines.form_study import SEED_NOISE
from benchmarks.engines.form_study import DecoderFormDesign
from benchmarks.engines.learned_priority_study import PriorityStudy
from benchmarks.engines.report_sink import ReportSink
from benchmarks.engines.solver_run import SolverRun
from benchmarks.engines.tsp_decoder_form_study import TspDecoderFormStudy
from benchmarks.engines.tsp_problems import BOUND_STEPS
from benchmarks.engines.tsp_problems import KINDS
from benchmarks.engines.tsp_problems import TspCost
from benchmarks.engines.tsp_problems import TspFamily
from benchmarks.engines.tsp_problems import TspInstance
from benchmarks.engines.tsp_problems import TspQuantileDecoder
from benchmarks.engines.tsp_problems import TspReport
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

PRIORS = ("nn", "twoopt")
DECISIVE_PAIRS = {
    "twoopt": ("twoopt_quantile-vs-twoopt_seed32", "twoopt_quantile-vs-plain"),
    "nn": ("nn_quantile-vs-nn_seed32", "nn_quantile-vs-plain"),
}
ARMS = (
    "plain",
    "nn_seed32",
    "nn_quantile",
    "twoopt_seed32",
    "twoopt_quantile",
    "twoopt_quantile_rerun",
)
PAIRS = (
    ("twoopt_quantile", "twoopt_seed32"),
    ("twoopt_quantile", "plain"),
    ("nn_quantile", "nn_seed32"),
    ("nn_quantile", "plain"),
    ("twoopt_quantile_rerun", "twoopt_quantile"),
)
TEMPERATURES = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
RERUN_OFFSET = 7919


class TsqDecision:
    """Apply the decision rule fixed before the quantile-codec study.

    Responsibilities:
    - Decide from paired statistics whether the quantile decoder clearly beats 32-order seeding
      and plain RKO, for both priors, every family and every budget.
    """

    @staticmethod
    def holds(tables: dict[str, dict[str, dict[str, dict[str, float]]]], prior: str) -> bool:
        """Check that the quantile arm beats its listed comparisons everywhere.

        Args:
            tables (dict[str, dict[str, dict[str, dict[str, float]]]]): Paired statistics keyed by
                family, budget label and `arm-vs-arm` pair.
            prior (str): `nn` or `twoopt`.

        Returns:
            bool: Whether every family and budget has `ci_high` below zero on both of the
            prior's decisive pairs.
        """
        return all(
            budget[pair]["ci_high"] < 0
            for family in tables.values()
            for budget in family.values()
            for pair in DECISIVE_PAIRS[prior]
        )

    @staticmethod
    def verdict(tables: dict[str, dict[str, dict[str, dict[str, float]]]]) -> dict[str, Any]:
        """Evaluate the rule on paired comparisons.

        Args:
            tables (dict[str, dict[str, dict[str, dict[str, float]]]]): Statistics keyed by
                family, budget label and `arm-vs-arm` pair, each holding `difference`, `ci_low`
                and `ci_high`.

        Returns:
            dict[str, Any]: `primary_holds` for the 2-opt prior, `secondary_holds` for the
            nearest-neighbour prior, the mean absolute rerun difference `noise_floor` and the
            `advance` flag, true only if both hold.
        """
        primary = TsqDecision.holds(tables, "twoopt")
        secondary = TsqDecision.holds(tables, "nn")
        noise = [
            abs(budget["twoopt_quantile_rerun-vs-twoopt_quantile"]["difference"])
            for family in tables.values()
            for budget in family.values()
        ]
        return {
            "primary_holds": primary,
            "secondary_holds": secondary,
            "noise_floor": float(np.mean(noise)),
            "advance": primary and secondary,
        }


class TspQuantileStudy:
    """Run the plain, seeded and quantile arms, tune the temperature and summarize.

    Responsibilities:
    - Build decoders and seed keys for each arm.
    - Verify every returned key vector independently of the solver.

    Attributes:
        design (DecoderFormDesign): Fixed design.
        bounds (dict[str, float]): Cached lower bound per instance name.
    """

    def __init__(self, design: DecoderFormDesign) -> None:
        """Store the design and start an empty bound cache.

        Args:
            design (DecoderFormDesign): Study design; `n_items` is the number of cities.
        """
        self.design = design
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
    def parse(arm: str) -> tuple[str | None, str]:
        """Split an arm name into its prior and mechanism.

        Args:
            arm (str): One of `ARMS`.

        Returns:
            tuple[str | None, str]: Prior (None for plain) and mechanism (`none`, `seed32`,
            `quantile` or `quantile_rerun`).

        Raises:
            ValueError: If the arm is unknown.
        """
        if arm == "plain":
            return None, "none"
        prior, _, mechanism = arm.partition("_")
        if prior in PRIORS and mechanism in ("seed32", "quantile", "quantile_rerun"):
            return prior, mechanism
        raise ValueError(f"Unknown arm {arm!r}.")

    def setup(
        self, arm: str, instance: TspInstance, temperatures: dict[str, float], seed: int
    ) -> tuple[Callable[[np.ndarray], Any], list[list[float]] | None, float]:
        """Build an arm's decoder and seed keys and time their preparation.

        Args:
            arm (str): One of `ARMS`.
            instance (TspInstance): Instance.
            temperatures (dict[str, float]): Chosen softmax temperature per prior.
            seed (int): Perturbation seed.

        Returns:
            tuple[Callable[[np.ndarray], Any], list[list[float]] | None, float]: Decoder, initial
            keys or None, and preparation seconds.
        """
        prior, mechanism = self.parse(arm)
        if prior is None:
            return TspQuantileStudy.plain_decoder(instance), None, 0.0
        started = time.perf_counter()
        tour = TspDecoderFormStudy.scores(prior, instance)
        if mechanism.startswith("quantile"):
            decoder, keys = TspQuantileDecoder.build(instance, tour, temperatures[prior]), None
        else:
            decoder = TspQuantileStudy.plain_decoder(instance)
            keys = TspFamily.perturbed_keys(tour, SEED_COUNT, SEED_NOISE, seed)
        return decoder, keys, time.perf_counter() - started

    @staticmethod
    def plain_decoder(instance: TspInstance) -> Callable[[np.ndarray], Any]:
        """Build the plain sort decoder.

        Args:
            instance (TspInstance): Instance.

        Returns:
            Callable[[np.ndarray], Any]: Decoder returning a visiting order.
        """
        return TspDecoderFormStudy.assemble(instance, None, 1.0, 0)

    def solve(
        self,
        instance: TspInstance,
        arm: str,
        context: tuple[dict[str, float], float],
        seed: int,
    ) -> dict[str, Any]:
        """Run one arm on one instance within one allowance.

        Args:
            instance (TspInstance): Instance.
            arm (str): One of `ARMS`.
            context (tuple[dict[str, float], float]): Chosen temperatures and the allowance.
            seed (int): Instance seed; the rerun arm uses a different solver seed.

        Returns:
            dict[str, Any]: Arm, budget, tour length, verification and phase timings.
        """
        temperatures, budget = context
        decoder, keys, prep = self.setup(arm, instance, temperatures, seed)
        allowance = PriorityStudy.search_allowance(budget, prep)
        started = time.perf_counter()
        solver_seed = seed + RERUN_OFFSET if arm.endswith("_rerun") else seed
        length, verified = SolverRun.solve(
            decoder,
            TspCost(instance),
            instance.n_cities,
            (self.design.execution, self.design.algorithms),
            (solver_seed, allowance),
            keys,
        )
        if instance.name not in self.bounds:
            self.bounds[instance.name] = TspFamily.lower_bound(instance, BOUND_STEPS)
        family = instance.name.split("-")[0]
        return {
            "instance": instance.name,
            "family": family,
            "arm": arm,
            "budget": budget,
            "length": length,
            "bound": self.bounds[instance.name],
            "verified": verified,
            "prep_seconds": prep,
            "allowance": allowance,
            "wall_seconds": time.perf_counter() - started + prep,
        }

    @staticmethod
    def progress(row: dict[str, Any]) -> None:
        """Print one line per finished trial to standard error.

        Args:
            row (dict[str, Any]): Trial record with `instance`, `budget` and `arm`.
        """
        sys.stderr.write(f"trial {row['instance']} {row['budget']:g}s {row['arm']}\n")
        sys.stderr.flush()

    def choose(self, kind: str, prior: str) -> float:
        """Pick the temperature with the lowest mean tour length over the bound on validation.

        Args:
            kind (str): Instance family.
            prior (str): `nn` or `twoopt`.

        Returns:
            float: Chosen temperature.
        """
        budget = min(self.design.budgets)
        means = {}
        for temperature in TEMPERATURES:
            ratios = []
            for offset in range(self.design.validation_instances):
                seed = self.design.validation_base + offset
                row = self.solve(
                    self.instance(kind, seed),
                    f"{prior}_quantile",
                    ({prior: temperature}, budget),
                    seed,
                )
                ratios.append(row["length"] / row["bound"])
            means[temperature] = float(np.mean(ratios))
        return min(means, key=lambda value: means[value])

    def evaluate(
        self, kind: str, temperatures: dict[str, float], seed: int
    ) -> list[dict[str, Any]]:
        """Run every arm at every budget on one test instance.

        Args:
            kind (str): Instance family.
            temperatures (dict[str, float]): Chosen temperature per prior.
            seed (int): Test instance seed.

        Returns:
            list[dict[str, Any]]: One record per arm and budget.
        """
        instance = self.instance(kind, seed)
        rows = []
        for budget in self.design.budgets:
            for arm in ARMS:
                rows.append(self.solve(instance, arm, (temperatures, budget), seed))
                self.progress(rows[-1])
        return rows

    @staticmethod
    def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add the gap to the shortest tour any arm found and the gap to the lower bound.

        Args:
            rows (list[dict[str, Any]]): Trial records with `length` and `bound`.

        Returns:
            list[dict[str, Any]]: Records with `gap` and `bound_gap` in percent.
        """
        return TspReport.normalize(rows)

    def summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare arm pairs per family and budget and apply the decision rule.

        Args:
            rows (list[dict[str, Any]]): Normalized trial records.

        Returns:
            dict[str, Any]: Mean gaps, paired statistics and the verdict.
        """
        means, tables = ReportSink.tables(KINDS, ARMS, PAIRS, self.design.budgets, rows)
        return {"mean_gap": means, "paired": tables, "verdict": TsqDecision.verdict(tables)}

    def run(self) -> dict[str, Any]:
        """Choose temperatures, evaluate every test instance and summarize.

        Returns:
            dict[str, Any]: Chosen temperatures, raw trials and the summary.
        """
        temperatures = {
            kind: {prior: self.choose(kind, prior) for prior in PRIORS} for kind in KINDS
        }
        rows: list[dict[str, Any]] = []
        for kind in KINDS:
            for offset in range(self.design.test_instances):
                rows.extend(self.evaluate(kind, temperatures[kind], self.design.test_base + offset))
        rows = self.normalize(rows)
        return {"temperatures": temperatures, "trials": rows, "summary": self.summarize(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the quantile-codec study and write its JSON report.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n-items", type=int, default=100)
    parser.add_argument("--test-instances", type=int, default=12)
    parser.add_argument("--validation-instances", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/tsp-quantile-study.json"))
    options = parser.parse_args(argv)
    design = DecoderFormDesign(
        n_items=options.n_items,
        test_instances=options.test_instances,
        validation_instances=options.validation_instances,
    )
    report = {"design": asdict(design)} | TspQuantileStudy(design).run()
    ReportSink.write(report, options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
