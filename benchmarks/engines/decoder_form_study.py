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

import argparse
import json
import numpy as np
import sys
import time
from benchmarks.engines.learned_priority_study import PriorityStudy
from benchmarks.engines.packing_problems import PackingDecoder
from benchmarks.engines.packing_problems import PackingFamily
from benchmarks.engines.packing_problems import PackingInstance
from benchmarks.engines.packing_problems import PackingRun
from benchmarks.engines.priority_decoder_study import PairedStatistics
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
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
SPREADS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
SEED_COUNT = 32
SEED_NOISE = 0.25
RERUN_OFFSET = 7919


@dataclass(slots=True)
class DecoderFormDesign:
    """Fixed numerical design of the study.

    Responsibilities:
    - Keep size, budgets and the disjoint validation and test seed ranges in one place.

    Attributes:
        n_items (int): Items per instance.
        budgets (tuple[float, ...]): Per-instance wall allowances in seconds.
        test_instances (int): Held-out instances per family.
        validation_instances (int): Instances used to choose spreads.
        validation_base (int): First validation seed.
        test_base (int): First test seed.
        execution (str): Python-engine execution mode passed to the solver.
        algorithms (tuple[str, ...] | None): Portfolio override; None keeps the default portfolio.
    """

    n_items: int = 250
    budgets: tuple[float, ...] = (2.0, 10.0)
    test_instances: int = 12
    validation_instances: int = 6
    validation_base: int = 6500
    test_base: int = 8000
    execution: str = "process"
    algorithms: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Require disjoint validation and test seeds.

        Raises:
            ValueError: If the two seed ranges overlap.
        """
        validation = range(self.validation_base, self.validation_base + self.validation_instances)
        test = range(self.test_base, self.test_base + self.test_instances)
        if validation.start < test.stop and test.start < validation.stop:
            raise ValueError("Validation and test seeds must be disjoint.")


class DecoderFormDecision:
    """Apply the decision rule fixed before the study.

    Responsibilities:
    - Decide from paired statistics whether the residual form clearly beats fair seeding.
    """

    @staticmethod
    def holds(
        tables: dict[str, dict[str, dict[str, dict[str, float]]]],
        first: str,
        others: tuple[str, ...],
    ) -> bool:
        """Check that a residual arm beats each listed arm with an interval below zero everywhere.

        Args:
            tables (dict[str, dict[str, dict[str, dict[str, float]]]]): Paired statistics keyed by
                family, budget label and `arm-vs-arm` pair.
            first (str): Residual arm name.
            others (tuple[str, ...]): Arms the residual arm is compared with.

        Returns:
            bool: Whether every family, budget and pair has `ci_high` below zero.
        """
        return all(
            budget[f"{first}-vs-{other}"]["ci_high"] < 0
            for family in tables.values()
            for budget in family.values()
            for other in others
        )

    @staticmethod
    def verdict(tables: dict[str, dict[str, dict[str, dict[str, float]]]]) -> dict[str, Any]:
        """Evaluate the rule on paired comparisons.

        Args:
            tables (dict[str, dict[str, dict[str, dict[str, float]]]]): Statistics keyed by
                family, budget label and `arm-vs-arm` pair, each holding `difference`, `ci_low`
                and `ci_high`.

        Returns:
            dict[str, Any]: `primary_holds` for the dual prior, `secondary_holds` for the ratio
            prior, the mean absolute rerun difference `noise_floor` and the `advance` flag,
            which equals `primary_holds`.
        """
        primary = DecoderFormDecision.holds(tables, "dual_resid", ("dual_seed32", "plain"))
        secondary = DecoderFormDecision.holds(tables, "ratio_resid", ("ratio_seed32", "plain"))
        noise = [
            abs(budget["dual_resid_rerun-vs-dual_resid"]["difference"])
            for family in tables.values()
            for budget in family.values()
        ]
        return {
            "primary_holds": primary,
            "secondary_holds": secondary,
            "noise_floor": float(np.mean(noise)),
            "advance": primary,
        }


class DecoderFormStudy:
    """Run the arms, choose spreads and summarize.

    Responsibilities:
    - Build each arm's decoder and seed keys and charge their preparation to the allowance.
    - Verify every returned key vector independently of the solver.

    Attributes:
        design (DecoderFormDesign): Fixed design.
    """

    def __init__(self, design: DecoderFormDesign) -> None:
        """Store the design.

        Args:
            design (DecoderFormDesign): Study design.
        """
        self.design = design

    def instance(self, kind: str, seed: int) -> PackingInstance:
        """Generate an instance at the design size.

        Args:
            kind (str): Instance family.
            seed (int): Instance seed.

        Returns:
            PackingInstance: Generated instance.
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
    def parse(arm: str) -> tuple[str | None, str]:
        """Split an arm name into its prior and mechanism.

        Args:
            arm (str): One of `ARMS`.

        Returns:
            tuple[str | None, str]: Prior (None for plain) and mechanism (`none`, `seed1`,
            `seed32`, `resid` or `resid_rerun`).

        Raises:
            ValueError: If the arm is unknown.
        """
        if arm == "plain":
            return None, "none"
        prior, _, mechanism = arm.partition("_")
        if prior in PRIORS and mechanism in ("seed1", "seed32", "resid", "resid_rerun"):
            return prior, mechanism
        raise ValueError(f"Unknown arm {arm!r}.")

    def setup(
        self, arm: str, instance: PackingInstance, spreads: dict[str, float], seed: int
    ) -> tuple[PackingDecoder, list[list[float]] | None, float]:
        """Build an arm's decoder and seed keys and time their preparation.

        Args:
            arm (str): One of `ARMS`.
            instance (PackingInstance): Packing instance.
            spreads (dict[str, float]): Chosen residual spread per prior.
            seed (int): Perturbation seed.

        Returns:
            tuple[PackingDecoder, list[list[float]] | None, float]: Decoder, initial keys or
            None, and preparation seconds.
        """
        prior, mechanism = self.parse(arm)
        if prior is None:
            return PackingDecoder(instance), None, 0.0
        started = time.perf_counter()
        scores = self.scores(prior, instance)
        if mechanism.startswith("resid"):
            decoder, keys = PackingDecoder(instance, scores, spreads[prior]), None
        else:
            count = 1 if mechanism == "seed1" else SEED_COUNT
            keys = PackingFamily.perturbed_keys(scores, count, SEED_NOISE, seed)
            decoder = PackingDecoder(instance)
        return decoder, keys, time.perf_counter() - started

    def solve(
        self,
        instance: PackingInstance,
        arm: str,
        context: tuple[dict[str, float], float],
        seed: int,
    ) -> dict[str, Any]:
        """Run one arm on one instance within one allowance.

        Args:
            instance (PackingInstance): Packing instance.
            arm (str): One of `ARMS`.
            context (tuple[dict[str, float], float]): Chosen spreads and the allowance.
            seed (int): Instance seed; the rerun arm uses a different solver seed.

        Returns:
            dict[str, Any]: Arm, budget, profit, verification and phase timings.
        """
        spreads, budget = context
        decoder, keys, prep = self.setup(arm, instance, spreads, seed)
        allowance = PriorityStudy.search_allowance(budget, prep)
        started = time.perf_counter()
        observed, verified = PackingRun.solve(
            instance,
            decoder,
            keys,
            (seed + RERUN_OFFSET if arm.endswith("_rerun") else seed, allowance),
            (self.design.execution, self.design.algorithms),
        )
        measured = {
            "profit": observed,
            "bound": PackingFamily.dual(instance)[1],
            "verified": verified,
            "prep_seconds": prep,
            "allowance": allowance,
            "wall_seconds": time.perf_counter() - started + prep,
            "spread": decoder.spread if decoder.scores is not None else None,
        }
        return self.identity(instance, arm, budget) | measured

    @staticmethod
    def progress(row: dict[str, Any]) -> None:
        """Print one line per finished trial to standard error.

        Args:
            row (dict[str, Any]): Trial record with `instance`, `budget` and `arm`.
        """
        sys.stderr.write(f"trial {row['instance']} {row['budget']:g}s {row['arm']}\n")
        sys.stderr.flush()

    @staticmethod
    def identity(instance: PackingInstance, arm: str, budget: float) -> dict[str, Any]:
        """Name the instance, family, arm and budget a record belongs to.

        Args:
            instance (PackingInstance): Packing instance.
            arm (str): One of `ARMS`.
            budget (float): Per-instance allowance in seconds.

        Returns:
            dict[str, Any]: The four identifying fields.
        """
        family = instance.name.split("-")[0]
        return {"instance": instance.name, "family": family, "arm": arm, "budget": budget}

    @staticmethod
    def save(report: dict[str, Any], output: Path) -> None:
        """Write the JSON report and print the verdict.

        Args:
            report (dict[str, Any]): Design, spreads, trials and summary.
            output (Path): Destination file; parent directories are created.
        """
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        sys.stdout.write(json.dumps(report["summary"]["verdict"], indent=2) + "\n")

    def choose_spread(self, kind: str, prior: str) -> float:
        """Pick the residual spread with the best mean profit over the dual bound on validation.

        Args:
            kind (str): Instance family.
            prior (str): `ratio` or `dual`.

        Returns:
            float: Chosen spread.
        """
        budget = min(self.design.budgets)
        means = {}
        for spread in SPREADS:
            ratios = []
            for offset in range(self.design.validation_instances):
                seed = self.design.validation_base + offset
                row = self.solve(
                    self.instance(kind, seed), f"{prior}_resid", ({prior: spread}, budget), seed
                )
                ratios.append(row["profit"] / row["bound"])
            means[spread] = float(np.mean(ratios))
        return max(means, key=lambda value: means[value])

    def evaluate(self, kind: str, spreads: dict[str, float], seed: int) -> list[dict[str, Any]]:
        """Run every arm at every budget on one test instance.

        Args:
            kind (str): Instance family.
            spreads (dict[str, float]): Chosen residual spread per prior.
            seed (int): Test instance seed.

        Returns:
            list[dict[str, Any]]: One record per arm and budget.
        """
        instance = self.instance(kind, seed)
        rows = []
        for budget in self.design.budgets:
            for arm in ARMS:
                rows.append(self.solve(instance, arm, (spreads, budget), seed))
                self.progress(rows[-1])
        return rows

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

    @staticmethod
    def column(
        rows: list[dict[str, Any]], kind: str, budget: float, field: str
    ) -> dict[str, list[float]]:
        """Collect one field of every arm on one family and budget in instance order.

        Args:
            rows (list[dict[str, Any]]): Normalized trial records.
            kind (str): Instance family.
            budget (float): Per-instance allowance in seconds.
            field (str): `gap` or `bound_gap`.

        Returns:
            dict[str, list[float]]: Values per arm.
        """
        columns: dict[str, list[float]] = {arm: [] for arm in ARMS}
        for row in rows:
            if row["family"] == kind and row["budget"] == budget:
                columns[row["arm"]].append(row[field])
        return columns

    def summarize(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Compare arm pairs per family and budget and apply the decision rule.

        Args:
            rows (list[dict[str, Any]]): Normalized trial records.

        Returns:
            dict[str, Any]: Mean gaps, paired statistics and the verdict.
        """
        tables: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
        means: dict[str, dict[str, float]] = {}
        for kind in KINDS:
            tables[kind] = {}
            for budget in self.design.budgets:
                gaps = self.column(rows, kind, budget, "gap")
                bound = self.column(rows, kind, budget, "bound_gap")
                means[f"{kind}@{budget:g}s"] = {arm: float(np.mean(gaps[arm])) for arm in ARMS} | {
                    f"{arm}|bound": float(np.mean(bound[arm])) for arm in ARMS
                }
                tables[kind][f"{budget:g}s"] = {
                    f"{a}-vs-{b}": PairedStatistics.compare(gaps[a], gaps[b], 10000, 0)
                    for a, b in PAIRS
                }
        return {"mean_gap": means, "paired": tables, "verdict": DecoderFormDecision.verdict(tables)}

    def run(self) -> dict[str, Any]:
        """Choose spreads, evaluate every test instance and summarize.

        Returns:
            dict[str, Any]: Chosen spreads, raw trials and the summary.
        """
        spreads = {
            kind: {prior: self.choose_spread(kind, prior) for prior in PRIORS} for kind in KINDS
        }
        rows: list[dict[str, Any]] = []
        for kind in KINDS:
            for offset in range(self.design.test_instances):
                rows.extend(self.evaluate(kind, spreads[kind], self.design.test_base + offset))
        rows = self.normalize(rows)
        return {"spreads": spreads, "trials": rows, "summary": self.summarize(rows)}


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study and write its JSON report.

    Args:
        argv (Sequence[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-items", type=int, default=250)
    parser.add_argument("--test-instances", type=int, default=12)
    parser.add_argument("--output", type=Path, default=Path("results/decoder-form-study.json"))
    options = parser.parse_args(argv)
    design = DecoderFormDesign(n_items=options.n_items, test_instances=options.test_instances)
    report = {"design": asdict(design)} | DecoderFormStudy(design).run()
    DecoderFormStudy.save(report, options.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
