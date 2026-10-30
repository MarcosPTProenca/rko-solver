"""Shared machinery for studies that compare a prior inside a decoder with seeding it.

Objective:
- Hold the design, decision rule and arm runner that the packing and the traveling-salesman
  decoder-form studies have in common, so each study only supplies its problem.

Technical approach:
- `FormStudy` is generic over the instance type. It runs eight arms per instance (plain RKO,
  each prior seeded with one key vector and with 32 perturbed orders, each prior in a residual
  decoder, and a rerun of the primary residual arm), tunes each residual spread and optional
  seeding noise on validation instances at the shortest budget, verifies every returned key
  vector independently of the solver, compares arm pairs with paired bootstrap intervals and
  applies a decision rule fixed before any result.
- A subclass supplies instances, priors, the decoder, seed keys, one measured solve and the
  gap normalization.

Flow (pseudocode):
1. Tune each residual spread and seeding noise per family and prior on validation instances.
2. Solve every test instance with every arm at every budget; charge prior computation.
3. Normalize gaps, compare arm pairs over instances and apply the decision rule.

Important criteria:
- Validation and test seed ranges are disjoint.
- The decision rule is a pure function of the paired statistics.

Inputs/Outputs:
- Input: a design and a subclass of `FormStudy`.
- Output: a JSON-ready report with tuned values, every trial, pair statistics and the verdict.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import sys
import time
from abc import ABC
from abc import abstractmethod
from benchmarks.engines.form_decision import DecoderFormDecision
from benchmarks.engines.learned_priority_study import PriorityStudy
from benchmarks.engines.priority_decoder_study import PairedStatistics
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Protocol

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


class StudyInstance(Protocol):
    """An instance that carries a `<family>-<size>-<seed>` name."""

    @property
    def name(self) -> str:
        """Return the instance name.

        Returns:
            str: `<family>-<size>-<seed>`.
        """
        ...


class FormStudy[InstanceT: StudyInstance](ABC):
    """Run the arms, choose spreads and summarize.

    Responsibilities:
    - Build each arm's decoder and seed keys and charge their preparation to the allowance.
    - Verify every returned key vector independently of the solver.

    Attributes:
        design (DecoderFormDesign): Fixed design.
        kinds (tuple[str, ...]): Instance families.
        priors (tuple[str, ...]): Prior names.
        arms (tuple[str, ...]): Arm names.
        pairs (tuple[tuple[str, str], ...]): Arm pairs compared in the paired statistics.
        decision (tuple[str, str, bool]): Primary prior, secondary prior and whether both decide.
        spread_grid (tuple[float, ...]): Candidate residual spreads.
        noises (tuple[float, ...]): Seeding-noise grid; empty keeps the fixed default noise.
    """

    kinds: tuple[str, ...]
    priors: tuple[str, ...]
    arms: tuple[str, ...]
    pairs: tuple[tuple[str, str], ...]
    decision: tuple[str, str, bool]
    spread_grid: tuple[float, ...]
    noises: tuple[float, ...]

    def __init__(self, design: DecoderFormDesign) -> None:
        """Store the design.

        Args:
            design (DecoderFormDesign): Study design.
        """
        self.design = design

    @abstractmethod
    def instance(self, kind: str, seed: int) -> InstanceT:
        """Generate an instance at the design size.

        Args:
            kind (str): Instance family.
            seed (int): Instance seed.

        Returns:
            Any: Generated instance.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def scores(prior: str, instance: InstanceT) -> np.ndarray:
        """Compute one prior's item scores.

        Args:
            prior (str): `ratio` or `dual`.
            instance (InstanceT): Instance.

        Returns:
            np.ndarray: Positive score per item.

        Raises:
            ValueError: If the prior is unknown.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def assemble(
        instance: InstanceT, scores: np.ndarray | None, spread: float, seed: int
    ) -> Callable[[np.ndarray], Any]:
        """Build the decoder, plain when there is no prior.

        Args:
            instance (InstanceT): Instance.
            scores (np.ndarray | None): Prior, or None for the plain decoder.
            spread (float): Residual scale.
            seed (int): Unused by the packing decoder.

        Returns:
            Callable[[np.ndarray], Any]: Decoder.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def seeds(
        instance: InstanceT, scores: np.ndarray, count: int, noise: float, seed: int
    ) -> list[list[float]]:
        """Encode the prior as one key vector and noisy variants of it.

        Args:
            instance (InstanceT): Instance.
            scores (np.ndarray): Prior.
            count (int): Number of key vectors.
            noise (float): Perturbation scale.
            seed (int): Perturbation seed.

        Returns:
            list[list[float]]: Key vectors, the exact prior first.
        """
        raise NotImplementedError

    @abstractmethod
    def measure(
        self,
        instance: InstanceT,
        decoder: Callable[[np.ndarray], Any],
        keys: list[list[float]] | None,
        timing: tuple[int, float],
    ) -> dict[str, Any]:
        """Solve once and report the verified objective and its reference value.

        Args:
            instance (InstanceT): Instance.
            decoder (Callable[[np.ndarray], Any]): Decoder.
            keys (list[list[float]] | None): Seed keys or None.
            timing (tuple[int, float]): Solver seed and search seconds.

        Returns:
            dict[str, Any]: `profit`, `bound` and `verified`.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def badness(row: dict[str, Any]) -> float:
        """Score a validation trial, lower being better.

        Args:
            row (dict[str, Any]): Trial record.

        Returns:
            float: Negative profit over the dual bound.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add the gap to the best profit any arm found and the gap to the dual bound.

        Args:
            rows (list[dict[str, Any]]): Trial records with `profit` and `bound`.

        Returns:
            list[dict[str, Any]]: Records with `gap` and `bound_gap` in percent.
        """
        raise NotImplementedError

    @classmethod
    def parse(cls, arm: str) -> tuple[str | None, str]:
        """Split an arm name into its prior and mechanism.

        Args:
            arm (str): One of the study's arms.

        Returns:
            tuple[str | None, str]: Prior (None for plain) and mechanism (`none`, `seed1`,
            `seed32`, `resid` or `resid_rerun`).

        Raises:
            ValueError: If the arm is unknown.
        """
        if arm == "plain":
            return None, "none"
        prior, _, mechanism = arm.partition("_")
        if prior in cls.priors and mechanism in ("seed1", "seed32", "resid", "resid_rerun"):
            return prior, mechanism
        raise ValueError(f"Unknown arm {arm!r}.")

    def setup(
        self, arm: str, instance: InstanceT, tuned: dict[str, float], seed: int
    ) -> tuple[Callable[[np.ndarray], Any], list[list[float]] | None, float]:
        """Build an arm's decoder and seed keys and time their preparation.

        Args:
            arm (str): One of the study's arms.
            instance (InstanceT): Instance.
            tuned (dict[str, float]): Chosen residual spread per prior and, where tuned,
                `<prior>_noise`.
            seed (int): Perturbation seed.

        Returns:
            tuple[Callable[[np.ndarray], Any], list[list[float]] | None, float]: Decoder,
            initial keys or None, and preparation seconds.
        """
        prior, mechanism = self.parse(arm)
        if prior is None:
            return self.assemble(instance, None, 0.0, 0), None, 0.0
        started = time.perf_counter()
        scores = self.scores(prior, instance)
        if mechanism.startswith("resid"):
            decoder, keys = self.assemble(instance, scores, tuned[prior], 0), None
        else:
            count = 1 if mechanism == "seed1" else SEED_COUNT
            keys = self.seeds(
                instance, scores, count, tuned.get(f"{prior}_noise", SEED_NOISE), seed
            )
            decoder = self.assemble(instance, None, 0.0, 0)
        return decoder, keys, time.perf_counter() - started

    def solve(
        self,
        instance: InstanceT,
        arm: str,
        context: tuple[dict[str, float], float],
        seed: int,
    ) -> dict[str, Any]:
        """Run one arm on one instance within one allowance.

        Args:
            instance (InstanceT): Instance.
            arm (str): One of `ARMS`.
            context (tuple[dict[str, float], float]): Chosen spreads and the allowance.
            seed (int): Instance seed; the rerun arm uses a different solver seed.

        Returns:
            dict[str, Any]: Arm, budget, profit, verification and phase timings.
        """
        tuned, budget = context
        decoder, keys, prep = self.setup(arm, instance, tuned, seed)
        allowance = PriorityStudy.search_allowance(budget, prep)
        started = time.perf_counter()
        solver_seed = seed + RERUN_OFFSET if arm.endswith("_rerun") else seed
        measured = self.measure(instance, decoder, keys, (solver_seed, allowance))
        prior, mechanism = self.parse(arm)
        measured |= {
            "prep_seconds": prep,
            "allowance": allowance,
            "wall_seconds": time.perf_counter() - started + prep,
            "spread": tuned[prior] if prior and mechanism.startswith("resid") else None,
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
    def identity(instance: InstanceT, arm: str, budget: float) -> dict[str, Any]:
        """Name the instance, family, arm and budget a record belongs to.

        Args:
            instance (InstanceT): Instance.
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

    @classmethod
    def execute(
        cls, argv: Sequence[str] | None, output: Path, design: DecoderFormDesign | None = None
    ) -> int:
        """Parse arguments, run the study class and write its JSON report.

        Args:
            argv (Sequence[str] | None): Arguments, or None for `sys.argv`.
            output (Path): Default output path.
            design (DecoderFormDesign | None): Default design; command-line options override it.

        Returns:
            int: Process exit code.
        """
        base = design or DecoderFormDesign()
        parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
        parser.add_argument("--n-items", type=int, default=base.n_items)
        parser.add_argument("--test-instances", type=int, default=base.test_instances)
        parser.add_argument("--validation-instances", type=int, default=base.validation_instances)
        parser.add_argument("--output", type=Path, default=output)
        options = parser.parse_args(argv)
        chosen = replace(
            base,
            n_items=options.n_items,
            test_instances=options.test_instances,
            validation_instances=options.validation_instances,
        )
        report = {"design": asdict(chosen)} | cls(chosen).run()
        cls.save(report, options.output)
        return 0

    def choose(self, kind: str, arm: str, key: str, grid: tuple[float, ...]) -> float:
        """Pick the grid value with the best mean validation score at the shortest budget.

        Args:
            kind (str): Instance family.
            arm (str): Arm that the tuned value drives.
            key (str): Name of the tuned value in the `tuned` mapping.
            grid (tuple[float, ...]): Candidate values.

        Returns:
            float: Chosen value.
        """
        budget = min(self.design.budgets)
        means = {}
        for value in grid:
            scores = []
            for offset in range(self.design.validation_instances):
                seed = self.design.validation_base + offset
                row = self.solve(self.instance(kind, seed), arm, ({key: value}, budget), seed)
                scores.append(self.badness(row))
            means[value] = float(np.mean(scores))
        return min(means, key=lambda value: means[value])

    def tune(self, kind: str) -> dict[str, float]:
        """Choose every tunable value of one family on validation instances.

        Args:
            kind (str): Instance family.

        Returns:
            dict[str, float]: Residual spread per prior and, where a noise grid exists,
            `<prior>_noise`.
        """
        tuned = {
            prior: self.choose(kind, f"{prior}_resid", prior, self.spread_grid)
            for prior in self.priors
        }
        if self.noises:
            tuned |= {
                f"{prior}_noise": self.choose(
                    kind, f"{prior}_seed32", f"{prior}_noise", self.noises
                )
                for prior in self.priors
            }
        return tuned

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
            for arm in self.arms:
                rows.append(self.solve(instance, arm, (spreads, budget), seed))
                self.progress(rows[-1])
        return rows

    @classmethod
    def column(
        cls, rows: list[dict[str, Any]], kind: str, budget: float, field: str
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
        columns: dict[str, list[float]] = {arm: [] for arm in cls.arms}
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
        for kind in self.kinds:
            tables[kind] = {}
            for budget in self.design.budgets:
                gaps = self.column(rows, kind, budget, "gap")
                bound = self.column(rows, kind, budget, "bound_gap")
                means[f"{kind}@{budget:g}s"] = {
                    arm: float(np.mean(gaps[arm])) for arm in self.arms
                } | {f"{arm}|bound": float(np.mean(bound[arm])) for arm in self.arms}
                tables[kind][f"{budget:g}s"] = {
                    f"{a}-vs-{b}": PairedStatistics.compare(gaps[a], gaps[b], 10000, 0)
                    for a, b in self.pairs
                }
        return {
            "mean_gap": means,
            "paired": tables,
            "verdict": DecoderFormDecision.verdict(tables, *self.decision),
        }

    def run(self) -> dict[str, Any]:
        """Choose spreads, evaluate every test instance and summarize.

        Returns:
            dict[str, Any]: Chosen spreads, raw trials and the summary.
        """
        spreads = {kind: self.tune(kind) for kind in self.kinds}
        rows: list[dict[str, Any]] = []
        for kind in self.kinds:
            for offset in range(self.design.test_instances):
                rows.extend(self.evaluate(kind, spreads[kind], self.design.test_base + offset))
        rows = self.normalize(rows)
        return {"spreads": spreads, "trials": rows, "summary": self.summarize(rows)}
