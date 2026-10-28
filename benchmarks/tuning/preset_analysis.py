"""Family-balanced training selection and predeclared confirmation gates for one preset.

Objective:
- Select one fixed simulated-annealing configuration without test-set input, then check its
  generality with paired evidence and a bounded acceptable regression in every test cell.

Technical approach:
- Training ranks a fixed candidate pool by normalized gap, averaging cells within each family
  before averaging families. Candidates whose worst training-cell mean loss exceeds the margin
  are ineligible; the default remains an eligible fallback.
- Confirmation evaluates only the locked candidate and default. Paired instance bootstraps reuse
  index draws across evaluation budgets within a context. The overall one-sided 95% upper bound
  must be negative, and Bonferroni-adjusted one-sided upper bounds across all cells must not
  exceed the regression margin. Percentile-bootstrap coverage is approximate, not guaranteed.

Flow (pseudocode):
1. Validate complete, finite training records and rank eligible candidates.
2. Lock the selected configuration before generating test instances.
3. Validate paired test records and compute the fixed acceptance gates.

Important criteria:
- A margin of 0.02 is in normalized-gap units, not two percent of raw objective value.
- Test data never influence ranking; failed gates do not trigger candidate reselection.
- Different sizes use disjoint context seed offsets; budgets share instances within a context.

Inputs/Outputs:
- Input: a frozen design and training or confirmation score records.
- Output: deterministic training rankings and a confirmation decision with per-cell bounds.
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

Case = tuple[str, int, int]
ScoreMap = dict[str, dict[Case, np.ndarray]]


@dataclass(frozen=True)
class PresetDesign:
    """Store the immutable experimental design independently of machine worker counts.

    Responsibilities:
    - Define instance sizes, budgets, disjoint seed contexts and acceptance thresholds.

    Attributes:
        sizes (tuple): Family names with the dimensions tested in each family.
        budgets (tuple[int, ...]): Evaluations per solver run.
        train (int): Training instances per context.
        test (int): Untouched confirmation instances per context.
        train_seeds (int): Solver seeds per training instance.
        test_seeds (int): Solver seeds per confirmation instance.
        seed_offset (int): Base of the new context cohort offsets.
        margin (float): Largest acceptable normalized-gap regression per test cell.
        alpha (float): Nominal one-sided error rate before cell multiplicity adjustment.
        resamples (int): Paired bootstrap draws.
    """

    sizes: tuple[tuple[str, tuple[int, ...]], ...] = (
        ("kp", (100, 600)),
        ("tsp", (30, 100)),
        ("rastrigin", (10, 60)),
    )
    budgets: tuple[int, ...] = (50_000, 500_000)
    train: int = 6
    test: int = 12
    train_seeds: int = 2
    test_seeds: int = 5
    seed_offset: int = 2_000_000
    margin: float = 0.02
    alpha: float = 0.05
    resamples: int = 20_000

    def __post_init__(self) -> None:
        """Reject designs with overlapping roles, missing contexts or invalid thresholds.

        Raises:
            ValueError: If a count, context, budget or threshold is invalid.
        """
        if not 1 <= self.train <= 100 or not 1 <= self.test <= 100:
            raise ValueError("Instance counts must lie between 1 and 100.")
        if min(self.train_seeds, self.test_seeds, self.resamples) < 1:
            raise ValueError("Seed counts and resamples must be positive.")
        if not self.budgets or min(self.budgets) < 1 or len(set(self.budgets)) != len(self.budgets):
            raise ValueError("Evaluation budgets must be positive and unique.")
        if self.margin < 0 or not 0 < self.alpha < 1 or self.seed_offset < 2_000_000:
            raise ValueError("Invalid regression margin, error rate or fresh-cohort offset.")
        self._validate_contexts()

    def _validate_contexts(self) -> None:
        """Reject unknown families, empty size lists and repeated contexts.

        Raises:
            ValueError: If a context is unknown, empty, nonpositive or duplicated.
        """
        families = [family for family, _ in self.sizes]
        if not families or len(families) != len(set(families)):
            raise ValueError("Family names must be nonempty and unique.")
        for family, sizes in self.sizes:
            if family not in ("kp", "tsp", "rastrigin") or not sizes or min(sizes) < 1:
                raise ValueError("Unknown family or invalid size.")
        if len(self.contexts) != len(set(self.contexts)):
            raise ValueError("Contexts must be unique.")

    @property
    def contexts(self) -> tuple[tuple[str, int], ...]:
        """Return family-size contexts in their frozen order.

        Returns:
            tuple[tuple[str, int], ...]: Contexts that receive distinct cohort offsets.
        """
        return tuple((family, size) for family, sizes in self.sizes for size in sizes)

    @property
    def cases(self) -> tuple[Case, ...]:
        """Return the full Cartesian product of contexts and evaluation budgets.

        Returns:
            tuple[Case, ...]: Family, size and evaluation-budget keys.
        """
        return tuple((f, s, budget) for f, s in self.contexts for budget in self.budgets)

    def offset(self, family: str, size: int) -> int:
        """Assign a separate instance seed namespace to one family-size context.

        Args:
            family (str): Problem family.
            size (int): Problem dimension.

        Returns:
            int: Cohort offset with room for both roles and their instance indices.

        Raises:
            ValueError: If the context is absent from the design.
        """
        return self.seed_offset + 10_000 * self.contexts.index((family, size))


class PresetAnalysis:
    """Compute training-only rankings and the locked candidate's confirmation decision.

    Responsibilities:
    - Reject partial evidence and keep family weights independent of context counts.
    - Apply the acceptance rule without changing its thresholds after seeing test data.

    Attributes:
        design (PresetDesign): Frozen counts, budgets and decision thresholds.
    """

    def __init__(self, design: PresetDesign) -> None:
        """Store the frozen design.

        Args:
            design (PresetDesign): Study specification.
        """
        self.design = design

    @staticmethod
    def balanced_mean(values: dict[Case, float]) -> float:
        """Average cells within families before averaging families equally.

        Args:
            values (dict[Case, float]): Per-cell mean scores or differences.

        Returns:
            float: Family-balanced mean.
        """
        grouped: dict[str, list[float]] = defaultdict(list)
        for (family, _, _), value in values.items():
            grouped[family].append(value)
        return float(np.mean([np.mean(items) for items in grouped.values()]))

    def matrices(self, records: list[dict[str, Any]], candidates: set[str], phase: str) -> ScoreMap:
        """Validate complete evidence and index score vectors by candidate and test cell.

        Args:
            records (list[dict[str, Any]]): Score records from exactly one phase.
            candidates (set[str]): Candidate identifiers expected in this phase.
            phase (str): `train` or `test`.

        Returns:
            ScoreMap: Per-instance score arrays keyed by candidate and cell.

        Raises:
            ValueError: If evidence is incomplete, duplicated, nonfinite or from another phase.
        """
        if phase not in ("train", "test"):
            raise ValueError("Unknown evidence phase.")
        expected = {(candidate, *case) for candidate in candidates for case in self.design.cases}
        found: ScoreMap = {candidate: {} for candidate in candidates}
        seen = set()
        for row in records:
            key = (row["candidate"], row["family"], row["size"], row["evaluations"])
            if key not in expected or key in seen:
                raise ValueError("Unexpected candidate, cell or duplicate record.")
            seen.add(key)
            found[key[0]][key[1:]] = self._score_vector(row, phase)
        if seen != expected:
            raise ValueError("Incomplete evidence: every candidate must cover every cell.")
        return found

    def _score_vector(self, row: dict[str, Any], phase: str) -> np.ndarray:
        """Read one finite score vector from the requested data role.

        Args:
            row (dict[str, Any]): A scoring record.
            phase (str): Expected training or confirmation phase.

        Returns:
            np.ndarray: One finite score per expected instance.

        Raises:
            ValueError: If the role, vector length or numeric values are invalid.
        """
        if row["phase"] != phase:
            raise ValueError("Unexpected evidence phase.")
        count = self.design.train if phase == "train" else self.design.test
        scores = np.asarray(row["scores"], dtype=float)
        if scores.shape != (count,) or not np.isfinite(scores).all():
            raise ValueError("Wrong instance count or nonfinite scores.")
        return scores

    def ranking(self, records: list[dict[str, Any]], candidates: set[str]) -> list[dict[str, Any]]:
        """Rank training-eligible candidates by family-balanced normalized gap.

        Args:
            records (list[dict[str, Any]]): Complete training-only evidence.
            candidates (set[str]): Identifiers including `default`.

        Returns:
            list[dict[str, Any]]: Eligible candidates first, then score and deterministic tie order.

        Raises:
            ValueError: If training evidence is incomplete or invalid.
        """
        if "default" not in candidates:
            raise ValueError("Training selection requires the default baseline.")
        matrices = self.matrices(records, candidates, "train")
        default = matrices["default"]
        ranked = []
        for candidate, values in matrices.items():
            means = {case: float(scores.mean()) for case, scores in values.items()}
            worst = max(float((values[case] - default[case]).mean()) for case in values)
            ranked.append(
                {
                    "candidate": candidate,
                    "score": self.balanced_mean(means),
                    "worst_delta": worst,
                    "eligible": worst <= self.design.margin,
                }
            )
        return sorted(
            ranked,
            key=lambda r: (
                not r["eligible"],
                r["score"],
                r["candidate"] != "default",
                r["candidate"],
            ),
        )

    def confirmation(self, records: list[dict[str, Any]], selected: str) -> dict[str, Any]:
        """Apply the frozen overall-improvement and per-cell regression gates.

        Args:
            records (list[dict[str, Any]]): Complete paired confirmation evidence.
            selected (str): Training-locked candidate identifier.

        Returns:
            dict[str, Any]: Overall difference, upper bound, cell bounds and acceptance decision.

        Raises:
            ValueError: If paired confirmation evidence is incomplete or invalid.
        """
        data = self.matrices(records, {"default", selected}, "test")
        differences = {
            case: data[selected][case] - data["default"][case] for case in self.design.cases
        }
        rng = np.random.default_rng(0)
        family_draws = {family: np.zeros(self.design.resamples) for family, _ in self.design.sizes}
        cells = []
        quantile = 1 - self.design.alpha / len(self.design.cases)
        for family, size in self.design.contexts:
            indices = rng.integers(self.design.test, size=(self.design.resamples, self.design.test))
            family_sizes = next(sizes for name, sizes in self.design.sizes if name == family)
            for budget in self.design.budgets:
                values = differences[(family, size, budget)]
                draws = values[indices].mean(axis=1)
                family_draws[family] += draws / (len(family_sizes) * len(self.design.budgets))
                upper = float(np.quantile(draws, quantile))
                cells.append(
                    {
                        "family": family,
                        "size": size,
                        "evaluations": budget,
                        "delta": float(values.mean()),
                        "upper": upper,
                        "passed": upper <= self.design.margin,
                    }
                )
        overall = np.mean(list(family_draws.values()), axis=0)
        upper = float(np.quantile(overall, 1 - self.design.alpha))
        delta = self.balanced_mean(
            {case: float(values.mean()) for case, values in differences.items()}
        )
        return {
            "delta": delta,
            "upper": upper,
            "cells": cells,
            "passed": upper < 0 and all(cell["passed"] for cell in cells),
            "margin": self.design.margin,
            "alpha": self.design.alpha,
            "cell_quantile": quantile,
            "resamples": self.design.resamples,
        }
