"""Meaningful-win indicators and class-level decisions for the expanded utility study.

Objective:
- Turn paired fixed-budget scores into one win per instance and six multiplicity-controlled
  class decisions, exactly as the registered protocol defines them.

Technical approach:
- Take each arm's median over its solver seeds, then apply an exact integer percentage rule
  for item selection and routing or a floored multiplicative rule for continuous scores.
- Require a win in every primary profile, test the win fraction with Hoeffding's bound and
  adjust the family of claims with Holm's step-down procedure.

Flow (pseudocode):
1. For each instance and profile, compare the candidate median with the baseline median.
2. An instance wins only if it wins in every profile; ties, zeros and failures never win.
3. Compute `p = exp(-2 n (r - 1/2)^2)` for `r > 1/2`, adjust by Holm and classify each claim.

Important criteria:
- Every planned instance enters; an incomplete claim is reported, never dropped.
- Percentages are compared by multiplication, never by dividing by a near-zero gap.
- The continuous floor is a reproducibility guard, not a regression margin.

Inputs/Outputs:
- Input: per-seed scores, feasibility flags and per-claim win lists.
- Output: booleans, p-values and claim status labels.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping
from collections.abc import Sequence

SEEDS = 5
INSTANCES_PER_CLASS = 60
ALPHA = 0.05
ATOL = 1e-8
RTOL = 1e-10
PERCENT = {"item": 5, "route": 1, "continuous": 5}
EVIDENCE = "evidence"
INSUFFICIENT = "insufficient"
INCOMPLETE = "incomplete"


class UtilityAnalysis:
    """Evaluate wins, Hoeffding p-values and Holm-adjusted class claims.

    Responsibilities:
    - Apply the exact per-kind benefit rule to a pair of seed-median scores.
    - Combine profiles, test win prevalence and control the family-wise error.
    """

    @staticmethod
    def seed_median(scores: Sequence[float]) -> float:
        """Take the median of exactly the planned number of finite solver-seed scores.

        Args:
            scores (Sequence[float]): One score per paired solver seed.

        Returns:
            float: Median; an exact integer when the scores are integers.

        Raises:
            ValueError: If the count differs from the plan or a score is not finite.
        """
        if len(scores) != SEEDS:
            raise ValueError(f"expected {SEEDS} solver seeds, got {len(scores)}")
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("scores must be finite")
        return statistics.median(scores)

    @staticmethod
    def profile_win(
        kind: str,
        baseline: Sequence[float],
        candidate: Sequence[float],
        feasible: Sequence[bool] | None = None,
    ) -> bool:
        """Decide whether the candidate meets the minimum benefit in one profile.

        Args:
            kind (str): `item`, `route` or `continuous`.
            baseline (Sequence[float]): Baseline score per solver seed.
            candidate (Sequence[float]): Candidate score per solver seed.
            feasible (Sequence[bool] | None): Candidate feasibility per seed, if it applies.

        Returns:
            bool: True only for a resolved win; ties, zero baselines and infeasible candidates
            are non-wins.

        Raises:
            ValueError: If the kind is unknown, a count is wrong or a score is invalid.
        """
        if kind not in PERCENT:
            raise ValueError(f"unknown kind {kind!r}")
        if kind == "continuous":
            return UtilityAnalysis._continuous_win(baseline, candidate)
        if feasible is not None and not all(feasible):
            return False
        base = UtilityAnalysis._integral(UtilityAnalysis.seed_median(baseline))
        cand = UtilityAnalysis._integral(UtilityAnalysis.seed_median(candidate))
        return cand < base and cand * 100 <= base * (100 - PERCENT[kind])

    @staticmethod
    def _integral(value: float) -> int:
        """Return an exactly integral score as an int.

        Args:
            value (float): Median of integer-valued scores.

        Returns:
            int: The same value.

        Raises:
            ValueError: If the value is not integral.
        """
        if value != int(value):
            raise ValueError(f"expected an integral score, got {value}")
        return int(value)

    @staticmethod
    def _continuous_win(baseline: Sequence[float], candidate: Sequence[float]) -> bool:
        """Apply the multiplicative 5% rule and the reproducibility floor.

        Args:
            baseline (Sequence[float]): Baseline residual per solver seed.
            candidate (Sequence[float]): Candidate residual per solver seed.

        Returns:
            bool: True only when both the percentage and the floor are met.

        Raises:
            ValueError: If a residual is below zero by more than the floor or is not finite.
        """
        values = []
        for scores in (baseline, candidate):
            median = UtilityAnalysis.seed_median(scores)
            if median < -ATOL:
                raise ValueError(f"invalid residual {median} below the known optimum")
            values.append(max(median, 0.0))
        base, cand = values
        floor = max(ATOL, RTOL * abs(base))
        return cand * 100 <= base * (100 - PERCENT["continuous"]) and base - cand > floor

    @staticmethod
    def instance_win(profiles: Mapping[str, bool]) -> bool:
        """Require a win in every primary profile.

        Args:
            profiles (Mapping[str, bool]): Win flag per profile.

        Returns:
            bool: True only when the mapping is non-empty and every flag is true.
        """
        return bool(profiles) and all(profiles.values())

    @staticmethod
    def hoeffding_p(wins: int, n: int) -> float:
        """Bound the probability of at least this win fraction when the true share is 1/2.

        Args:
            wins (int): Number of meaningful wins.
            n (int): Number of instances.

        Returns:
            float: `exp(-2 n (r - 1/2)^2)` for `r > 1/2`, otherwise one.

        Raises:
            ValueError: If the counts are inconsistent.
        """
        if n < 1 or not 0 <= wins <= n:
            raise ValueError("wins must lie within a positive instance count")
        share = wins / n
        return math.exp(-2 * n * (share - 0.5) ** 2) if share > 0.5 else 1.0

    @staticmethod
    def holm(p_values: Mapping[str, float]) -> dict[str, float]:
        """Return Holm step-down adjusted p-values for a fixed family.

        Args:
            p_values (Mapping[str, float]): Raw p-value per registered claim.

        Returns:
            dict[str, float]: Adjusted p-value per claim, capped at one and monotone.

        Raises:
            ValueError: If the family is empty or a p-value is outside [0, 1].
        """
        if not p_values or not all(0.0 <= value <= 1.0 for value in p_values.values()):
            raise ValueError("a non-empty family of p-values in [0, 1] is required")
        adjusted: dict[str, float] = {}
        running = 0.0
        size = len(p_values)
        for rank, (claim, value) in enumerate(sorted(p_values.items(), key=lambda kv: kv[1])):
            running = max(running, min(1.0, (size - rank) * value))
            adjusted[claim] = running
        return adjusted

    @classmethod
    def decide(
        cls, claims: Mapping[str, Sequence[bool]], expected: int = INSTANCES_PER_CLASS
    ) -> dict[str, str]:
        """Classify every registered claim as evidence, insufficient or incomplete.

        Args:
            claims (Mapping[str, Sequence[bool]]): Instance win flags per registered claim.
            expected (int): Planned instance count per claim.

        Returns:
            dict[str, str]: Status per claim; an incomplete claim keeps its family slot with
            p-value one.

        Raises:
            ValueError: If the family is empty.
        """
        if not claims:
            raise ValueError("the registered family of claims is empty")
        complete = {name: len(wins) == expected for name, wins in claims.items()}
        raw = {
            name: cls.hoeffding_p(sum(wins), expected) if complete[name] else 1.0
            for name, wins in claims.items()
        }
        adjusted = cls.holm(raw)
        return {name: cls._status(complete[name], adjusted[name]) for name in claims}

    @staticmethod
    def _status(complete: bool, adjusted: float) -> str:
        """Label one claim from its completeness and adjusted p-value.

        Args:
            complete (bool): Whether every planned instance is present.
            adjusted (float): Holm-adjusted p-value.

        Returns:
            str: `incomplete`, `evidence` or `insufficient`.
        """
        if not complete:
            return INCOMPLETE
        return EVIDENCE if adjusted <= ALPHA else INSUFFICIENT
