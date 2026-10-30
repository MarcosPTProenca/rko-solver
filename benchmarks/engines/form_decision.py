"""Decision rule for the decoder-form studies, a pure function of paired statistics.

Objective:
- Decide, from paired comparisons fixed before any result, whether a prior inside the residual
  decoder clearly beats seeding the same prior and beats plain RKO.

Technical approach:
- A residual arm passes only if the upper end of its paired 95 percent interval is below zero
  against 32-order seeding and against plain RKO in every family and budget.
- The primary prior decides; the secondary prior decides too when `both` is set. The mean absolute
  difference of the rerun arm gives a noise floor.

Flow (pseudocode):
1. For each decisive prior, check every family, budget and listed comparison.
2. Combine the checks into the advance flag.
3. Average the absolute rerun differences into the noise floor.

Important criteria:
- Nothing here reads data; the rule cannot be tuned after seeing results.

Inputs/Outputs:
- Input: nested statistics keyed by family, budget label and `arm-vs-arm`.
- Output: a dictionary with both holds, the noise floor and the advance flag.
"""

from __future__ import annotations

import numpy as np
from typing import Any


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
    def verdict(
        tables: dict[str, dict[str, dict[str, dict[str, float]]]],
        primary: str = "dual",
        secondary: str = "ratio",
        both: bool = False,
    ) -> dict[str, Any]:
        """Evaluate the rule on paired comparisons.

        Args:
            tables (dict[str, dict[str, dict[str, dict[str, float]]]]): Statistics keyed by
                family, budget label and `arm-vs-arm` pair, each holding `difference`, `ci_low`
                and `ci_high`.
            primary (str): Prior whose residual arm decides advancement.
            secondary (str): Second prior, reported and decisive only when `both` is set.
            both (bool): Whether advancement needs both priors to hold.

        Returns:
            dict[str, Any]: `primary_holds`, `secondary_holds`, the mean absolute rerun
            difference `noise_floor` and the `advance` flag.
        """
        first = DecoderFormDecision.holds(
            tables, f"{primary}_resid", (f"{primary}_seed32", "plain")
        )
        second = DecoderFormDecision.holds(
            tables, f"{secondary}_resid", (f"{secondary}_seed32", "plain")
        )
        noise = [
            abs(budget[f"{primary}_resid_rerun-vs-{primary}_resid"]["difference"])
            for family in tables.values()
            for budget in family.values()
        ]
        return {
            "primary_holds": first,
            "secondary_holds": second,
            "noise_floor": float(np.mean(noise)),
            "advance": first and second if both else first,
        }
