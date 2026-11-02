"""Shared bounded value types for the resource-capped contextual training pilot.

Objective:
- Keep training settings, budget-exhaustion signaling and paired evaluation rows reusable.

Technical approach:
- Validate every bound eagerly in frozen dataclasses so invalid settings fail immediately.

Flow (pseudocode):
1. Validate requested family, size, step and processor-second bounds at construction.
2. Carry one instance-aggregated construction observation per evaluated panel row.

Important criteria:
- No behavior beyond value validation; the training lifecycle lives in `run.py`.

Inputs/Outputs:
- Input: raw family, size, step, seed and processor-second settings.
- Output: validated frozen settings and evaluation records.
"""

from __future__ import annotations

from dataclasses import dataclass


class TrainingBudgetExpired(RuntimeError):
    """Stop a training job whose total process processor allowance is exhausted.

    Responsibilities:
    - Keep budget exhaustion distinct from an unexpected numerical or implementation failure.
    """


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Freeze the supported initial training pilot settings.

    Responsibilities:
    - Prevent an invocation from silently expanding the approved first training envelope.

    Attributes:
        family (str): Tour or covering feature family.
        size (int): Number of cities or covering rows.
        steps (int): Maximum policy-gradient updates, at most four hundred.
        cpu_seconds (float): Total process processor allowance, at most fifteen hundred seconds.
        seed (int): Local parameter and trajectory initialization.
    """

    family: str
    size: int = 100
    steps: int = 400
    cpu_seconds: float = 1500.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate the explicitly bounded first-pilot settings.

        Raises:
            ValueError: If a requested setting exceeds the supported pilot envelope.
        """
        if self.family not in ("tour", "cover"):
            raise ValueError("Training family must be 'tour' or 'cover'.")
        if not 2 <= self.size <= 500:
            raise ValueError("Training size must lie between 2 and 500.")
        if not 1 <= self.steps <= 400:
            raise ValueError("This pilot supports 1..400 policy-gradient updates.")
        if not 1.0 <= self.cpu_seconds <= 1500.0:
            raise ValueError("This pilot supports 1..1500 processor seconds per family.")
        if self.seed < 0:
            raise ValueError("Training seed must be nonnegative.")


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Carry one instance-aggregated construction observation.

    Responsibilities:
    - Pair comparisons by original data identity rather than row order.

    Attributes:
        seed (int): Instance generation initialization.
        instance_hash (str): Original data fingerprint.
        reference (float): Fixed analytic greedy reference cost.
        mean_cost (float): Mean independently checked construction cost.
        normalized_cost (float): Mean cost divided by the fixed positive reference.
        unique_solutions (int): Distinct solution tuples among the four sampled constructions.
    """

    seed: int
    instance_hash: str
    reference: float
    mean_cost: float
    normalized_cost: float
    unique_solutions: int
