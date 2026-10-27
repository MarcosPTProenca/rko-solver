"""Budget-metered objective that every tuner calls.

Objective:
- Give all tuners one equal, enforced budget of solver runs and one interface to spend it.

Technical approach:
- `TuningObjective` wraps a runner `(configuration, instance index, seed) -> score` and counts
  every call; when the budget is spent the next call raises `BudgetExhausted`, so no tuner can
  use more runs than another and none needs to trust itself to stop.
- Run seeds are a function of the instance only, so two configurations scored on the same
  instance share the random stream of the solver (common random numbers), which lowers the noise
  of comparisons between them.

Flow (pseudocode):
1. A tuner calls `run` for a configuration and an instance.
2. The counter is checked and incremented, then the runner is called.
3. After the last allowed run the next call raises `BudgetExhausted`.

Important criteria:
- The counter counts attempted runs exactly once, including runs of repeated configurations.
- The runner never receives held-out instances; they are outside `n_instances`.

Inputs/Outputs:
- Input: a runner, the number of training instances, a budget and a base seed.
- Output: scores (lower is better) and the number of runs used.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence


class BudgetExhausted(Exception):
    """Signal that the budget of solver runs is spent."""


class TuningObjective:
    """Count and limit the solver runs a tuner performs.

    Responsibilities:
    - Forward runs to the runner while the budget lasts and refuse afterwards.
    - Provide the shared per-instance seeds and a mean over an instance subset.

    Attributes:
        runner (Callable[[dict[str, float], int, int], float]): Scores one run.
        n_instances (int): Number of training instances available.
        budget (int): Maximum number of runs.
        base_seed (int): Offset of the per-instance seeds.
        runs_used (int): Runs performed so far.
    """

    def __init__(
        self,
        runner: Callable[[dict[str, float], int, int], float],
        n_instances: int,
        budget: int,
        base_seed: int = 0,
    ) -> None:
        """Create a metered objective.

        Args:
            runner (Callable[[dict[str, float], int, int], float]): Scores one run.
            n_instances (int): Number of training instances.
            budget (int): Maximum number of runs, at least one.
            base_seed (int): Offset of the per-instance seeds.

        Raises:
            ValueError: If the budget or the instance count is not positive.
        """
        if budget < 1 or n_instances < 1:
            raise ValueError("Budget and instance count must be positive.")
        self.runner = runner
        self.n_instances = n_instances
        self.budget = budget
        self.base_seed = base_seed
        self.runs_used = 0

    @property
    def remaining(self) -> int:
        """Return the runs still allowed.

        Returns:
            int: Budget minus runs used.
        """
        return self.budget - self.runs_used

    def run(self, config: dict[str, float], instance: int) -> float:
        """Score one configuration on one training instance.

        Args:
            config (dict[str, float]): Offline parameters.
            instance (int): Training instance index.

        Returns:
            float: Normalized gap; lower is better.

        Raises:
            BudgetExhausted: If no run is left.
            IndexError: If the instance index is out of range.
        """
        if not 0 <= instance < self.n_instances:
            raise IndexError(f"Instance {instance} is not a training instance.")
        if self.runs_used >= self.budget:
            raise BudgetExhausted
        self.runs_used += 1
        return self.runner(config, instance, self.base_seed + instance)

    def mean(self, config: dict[str, float], instances: Sequence[int]) -> float:
        """Average the scores of a configuration over some instances.

        Args:
            config (dict[str, float]): Offline parameters.
            instances (Sequence[int]): Training instance indices.

        Returns:
            float: Mean normalized gap.

        Raises:
            BudgetExhausted: If the budget ends before every instance is scored.
        """
        scores = [self.run(config, instance) for instance in instances]
        return sum(scores) / len(scores)
