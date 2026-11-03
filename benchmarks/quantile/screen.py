"""Shared-weight RKO search contracts for the frozen-model F5 development screen.

Objective:
- Isolate the learned scorer's construction value from the chosen search method.

Technical approach:
- Hold `SolverBridge.decode`/`SolverBridge.cost` fixed per arm and vary only the
  RKO algorithm selection and whether the frozen scorer is active in the codec.
- Every arm receives the same common wall-time allowance (10s or 60s) and one
  process worker, so the three-member portfolio arm divides that one allowance
  across its members the same way `n_jobs=1` already divides remaining time
  for queued single-worker algorithms; no arm's allowance is multiplied.

Flow (pseudocode):
1. Build one solver per arm sharing the same bridge-derived callbacks.
2. Reconstruct the same shared reference trace through each arm's own codec.
3. Run each arm under the common time allowance and report its result.

Important criteria:
- Every arm receives the same checked starting trace and the same wall-time budget.
- The classical arm never sees the frozen scorer; every other arm does.

Inputs/Outputs:
- Input: a frozen scorer, a task instance and a common time allowance.
- Output: one `RKOResult` per arm.
"""

from __future__ import annotations

from benchmarks.quantile.codec import TrajectoryCodec
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.solver import SolverBridge
from benchmarks.quantile.tasks import Cover
from benchmarks.quantile.tasks import Tour
from rko_solver import RKO
from rko_solver import RKOResult

ARMS = ("classical", "independent", "genetic", "portfolio")
PORTFOLIO_ALGORITHMS = ("BRKGA", "SA", "ILS")
ARM_ALGORITHMS = {
    "classical": ("BRKGA",),
    "genetic": ("BRKGA",),
    "independent": ("MultiStart",),
    "portfolio": PORTFOLIO_ALGORITHMS,
}


class ArmBuilder:
    """Build one same-checkpoint RKO solver per study arm.

    Responsibilities:
    - Share one checked starting trace and wall-time allowance across every arm.
    - Keep the classical arm's codec scorer-free and every other arm's scorer frozen.

    Attributes:
        task (Tour | Cover): Fixed task instance for this trial.
        scorer (FrozenScorer): Frozen learned scorer for non-classical arms.
    """

    task: Tour | Cover
    scorer: FrozenScorer

    def __init__(self, task: Tour | Cover, scorer: FrozenScorer) -> None:
        """Store the fixed trial inputs without building any solver yet.

        Args:
            task (Tour | Cover): Fixed task instance for this trial.
            scorer (FrozenScorer): Frozen learned scorer, matching the task family.

        Raises:
            ValueError: If the scorer family does not match the task.
        """
        family = "tour" if isinstance(task, Tour) else "cover"
        if scorer.family != family:
            raise ValueError("The scorer family must match the task.")
        self.task, self.scorer = task, scorer

    def bridge(self, arm: str) -> SolverBridge:
        """Build the bridge for one arm, with or without the frozen scorer.

        Args:
            arm (str): One of `ARMS`.

        Returns:
            SolverBridge: Classical arm uses the analytic codec; every other arm
                uses the frozen-scorer codec.

        Raises:
            ValueError: If `arm` is not a recognized study arm.
        """
        if arm not in ARMS:
            raise ValueError(f"Unknown study arm: {arm!r}.")
        scorer = None if arm == "classical" else self.scorer
        return SolverBridge(TrajectoryCodec(self.task, scorer=scorer))

    def reference_trace(self) -> tuple[int, ...]:
        """Select the better of the classical and neural greedy traces as the shared start.

        Returns:
            tuple[int, ...]: Whichever checked greedy trace has the lower cost; every arm
                re-encodes this same trace through its own codec, so every arm starts from
                an equally good solution (D8) rather than its own independently constructed
                greedy reference.
        """
        classical = self.bridge("classical").codec.greedy()
        neural = self.bridge("independent").codec.greedy()
        return classical.trace if classical.cost <= neural.cost else neural.trace

    def initial_key(self, arm: str, reference: tuple[int, ...]) -> list[float]:
        """Encode the shared reference trace through one arm's own codec.

        Args:
            arm (str): One of `ARMS`.
            reference (tuple[int, ...]): Shared trace from `reference_trace`.

        Returns:
            list[float]: One admissible checked starting key vector for this arm's codec.
        """
        return self.bridge(arm).codec.encode(reference).tolist()

    def solver(self, arm: str, seed: int, reference: tuple[int, ...]) -> RKO:
        """Build the one-process-worker solver for an arm under the common allowance.

        Args:
            arm (str): One of `ARMS`.
            seed (int): Search seed.
            reference (tuple[int, ...]): Shared starting trace from `reference_trace`.

        Returns:
            RKO: Configured solver; not yet solved. `n_jobs=1` makes every arm run
                under one process worker, so the three-member portfolio arm divides
                the common wall-time allowance across its members exactly as any
                single-worker multi-algorithm queue does; the allowance is never
                multiplied by the member count.
        """
        bridge = self.bridge(arm)
        return RKO(
            bridge.decode,
            bridge.cost,
            self.task.horizon,
            algorithms=list(ARM_ALGORITHMS[arm]),
            adaptation="fixed",
            execution="process",
            n_jobs=1,
            initial_keys=[self.initial_key(arm, reference)],
            random_state=seed,
        )

    def run(self, arm: str, seed: int, time_limit: float, reference: tuple[int, ...]) -> RKOResult:
        """Build and solve one arm's solver under the common wall-time allowance.

        Args:
            arm (str): One of `ARMS`.
            seed (int): Search seed.
            time_limit (float): Common allowance (10 or 60 seconds) shared by every arm.
            reference (tuple[int, ...]): Shared starting trace from `reference_trace`.

        Returns:
            RKOResult: Completed result.

        Raises:
            ValueError: If `arm`, settings or produced keys are invalid.
            RuntimeError: If a worker fails or exits unexpectedly.
        """
        return self.solver(arm, seed, reference).solve(time_limit)
