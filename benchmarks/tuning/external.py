"""Tuners that wrap other optimizers: Optuna's TPE and RKO itself.

Objective:
- Put a widely used hyperparameter optimizer and the library's own metaheuristics through the
  same budgeted protocol as the tuners written for this study.

Technical approach:
- `OptunaTPE` runs a Tree-structured Parzen Estimator study. Each trial scores its configuration
  on the training instances in a fixed order and reports the running mean, so Optuna's median
  pruner can stop a poor trial early and the saved runs fund more trials. The chosen
  configuration is the best trial that was scored on every instance.
- `RKOTuner` is RKO applied to itself: a decoder maps random keys to a configuration of the
  shared unit cube and the cost is the mean score on a fixed random subset of the training
  instances. It uses an SA and a BRKGA search sequentially, each with half of the evaluations.

Flow (pseudocode):
1. Receive a metered objective, a space and a seed.
2. Optuna: loop trials until the budget ends; RKO: solve with an evaluation limit that fits it.
3. Return the best fully scored configuration.

Important criteria:
- Both stop on `BudgetExhausted`: a trial cut by the end of the budget is discarded, and the RKO
  cost returns a large constant after the budget ends so the outer search can finish cleanly.
- Randomness comes only from the given seed.

Inputs/Outputs:
- Input: a `TuningObjective`, a `ParameterSpace` and an integer seed.
- Output: the chosen configuration dictionary.
"""

from __future__ import annotations

import numpy as np
import optuna
from benchmarks.tuning.objective import BudgetExhausted
from benchmarks.tuning.objective import TuningObjective
from benchmarks.tuning.spaces import ParameterSpace
from benchmarks.tuning.tuners import Tuner
from rko_solver import RKO
from rko_solver import Callbacks

EXHAUSTED_COST = 1e6


class OptunaTPE(Tuner):
    """Optuna's TPE sampler with median pruning over the training instances.

    Responsibilities:
    - Run a study within the run budget and return the best fully scored trial.

    Attributes:
        startup (int): Random trials before the model is used and pruning starts.
        warmup (int): Instances scored before a trial may be pruned.
    """

    name = "optuna_tpe"

    def __init__(self, startup: int = 10, warmup: int = 3) -> None:
        """Create the method.

        Args:
            startup (int): Random startup trials, at least one.
            warmup (int): Instances before pruning is allowed, at least one.

        Raises:
            ValueError: If a parameter is out of range.
        """
        if startup < 1 or warmup < 1:
            raise ValueError("startup and warmup must be positive.")
        self.startup = startup
        self.warmup = warmup

    def tune(
        self, objective: TuningObjective, space: ParameterSpace, seed: int
    ) -> dict[str, float]:
        """Run the study until the budget ends.

        Args:
            objective (TuningObjective): Metered objective.
            space (ParameterSpace): Parameters to tune.
            seed (int): Sampler seed.

        Returns:
            dict[str, float]: Best trial scored on every training instance.
        """
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        study = optuna.create_study(
            sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=self.startup),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.startup, n_warmup_steps=self.warmup - 1
            ),
        )

        def trial_objective(trial: optuna.Trial) -> float:
            """Score one trial instance by instance, reporting the running mean.

            Args:
                trial (optuna.Trial): Trial being evaluated.

            Returns:
                float: Mean score over every training instance.

            Raises:
                optuna.TrialPruned: If the trial is pruned or the budget ends.
            """
            point = [trial.suggest_float(f"x{i}", 0.0, 1.0) for i in range(space.dimension)]
            config = space.decode(point)
            scores: list[float] = []
            for instance in range(objective.n_instances):
                try:
                    scores.append(objective.run(config, instance))
                except BudgetExhausted:
                    study.stop()
                    raise optuna.TrialPruned from None
                trial.report(float(np.mean(scores)), instance)
                if trial.should_prune():
                    raise optuna.TrialPruned
            trial.set_user_attr("config", config)
            return float(np.mean(scores))

        study.optimize(trial_objective, n_trials=objective.budget)
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not complete:
            return space.decode([0.5] * space.dimension)
        return dict(min(complete, key=lambda t: t.value).user_attrs["config"])


class RKOTuner(Tuner):
    """RKO tuning RKO: a random-key search over the configuration cube.

    Responsibilities:
    - Minimize the mean training score of a configuration with an SA and a BRKGA search.

    Attributes:
        instances (int): Training instances scored per evaluation.
    """

    name = "rko"

    def __init__(self, instances: int = 4) -> None:
        """Create the method.

        Args:
            instances (int): Training instances per evaluation, at least one.

        Raises:
            ValueError: If the count is not positive.
        """
        if instances < 1:
            raise ValueError("instances must be positive.")
        self.instances = instances

    def tune(
        self, objective: TuningObjective, space: ParameterSpace, seed: int
    ) -> dict[str, float]:
        """Solve the configuration problem with RKO under the run budget.

        Args:
            objective (TuningObjective): Metered objective.
            space (ParameterSpace): Parameters to tune.
            seed (int): Seed of the outer search and of the instance subset.

        Returns:
            dict[str, float]: Best configuration found by the outer search.
        """
        rng = np.random.default_rng(seed)
        count = min(self.instances, objective.n_instances)
        subset = [int(i) for i in rng.choice(objective.n_instances, count, replace=False)]
        evaluations = max(2, objective.budget // count // 2)

        def decode(keys: np.ndarray) -> dict[str, float]:
            """Map random keys to a configuration.

            Args:
                keys (np.ndarray): Random keys of the outer search.

            Returns:
                dict[str, float]: Decoded configuration.
            """
            return space.decode([float(key) for key in keys])

        def cost(config: dict[str, float]) -> float:
            """Score a configuration, or return a large constant when the budget is spent.

            Args:
                config (dict[str, float]): Decoded configuration.

            Returns:
                float: Mean training score.
            """
            try:
                return objective.mean(config, subset)
            except BudgetExhausted:
                return EXHAUSTED_COST

        result = RKO(
            Callbacks(decode, cost, space.dimension),
            algorithms=["SA", "BRKGA"],
            adaptation="fixed",
            engine="python",
            execution="sequential",
            max_evaluations=evaluations,
            random_state=seed,
        ).solve(3600.0)
        return space.decode([float(key) for key in result.keys])
