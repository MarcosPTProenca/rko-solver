"""Container-only owned-model construction and matched-control evaluation.

Objective:
- Run source construction through the protected original model evidence pipeline.
Technical approach:
- Restore bounded owned data before source loading, then use the explicit trusted hook.
- Retain both the actual score incumbent and independent feasible original objective.
Flow (pseudocode):
1. Validate source/model identity inside the already-qualified external runtime.
2. Run fixed sequential SA or matched random/fixed/check controls.
3. Emit bounded witnesses and actual completed counts for independent parent verification.
Important criteria:
- Never invoke source evaluation outside the qualified worker.
- No native frontend, provider, reference optimum or external optimizer is required.
Inputs/Outputs:
- Input: bounded controller job with owned model data and source.
- Output: numerical witnesses, counters and resource measurements only.
"""

import dataclasses
import hashlib
import numpy as np
import resource
import time
import typing
from collections.abc import Iterator
from rko_solver import RKO
from rko_solver.components.random_streams import RandomStreams
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.experimental.self_evolve.model_data import MODEL_FINALIZATION_ALLOWANCE
from rko_solver.experimental.self_evolve.model_data import ModelProblem
from rko_solver.modeling.construction import ConstructionFn
from rko_solver.modeling.construction import ModelConstruction
from rko_solver.modeling.factory import ModelFactory
from rko_solver.modeling.factory import ModelOptions
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.records import ModelEvaluation


class _Stopped(Exception):
    """Stop between complete model invocations while retaining prior actual witnesses.

    Responsibilities:
    - Keep a partial construction from becoming a reported completed evaluation.

    Attributes:
        args (tuple): Standard cooperative stop details.
    """


class ModelJob:
    """Own one isolated model invocation and bounded evidence, never candidate trajectories.

    Responsibilities:
    - Bind candidate construction without replacing the mathematical evaluator.
    - Preserve original keys, values, objective, score and feasibility separately.

    Attributes:
        problem (ModelProblem): Protected original source and numerical policy.
        started (float): Entire inner job origin, including preparation.
        deadline (float): Frozen cooperative invocation deadline.
        solver (RKO): Ordinary model solver with the fixed control settings.
        incumbent (ModelEvaluation | None): Actual score-selected completed witness.
        feasible (ModelEvaluation | None): Independent original-objective feasible witness.
    """

    def __init__(self, job: dict[str, typing.Any]) -> None:
        """Restore original semantics before loading any source in the isolated worker.

        Args:
            job (dict): Trusted bounded worker envelope, never a provider reply.

        Raises:
            ValueError: If original source, artifact or numerical policy is incompatible.
            Exception: If source initialization fails inside isolation.
        """
        self.started = time.perf_counter()
        self.finalization_allowance = min(MODEL_FINALIZATION_ALLOWANCE, job["seconds"] / 2)
        self.deadline = self.started + job["seconds"] - self.finalization_allowance
        self.problem = ModelProblem(job["data"])
        artifact = DecoderArtifact.create("linear_model", job["source"])
        construction: ConstructionFn | None = None
        if self.problem.dimension:
            namespace: dict[str, typing.Any] = {}
            exec(compile(artifact.source, "decoder.py", "exec"), namespace)
            construction = namespace["decode"]
        options: ModelOptions = {
            **self.problem.data["policy"],
            "construction": construction,
            "construction_deadline": self.deadline if construction is not None else None,
            "algorithms": ["SA"],
            "adaptation": "fixed",
            "engine": "python",
            "execution": "sequential",
            "prefill_pool": False,
            "pool_size": 4,
            "random_state": job["seed"],
        }
        self.solver = ModelFactory.create_owned(RKO, self.problem.source, options)
        owner = getattr(self.solver.decoder, "__self__", None)
        if not isinstance(owner, (CompiledLinearModel, ModelConstruction)):
            raise ValueError("Invalid protected model callback owner.")
        self.callbacks = owner
        self.incumbent: ModelEvaluation | None = None
        self.feasible: ModelEvaluation | None = None
        self.evaluations = 0
        self.infeasible = 0
        self.outputs: set[str] = set()

    def _check(self) -> None:
        """Permit the existing first-completion grace, then stop at the owner deadline.

        Raises:
            _Stopped: If a previous completion exists and time is exhausted.
        """
        if self.evaluations and time.perf_counter() >= self.deadline:
            raise _Stopped

    def _record(self, value: ModelEvaluation) -> None:
        """Retain bounded completed counts and independent original-objective witnesses.

        Args:
            value (ModelEvaluation): Protected completed numerical invocation.
        """
        self.evaluations += 1
        self.infeasible += not value.feasible
        if len(self.outputs) < 256:
            self.outputs.add(hashlib.sha256((np.asarray(value.values) + 0.0).tobytes()).hexdigest())
        if self.incumbent is None or value.score < self.incumbent.score:
            self.incumbent = value
        if value.feasible and (
            self.feasible is None
            or (
                value.objective > self.feasible.objective
                if self.problem.source.maximize
                else value.objective < self.feasible.objective
            )
        ):
            self.feasible = value

    def _rko(self) -> None:
        """Use the ordinary model runtime to retain actual completed optimizer evidence.

        Raises:
            ValueError: If ordinary model execution returns inconsistent/missing evidence.
        """
        result = self.solver.solve(time_limit=max(1e-9, self.deadline - time.perf_counter()))
        model = result.model
        if model is None:
            raise ValueError("Missing protected model result.")
        self.incumbent = model.search_incumbent.evaluation
        self.feasible = model.best_feasible.evaluation if model.best_feasible is not None else None
        if model.statistics is not None:
            stats = model.statistics
            self.evaluations, self.infeasible = stats.evaluations, stats.infeasible_evaluations
            self.outputs = set(stats.outputs or ())
        elif model.no_search:
            self._record(self.incumbent)
        else:
            raise ValueError("Missing completed construction counters.")
        if result.n_evaluations != self.evaluations:
            raise ValueError("Model completion counts disagree with ordinary search evidence.")

    def _controls(self, mode: str, seed: int) -> None:
        """Run declared matched controls, with all construction/checking work inside time.

        Args:
            mode (str): Check, random or fixed-key mode; no invented generic human baseline.
            seed (int): Declared uint32 control seed.

        Raises:
            ValueError: If mode or a constructor output is invalid.
            Exception: Unexpected constructor failures propagate as failed jobs.
        """
        for keys in self._vectors(mode, seed):
            try:
                self._check()
                value = self.callbacks.decode(keys, _budget_check=self._check)
            except _Stopped:
                break
            self._record(value)
            if self.problem.dimension == 0:
                break

    def _vectors(self, mode: str, seed: int) -> Iterator[np.ndarray]:
        """Yield only the fixed declared probes or seeded matched-control chromosomes.

        Args:
            mode (str): Check, random or fixed control.
            seed (int): Declared control random seed.

        Yields:
            np.ndarray: One original unit-interval key vector.

        Raises:
            ValueError: If the mode has no declared model control.
        """
        rng = RandomStreams(seed).numpy
        dimension = self.problem.dimension
        fixed = np.full(dimension, 0.5)
        if mode == "check":
            yield from (
                np.zeros(dimension),
                np.ones(dimension),
                fixed,
                np.linspace(0, 1, dimension),
                rng.random(dimension),
            )
            return
        if mode not in ("random", "fixed"):
            raise ValueError("Model controls require check, random or fixed mode.")
        while True:
            yield fixed if mode == "fixed" else rng.random(dimension)

    @staticmethod
    def _witness(value: ModelEvaluation | None) -> dict[str, typing.Any] | None:
        """Serialize one bounded real witness, never a source-provided claim dictionary.

        Args:
            value (ModelEvaluation | None): Actual completed invocation or absent feasible result.

        Returns:
            dict | None: Portable original keys/values/numerical evidence.
        """
        if value is None:
            return None
        return {
            "keys": list(value.keys),
            "values": list(value.values),
            "objective": value.objective,
            "score": value.score,
            "feasible": value.feasible,
            "repair": dataclasses.asdict(value.repair),
            "rows": [dataclasses.asdict(row) for row in value.rows],
        }

    def execute(self, job: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Emit primary feasible evidence separately from the actual score incumbent.

        Args:
            job (dict): Frozen mode/seed envelope.

        Returns:
            dict: Bounded witnesses and actual completed resource/counter observations.

        Raises:
            ValueError: If no completed invocation exists.
            Exception: Candidate or optimizer errors propagate to the worker error envelope.
        """
        if job["mode"] == "rko":
            self._rko()
        else:
            self._controls(job["mode"], job["seed"])
        if job["mode"] == "check" and self.evaluations != (1 if self.problem.dimension == 0 else 5):
            raise ValueError("Incomplete required model endpoint/control probe panel.")
        primary = self.feasible if self.feasible is not None else self.incumbent
        if primary is None:
            raise ValueError("No completed model invocation.")
        return {
            "status": "ok",
            "finalization_allowance_seconds": self.finalization_allowance,
            "no_search": self.problem.dimension == 0,
            "construction_used": self.problem.dimension != 0,
            "model_source_fingerprint": self.problem.source.fingerprint,
            "model_policy_fingerprint": self.problem.policy_fingerprint,
            "objective": primary.objective,
            "score": primary.score,
            "feasible": primary.feasible,
            "solution": list(primary.values),
            "keys": list(primary.keys),
            "search_incumbent": self._witness(self.incumbent),
            "best_feasible": self._witness(self.feasible),
            "evaluations": self.evaluations,
            "infeasible_evaluations": self.infeasible,
            "distinct_outputs_capped": len(self.outputs),
            "wall_seconds": time.perf_counter() - self.started,
            "cpu_seconds": resource.getrusage(resource.RUSAGE_SELF).ru_utime
            + resource.getrusage(resource.RUSAGE_SELF).ru_stime,
            "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        }
