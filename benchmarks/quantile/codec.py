"""Deterministic constructive quantile mapping and admissible-trace reconstruction.

Objective:
- Turn explicit decision keys into checked solutions and recover keys from legal traces.

Technical approach:
- Replay exact task rules and use one shared numerical interval implementation.

Flow (pseudocode):
1. Validate the fixed-size key vector or external action trace.
2. Select legal actions or encode their strict interval midpoints.
3. Finish and independently verify a terminal construction.

Important criteria:
- Unused keys are neutral padding; optional postprocessing is outside the trace inverse.

Inputs/Outputs:
- Input: a validated task and decision keys or an admissible action trace.
- Output: checked immutable candidates or reconstructed keys.
"""

from __future__ import annotations

import numpy as np
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.model import LinearRule
from benchmarks.quantile.probabilities import Intervals
from benchmarks.quantile.tasks import Cover
from benchmarks.quantile.tasks import Tour
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar


class ConstructionFailure(ValueError):
    """Signal an exact task contract that cannot complete the current construction.

    Responsibilities:
    - Keep semantic failures distinct from successful candidates and invalid key inputs.
    """

    status: ClassVar[str] = "failed_construction"


class HorizonExhausted(ConstructionFailure):
    """Signal a nonterminal construction after all declared decision keys were consumed.

    Responsibilities:
    - Expose an insufficient adapter horizon without inventing a penalized solution.
    """

    status: ClassVar[str] = "horizon_exhausted"


@dataclass(frozen=True, slots=True)
class Candidate:
    """Carry one decoded solution together with its exact action trace and checked cost.

    Responsibilities:
    - Preserve the identity of an evaluated construction without mutable last-result state.

    Attributes:
        solution (tuple[int, ...]): Complete task solution.
        trace (tuple[int, ...]): Original ordered legal-action trajectory.
        cost (float): Independently recomputed objective.
        boundary_repairs (int): Exact upper endpoints normalized by the solver bridge.
        status (str): Successful construction classification.
    """

    solution: tuple[int, ...]
    trace: tuple[int, ...]
    cost: float
    boundary_repairs: int = 0
    status: ClassVar[str] = "feasible"


@dataclass(frozen=True, slots=True, eq=False)
class TrajectoryCodec:
    """Construct solutions and re-encode admissible traces with the same intervals.

    Responsibilities:
    - Keep task semantics and numerical configuration fixed during evaluation.
    - Preserve every admissible decision through an analytic midpoint inverse.

    Attributes:
        task (Tour | Cover): Exact validated task adapter.
        temperature (float): Fixed positive action-score temperature.
        mixture (float): Fixed uniform probability mixture.
        scorer (FrozenScorer | LinearRule | None): Optional fixed contextual scoring model.
    """

    task: Tour | Cover
    temperature: float = 1.0
    mixture: float = 0.05
    scorer: FrozenScorer | LinearRule | None = None

    def __post_init__(self) -> None:
        """Validate fixed numerical configuration and the scorer's feature family.

        Raises:
            ValueError: If temperature, mixture or the model's feature family is invalid.
        """
        Intervals.validate(np.zeros(1), self.temperature, self.mixture)
        family = "tour" if isinstance(self.task, Tour) else "cover"
        if self.scorer is not None and self.scorer.family != family:
            raise ValueError("The scorer family does not match the task feature schema.")

    def intervals(self, trace: tuple[int, ...], actions: tuple[int, ...]) -> Intervals:
        """Construct probabilities over the current ordered legal-action list.

        Args:
            trace (tuple[int, ...]): Partial construction trace.
            actions (tuple[int, ...]): Ordered legal actions.

        Returns:
            Intervals: Shared selection and reconstruction intervals.

        Raises:
            ValueError: If task data or action probabilities are invalid.
        """
        scores = (
            self.task.scores(trace, actions)
            if self.scorer is None
            else self.scorer.scores(self.task.features(trace, actions))
        )
        return Intervals.from_logits(scores, self.temperature, self.mixture)

    def decode(self, keys: np.ndarray) -> Candidate:
        """Decode a fixed-size decision-key vector and independently check its solution.

        Args:
            keys (np.ndarray): Decision keys, including unused neutral padding.

        Returns:
            Candidate: Checked solution, original trace and exact objective.

        Raises:
            ValueError: If decisions or the resulting solution are invalid.
            ConstructionFailure: If a nonterminal state has no legal actions.
            HorizonExhausted: If the declared horizon cannot complete construction.
        """
        keys = self.validate_keys(keys)
        trace = self._construct(lambda position, interval: interval.select(float(keys[position])))
        solution = self.task.finish(trace)
        return Candidate(solution, trace, self.task.check(solution))

    def greedy(self) -> Candidate:
        """Construct a checked reference using each highest-probability legal action.

        Returns:
            Candidate: Deterministic feasible baseline or frozen-policy greedy construction.

        Raises:
            ConstructionFailure: If a nonterminal state has no legal action.
            HorizonExhausted: If the declared task horizon is insufficient.
            ValueError: If probabilities or the resulting solution are invalid.
        """
        trace = self._construct(lambda _, interval: int(np.argmax(interval.probabilities)))
        solution = self.task.finish(trace)
        return Candidate(solution, trace, self.task.check(solution))

    def _construct(self, select: Callable[[int, Intervals], int]) -> tuple[int, ...]:
        """Replay the shared legal-action loop with one caller-supplied selection rule.

        Args:
            select (Callable[[int, Intervals], int]): Maps a decision position and its
                current legal-action interval to the chosen action's index.

        Returns:
            tuple[int, ...]: Complete admissible action trace.

        Raises:
            ConstructionFailure: If a nonterminal state has no legal action.
            HorizonExhausted: If the declared task horizon is insufficient.
            ValueError: If probabilities or a selected index are invalid.
        """
        trace: tuple[int, ...] = ()
        for position in range(self.task.horizon):
            if self.task.terminal(trace):
                break
            actions = self.task.legal(trace)
            if not actions:
                raise ConstructionFailure("A nonterminal state has no legal actions.")
            index = select(position, self.intervals(trace, actions))
            trace = (*trace, actions[index])
        if not self.task.terminal(trace):
            raise HorizonExhausted("The task did not terminate within its declared horizon.")
        return trace

    def encode(self, trace: tuple[int, ...]) -> np.ndarray:
        """Encode a complete admissible trace using strict cumulative-interval midpoints.

        Args:
            trace (tuple[int, ...]): Exact ordered decisions before postprocessing.

        Returns:
            np.ndarray: Decision keys with neutral one-half padding.

        Raises:
            ValueError: If the trace or a midpoint is invalid.
        """
        if len(trace) > self.task.horizon:
            raise ValueError("The trace exceeds the task horizon.")
        if any(type(action) is not int and not isinstance(action, np.integer) for action in trace):
            raise ValueError("The trace must contain integer action identifiers.")
        keys = np.full(self.task.horizon, 0.5)
        prefix = self._encode_prefix(keys, trace)
        if not self.task.terminal(prefix):
            raise ValueError("The trace is incomplete.")
        self.task.finish(prefix)
        return keys

    def _encode_prefix(self, keys: np.ndarray, trace: tuple[int, ...]) -> tuple[int, ...]:
        """Fill one midpoint key per decision and return the resulting prefix.

        Args:
            keys (np.ndarray): Mutable decision-key buffer to fill in place.
            trace (tuple[int, ...]): Candidate ordered decisions to replay.

        Returns:
            tuple[int, ...]: The prefix reached after replaying every decision.

        Raises:
            ValueError: If the trace continues after termination or contains a nonlegal action.
        """
        prefix: tuple[int, ...] = ()
        for position, action in enumerate(trace):
            if self.task.terminal(prefix):
                raise ValueError("The trace continues after task termination.")
            actions = self.task.legal(prefix)
            if action not in actions:
                raise ValueError("The trace contains a nonlegal action.")
            keys[position] = self.intervals(prefix, actions).midpoint(actions.index(action))
            prefix = (*prefix, action)
        return prefix

    def validate_keys(self, keys: np.ndarray) -> np.ndarray:
        """Detach and validate the complete fixed-size decision vector before construction.

        Args:
            keys (np.ndarray): External keys including neutral padding.

        Returns:
            np.ndarray: Owned finite keys of the task's exact dimension.

        Raises:
            ValueError: If any key or the vector shape is invalid.
        """
        values = np.array(keys, dtype=np.float64, copy=True)
        if values.shape != (self.task.horizon,):
            raise ValueError("Decision keys must match the task horizon.")
        if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values >= 1.0):
            raise ValueError("Decision keys must be finite and in [0, 1).")
        return values
