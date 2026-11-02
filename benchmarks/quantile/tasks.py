"""Exact task semantics for tour and positive-cost covering construction.

Objective:
- Separate feasibility and finite-horizon completion from scoring-model quality.

Technical approach:
- Represent a construction state by its immutable selected-action trace.
- Recompute independent solution objectives from original task data.

Flow (pseudocode):
1. Validate and snapshot task data.
2. Enumerate ordered legal actions from the current trace.
3. Finish a terminal trace and independently check its solution.

Important criteria:
- Tour graphs are complete and symmetric; every covering row must be coverable.

Inputs/Outputs:
- Input: original coordinates or binary incidence with positive costs.
- Output: exact task semantics, action features and independent objectives.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from dataclasses import field


def _integral(values: tuple[int, ...]) -> bool:
    """Tell whether every candidate identifier is a native or NumPy integer.

    Args:
        values (tuple[int, ...]): Candidate action or solution identifiers.

    Returns:
        bool: True only if every value has an accepted integer type.
    """
    return not any(type(value) is not int and not isinstance(value, np.integer) for value in values)


@dataclass(frozen=True, slots=True, eq=False)
class Tour:
    """Construct complete symmetric tours with city zero fixed as the start.

    Responsibilities:
    - Expose unvisited cities as exact legal actions.
    - Check complete tours independently from the construction state.

    Attributes:
        coordinates (np.ndarray): Detached two-dimensional city coordinates.
        distances (np.ndarray): Cached complete Euclidean distance matrix.
    """

    coordinates: np.ndarray
    distances: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate geometry and retain detached read-only arrays.

        Raises:
            ValueError: If coordinates or their pairwise distances are invalid.
        """
        coordinates = np.array(self.coordinates, dtype=np.float64, copy=True)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) < 2:
            raise ValueError("Tour coordinates must have shape (cities >= 2, 2).")
        if not np.isfinite(coordinates).all():
            raise ValueError("Tour coordinates must be finite.")
        with np.errstate(over="ignore", invalid="ignore"):
            delta = coordinates[:, None, :] - coordinates[None, :, :]
            distances = np.hypot(delta[:, :, 0], delta[:, :, 1])
        if not np.isfinite(distances).all():
            raise ValueError("Tour coordinates produce nonfinite distances.")
        coordinates.setflags(write=False)
        distances.setflags(write=False)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "distances", distances)

    @property
    def horizon(self) -> int:
        """Return the number of explicit city-selection decisions.

        Returns:
            int: Number of cities excluding the fixed start.
        """
        return len(self.coordinates) - 1

    def legal(self, trace: tuple[int, ...]) -> tuple[int, ...]:
        """Enumerate unvisited noninitial cities in stable identifier order.

        Args:
            trace (tuple[int, ...]): Selected noninitial cities.

        Returns:
            tuple[int, ...]: Ordered legal next cities.

        Raises:
            ValueError: If a trace repeats or includes a nonselectable city.
        """
        if not _integral(trace):
            raise ValueError("The tour trace must contain integer actions.")
        if len(set(trace)) != len(trace) or not set(trace).issubset(
            range(1, len(self.coordinates))
        ):
            raise ValueError("The tour trace must contain distinct noninitial cities.")
        return tuple(city for city in range(1, len(self.coordinates)) if city not in trace)

    def terminal(self, trace: tuple[int, ...]) -> bool:
        """Check whether every noninitial city has been selected.

        Args:
            trace (tuple[int, ...]): Selected noninitial cities.

        Returns:
            bool: Whether the construction has reached its finite horizon.
        """
        return not self.legal(trace)

    def features(self, trace: tuple[int, ...], actions: tuple[int, ...]) -> np.ndarray:
        """Describe legal next cities with state-dependent normalized geometry.

        Args:
            trace (tuple[int, ...]): Valid partial city trace.
            actions (tuple[int, ...]): Ordered legal next cities.

        Returns:
            np.ndarray: Current distance, home distance, remaining mean distance and progress.

        Raises:
            ValueError: If trace or actions are invalid.
        """
        legal = self.legal(trace)
        if not set(actions).issubset(legal):
            raise ValueError("Tour features require legal actions.")
        indices = np.asarray(actions, dtype=np.int64)
        current = trace[-1] if trace else 0
        scale = max(float(self.distances.max()), np.finfo(float).eps)
        return np.column_stack(
            (
                self.distances[current, indices] / scale,
                self.distances[0, indices] / scale,
                self.distances[np.ix_(indices, np.asarray(legal))].mean(axis=1) / scale,
                np.full(len(indices), len(trace) / self.horizon),
            )
        )

    def scores(self, trace: tuple[int, ...], actions: tuple[int, ...]) -> np.ndarray:
        """Prefer nearby legal cities without a learned model.

        Args:
            trace (tuple[int, ...]): Valid partial tour.
            actions (tuple[int, ...]): Ordered legal cities.

        Returns:
            np.ndarray: Analytic distance-preference logits.

        Raises:
            ValueError: If the construction state or actions are invalid.
        """
        return -10.0 * self.features(trace, actions)[:, 0]

    def finish(self, trace: tuple[int, ...]) -> tuple[int, ...]:
        """Close a checked complete tour at city zero.

        Args:
            trace (tuple[int, ...]): Complete city-selection trace.

        Returns:
            tuple[int, ...]: Closed tour beginning and ending at zero.

        Raises:
            ValueError: If the trace does not form a complete tour.
        """
        solution = (0, *trace, 0)
        self.check(solution)
        return solution

    def check(self, solution: tuple[int, ...]) -> float:
        """Independently validate and sum a closed Euclidean tour.

        Args:
            solution (tuple[int, ...]): Closed sequence containing every city once.

        Returns:
            float: Euclidean tour length recomputed from original coordinates.

        Raises:
            ValueError: If any city, closure or visit count is invalid.
        """
        if not _integral(solution):
            raise ValueError("A tour must contain integer city identifiers.")
        if len(solution) != len(self.coordinates) + 1 or solution[0] != solution[-1]:
            raise ValueError("A tour must be closed and contain the required number of cities.")
        if sorted(solution[:-1]) != list(range(len(self.coordinates))):
            raise ValueError("A tour must visit every city exactly once before closure.")
        return math.fsum(
            math.hypot(*(self.coordinates[a] - self.coordinates[b]))
            for a, b in zip(solution[:-1], solution[1:], strict=True)
        )


@dataclass(frozen=True, slots=True, eq=False)
class Cover:
    """Construct positive-cost covers using strictly coverage-increasing actions.

    Responsibilities:
    - Expose unselected columns that cover an uncovered row.
    - Validate full coverage independently from legal-action construction.

    Attributes:
        incidence (np.ndarray): Binary row-by-column coverage matrix.
        costs (np.ndarray): Positive original column costs.
    """

    incidence: np.ndarray
    costs: np.ndarray

    def __post_init__(self) -> None:
        """Validate and snapshot a fully coverable binary matrix and positive costs.

        Raises:
            ValueError: If incidence, coverability or costs violate the task contract.
        """
        incidence = np.array(self.incidence, copy=True)
        self._validate_incidence(incidence)
        costs = self._validated_costs(incidence)
        incidence = incidence.astype(bool)
        incidence.setflags(write=False)
        costs.setflags(write=False)
        object.__setattr__(self, "incidence", incidence)
        object.__setattr__(self, "costs", costs)

    @staticmethod
    def _validate_incidence(incidence: np.ndarray) -> None:
        """Check that the raw incidence matrix is a nonempty, binary, fully coverable grid.

        Args:
            incidence (np.ndarray): Raw candidate incidence matrix, not yet detached.

        Raises:
            ValueError: If the matrix shape, values or coverability are invalid.
        """
        if incidence.ndim != 2 or min(incidence.shape) == 0:
            raise ValueError("Cover incidence must be a nonempty row-by-column matrix.")
        if not np.isin(incidence, [0, 1]).all() or not incidence.any(axis=1).all():
            raise ValueError("Cover incidence must be binary with every row coverable.")

    def _validated_costs(self, incidence: np.ndarray) -> np.ndarray:
        """Validate the raw cost vector against the matching incidence column count.

        Args:
            incidence (np.ndarray): Already-validated raw candidate incidence matrix.

        Returns:
            np.ndarray: Owned finite positive-cost vector.

        Raises:
            ValueError: If costs are the wrong shape, nonfinite or nonpositive.
        """
        costs = np.array(self.costs, dtype=np.float64, copy=True)
        if costs.shape != (incidence.shape[1],) or not np.isfinite(costs).all():
            raise ValueError("Cover costs must be a matching finite vector.")
        with np.errstate(over="ignore"):
            valid_costs = bool(np.all(costs > 0.0) and np.isfinite(costs.sum()))
        if not valid_costs:
            raise ValueError("Cover costs must be positive with a finite total.")
        return costs

    @property
    def horizon(self) -> int:
        """Return the maximum number of selected-column decisions.

        Returns:
            int: Number of available columns.
        """
        return len(self.costs)

    def covered(self, trace: tuple[int, ...]) -> np.ndarray:
        """Compute covered rows for a construction trace.

        Args:
            trace (tuple[int, ...]): Selected column identifiers.

        Returns:
            np.ndarray: Boolean coverage for each row.

        Raises:
            ValueError: If actions are invalid or fail to increase coverage.
        """
        if not _integral(trace):
            raise ValueError("A cover trace must contain integer actions.")
        if len(set(trace)) != len(trace) or not set(trace).issubset(range(self.horizon)):
            raise ValueError("A cover trace must contain distinct valid columns.")
        covered = np.zeros(len(self.incidence), dtype=bool)
        for column in trace:
            if not np.any(self.incidence[~covered, column]):
                raise ValueError("A cover trace must stop at coverage and make positive progress.")
            covered |= self.incidence[:, column]
        return covered

    def terminal(self, trace: tuple[int, ...]) -> bool:
        """Check whether every row is covered.

        Args:
            trace (tuple[int, ...]): Selected columns.

        Returns:
            bool: Whether construction must stop immediately.
        """
        return bool(np.all(self.covered(trace)))

    def legal(self, trace: tuple[int, ...]) -> tuple[int, ...]:
        """Enumerate unselected columns with strictly positive uncovered gain.

        Args:
            trace (tuple[int, ...]): Selected columns.

        Returns:
            tuple[int, ...]: Ordered legal column identifiers.
        """
        gain = np.sum(self.incidence[~self.covered(trace)], axis=0)
        return tuple(
            column for column in range(self.horizon) if gain[column] and column not in trace
        )

    def features(self, trace: tuple[int, ...], actions: tuple[int, ...]) -> np.ndarray:
        """Describe legal columns with dynamic gain and original normalized costs.

        Args:
            trace (tuple[int, ...]): Valid selected-column trace.
            actions (tuple[int, ...]): Ordered legal next columns.

        Returns:
            np.ndarray: Gain fraction, relative cost, gain-to-cost ratio and progress.

        Raises:
            ValueError: If the state or requested actions are invalid.
        """
        if not set(actions).issubset(self.legal(trace)):
            raise ValueError("Cover features require legal actions.")
        indices = np.asarray(actions, dtype=np.int64)
        covered = self.covered(trace)
        gain = self.incidence[~covered][:, indices].sum(axis=0) / len(self.incidence)
        relative_cost = self.costs[indices] / self.costs.mean()
        return np.column_stack(
            (gain, relative_cost, gain / relative_cost, np.full(len(indices), covered.mean()))
        )

    def scores(self, trace: tuple[int, ...], actions: tuple[int, ...]) -> np.ndarray:
        """Prefer uncovered gain per cost using a conventional analytic construction rule.

        Args:
            trace (tuple[int, ...]): Valid partial cover.
            actions (tuple[int, ...]): Ordered legal columns.

        Returns:
            np.ndarray: Logarithms of relative gain-per-cost scores.

        Raises:
            ValueError: If the construction state or actions are invalid.
        """
        return np.log(self.features(trace, actions)[:, 2])

    def finish(self, trace: tuple[int, ...]) -> tuple[int, ...]:
        """Return a checked cover in canonical column order.

        Args:
            trace (tuple[int, ...]): Terminal selected-column trace.

        Returns:
            tuple[int, ...]: Sorted selected columns.

        Raises:
            ValueError: If the trace does not cover every row exactly as declared.
        """
        self.covered(trace)
        self.check(trace)
        return tuple(sorted(trace))

    def check(self, solution: tuple[int, ...]) -> float:
        """Independently verify row coverage and sum original column costs.

        Args:
            solution (tuple[int, ...]): Distinct selected column identifiers.

        Returns:
            float: Total selected-column cost.

        Raises:
            ValueError: If identifiers repeat, are invalid, or leave a row uncovered.
        """
        if not _integral(solution):
            raise ValueError("A cover must contain integer column identifiers.")
        if len(set(solution)) != len(solution) or not set(solution).issubset(range(self.horizon)):
            raise ValueError("A cover must contain distinct valid column identifiers.")
        columns = np.asarray(solution, dtype=np.int64)
        if not np.any(self.incidence[:, columns], axis=1).all():
            raise ValueError("The cover leaves at least one row uncovered.")
        return math.fsum(float(self.costs[column]) for column in solution)
