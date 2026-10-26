"""Trusted problem contracts, independent objectives and reproducible pilot instances.

Objective:
- Evaluate knapsack, total tardiness and weighted set cover without trusting decoder claims.
Technical approach:
- Require bounded positive-integer data and unique integer decision indices.
- Recompute objective/feasibility and exact small-instance reference values independently.
Flow (pseudocode):
1. Validate owned instance data and solution structure.
2. Recompute constraints and original objective from selected indices.
3. Return original objective, feasibility and a fixed feasibility-first search score.
Important criteria:
- Generated data are research fixtures, not evidence of broad solver superiority.
Inputs/Outputs:
- Input: family/data and selected indices or a job permutation.
- Output: independently computed evidence; no candidate-supplied costs are accepted.
"""

from __future__ import annotations

import json
import math
import numpy as np
from ...components.random_streams import RandomStreams
from .artifacts import FAMILIES
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Problem:
    """Own validated pilot problem data and independent numerical semantics.

    Responsibilities:
    - Check input domains and recompute solution evidence.

    Attributes:
        family (str): Problem family.
        data (dict[str, Any]): Owned JSON-compatible instance snapshot.
    """

    family: str
    data: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate bounded instance data and detach it from the caller.

        Raises:
            ValueError: If data violate the supported positive-integer envelope.
        """
        if self.family not in FAMILIES or not isinstance(self.data, dict):
            raise ValueError("Unsupported problem family/data.")
        data = self.data
        expected = {
            "knapsack": {"weights", "profits", "capacity"},
            "scheduling": {"processing_times", "due_dates"},
            "set_cover": {"subsets", "costs", "universe_size"},
        }[self.family]
        if set(data) != expected:
            raise ValueError("Instance fields do not match the problem contract.")
        for name, value in data.items():
            self._validate_field(name, value)
        count = self._validate_dimensions()
        if not 1 <= count <= 1000:
            raise ValueError("Pilot dimension must lie between 1 and 1000.")
        encoded = json.dumps(data, allow_nan=False)
        if len(encoded.encode("utf-8")) > 24576:
            raise ValueError("Instance data exceed the 24 KiB worker envelope.")
        object.__setattr__(self, "data", json.loads(encoded))

    @staticmethod
    def _validate_field(name: str, value: object) -> None:
        """Validate one scalar or sequence before family-level alignment checks.

        Args:
            name (str): Declared instance field.
            value (object): Untrusted field contents.

        Raises:
            ValueError: If container sizes or integer domains are invalid.
        """
        if name in {"capacity", "universe_size"}:
            if type(value) is not int:
                raise ValueError("Instance scalar fields must be integers.")
            Problem._validate_integers(name, [value])
            return
        if not isinstance(value, list) or len(value) > 1000:
            raise ValueError("Instance sequences must be bounded lists.")
        if name == "subsets":
            Problem._validate_subsets(value)
            flat = [v for subset in value for v in subset]
        else:
            flat = value
        Problem._validate_integers(name, flat)

    @staticmethod
    def _validate_subsets(value: list[Any]) -> None:
        """Check nested set-cover containers and the total membership envelope.

        Args:
            value (list[Any]): Untrusted subset containers.

        Raises:
            ValueError: If subsets are not bounded lists or contain too many members.
        """
        if any(not isinstance(v, list) or len(v) > 10000 for v in value):
            raise ValueError("Subsets must be bounded lists.")
        if sum(map(len, value)) > 8192:
            raise ValueError("Set cover input exceeds the bounded membership envelope.")

    @staticmethod
    def _validate_integers(name: str, flat: list[Any]) -> None:
        """Check the strict integer domain of a flattened field.

        Args:
            name (str): Declared field name.
            flat (list[Any]): Scalar values in input order.

        Raises:
            ValueError: If values are absent or outside the field domain.
        """
        minimum = 0 if name in {"due_dates", "subsets"} else 1
        if not flat or any(type(v) is not int or not minimum <= v <= 100000 for v in flat):
            raise ValueError("Instance values must be bounded integers in their domains.")

    def _validate_dimensions(self) -> int:
        """Check family alignment after all individual fields are validated.

        Returns:
            int: Number of items, jobs or subsets.

        Raises:
            ValueError: If parallel arrays or set-cover domains do not align.
        """
        data = self.data
        if self.family == "knapsack":
            count = len(data["weights"])
            if count != len(data["profits"]):
                raise ValueError("Knapsack weights/profits must align.")
        elif self.family == "scheduling":
            count = len(data["processing_times"])
            if count != len(data["due_dates"]):
                raise ValueError("Scheduling processing times/due dates must align.")
        else:
            count = len(data["costs"])
            if count != len(data["subsets"]) or data["universe_size"] > 10000:
                raise ValueError("Set cover dimensions exceed the supported envelope.")
            self._validate_coverage()
        return count

    def _validate_coverage(self) -> None:
        """Require unique in-range memberships and full available universe coverage.

        Raises:
            ValueError: If a subset is malformed or the union omits an element.
        """
        data = self.data
        for subset in data["subsets"]:
            if len(subset) != len(set(subset)) or any(v >= data["universe_size"] for v in subset):
                raise ValueError("Set cover contains duplicate/out-of-range elements.")
        if set().union(*map(set, data["subsets"])) != set(range(data["universe_size"])):
            raise ValueError("Pilot set cover instances must have full coverage available.")

    @property
    def dimension(self) -> int:
        """Return the instance-dependent key dimension.

        Returns:
            int: Number of items, jobs or subsets.
        """
        name = {"knapsack": "weights", "scheduling": "processing_times", "set_cover": "costs"}[
            self.family
        ]
        return len(self.data[name])

    def evaluate(self, solution: object) -> dict[str, Any]:
        """Recompute feasibility and objective from decision indices alone.

        Args:
            solution (object): Unique index sequence, a full permutation for scheduling.

        Returns:
            dict[str, Any]: Original objective, feasibility and fixed scalar score.

        Raises:
            ValueError: If solution structure or index domain is invalid.
        """
        if (
            not isinstance(solution, (list, tuple))
            or any(type(i) is not int or not 0 <= i < self.dimension for i in solution)
            or len(solution) != len(set(solution))
        ):
            raise ValueError("Solution must contain unique in-range integer indices.")
        return self._evaluate_indices(solution)

    def _evaluate_indices(self, solution: list[int] | tuple[int, ...]) -> dict[str, Any]:
        """Compute family evidence after strict decision-index validation.

        Args:
            solution (list[int] | tuple[int, ...]): Validated unique indices.

        Returns:
            dict[str, Any]: Original objective and feasibility-first score.

        Raises:
            ValueError: If a scheduling permutation is incomplete.
        """
        data = self.data
        if self.family == "knapsack":
            objective = sum(data["profits"][i] for i in solution)
            violation = max(0, sum(data["weights"][i] for i in solution) - data["capacity"])
            normalized = 1 - objective / sum(data["profits"])
        elif self.family == "scheduling":
            if len(solution) != self.dimension:
                raise ValueError("Scheduling requires a complete permutation.")
            elapsed, objective = 0, 0
            for i in solution:
                elapsed += data["processing_times"][i]
                objective += max(0, elapsed - data["due_dates"][i])
            violation = 0
            normalized = objective / (1 + self.dimension * sum(data["processing_times"]))
        else:
            covered = set().union(*(set(data["subsets"][i]) for i in solution))
            violation = data["universe_size"] - len(covered)
            objective = sum(data["costs"][i] for i in solution)
            normalized = objective / sum(data["costs"])
        return {
            "objective": objective,
            "feasible": violation == 0,
            "violation": violation,
            "score": 0.5 * normalized if violation == 0 else 1 + violation,
        }

    def constructive(self) -> list[int]:
        """Build the frozen competent constructive baseline with index-stable ties.

        Returns:
            list[int]: Feasible decision indices or job permutation.
        """
        data, remaining, selected = self.data, list(range(self.dimension)), []
        if self.family == "knapsack":
            capacity = data["capacity"]
            for i in sorted(remaining, key=lambda i: (-data["profits"][i] / data["weights"][i], i)):
                if data["weights"][i] <= capacity:
                    selected.append(i)
                    capacity -= data["weights"][i]
        elif self.family == "scheduling":
            elapsed = 0
            while remaining:
                mean = sum(data["processing_times"][i] for i in remaining) / len(remaining)
                job = min(
                    remaining,
                    key=lambda i: (
                        -math.exp(
                            -max(0, data["due_dates"][i] - data["processing_times"][i] - elapsed)
                            / (2 * mean)
                        )
                        / data["processing_times"][i],
                        i,
                    ),
                )
                selected.append(job)
                remaining.remove(job)
                elapsed += data["processing_times"][job]
        else:
            uncovered = set(range(data["universe_size"]))
            while uncovered:
                useful = [i for i in remaining if uncovered.intersection(data["subsets"][i])]
                index = min(
                    useful,
                    key=lambda i: (
                        data["costs"][i] / len(uncovered.intersection(data["subsets"][i])),
                        i,
                    ),
                )
                selected.append(index)
                remaining.remove(index)
                uncovered.difference_update(data["subsets"][index])
        return selected

    def optimum(self) -> int:
        """Compute an independent exact reference only within bounded pilot sizes.

        Returns:
            int: Exact optimum of the declared integer problem.

        Raises:
            ValueError: If dynamic-programming work exceeds the reference envelope.
        """
        data, n = self.data, self.dimension
        if self.family == "knapsack":
            return self._knapsack_optimum()
        size = n if self.family == "scheduling" else data["universe_size"]
        if size > 16 or n * (1 << size) > 2000000:
            raise ValueError("Reference exceeds the exact-work envelope.")
        values = [math.inf] * (1 << size)
        values[0] = 0
        if self.family == "scheduling":
            self._scheduling_reference(values, n)
        else:
            masks = [sum(1 << x for x in subset) for subset in data["subsets"]]
            for mask in range(1 << size):
                for i, subset in enumerate(masks):
                    values[mask | subset] = min(
                        values[mask | subset], values[mask] + data["costs"][i]
                    )
        return int(values[-1])

    def _knapsack_optimum(self) -> int:
        """Solve bounded knapsack by reverse-capacity dynamic programming.

        Returns:
            int: Exact maximum profit.

        Raises:
            ValueError: If capacity work exceeds the reference envelope.
        """
        data = self.data
        capacity = data["capacity"]
        if self.dimension * capacity > 2000000:
            raise ValueError("Knapsack reference exceeds the exact-work envelope.")
        values = [0] * (capacity + 1)
        for weight, profit in zip(data["weights"], data["profits"], strict=True):
            for size in range(capacity, weight - 1, -1):
                values[size] = max(values[size], values[size - weight] + profit)
        return values[-1]

    def _scheduling_reference(self, values: list[float], n: int) -> None:
        """Fill subset-state tardiness costs in increasing mask order.

        Args:
            values (list[float]): Dynamic-programming costs updated in place.
            n (int): Validated bounded job count.
        """
        data = self.data
        elapsed = [0] * (1 << n)
        for mask in range(1, 1 << n):
            bit = mask & -mask
            elapsed[mask] = elapsed[mask ^ bit] + data["processing_times"][bit.bit_length() - 1]
            for i in range(n):
                if mask & (1 << i):
                    values[mask] = min(
                        values[mask],
                        values[mask ^ (1 << i)] + max(0, elapsed[mask] - data["due_dates"][i]),
                    )

    @classmethod
    def generate(cls, family: str, seed: int, size: int = 12) -> Problem:
        """Create deterministic original development fixtures with available exact references.

        Args:
            family (str): Supported family.
            seed (int): Nonnegative independent instance seed.
            size (int): Requested bounded dimension.

        Returns:
            Problem: Validated feasible-source instance.

        Raises:
            ValueError: If dimensions/family are unsupported.
        """
        cls._validate_generation(seed, size)
        rng = RandomStreams(seed).numpy
        if family == "knapsack":
            weights = rng.randint(1, 40, size).tolist()
            data = {
                "weights": weights,
                "profits": rng.randint(1, 70, size).tolist(),
                "capacity": max(1, sum(weights) // 3),
            }
        elif family == "scheduling":
            times = rng.randint(1, 30, size).tolist()
            data = {
                "processing_times": times,
                "due_dates": rng.randint(0, max(1, sum(times)), size).tolist(),
            }
        elif family == "set_cover":
            universe = min(size, 12)
            subsets = [[i] for i in range(universe)]
            subsets.extend(
                [
                    np.flatnonzero(rng.random(universe) < 0.35).tolist() or [0]
                    for _ in range(size - universe)
                ]
            )
            data = {
                "subsets": subsets,
                "costs": rng.randint(1, 30, size).tolist(),
                "universe_size": universe,
            }
        else:
            raise ValueError("Unsupported generator family.")
        return cls(family, data)

    @staticmethod
    def _validate_generation(seed: int, size: int) -> None:
        """Validate deterministic fixture seed and dimension bounds.

        Args:
            seed (int): Requested uint32 stream seed.
            size (int): Requested problem dimension.

        Raises:
            ValueError: If either argument is not a strict bounded integer.
        """
        if (
            type(seed) is not int
            or not 0 <= seed < 2**32
            or type(size) is not int
            or not 2 <= size <= 100
        ):
            raise ValueError("Invalid generator seed/size.")

    @classmethod
    def seed_source(cls, family: str) -> str:
        """Return explicit reference seed source, not an LLM-generated initial decoder.

        Args:
            family (str): Supported family.

        Returns:
            str: Stable key-order scheduling or threshold selection module.

        Raises:
            ValueError: If family is unknown.
        """
        if family not in FAMILIES:
            raise ValueError("Unsupported seed family.")
        body = (
            "return sorted(range(len(keys)), key=lambda i: (keys[i], i))"
            if family == "scheduling"
            else "return [i for i in range(len(keys)) if keys[i] >= 0.5]"
        )
        return (
            'def decode(keys, instance):\n    """Decode the explicit reference seed."""\n    '
            + body
            + "\n"
        )
