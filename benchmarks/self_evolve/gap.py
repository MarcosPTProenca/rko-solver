"""Original bounded generalized-assignment fixtures and independent exhaustive references.

Objective:
- Validate mathematical-model construction without a GAP-specific optimizer implementation.
Technical approach:
- Keep original integer costs, resource uses and capacities separate from compiled models.
- Reuse existing native model builders and enumerate complete assignments only as a test oracle.
Flow (pseudocode):
1. Own six-job or unseen seven-job original data.
2. Build binary assignment equalities and machine-capacity inequalities.
3. Enumerate bounded complete assignments independently of RKO and candidate code.
Important criteria:
- Oracle witnesses are never constructor inputs or search-pool injections.
- These tiny fixtures are correctness controls, not performance or superiority evidence.
Inputs/Outputs:
- Input: fixed original integer data and an explicit fixture variant.
- Output: frontend-independent formulation and exact small reference objective/witness.
"""

import itertools
import math
from benchmarks.modeling.schema import CaseRow
from benchmarks.modeling.schema import CaseVariable
from benchmarks.modeling.schema import PracticalCase
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeneralizedAssignment:
    """Own original job/machine coefficients independently of numerical model extraction.

    Responsibilities:
    - Provide transparent bounded formulations and independent integer arithmetic checks.

    Attributes:
        costs (tuple): Assignment costs indexed by job, then machine.
        resources (tuple): Resource use in the same original coordinates.
        capacities (tuple[int, ...]): Machine capacities.
    """

    costs: tuple[tuple[int, ...], ...]
    resources: tuple[tuple[int, ...], ...]
    capacities: tuple[int, ...]

    @classmethod
    def fixture(cls, variant: int = 0, *, unseen: bool = False) -> GeneralizedAssignment:
        """Return explicit development variants or a different-size offline replay fixture.

        Args:
            variant (int): One of three predeclared development capacity profiles.
            unseen (bool): Add a seventh job only for offline portability tests.

        Returns:
            GeneralizedAssignment: Original reproducible six- or seven-job data.

        Raises:
            ValueError: If a variant or flag has an invalid type/value.
        """
        if type(variant) is not int or variant not in (0, 1, 2) or type(unseen) is not bool:
            raise ValueError("Expected a declared GAP fixture variant.")
        costs = ((9, 2, 7), (6, 4, 3), (5, 8, 1), (7, 6, 4), (3, 5, 9), (8, 1, 6))
        resources = ((4, 3, 5), (3, 4, 2), (2, 5, 4), (5, 2, 3), (3, 2, 4), (4, 3, 2))
        capacities = ((8, 8, 8), (9, 7, 8), (7, 9, 8))[variant]
        if unseen:
            costs += ((4, 7, 2),)
            resources += ((3, 2, 2),)
            capacities = (10, 10, 10)
        return cls(costs, resources, capacities)

    def evaluate(self, machines: tuple[int, ...]) -> tuple[int, bool]:
        """Check original costs/capacities using complete job-to-machine choices only.

        Args:
            machines (tuple[int, ...]): Exactly one machine index per original job.

        Returns:
            tuple[int, bool]: Original objective and capacity feasibility.

        Raises:
            ValueError: If the assignment has invalid size or machine indices.
        """
        if len(machines) != len(self.costs) or any(
            type(machine) is not int or not 0 <= machine < len(self.capacities)
            for machine in machines
        ):
            raise ValueError("GAP requires one in-range machine per job.")
        loads = [0] * len(self.capacities)
        objective = 0
        for job, machine in enumerate(machines):
            loads[machine] += self.resources[job][machine]
            objective += self.costs[job][machine]
        return objective, all(a <= b for a, b in zip(loads, self.capacities, strict=True))

    def reference(self) -> tuple[int, tuple[int, ...]]:
        """Enumerate the full bounded original problem without using compiled evaluator data.

        Returns:
            tuple[int, tuple[int, ...]]: Exact optimum and a complete original witness.

        Raises:
            ValueError: If enumeration is too large or the fixture has no feasible assignment.
        """
        machines, jobs = len(self.capacities), len(self.costs)
        if machines**jobs > 2000000:
            raise ValueError("GAP reference exceeds its exact enumeration envelope.")
        best: tuple[int, tuple[int, ...]] | None = None
        for assignment in itertools.product(range(machines), repeat=jobs):
            cost, feasible = self.evaluate(assignment)
            if feasible and (best is None or cost < best[0]):
                best = cost, assignment
        if best is None:
            raise ValueError("The complete GAP fixture has no feasible assignment.")
        return best

    def formulation(self, name: str = "gap") -> PracticalCase:
        """Build original binary linear records without solving or compiling them.

        Args:
            name (str): Public fixture name, not a candidate lookup/reference channel.

        Returns:
            PracticalCase: Original formulation accepted by existing native builders.
        """
        machines, jobs = len(self.capacities), len(self.costs)
        variables = tuple(
            CaseVariable(f"x_{i}_{j}", 0, 1, "binary") for i in range(jobs) for j in range(machines)
        )
        rows = tuple(
            CaseRow(f"job_{i}", tuple((i * machines + j, 1) for j in range(machines)), 1, 1)
            for i in range(jobs)
        )
        rows += tuple(
            CaseRow(
                f"capacity_{j}",
                tuple((i * machines + j, self.resources[i][j]) for i in range(jobs)),
                -math.inf,
                self.capacities[j],
            )
            for j in range(machines)
        )
        return PracticalCase(
            name,
            "IP",
            "original-rko-m6-fixture",
            "Original bounded GAP data; not an external benchmark.",
            variables,
            tuple(value for row in self.costs for value in row),
            rows,
            False,
            None,
        )
