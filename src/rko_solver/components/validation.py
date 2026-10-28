"""Semantic validation for all registered metaheuristic configurations.

Objective:
- Reject invalid scalar settings and every invalid online parameter combination.

Technical approach:
- Validate constructor attributes using named algorithm axes and numeric domains.
- Check joint population, cooling and interval constraints before execution.
- Require at least one BRKGA mutant per generation so fresh uniform keys keep exploring.

Flow (pseudocode):
1. Validate common problem and control settings.
2. Validate offline parameter values and coupled constraints.
3. Validate every reachable Cartesian online state when adaptation is enabled.

Important criteria:
- Integral-valued floats are accepted for discrete online grids; fractions are not.
- Booleans and nonfinite values are not numeric hyperparameters.

Inputs/Outputs:
- Input: fully initialized algorithm object and canonical algorithm name.
- Output: no mutation, or a parameter-specific ValueError.
"""

import math
from ..orchestration.limits import RunLimits
from ..parameters.loader import _NAME_TO_FILE
from ..parameters.loader import _ONLINE_AXES
from collections.abc import Callable
from itertools import product
from numbers import Real


class ParameterValidator:
    """Validate algorithm domains independently of engines and execution modes.

    Responsibilities:
    - Apply the same checks to direct constructors and configured portfolios.
    - Reject online states before stochastic selection can activate them.

    Attributes:
        INTEGER_MINIMUMS (dict[str, int]): Lower bounds for discrete parameters.
        PROBABILITIES (set[str]): Inclusive unit-interval parameters.
        POSITIVE (set[str]): Strictly positive scalar parameters.
    """

    INTEGER_MINIMUMS = {"sa_max": 1, "k_max": 1, "p": 2, "size_pop": 3, "p_size": 1}
    PROBABILITIES = {
        "pe",
        "pm",
        "rhoe",
        "prob_cros",
        "prob_mut",
        "beta_min",
        "beta_max",
        "alpha_grasp",
    }
    POSITIVE = {"t0", "hs", "he", "sigma"}

    @classmethod
    def check(cls, algorithm: object, name: str) -> Callable[[list[float]], bool] | None:
        """Validate constructor attributes and reachable online configurations.

        Args:
            algorithm (object): Fully initialized registered algorithm.
            name (str): Canonical algorithm identifier.

        Returns:
            Callable | None: Online feasibility predicate, or None in offline mode.

        Raises:
            ValueError: If settings are invalid or no feasible online state exists.
        """
        dimension = algorithm.tam_solution
        if type(dimension) is not int or dimension <= 0:
            raise ValueError(f"{name}: tam_solution must be a positive integer.")
        if not callable(algorithm.decoder) or not callable(algorithm.cost):
            raise ValueError(f"{name}: decoder and cost must be callable.")
        control = getattr(algorithm, "control", 0)
        if type(control) is not int or control not in (0, 1):
            raise ValueError(f"{name}: control must be integer 0 or 1.")
        axes = _ONLINE_AXES[_NAME_TO_FILE[name]]
        offline_names = ("hs", "he") if name == "GRASP" else axes
        values = {key: getattr(algorithm, key) for key in offline_names}
        if name == "BRKGA_CS":
            values["sigma"] = algorithm.sigma
        for key in cls.INTEGER_MINIMUMS.keys() & values.keys():
            if type(values[key]) is not int:
                raise ValueError(f"{name}: offline '{key}' must be an integer.")
        cls._values(name, values)
        cls._allocation(name, values, dimension)
        if control == 1:
            grid = algorithm.parameters
            if (
                not isinstance(grid, list)
                or len(grid) != len(axes)
                or any(not isinstance(axis, list) or not axis for axis in grid)
            ):
                raise ValueError(f"{name}: parameters must contain nonempty axes {axes}.")
            for state in product(*grid):
                cls._values(name, values | dict(zip(axes, state, strict=True)), check_joint=False)
            cls._allocation(name, dict(zip(axes, map(max, grid), strict=True)), dimension)

            def feasible(state: list[float]) -> bool:
                """Check a joint state without changing its parameter meanings.

                Args:
                    state (list[float]): Candidate Q-learning parameter vector.

                Returns:
                    bool: Whether the entire state satisfies joint constraints.
                """
                try:
                    cls._values(name, values | dict(zip(axes, state, strict=True)))
                except ValueError:
                    return False
                return True

            if not any(feasible(list(state)) for state in product(*grid)):
                raise ValueError(f"{name}: online parameters contain no feasible states.")
            return feasible
        return None

    @staticmethod
    def _allocation(name: str, values: dict[str, object], dimension: int) -> None:
        """Bound the population-like values and the memory their key vectors need.

        Args:
            name (str): Canonical algorithm identifier.
            values (dict[str, object]): Named parameter values, possibly non-numeric.
            dimension (int): Number of random keys.

        Raises:
            ValueError: If a population-like value exceeds the allocation bounds.
        """
        for key in RunLimits.POPULATION_KEYS & values.keys():
            value = values[key]
            if isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value):
                RunLimits.allocation(f"{name}.{key}", float(value), dimension)

    @classmethod
    def _values(
        cls,
        name: str,
        values: dict[str, object],
        *,
        check_joint: bool = True,
    ) -> None:
        """Check one offline or reachable online parameter state.

        Args:
            name (str): Canonical algorithm identifier.
            values (dict[str, object]): Named parameter values.
            check_joint (bool): Check coupled constraints as well as scalar domains.

        Raises:
            ValueError: If an individual or coupled constraint is violated.
        """
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError(f"{name}: '{key}' must be a finite number, not a boolean.")
            if key in cls.INTEGER_MINIMUMS and (
                int(value) != value or value < cls.INTEGER_MINIMUMS[key]
            ):
                raise ValueError(
                    f"{name}: '{key}' must be integral and >= {cls.INTEGER_MINIMUMS[key]}."
                )
            if key in cls.PROBABILITIES and not 0 <= value <= 1:
                raise ValueError(f"{name}: '{key}' must be in [0, 1].")
            if key in cls.POSITIVE and value <= 0:
                raise ValueError(f"{name}: '{key}' must be positive.")
            if key in ("alpha", "alpha_lns") and not 0 < value < 1:
                raise ValueError(f"{name}: '{key}' must be strictly between 0 and 1.")
            if key in ("c1", "c2", "w") and value < 0:
                raise ValueError(f"{name}: '{key}' must be nonnegative.")
        if not check_joint:
            return
        if "beta_max" in values and values["beta_min"] > values["beta_max"]:
            raise ValueError(f"{name}: beta_min must not exceed beta_max.")
        if "he" in values and values["he"] > values["hs"]:
            raise ValueError(f"{name}: he must not exceed hs.")
        if name in ("BRKGA", "BRKGA_CS"):
            cls._genetic_population(name, values)

    @staticmethod
    def _genetic_population(name: str, values: dict[str, object]) -> None:
        """Check the BRKGA population split into elite, mutants and crossover offspring.

        Args:
            name (str): `BRKGA` or `BRKGA_CS`.
            values (dict[str, object]): Named parameter values holding `p`, `pe` and `pm`.

        Raises:
            ValueError: If BRKGA has no mutant or no slot is left for crossover offspring.
        """
        population = int(values["p"])
        elite = max(1, int(population * values["pe"]))
        mutants = int(population * values["pm"]) if name == "BRKGA" else 0
        if name == "BRKGA" and mutants < 1:
            raise ValueError(f"{name}: p * pm must give at least one mutant per generation.")
        if elite + mutants >= population:
            raise ValueError(f"{name}: population must leave nonelite crossover offspring.")
