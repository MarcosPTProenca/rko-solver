"""Search spaces of the tunable offline parameters on a shared unit cube.

Objective:
- Describe, per algorithm, which parameters a tuner may change, over which ranges, and map a
  point of the unit cube to a configuration the solver accepts.

Technical approach:
- Each parameter has a range and a scale (linear, logarithmic, or logarithmic in `1 - value`
  for cooling factors near one) and may be integral. Every tuner searches the same unit cube,
  so no method gets a differently shaped problem.
- A repair step enforces the joint rules the solver validates (`beta_min <= beta_max`,
  `he <= hs`, a BRKGA population that leaves non-elite offspring and at least one mutant), so
  every point decodes to a valid configuration instead of an error.

Flow (pseudocode):
1. Look up the space of an algorithm.
2. Decode a unit-cube point into parameter values on their scales.
3. Repair joint constraints and return a plain dictionary.

Important criteria:
- The packaged default of every algorithm lies inside its space.
- The simulated-annealing baseline is pinned to the default the archived studies compared
  against, so changing the packaged preset never redefines a reference comparison or calibration.
- Decoding is a pure function of the point: equal points give equal configurations.
- The original profile remains unchanged; the expanded profile widens binding SA and BRKGA axes.

Inputs/Outputs:
- Input: an algorithm name and a sequence of floats in [0, 1].
- Output: a configuration dictionary usable as `hyperparameters={ALGORITHM: config}`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from rko_solver.parameters.loader import load_offline

STUDY_DEFAULTS: dict[str, dict[str, float]] = {
    "SA": {"sa_max": 1000.0, "t0": 1000000.0, "alpha": 0.99, "beta_min": 0.01, "beta_max": 0.05}
}


@dataclass(frozen=True)
class ParameterSpec:
    """One tunable parameter.

    Responsibilities:
    - Map a unit interval value to the parameter's range on its scale.

    Attributes:
        name (str): Parameter key accepted by the solver.
        low (float): Smallest value.
        high (float): Largest value.
        scale (str): `linear`, `log`, or `log1m` (logarithmic in `1 - value`).
        integer (bool): Whether the value is rounded to an integer.
    """

    name: str
    low: float
    high: float
    scale: str = "linear"
    integer: bool = False

    def decode(self, unit: float) -> float:
        """Map a unit value to this parameter's range.

        Args:
            unit (float): Value in [0, 1]; values outside are clipped.

        Returns:
            float: Parameter value as a plain `float` (the solver rejects numpy scalars),
                rounded when the parameter is integral.

        Raises:
            ValueError: If the scale is not one of the supported names.
        """
        point = min(1.0, max(0.0, float(unit)))
        if self.scale == "linear":
            value = self.low + point * (self.high - self.low)
        elif self.scale == "log":
            value = math.exp(
                math.log(self.low) + point * (math.log(self.high) - math.log(self.low))
            )
        elif self.scale == "log1m":
            top, bottom = 1.0 - self.low, 1.0 - self.high
            value = 1.0 - math.exp(math.log(top) + point * (math.log(bottom) - math.log(top)))
        else:
            raise ValueError(f"Unknown scale '{self.scale}'.")
        return float(round(value)) if self.integer else value


@dataclass(frozen=True)
class ParameterSpace:
    """The tunable parameters of one algorithm.

    Responsibilities:
    - Decode unit-cube points into valid configurations and report the default one.

    Attributes:
        algorithm (str): Canonical algorithm name.
        specs (tuple[ParameterSpec, ...]): Parameters in cube-axis order.
    """

    algorithm: str
    specs: tuple[ParameterSpec, ...]

    @property
    def dimension(self) -> int:
        """Return the number of cube axes.

        Returns:
            int: Number of tunable parameters.
        """
        return len(self.specs)

    def decode(self, point: Sequence[float]) -> dict[str, float]:
        """Decode a unit-cube point into a valid configuration.

        Args:
            point (Sequence[float]): One value in [0, 1] per parameter.

        Returns:
            dict[str, float]: Repaired configuration keyed by parameter name.

        Raises:
            ValueError: If the point has the wrong number of coordinates.
        """
        if len(point) != self.dimension:
            raise ValueError(f"{self.algorithm}: expected {self.dimension} coordinates.")
        values = {
            spec.name: spec.decode(unit) for spec, unit in zip(self.specs, point, strict=True)
        }
        return self._repair(values)

    def default(self) -> dict[str, float]:
        """Return the baseline configuration of the tunable parameters.

        Algorithms listed in `STUDY_DEFAULTS` return their pinned historical default, because
        the archived studies and their calibration scales were measured against it. Other
        algorithms return the packaged offline defaults.

        Returns:
            dict[str, float]: Baseline value per tunable parameter.
        """
        pinned = STUDY_DEFAULTS.get(self.algorithm)
        if pinned is not None:
            return {spec.name: pinned[spec.name] for spec in self.specs}
        packaged = load_offline(self.algorithm)
        return {spec.name: packaged[spec.name] for spec in self.specs}

    def _repair(self, values: dict[str, float]) -> dict[str, float]:
        """Enforce the joint rules the solver validates.

        Args:
            values (dict[str, float]): Decoded values.

        Returns:
            dict[str, float]: Values that pass the solver's joint validation.
        """
        if "beta_min" in values:
            values["beta_max"] = max(values["beta_max"], values["beta_min"])
        if "he" in values:
            values["he"] = min(values["he"], values["hs"])
        if "pm" in values:
            population = int(values["p"])
            elite = max(1, int(population * values["pe"]))
            room = (population - elite - 1) / population
            if int(population * values["pm"]) > population - elite - 1:
                values["pm"] = max(0.0, room * 0.999)
            values["pm"] = max(values["pm"], (1.0 + 1e-9) / population)
            values["p"] = float(population)
        return values


SPACES: dict[str, ParameterSpace] = {
    "SA": ParameterSpace(
        "SA",
        (
            ParameterSpec("sa_max", 10, 5000, "log", True),
            ParameterSpec("t0", 1e3, 1e8, "log"),
            ParameterSpec("alpha", 0.90, 0.999, "log1m"),
            ParameterSpec("beta_min", 0.001, 0.1, "log"),
            ParameterSpec("beta_max", 0.01, 0.5, "log"),
        ),
    ),
    "BRKGA": ParameterSpace(
        "BRKGA",
        (
            ParameterSpec("p", 10, 500, "log", True),
            ParameterSpec("pe", 0.05, 0.35),
            ParameterSpec("pm", 0.01, 0.30),
            ParameterSpec("rhoe", 0.5, 0.9),
        ),
    ),
    "GRASP": ParameterSpace(
        "GRASP",
        (
            ParameterSpec("hs", 0.01, 0.5, "log"),
            ParameterSpec("he", 1e-5, 1e-2, "log"),
        ),
    ),
}

SPACE_PROFILES: dict[str, dict[str, ParameterSpace]] = {
    "original": SPACES,
    "expanded": {
        **SPACES,
        "SA": ParameterSpace(
            "SA",
            (
                ParameterSpec("sa_max", 1, 5000, "log", True),
                ParameterSpec("t0", 0.01, 1e8, "log"),
                ParameterSpec("alpha", 0.5, 0.999, "log1m"),
                ParameterSpec("beta_min", 1e-5, 0.1, "log"),
                ParameterSpec("beta_max", 1e-4, 0.5, "log"),
            ),
        ),
        "BRKGA": ParameterSpace(
            "BRKGA", (ParameterSpec("p", 4, 2000, "log", True), *SPACES["BRKGA"].specs[1:])
        ),
    },
}
