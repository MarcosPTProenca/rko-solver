"""Acceptance and restart ablation for ILS on generated models (issue 113, measurement).

Objective:
- Measure whether late acceptance or a restart after stagnation changes best cost at equal
  wall time, against the shipped strict-improvement rule, before any option is added to `src/`.

Technical approach:
- Reuse the shipped `shake` perturbation and `rvnd` local search; only the acceptance rule and the
  restart policy differ between arms.
- Every family decodes keys with the same box mapping and scores a penalized minimization cost
  (objective plus a fixed weight per unit of row violation). Feasibility is checked independently.
- Arms: `strict` (perturb the best, keep only strict improvement; the shipped rule), `late`
  (perturb the current solution; accept when not worse than the cost recorded a list-length
  iterations earlier), and `restart` (strict, plus a fresh random start after a fixed number of
  non-improving iterations, best kept).

Flow (pseudocode):
1. Build a fixture; draw the initial keys from the seeded stream.
2. Run one arm for the wall budget: shake, local search, accept by the arm's rule, track best.
3. Record the best penalized cost, its feasibility and iterations; compare arms on paired seeds.

Important criteria:
- Equal budget is wall time per run; arms run one after another.
- Costs are compared only within a fixture and seed; the best objective gap is reported too.
- Binary and integer variables only: continuous-variable families are not decoded here.

Inputs/Outputs:
- Input: family, sizes, seeds, budget in seconds.
- Output: JSON records per arm, fixture and seed.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import sys
import time
from ..modeling.cases import PracticalModels
from ..modeling.hard_cases import HardCases
from ..modeling.key_feasibility import KeyFeasibilityStudy
from ..modeling.qualification_cases import QualificationCases
from collections import deque
from pathlib import Path
from rko_solver.components.local_search.rvnd import rvnd
from rko_solver.components.shaking.random import shake

PENALTY = 1000.0
LATE_LENGTH = 20
RESTART_AFTER = 30
ARMS = ("strict", "late", "restart")
FAMILIES = ("correlated_knapsack", "binary_chain", "ranged_integer")


class CaseDecoder:
    """Box-map keys to integer or binary values and score the penalized objective.

    Responsibilities:
    - Give every arm the same decoder and cost, so only acceptance differs.
    - Report the decoded objective, independent feasibility and gap for a key vector.

    Attributes:
        case (object): Fixture with variables, rows, objective and direction.
    """

    def __init__(self, case: object) -> None:
        """Store the fixture used for decoding and scoring."""
        self.case = case
        self.lower = np.array([v.lower for v in case.variables], dtype=float)
        self.width = np.array([v.upper - v.lower for v in case.variables], dtype=float)
        self.sign = -1.0 if case.maximize else 1.0
        self.objective = np.array(case.objective, dtype=float)

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Map keys to integer values in [lower, upper] with the shipped box rule."""
        return self.lower + np.minimum(np.floor(keys * (self.width + 1.0)), self.width)

    def cost(self, values: np.ndarray) -> float:
        """Return minimized penalized cost: signed objective plus a weight per row violation."""
        return float(self.sign * (self.objective @ values) + PENALTY * self._violation(values))

    def _violation(self, values: np.ndarray) -> float:
        """Sum the distance of every ranged row activity outside its interval."""
        total = 0.0
        for row in self.case.rows:
            activity = sum(c * values[i] for i, c in row.coefficients)
            total += max(0.0, row.lower - activity) + max(0.0, activity - row.upper)
        return total

    def report(self, keys: np.ndarray) -> dict[str, object]:
        """Return the decoded objective, independent feasibility and gap of a key vector."""
        values = self.decode(keys)
        labelled = {v.name: float(x) for v, x in zip(self.case.variables, values, strict=True)}
        checked = PracticalModels.check(self.case, labelled)
        objective = float(self.objective @ values)
        return {
            "objective": objective,
            "feasible": bool(checked["feasible"]),
            "gap": KeyFeasibilityStudy.gap(self.case, objective)
            if self.case.expected is not None
            else None,
        }


class AcceptanceStudy:
    """Run one ILS arm for a wall budget and return its best penalized cost.

    Responsibilities:
    - Implement strict, late-acceptance and restart policies around shared components.
    """

    @staticmethod
    def run(arm: str, decoder: CaseDecoder, n: int, seed: int, seconds: float) -> dict[str, object]:
        """Run one arm and return its best cost, feasibility and completed iterations.

        Args:
            arm (str): One of `strict`, `late` or `restart`.
            decoder (CaseDecoder): Shared decoder and cost.
            n (int): Number of key coordinates.
            seed (int): Seed for the initial keys and the shaking stream.
            seconds (float): Wall-clock budget.

        Returns:
            dict[str, object]: Arm, best cost, objective, feasibility, gap, iterations and restarts.
        """
        rng = np.random.default_rng(seed)
        np.random.seed(seed)
        state = AcceptanceStudy._start(rng.random(n), decoder)
        began = time.perf_counter()
        iterations = restarts = 0
        while time.perf_counter() - began < seconds:
            iterations += 1
            base = state["best"] if arm == "strict" else state["current"]
            shaken = shake(base[0].copy(), 0.10, 0.40)
            candidate, candidate_cost = rvnd(shaken, decoder.decode, decoder.cost)
            AcceptanceStudy._accept(arm, state, candidate, candidate_cost)
            if arm == "restart" and state["stalled"] >= RESTART_AFTER:
                restarts += 1
                state["stalled"] = 0
                state["current"] = AcceptanceStudy._start(rng.random(n), decoder)["current"]
        best_keys, best_cost = state["best"]
        return {
            "arm": arm,
            "best_cost": float(best_cost),
            **decoder.report(best_keys),
            "iterations": iterations,
            "restarts": restarts,
        }

    @staticmethod
    def _start(keys: np.ndarray, decoder: CaseDecoder) -> dict[str, object]:
        """Local-search a starting point and seed the best, current and history records."""
        start, cost = rvnd(keys, decoder.decode, decoder.cost)
        return {
            "best": (start.copy(), cost),
            "current": (start.copy(), cost),
            "history": deque([cost] * LATE_LENGTH, maxlen=LATE_LENGTH),
            "stalled": 0,
        }

    @staticmethod
    def _accept(arm: str, state: dict[str, object], candidate: np.ndarray, cost: float) -> None:
        """Track the best solution, then move the current solution under the arm's rule."""
        AcceptanceStudy._track_best(state, candidate, cost)
        if arm == "late":
            AcceptanceStudy._late(state, candidate, cost)
        elif arm == "restart" and cost < state["current"][1]:
            state["current"] = (candidate.copy(), cost)
        elif arm == "strict":
            state["current"] = state["best"]

    @staticmethod
    def _track_best(state: dict[str, object], candidate: np.ndarray, cost: float) -> None:
        """Replace the best when strictly better and count iterations without improvement."""
        if cost < state["best"][1]:
            state["best"] = (candidate.copy(), cost)
            state["stalled"] = 0
        else:
            state["stalled"] += 1

    @staticmethod
    def _late(state: dict[str, object], candidate: np.ndarray, cost: float) -> None:
        """Accept when not worse than the oldest remembered cost, then remember the current cost."""
        if cost <= state["history"][0] or cost <= state["current"][1]:
            state["current"] = (candidate.copy(), cost)
        state["history"].append(state["current"][1])


def fixture(family: str, size: int, seed: int) -> object:
    """Return the fixture of one family at one size and generation seed."""
    if family == "correlated_knapsack":
        return HardCases.build(size, 5, seed)
    return QualificationCases.build(family, size, 20261107)[0]


def main() -> None:
    """Run the arms on one family and write records as JSON.

    Raises:
        ValueError: If a numeric argument is not positive.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=(100,))
    parser.add_argument("--seeds", type=int, nargs="+", default=tuple(range(1, 11)))
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.seconds <= 0:
        raise ValueError("Seconds must be positive.")
    records = []
    for size in arguments.sizes:
        for seed in arguments.seeds:
            case = fixture(arguments.family, size, seed)
            decoder = CaseDecoder(case)
            order = list(ARMS)
            if seed % 2:
                order.reverse()
            for arm in order:
                record = AcceptanceStudy.run(
                    arm, decoder, len(case.variables), seed, arguments.seconds
                )
                record.update({"family": arguments.family, "case": case.name, "seed": seed})
                records.append(record)
                print(json.dumps(record), file=sys.stderr, flush=True)
    arguments.output.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
