"""Acceptance and restart ablation for ILS on generated hard knapsacks (issue 113, measurement).

Objective:
- Measure whether late acceptance or a restart after stagnation changes best cost at equal wall
  time, against the shipped strict-improvement rule, before any option is added to `src/`.

Technical approach:
- Reuse the shipped `shake` perturbation and `rvnd` local search; only the acceptance rule and the
  restart policy differ between arms.
- Binary random-key decoding by threshold at 0.5; penalized minimization cost with a fixed weight.
- Arms: `strict` (perturb the best, keep only strict improvement; the shipped rule), `late`
  (late acceptance: perturb the current solution, accept when not worse than the cost recorded a
  list-length iterations earlier), and `restart` (strict, plus a fresh random start after a fixed
  number of non-improving iterations, best kept).

Flow (pseudocode):
1. Build a fixture; draw the initial keys from the seeded stream.
2. Run one arm for the wall budget: shake, local search, accept by the arm's rule, track best.
3. Record the best penalized cost and the iterations completed; compare arms on paired seeds.

Important criteria:
- Equal budget is wall time per run; arms run one after another.
- Costs are compared only within a fixture and seed; the best feasible objective is reported too.
- One family (correlated knapsacks, binary) only: a screen, not the three-family ablation the issue
  asks for.

Inputs/Outputs:
- Input: item counts, seeds, budget in seconds and the arm parameters.
- Output: JSON records and a paired summary (mean difference, sign-test p-value).
"""

from __future__ import annotations

import argparse
import json
import numpy as np
import sys
import time
from ..modeling.hard_cases import HardCases
from collections import deque
from pathlib import Path
from rko_solver.components.local_search.rvnd import rvnd
from rko_solver.components.shaking.random import shake

ROWS = 5
PENALTY = 1000.0
LATE_LENGTH = 20
RESTART_AFTER = 30
ARMS = ("strict", "late", "restart")


class KnapsackDecoder:
    """Threshold-decode binary keys and score the penalized knapsack objective.

    Responsibilities:
    - Give every arm the same decoder and cost, so only acceptance differs.

    Attributes:
        case (object): Fixture with profits, rows and capacities.
    """

    def __init__(self, case: object) -> None:
        """Store the fixture used for decoding and scoring."""
        self.case = case
        self.weights = np.array(
            [
                [dict(r.coefficients).get(j, 0.0) for j in range(len(case.variables))]
                for r in case.rows
            ]
        )
        self.capacity = np.array([r.upper for r in case.rows])
        self.profits = np.array(case.objective)

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Return the 0/1 selection for keys at least one half."""
        return (keys >= 0.5).astype(float)

    def cost(self, selection: np.ndarray) -> float:
        """Return minimized penalized cost: negative profit plus a weight per unit of excess."""
        excess = np.maximum(0.0, self.weights @ selection - self.capacity).sum()
        return float(-(self.profits @ selection) + PENALTY * excess)


class AcceptanceStudy:
    """Run one ILS arm for a wall budget and return its best penalized cost.

    Responsibilities:
    - Implement strict, late-acceptance and restart policies around shared components.
    """

    @staticmethod
    def run(
        arm: str, decoder: KnapsackDecoder, n: int, seed: int, seconds: float
    ) -> dict[str, object]:
        """Run one arm and return its best cost, feasibility and completed iterations.

        Args:
            arm (str): One of `strict`, `late` or `restart`.
            decoder (KnapsackDecoder): Shared decoder and cost.
            n (int): Number of key coordinates (items).
            seed (int): Seed for the initial keys and the shaking stream.
            seconds (float): Wall-clock budget.

        Returns:
            dict[str, object]: Arm, best penalized cost, best feasibility, iterations and restarts.
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
        selection = decoder.decode(best_keys)
        feasible = bool(np.all(decoder.weights @ selection <= decoder.capacity + 1e-9))
        return {
            "arm": arm,
            "best_cost": float(best_cost),
            "best_profit": float(decoder.profits @ selection),
            "feasible": feasible,
            "iterations": iterations,
            "restarts": restarts,
        }

    @staticmethod
    def _start(keys: np.ndarray, decoder: KnapsackDecoder) -> dict[str, object]:
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


def main() -> None:
    """Run the arms on the requested fixtures and write records plus the paired summary.

    Raises:
        ValueError: If a numeric argument is not positive.
    """
    parser = argparse.ArgumentParser(description=__doc__)
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
            case = HardCases.build(size, ROWS, seed)
            decoder = KnapsackDecoder(case)
            order = list(ARMS)
            if seed % 2:
                order.reverse()
            for arm in order:
                record = AcceptanceStudy.run(arm, decoder, size, seed, arguments.seconds)
                record.update({"case": case.name, "seed": seed})
                records.append(record)
                print(json.dumps(record), file=sys.stderr, flush=True)
    arguments.output.write_text(json.dumps({"records": records}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
