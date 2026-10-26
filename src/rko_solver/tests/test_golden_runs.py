"""Golden regression runs for the reproducible engine configurations.

Objective:
- Detect any unintended change in what an algorithm computes, by comparing complete seeded
  runs with frozen reference results.

Technical approach:
- Run each algorithm alone, fixed and adaptive, on a small integer Knapsack with a seed and a
  `max_evaluations` limit that is far below the time limit, so the work done never depends on
  machine speed. Only configurations measured to be reproducible are frozen: the Python engine
  with sequential execution and the Rust engine with one worker. Threads and the C++ engine
  are excluded because their runs are not repeatable.
- Record the final cost, a digest of the key vector, the per-search evaluation count and the
  stop reason in `golden/runs.json`.

Flow (pseudocode):
1. Build every case from algorithm, adaptation and engine.
2. Solve it and reduce the result to cost, key digest, counts and reasons.
3. Compare with the reference, or rewrite the reference when updating on purpose.

Important criteria:
- A failure means behavior changed. If the change is intended, regenerate with
  `RKO_UPDATE_GOLDEN=1 uv run pytest src/rko_solver/tests/test_golden_runs.py` and review the
  diff of `golden/runs.json` like any other code change.
- The integer objective keeps the reference independent of floating-point summation order.

Inputs/Outputs:
- Input: a 60-item Knapsack, seed 11, 400 evaluations per algorithm.
- Output: pass/fail against `golden/runs.json`, or a rewritten reference.
"""

import hashlib
import json
import numpy as np
import os
import pytest
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver.orchestration.configuration import MH_SPECS

GOLDEN = Path(__file__).with_name("golden") / "runs.json"
PROFITS = [10 + (index * 37) % 91 for index in range(60)]
WEIGHTS = [5 + (index * 53) % 47 for index in range(60)]
CAPACITY = 700
ENGINES = {
    "python-sequential": {"engine": "python", "execution": "sequential"},
    "rust-one-worker": {"engine": "rust", "n_jobs": 1},
}
LIMIT = 400
SEED = 11
UPDATING = os.environ.get("RKO_UPDATE_GOLDEN") == "1"
native = pytest.importorskip("rko_solver._native")


def decode(keys: np.ndarray) -> np.ndarray:
    """Select items whose key exceeds one half.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: Boolean selection.
    """
    return np.asarray(keys) > 0.5


def cost(selection: np.ndarray) -> float:
    """Score a selection with a penalized negative profit.

    Args:
        selection (np.ndarray): Boolean selection.

    Returns:
        float: Cost to minimize.
    """
    chosen = np.asarray(selection)
    profit = int(np.dot(chosen, PROFITS))
    weight = int(np.dot(chosen, WEIGHTS))
    return float(100000 * max(0, weight - CAPACITY) - profit)


def solve(engine: str, algorithms: list[str] | None, adaptation: str | None) -> dict[str, object]:
    """Run one case and reduce the result to its comparable fingerprint.

    Args:
        engine (str): Key of `ENGINES`.
        algorithms (list[str] | None): Portfolio, or None for the packaged default.
        adaptation (str | None): `fixed`, `adaptive`, or None for the packaged default.

    Returns:
        dict[str, object]: Cost, key digest, per-search evaluations and stop reasons.
    """
    result = RKO(
        Callbacks(decode, cost, len(PROFITS)),
        algorithms=algorithms,
        adaptation=adaptation,
        max_evaluations=LIMIT,
        random_state=SEED,
        **ENGINES[engine],
    ).solve(120.0)
    digest = hashlib.sha256(",".join(repr(float(key)) for key in result.keys).encode())
    return {
        "cost": result.cost,
        "keys_sha256": digest.hexdigest()[:16],
        "searches": [
            [item.algorithm, item.n_evaluations, item.stop_reason] for item in result.searches
        ],
    }


def cases() -> dict[str, tuple[str, list[str] | None, str | None]]:
    """Enumerate every frozen case.

    Returns:
        dict[str, tuple]: Case name mapped to engine key, portfolio and adaptation.
    """
    found: dict[str, tuple[str, list[str] | None, str | None]] = {}
    for engine in ENGINES:
        for algorithm in MH_SPECS:
            for adaptation in ("fixed", "adaptive"):
                found[f"{engine}/{algorithm}/{adaptation}"] = (engine, [algorithm], adaptation)
        found[f"{engine}/default-portfolio"] = (engine, None, None)
    return found


def test_golden_runs_match_the_reference() -> None:
    """Every frozen case reproduces its recorded result exactly, or the file is rewritten."""
    measured = {name: solve(*case) for name, case in cases().items()}
    if UPDATING:
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(measured, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"golden reference rewritten with {len(measured)} cases")
    reference = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert sorted(reference) == sorted(measured), "cases added or removed; regenerate on purpose"
    changed = [name for name in sorted(measured) if measured[name] != reference[name]]
    assert not changed, "behavior changed for: " + ", ".join(changed)


def test_every_registered_algorithm_has_golden_coverage() -> None:
    """A new algorithm cannot be added without a frozen reference for it."""
    reference = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for engine in ENGINES:
        for algorithm in MH_SPECS:
            assert f"{engine}/{algorithm}/fixed" in reference
            assert f"{engine}/{algorithm}/adaptive" in reference
