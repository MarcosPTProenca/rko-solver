"""Independent scoring and executor contracts of the utility study.

Objective:
- Prove the study scores outside the solver and refuses any run that violates its plan.

Technical approach:
- Tiny item, routing and continuous instances with hand-checkable scores.
- Command construction and failure paths with a replaced subprocess boundary.
- Real end-to-end trials through the native benchmark and the public solver on tiny budgets.

Flow (pseudocode):
1. Score fixed key vectors and compare with the expected shortfall, tour and residual.
2. Build commands for each arm and profile; reject wrong echoes, modes and stops.
3. Run one real trial per executor and per profile and inspect the returned row.

Important criteria:
- A mismatching specification echo, calibration flag, worker count or stop reason aborts.

Inputs/Outputs:
- Input: temporary instance files and planned trials.
- Output: deterministic assertions; the native trials skip when the binary is not built.
"""

import json
import pytest
import subprocess
from benchmarks.algorithms.sa_specification import SaSpecification
from benchmarks.algorithms.utility_executors import AdaptiveExecutor
from benchmarks.algorithms.utility_executors import TemperatureExecutor
from benchmarks.algorithms.utility_executors import UtilityScoring
from benchmarks.algorithms.utility_inputs import UtilityInputs
from benchmarks.engines.gate_support import ConfigOverrides
from benchmarks.problems.knapsack_utils import exact_profit
from examples.run_kp_cpp_instances import read_kp_instance
from pathlib import Path
from typing import Any

BINARY = Path(__file__).parents[2] / "target" / "release" / "rko_bench"
needs_binary = pytest.mark.skipif(not BINARY.exists(), reason="native benchmark not built")


@pytest.fixture
def entries(tmp_path: Path) -> list[dict[str, Any]]:
    """Write one tiny instance per class and describe it as manifest entries.

    Args:
        tmp_path (Path): Scratch directory.

    Returns:
        list[dict[str, Any]]: Item, route and continuous entries.
    """
    item = tmp_path / "tiny.kp"
    item.write_text("4 10\n10 5\n40 4\n30 6\n50 3\n")
    route = tmp_path / "square.txt"
    route.write_text("5\n0 0\n3 0\n3 4\n0 4\n1 1\n")
    shifts = tmp_path / "shift.txt"
    shifts.write_text("6\n0.5\n-0.25\n1.0\n0.0\n-1.5\n0.75\n")
    optimum = exact_profit(read_kp_instance(item))
    return [
        {"class": "item", "instance": "item", "path": str(item), "reference": optimum},
        {"class": "route", "instance": "route", "path": str(route), "reference": None},
        {"class": "continuous", "instance": "cont", "path": str(shifts), "reference": None},
    ]


def trial(instance: str, profile: str, arm: str, seed: int = 20262001) -> dict[str, Any]:
    """Build a planned trial.

    Args:
        instance (str): Manifest instance name.
        profile (str): `evaluations` or `time`.
        arm (str): `baseline` or `candidate`.
        seed (int): Solver seed.

    Returns:
        dict[str, Any]: Trial mapping.
    """
    return {"instance": instance, "profile": profile, "arm": arm, "replicate": 0, "seed": seed}


def test_item_scorer_is_the_shortfall_to_the_exact_optimum(
    tmp_path: Path, entries: list[dict[str, Any]]
) -> None:
    """Score profit shortfall and capacity feasibility from the keys alone.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
    """
    scoring = UtilityScoring(UtilityInputs(tmp_path)).for_entry(entries[0])
    assert scoring.dimension == 4
    best = [0.0, 0.9, 0.0, 0.9]
    assert scoring.score(best) == 0 and scoring.feasible(best)
    assert scoring.score([0.0, 0.0, 0.0, 0.0]) == entries[0]["reference"]
    everything = [0.9, 0.9, 0.9, 0.9]
    assert scoring.feasible(everything) is False and scoring.score(everything) > 1000


def test_route_and_continuous_scorers_match_hand_values(
    tmp_path: Path, entries: list[dict[str, Any]]
) -> None:
    """Match a rounded tour and a shifted Rastrigin value computed by hand.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
    """
    scoring = UtilityScoring(UtilityInputs(tmp_path))
    route = scoring.for_entry(entries[1])
    assert route.dimension == 5 and route.score([0.1, 0.2, 0.3, 0.4, 0.5]) == 3 + 4 + 3 + 3 + 1
    cont = scoring.for_entry(entries[2])
    optimum_keys = [(shift + 5.12) / 10.24 for shift in (0.5, -0.25, 1.0, 0.0, -1.5, 0.75)]
    assert cont.score(optimum_keys) == pytest.approx(0.0, abs=1e-9)
    assert cont.score([0.5] * 6) > 1.0
    with pytest.raises(ValueError, match="unknown class"):
        scoring.for_entry({"class": "graph", "path": "x"})


def test_command_encodes_arm_profile_worker_and_packaged_specification(
    tmp_path: Path, entries: list[dict[str, Any]]
) -> None:
    """Pass calibration only to the candidate and the budget only to its own profile.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
    """
    executor = TemperatureExecutor(
        tmp_path / "bin", entries, UtilityScoring(UtilityInputs(tmp_path)), (10.0, 100_000)
    )
    spec = SaSpecification.for_dimension(4).native_arguments()
    base = executor.command(trial("item", "evaluations", "baseline"), 4)
    cand = executor.command(trial("item", "time", "candidate"), 4)
    assert base[2] == "600.0" and "--evaluations" in base and "--sa-t0-auto" not in base
    assert cand[2] == "10.0" and "--evaluations" not in cand and "--sa-t0-auto" in cand
    for command in (base, cand):
        assert command[command.index("--threads") + 1] == "1" and "--fixed" in command
        assert command[command.index("--algo") + 1] == "SA"
        assert command[command.index("--problem") + 1] == "kp"
        assert command[command.index("--sa-parameters") + 1] == spec[1]
    assert executor.command(trial("route", "time", "baseline"), 5)[4] == "tsp"
    assert executor.command(trial("cont", "time", "baseline"), 6)[4] == "rastrigin"


def fake_output(**override: object) -> str:
    """Build a native output that matches the plan, with optional deviations.

    Args:
        **override (object): Summary fields to change.

    Returns:
        str: Result line plus summary line.
    """
    summary: dict[str, Any] = {
        "cost": -90.0,
        "evaluations": 100,
        "elapsed": 0.01,
        "workers": 1,
        "stops": ["max_evaluations"],
        "sa_t0_auto": False,
        "adaptive": False,
        "sa_parameters": list(SaSpecification.for_dimension(4).values),
    }
    summary.update(override)
    keys = json.dumps({"keys": [0.0, 0.9, 0.0, 0.9], "objective": -90.0})
    return f"RKO_BENCH_RESULT {keys}\n{json.dumps(summary)}\n"


@pytest.mark.parametrize(
    "override",
    [
        {"sa_parameters": [1000.0, 1e6, 0.99, 0.01, 0.05]},
        {"sa_parameters": None},
        {"sa_t0_auto": True},
        {"adaptive": True},
        {"workers": 2},
        {"stops": ["time_limit"]},
        {"evaluations": 99},
        {"cost": 5.0},
    ],
)
def test_a_run_that_violates_the_plan_aborts(
    tmp_path: Path,
    entries: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
) -> None:
    """Refuse a wrong specification echo, mode, workers, stop, budget or objective.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
        monkeypatch (pytest.MonkeyPatch): Replaces only the process boundary.
        override (dict[str, Any]): Deviation injected into the native summary.
    """
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, fake_output(**override), ""),
    )
    executor = TemperatureExecutor(
        tmp_path / "bin", entries, UtilityScoring(UtilityInputs(tmp_path)), (10.0, 100)
    )
    with pytest.raises((RuntimeError, ValueError)):
        executor(trial("item", "evaluations", "baseline"))


def test_a_clean_run_returns_an_independently_scored_row(
    tmp_path: Path, entries: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Return score, feasibility and the run metadata when everything matches.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
        monkeypatch (pytest.MonkeyPatch): Replaces only the process boundary.
    """
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, fake_output(), ""),
    )
    executor = TemperatureExecutor(
        tmp_path / "bin", entries, UtilityScoring(UtilityInputs(tmp_path)), (10.0, 100)
    )
    row = executor(trial("item", "evaluations", "baseline"))
    assert row["score"] == 0 and row["feasible"] is True and row["evaluations"] == 100
    assert row["stop"] == "max_evaluations" and row["arm"] == "baseline"


def test_a_failing_process_aborts(
    tmp_path: Path, entries: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surface a nonzero exit as an error.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
        monkeypatch (pytest.MonkeyPatch): Replaces only the process boundary.
    """
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 3, "", "boom")
    )
    executor = TemperatureExecutor(
        tmp_path / "bin", entries, UtilityScoring(UtilityInputs(tmp_path))
    )
    with pytest.raises(RuntimeError, match="boom"):
        executor(trial("item", "evaluations", "baseline"))


@needs_binary
@pytest.mark.parametrize("instance", ["item", "route", "cont"])
@pytest.mark.parametrize("profile", ["evaluations", "time"])
def test_real_native_trials_honor_the_planned_budget_and_mode(
    tmp_path: Path, entries: list[dict[str, Any]], instance: str, profile: str
) -> None:
    """Run real baseline and candidate trials through the native benchmark.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
        instance (str): Manifest instance name.
        profile (str): Budget profile.
    """
    executor = TemperatureExecutor(
        BINARY, entries, UtilityScoring(UtilityInputs(tmp_path)), (0.2, 3000)
    )
    for arm in ("baseline", "candidate"):
        row = executor(trial(instance, profile, arm))
        assert row["stop"] == ("max_evaluations" if profile == "evaluations" else "time_limit")
        assert row["score"] >= 0 and (row["evaluations"] == 3000 or profile == "time")


@needs_binary
def test_equal_seeds_reproduce_the_evaluation_profile_exactly(
    tmp_path: Path, entries: list[dict[str, Any]]
) -> None:
    """Repeat one real evaluation-limited trial and require identical results.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
    """
    executor = TemperatureExecutor(
        BINARY, entries, UtilityScoring(UtilityInputs(tmp_path)), (0.2, 3000)
    )
    first = executor(trial("route", "evaluations", "candidate"))
    second = executor(trial("route", "evaluations", "candidate"))
    assert (first["score"], first["evaluations"]) == (second["score"], second["evaluations"])


@pytest.mark.parametrize("instance", ["item", "route", "cont"])
@pytest.mark.parametrize("profile", ["evaluations", "time"])
def test_real_adaptive_trials_toggle_only_the_schedule(
    tmp_path: Path, entries: list[dict[str, Any]], instance: str, profile: str
) -> None:
    """Run real trials through the public solver with the two copied configurations.

    Args:
        tmp_path (Path): Scratch directory.
        entries (list[dict[str, Any]]): Tiny instances.
        instance (str): Manifest instance name.
        profile (str): Budget profile.
    """
    params = {}
    for arm, value in (("baseline", "false"), ("candidate", "true")):
        directory = tmp_path / arm
        directory.mkdir()
        params[arm] = ConfigOverrides.parse([f"progress_schedule={value}"]).write(directory)
    executor = AdaptiveExecutor(
        entries, UtilityScoring(UtilityInputs(tmp_path)), params, (0.3, 500)
    )
    for arm in ("baseline", "candidate"):
        row = executor(trial(instance, profile, arm))
        assert row["stop"] == ("max_evaluations" if profile == "evaluations" else "time_limit")
        assert row["score"] >= 0 and row["evaluations"] > 0
