"""Frozen-input and native-output contracts of the offline SA screen.

Objective:
- Prevent data reuse and invalid budget measurements in an unshipped experiment.

Technical approach:
- Reconstruct the reserved cohorts and use synthetic native output.

Flow (pseudocode):
1. Compare the selected IDs with every previously reserved pool.
2. Verify deterministic shifts and native result validation.
3. Refuse mismatched objective values or stop reasons.

Important criteria:
- Invalid output is an error, not a silently dropped trial.

Inputs/Outputs:
- Input: temporary caches and synthetic native output.
- Output: deterministic assertions without solving an optimization problem.
"""

import hashlib
import json
import pytest
import random
from benchmarks.algorithms.sa_temperature_screen import Screen
from benchmarks.algorithms.sa_temperature_screen import ScreenInputs
from benchmarks.engines.gate_instances import GateInstance
from benchmarks.engines.gate_instances import GateInstances
from pathlib import Path


def test_screen_inputs_exclude_all_reserved_cohorts(tmp_path: Path) -> None:
    """Keep the new screen outside all three earlier pools.

    Args:
        tmp_path (Path): Temporary input cache root.
    """
    calibration = set(GateInstances(tmp_path, "calibration", 100).knapsack_ids())
    historical = set(GateInstances(tmp_path, "test", 100).knapsack_ids())
    calibration_ids = {number for _, number in calibration}
    test_ids = {number for _, number in historical}
    pool = [number for number in range(100) if number not in calibration_ids | test_ids]
    progress_ids = set(random.Random(20261121).sample(pool, 20))
    chosen = ScreenInputs(tmp_path).knapsack_ids()
    assert len(chosen) == len(set(chosen)) == 20
    assert not set(chosen) & (calibration | historical)
    assert not {number for _, number in chosen} & progress_ids
    assert chosen == ScreenInputs(tmp_path).knapsack_ids()


def test_screen_shift_namespace_is_repeatable(tmp_path: Path) -> None:
    """Generate bounded shifts under a namespace unused by previous gate splits.

    Args:
        tmp_path (Path): Generated input cache.
    """
    inputs = ScreenInputs(tmp_path)
    for dimension in (100, 300):
        path = inputs.rastrigin_path(dimension, 0)
        original = path.read_bytes()
        assert path.name.startswith("sa-screen-")
        assert inputs.rastrigin_path(dimension, 0).read_bytes() == original
        values = path.read_text().splitlines()
        assert int(values[0]) == dimension and len(values) == dimension + 1
        assert all(-2 <= float(value) <= 2 for value in values[1:])
    with pytest.raises(ValueError):
        inputs.rastrigin_path(30, 0)


def output(stops: list[str], evaluations: int = 100_000, cost: float = 0.5) -> str:
    """Construct a minimal native-output pair.

    Args:
        stops (list[str]): Native stop labels.
        evaluations (int): Reported evaluation count.
        cost (float): Reported native cost.

    Returns:
        str: Result-vector prefix followed by a native summary.
    """
    return (
        "RKO_BENCH_RESULT "
        + json.dumps({"keys": [0.5], "objective": cost})
        + "\n"
        + json.dumps(
            {
                "evaluations": evaluations,
                "stops": stops,
                "workers": 1,
                "adaptive": False,
                "sa_t0_auto": True,
                "cost": cost,
                "elapsed": 0.01,
            }
        )
    )


def test_native_result_is_scored_independently() -> None:
    """Accept an independently matching objective under the exact evaluation budget."""
    instance = GateInstance("rastrigin-100", "fake", "h", 0.0, None, sum)
    row = Screen.parse_result(output(["max_evaluations"]), instance)
    assert row["gap"] == row["normalized_gap"] == 0.5
    assert row["evaluations"] == 100_000


@pytest.mark.parametrize(
    "stdout",
    [
        output(["time_limit"]),
        output(["max_evaluations"], 99999),
        output(["max_evaluations"], cost=2),
    ],
)
def test_invalid_native_results_are_refused(stdout: str) -> None:
    """Refuse wrong stops, incomplete budgets and false objective values.

    Args:
        stdout (str): Invalid native output.
    """
    instance = GateInstance("rastrigin-100", "fake", "h", 0.0, None, sum)
    with pytest.raises(ValueError):
        Screen.parse_result(stdout, instance)


def test_existing_screen_output_is_never_overwritten(tmp_path: Path) -> None:
    """Protect an earlier experiment even when a caller reuses its output path.

    Args:
        tmp_path (Path): Isolated output directory.
    """
    destination = tmp_path / "existing.jsonl"
    destination.write_text("evidence\n")
    runner = Screen(tmp_path / "binary", ScreenInputs(tmp_path), destination)
    with pytest.raises(FileExistsError):
        runner.run()
    assert destination.read_text() == "evidence\n"


def test_screen_stops_on_the_first_invalid_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Abort instead of filling the queue with results after a native failure.

    Args:
        tmp_path (Path): Isolated binary and output directory.
        monkeypatch (pytest.MonkeyPatch): Replaces only the expensive trial boundary.
    """
    binary = tmp_path / "binary"
    binary.write_bytes(b"fake executable")
    instance = GateInstance("rastrigin-100", "fake", "h", 0.0, None, sum)
    monkeypatch.setattr(
        Screen, "cases", lambda self: iter([(instance, tmp_path / "input")]), raising=False
    )
    calls: list[bool] = []

    def invalid(*args: object, **kwargs: object) -> dict[str, object]:
        """Record an attempted trial and expose the native validation failure.

        Args:
            *args (object): Ignored trial arguments.
            **kwargs (object): Ignored trial options.

        Raises:
            RuntimeError: Always, modeling an invalid native measurement.
        """
        calls.append(True)
        raise RuntimeError("invalid native result")

    monkeypatch.setattr(Screen, "_trial", invalid, raising=False)
    runner = Screen(binary, ScreenInputs(tmp_path), tmp_path / "rows.jsonl")
    with pytest.raises(RuntimeError, match="invalid native result"):
        runner.run()
    assert calls == [True]


def test_generated_shift_hashes_are_disjoint_from_previous_cohorts(tmp_path: Path) -> None:
    """Verify actual generated bytes against all historical gate shift seeds.

    Args:
        tmp_path (Path): Isolated generated input cache.
    """
    inputs = ScreenInputs(tmp_path)
    for dimension in (100, 300):
        previous = set()
        for seed in (20261120, 20261111, 20261121):
            for index in range(100):
                generator = random.Random(f"{seed}:{dimension}:{index}")
                shifts = [round(generator.uniform(-2.0, 2.0), 3) for _ in range(dimension)]
                content = "\n".join([str(dimension), *(repr(value) for value in shifts)]) + "\n"
                previous.add(hashlib.sha256(content.encode()).hexdigest())
        current = {
            hashlib.sha256(inputs.rastrigin_path(dimension, index).read_bytes()).hexdigest()
            for index in range(20)
        }
        assert len(current) == 20
        assert not current & previous
