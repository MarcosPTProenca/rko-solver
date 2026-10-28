"""Knapsack benchmark contract tests.

Objective:
- Check instance validation and spawn-compatible problem callbacks.

Technical approach:
- Exercise public example helpers with small independently solved instances.

Flow (pseudocode):
1. Parse valid and malformed instances.
2. Serialize the objective and evaluate known solutions.
3. Check that invalid solution dimensions are rejected.

Important criteria:
- Serialization must preserve objective values and penalties.

Inputs/Outputs:
- Input: temporary instance files and binary decisions.
- Output: assertions on objective values and validation errors.
"""

import pickle
import pytest
from examples.run_kp_cpp_instances import decode_knapsack
from examples.run_kp_cpp_instances import default_instances_dir
from examples.run_kp_cpp_instances import default_output_csv
from examples.run_kp_cpp_instances import evaluate_solution
from examples.run_kp_cpp_instances import make_cost
from examples.run_kp_cpp_instances import read_kp_instance
from examples.run_kp_cpp_instances import run_single_instance
from pathlib import Path


def test_default_instances_live_in_the_examples_directory() -> None:
    """Keep the benchmark runnable from a checkout without a separately cloned C++ directory."""
    directory = default_instances_dir()
    assert directory == Path(__file__).resolve().parents[2] / "examples" / "instances"
    assert read_kp_instance(directory / "kp10.txt").n_items == 10
    assert read_kp_instance(directory / "kp50.txt").n_items == 50


@pytest.mark.parametrize("filename, optimum", [("kp10.txt", 31), ("kp50.txt", 149)])
def test_bundled_instances_match_exact_reference(filename: str, optimum: int) -> None:
    """Verify reference objectives independently of stochastic solver behavior.

    Args:
        filename (str): Bundled reference instance filename.
        optimum (int): Known optimal profit for the instance.
    """
    instance = read_kp_instance(default_instances_dir() / filename)
    values = [0] * (instance.capacity + 1)
    for profit, weight in zip(instance.benefits, instance.weights, strict=True):
        for capacity in range(instance.capacity, weight - 1, -1):
            values[capacity] = max(values[capacity], values[capacity - weight] + profit)
    assert values[-1] == optimum


def test_default_output_uses_the_callers_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Avoid writing benchmark results into an installed package directory.

    Args:
        tmp_path (Path): Caller-selected working directory.
        monkeypatch (pytest.MonkeyPatch): Change the working directory temporarily.
    """
    monkeypatch.chdir(tmp_path)
    assert default_output_csv() == tmp_path / "results" / "kp_results.csv"


def test_objective_survives_pickle_round_trip(tmp_path: Path) -> None:
    """Preserve known feasible and penalized objectives after serialization.

    Args:
        tmp_path (Path): Temporary directory for the benchmark fixture.
    """
    path = tmp_path / "tiny.txt"
    path.write_text("3 5\n6 1\n10 2\n12 3\n", encoding="utf-8")
    instance = read_kp_instance(path)
    objective = pickle.loads(pickle.dumps(make_cost(instance)))
    assert objective([0, 1, 1]) == -22.0
    assert objective([1, 1, 1]) == 99972.0
    assert evaluate_solution([0, 1, 1], instance) == (22, 5, True)
    assert decode_knapsack([0.5, 0.6, 0.9]) == [0, 1, 1]


@pytest.mark.parametrize("text", ["0 5", "1 -1 6 1", "1 5 6 0", "1 5 -6 1", "1 5 6 1 99"])
def test_reader_rejects_malformed_instances(tmp_path: Path, text: str) -> None:
    """Reject invalid headers, item values and unexpected trailing data.

    Args:
        tmp_path (Path): Temporary directory for the benchmark fixture.
        text (str): Malformed input contents.
    """
    path = tmp_path / "invalid.txt"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        read_kp_instance(path)


@pytest.mark.parametrize("solution", [[1], [0, 1, 2], [0, 1, 1, 0]])
def test_objective_rejects_invalid_decisions(tmp_path: Path, solution: list[int]) -> None:
    """Reject truncated, nonbinary and oversized solution vectors.

    Args:
        tmp_path (Path): Temporary directory for the benchmark fixture.
        solution (list[int]): Invalid binary decisions.
    """
    path = tmp_path / "tiny.txt"
    path.write_text("3 5 6 1 10 2 12 3", encoding="utf-8")
    with pytest.raises(ValueError):
        make_cost(read_kp_instance(path))(solution)


def test_rust_engine_reaches_the_known_optimum_with_independent_scoring() -> None:
    """Solve a bundled instance in the compiled engine and re-score it in Python."""
    pytest.importorskip("rko_solver._native")
    result = run_single_instance(
        default_instances_dir() / "kp10.txt", 0.2, None, engine="rust", random_state=3
    )
    assert result.feasible and result.profit == 31
    assert result.cost == -31.0
