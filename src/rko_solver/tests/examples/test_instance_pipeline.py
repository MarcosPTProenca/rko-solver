"""File-backed problem adapter and parent-only instance loading contracts.

Objective:
- Verify a complete reader-to-solution pipeline without changing solver behavior.

Technical approach:
- Use small JSON fixtures, fixed priority keys and actual managed execution modes.
- Delete the input after reading to detect accidental search-time file access.

Flow (pseudocode):
1. Parse valid/invalid instance data before constructing search workers.
2. Check deterministic decoding/objective use of loaded processing times/due dates.
3. Run guarded example entry points and standalone callbacks with owned instance data.

Important criteria:
- Tiny budgets exercise usability, not optimization-quality or performance claims.
- Parsing happens once in the caller; spawned callbacks carry only immutable data.

Inputs/Outputs:
- Input: small scheduling JSON files and bounded solve settings.
- Output: assertions about schema errors, adapter state, output and callback isolation.
"""

import json
import numpy as np
import pickle
import pytest
import subprocess
import sys
from pathlib import Path
from rko_solver.examples import instance_pipeline
from rko_solver.examples.instance_pipeline import SchedulingInstance
from rko_solver.examples.instance_pipeline import SchedulingProblem
from rko_solver.examples.instance_pipeline import read_data
from rko_solver.metaheuristics import SA
from unittest.mock import Mock


@pytest.fixture
def instance_file(tmp_path: Path) -> Path:
    """Create a complete isolated scheduling instance.

    Args:
        tmp_path (Path): Test input directory.

    Returns:
        Path: UTF-8 JSON instance file.
    """
    path = tmp_path / "instance.json"
    path.write_text(
        json.dumps(
            {
                "name": "fixture",
                "processing_times": [3, 2, 5, 1],
                "due_dates": [4, 6, 7, 3],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_decoder_and_cost_use_loaded_data(instance_file: Path) -> None:
    """Demonstrate data-dependent construction and scoring from one snapshot.

    Args:
        instance_file (Path): Scheduling JSON fixture.
    """
    problem = SchedulingProblem(read_data(instance_file))
    solution = problem.decode(np.array([0.2, 0.9, 0.7, 0.1]))
    assert problem.n_variables == 4
    assert solution == ((3, 1), (0, 4), (2, 9), (1, 11))
    assert problem.cost(solution) == 7.0
    assert problem.decode(np.full(4, 0.5)) == ((0, 3), (1, 5), (2, 10), (3, 11))
    different_times = SchedulingProblem(
        SchedulingInstance("short jobs", (1, 1, 1, 1), (4, 6, 7, 3))
    )
    assert different_times.decode(np.array([0.2, 0.9, 0.7, 0.1])) != solution
    later_dates = SchedulingProblem(SchedulingInstance("later dates", (3, 2, 5, 1), (100,) * 4))
    assert later_dates.cost(solution) == 0.0
    restored = pickle.loads(pickle.dumps(problem))
    assert restored.instance == problem.instance
    assert restored.decode(np.array([0.2, 0.9, 0.7, 0.1])) == solution


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        {},
        {"name": "bad"},
        {"name": "", "processing_times": [1], "due_dates": [1]},
        {"name": "bad", "processing_times": [], "due_dates": []},
        {"name": "bad", "processing_times": [1, 2], "due_dates": [1]},
        {"name": "bad", "processing_times": [True], "due_dates": [1]},
        {"name": "bad", "processing_times": [0], "due_dates": [1]},
        {"name": "bad", "processing_times": [1.5], "due_dates": [1]},
        {"name": "bad", "processing_times": [1], "due_dates": [-1]},
        {"name": "bad", "processing_times": [1], "due_dates": [True]},
        {"name": "bad", "processing_times": "1", "due_dates": [1]},
        {"name": "bad", "processing_times": [1], "due_dates": [1], "extra": 1},
    ],
)
def test_bad_input_fails_before_solver_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid: object,
) -> None:
    """Reject malformed schemas without allocating a solver or worker.

    Args:
        tmp_path (Path): Isolated input directory.
        monkeypatch (pytest.MonkeyPatch): Detects unwanted solver construction.
        invalid (object): Invalid JSON-compatible instance.
    """
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(invalid), encoding="utf-8")
    solver = Mock(side_effect=AssertionError("must not construct solver"))
    monkeypatch.setattr(instance_pipeline, "RKO", solver)
    with pytest.raises(ValueError):
        instance_pipeline.main([str(path)])
    solver.assert_not_called()


def test_reader_surfaces_file_and_json_errors(tmp_path: Path) -> None:
    """Propagate missing files and invalid JSON distinctly from solve failures.

    Args:
        tmp_path (Path): Isolated instance directory.
    """
    with pytest.raises(OSError):
        read_data(tmp_path / "absent.json")
    path = tmp_path / "bad.json"
    path.write_text("not JSON", encoding="utf-8")
    with pytest.raises(ValueError):
        read_data(path)


@pytest.mark.parametrize("keys", [[], [0.1], [[0.1] * 4], [0.1, 0.2, 0.3, float("nan")]])
def test_decoder_rejects_invalid_dimensions_and_values(instance_file: Path, keys: object) -> None:
    """Check the adapter's own numerical boundary without introducing a domain checker.

    Args:
        instance_file (Path): Valid input file.
        keys (object): Invalid priority vector.
    """
    problem = SchedulingProblem(read_data(instance_file))
    with pytest.raises(ValueError, match="one finite key"):
        problem.decode(keys)


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_reader_runs_once_outside_all_callback_loops(
    instance_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    execution: str,
) -> None:
    """Run a real portfolio after deleting the source file in the parent reader.

    Args:
        instance_file (Path): Disposable input file.
        monkeypatch (pytest.MonkeyPatch): Installs a parent-only reader spy.
        capsys (pytest.CaptureFixture[str]): Captures the example's result summary.
        execution (str): Managed execution mode.
    """
    original = instance_pipeline.read_data

    def load_once(path: Path) -> SchedulingInstance:
        """Load and remove the input so callbacks cannot reopen it.

        Args:
            path (Path): Instance file to read once.

        Returns:
            SchedulingInstance: Detached immutable data.
        """
        instance = original(path)
        path.unlink()
        return instance

    reader = Mock(side_effect=load_once)
    monkeypatch.setattr(instance_pipeline, "read_data", reader)
    arguments = [
        str(instance_file),
        "--execution",
        execution,
        "--algorithms",
        "SA",
        "PSO",
        "--time-limit",
        "0.2",
    ]
    if execution == "process":
        arguments.extend(["--n-jobs", "2"])
    instance_pipeline.main(arguments)
    reader.assert_called_once_with(instance_file)
    output = capsys.readouterr().out
    assert "Instance: fixture" in output
    assert "Total tardiness:" in output
    assert "Schedule (job index, completion time):" in output
    assert not instance_file.exists()


def test_same_adapter_works_with_standalone_algorithm(instance_file: Path) -> None:
    """Use the class method directly without creating a pool or new adapter interface.

    Args:
        instance_file (Path): Valid scheduling fixture.
    """
    problem = SchedulingProblem(read_data(instance_file))
    solver = SA(
        problem.decode,
        problem.cost,
        problem.n_variables,
        adaptation="fixed",
        execution="sequential",
        random_state=42,
    )
    result = solver.solve(time_limit=0.02)
    assert result.cost == problem.cost(problem.decode(np.asarray(result.keys)))
    assert result.algorithms == ("SA",)


def test_documented_file_to_solution_script_runs(tmp_path: Path) -> None:
    """Run the exact tutorial code beside its documented input outside the checkout.

    Args:
        tmp_path (Path): Isolated application directory.
    """
    root = Path(__file__).resolve().parents[4]
    text = (root / "docs" / "problem-adapters.md").read_text(encoding="utf-8")
    data = text.split("```json\n", 1)[1].split("```", 1)[0]
    code = text.split("```python\n", 1)[1].split("```", 1)[0]
    (tmp_path / "my_instance.json").write_text(data, encoding="utf-8")
    script = tmp_path / "solve_schedule.py"
    script.write_text(code, encoding="utf-8")
    process = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    assert "Schedule (job index, completion time):" in process.stdout
    assert "Total tardiness:" in process.stdout
