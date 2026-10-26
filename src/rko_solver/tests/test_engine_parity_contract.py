"""Parameter and metadata parity contract across the Python, Rust and C++ engines.

Objective:
- Pin that the same TOML files, overrides and validation rules produce the same configuration,
  report vocabulary and errors in every engine, the first slice of the engine parity contract.

Technical approach:
- Run each registered algorithm on every available engine with a small `max_evaluations` and
  compare the reported parameter snapshots, adaptive grids, controls and stop vocabulary.
- Overrides are taken from each algorithm's own online grid, which guarantees valid values, and
  custom TOML directories are copies of the packaged ones with one edited value.

Flow (pseudocode):
1. Build the overrides or the edited TOML directory for an algorithm.
2. Solve with each engine and collect the per-search metadata.
3. Assert that every engine reports exactly what the Python engine reports.

Important criteria:
- Parameters are snapshotted in Python for every engine, so equality also proves that the
  compiled engines receive overrides and custom directories instead of silently using defaults.
- Algorithms that depend on engine-specific behavior are not compared here, only configuration.

Inputs/Outputs:
- Input: the packaged parameter files and a 60-item Knapsack.
- Output: exact equality of metadata and error messages between engines.
"""

import numpy as np
import pytest
import shutil
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.parameters.loader import default_params_dir
from rko_solver.parameters.loader import load_offline
from rko_solver.parameters.loader import load_online_named

native = pytest.importorskip("rko_solver._native")
PROFITS = [10 + (index * 37) % 91 for index in range(60)]
WEIGHTS = [5 + (index * 53) % 47 for index in range(60)]
VOCABULARY = {"completed", "time_limit", "max_evaluations", "cancelled", "not_started"}
ENGINES = {
    "python": {"engine": "python", "execution": "sequential"},
    "rust": {"engine": "rust", "n_jobs": 1},
    "cpp": {"engine": "cpp"},
}
PARAMETRIZED = [name for name in MH_SPECS if load_offline(name)]


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
    return float(100000 * max(0, weight - 700) - profit)


def available() -> list[str]:
    """List the engines present in this build.

    Returns:
        list[str]: Engine keys, without `cpp` when it was not compiled.
    """
    return [name for name in ENGINES if name != "cpp" or native.cpp_available()]


def solve(engine: str, algorithm: str, **options: object) -> object:
    """Run one algorithm on one engine with a small evaluation limit.

    Args:
        engine (str): Key of `ENGINES`.
        algorithm (str): Registered algorithm name.
        **options (object): Extra `RKO` settings.

    Returns:
        object: The `RKOResult`.
    """
    return RKO(
        Callbacks(decode, cost, len(PROFITS)),
        algorithms=[algorithm],
        max_evaluations=60,
        random_state=3,
        **ENGINES[engine],
        **options,
    ).solve(60.0)


def overrides(name: str) -> dict[str, float]:
    """Choose a valid non-default value for every offline parameter of an algorithm.

    Args:
        name (str): Registered algorithm name.

    Returns:
        dict[str, float]: Parameter name to a value taken from its online grid.
    """
    defaults, grid = load_offline(name), load_online_named(name)
    chosen: dict[str, float] = {}
    for key, default in defaults.items():
        value = next(item for item in grid[key] if item != default)
        chosen[key] = type(default)(value)
    return chosen


@pytest.mark.parametrize("algorithm", PARAMETRIZED)
def test_overrides_reach_every_engine_with_the_same_snapshot(algorithm: str) -> None:
    """Hyperparameter overrides are reported identically by every engine.

    Args:
        algorithm (str): Algorithm with at least one offline parameter.
    """
    wanted = overrides(algorithm)
    snapshots = {
        engine: dict(
            solve(
                engine,
                algorithm,
                adaptation="fixed",
                hyperparameters={algorithm: wanted},
            )
            .searches[0]
            .parameters
        )
        for engine in available()
    }
    for engine, snapshot in snapshots.items():
        assert {key: snapshot[key] for key in wanted} == wanted, engine
    assert len({tuple(sorted(item.items())) for item in snapshots.values()}) == 1


@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("algorithm", PARAMETRIZED)
def test_overrides_change_the_seeded_trajectory_not_only_the_report(
    engine: str, algorithm: str
) -> None:
    """A seeded run with overridden parameters differs from the default run.

    The report snapshot is built in Python, so only a changed trajectory proves that the engine
    consumed the values. ILS needs a large budget because its first descent takes thousands of
    evaluations before any perturbation uses `beta_min` and `beta_max`. The C++ engine is not
    compared because its runs are not repeatable even with a seed.

    Args:
        engine (str): Seeded engine under test.
        algorithm (str): Algorithm with at least one offline parameter.
    """
    budget = 60000 if algorithm == "ILS" else 400
    options = {"adaptation": "fixed", "max_evaluations": budget}

    def run(**extra: object) -> tuple[float, tuple[float, ...]]:
        """Run the seeded case and return its cost and keys.

        Args:
            **extra (object): Additional `RKO` settings.

        Returns:
            tuple[float, tuple[float, ...]]: Final cost and key vector.
        """
        result = RKO(
            Callbacks(decode, cost, len(PROFITS)),
            algorithms=[algorithm],
            random_state=3,
            **ENGINES[engine],
            **{**options, **extra},
        ).solve(120.0)
        return result.cost, result.keys

    assert run() != run(hyperparameters={algorithm: overrides(algorithm)})


def test_a_custom_parameter_directory_reaches_every_engine(tmp_path: Path) -> None:
    """Edited TOML defaults are honored by every engine, not only by the Python one.

    Args:
        tmp_path (Path): Directory for the edited copy of the packaged TOML files.
    """
    shutil.copytree(default_params_dir(), tmp_path / "params")
    sa = tmp_path / "params" / "sa.toml"
    sa.write_text(sa.read_text(encoding="utf-8").replace("sa_max   = 1000", "sa_max   = 77"))
    seen = {
        engine: dict(
            solve(engine, "SA", adaptation="fixed", params_dir=tmp_path / "params")
            .searches[0]
            .parameters
        )["sa_max"]
        for engine in available()
    }
    assert seen == dict.fromkeys(available(), 77)


@pytest.mark.parametrize("algorithm", PARAMETRIZED)
def test_adaptive_grids_and_controls_match_across_engines(algorithm: str) -> None:
    """Online grids and the effective control are the same on every engine.

    Args:
        algorithm (str): Algorithm with at least one parameter axis.
    """
    reports = {
        engine: solve(engine, algorithm, adaptation="adaptive").searches[0]
        for engine in available()
    }
    reference = reports["python"]
    for engine, report in reports.items():
        assert report.control == reference.control, engine
        assert report.parameter_grid == reference.parameter_grid, engine
        assert report.parameters == reference.parameters, engine


@pytest.mark.parametrize(
    "bad",
    [
        {"SA": {"sa_max": -1}},
        {"SA": {"unknown_key": 1}},
        {"NotAnAlgorithm": {"x": 1}},
    ],
)
def test_invalid_overrides_fail_with_the_same_error_everywhere(bad: dict[str, object]) -> None:
    """Validation happens before dispatch, so every engine raises the same error.

    Args:
        bad (dict[str, object]): Invalid hyperparameter mapping.
    """
    errors = set()
    for engine in available():
        with pytest.raises(Exception) as caught:
            solve(engine, "SA", adaptation="fixed", hyperparameters=bad)
        errors.add((type(caught.value).__name__, str(caught.value)))
    assert len(errors) == 1


@pytest.mark.parametrize("engine", available())
def test_every_engine_reports_the_same_metadata_fields(engine: str) -> None:
    """Search reports use one vocabulary and carry the fields the contract names.

    Args:
        engine (str): Engine under test.
    """
    result = solve(engine, "SA", adaptation="fixed", history_limit=0)
    report = result.searches[0]
    assert report.algorithm == "SA" and report.control == 0
    assert report.stop_reason in VOCABULARY and report.stop_reason == "max_evaluations"
    assert report.n_evaluations == 60 and report.random_state is not None
    assert report.time_limit > 0 and report.elapsed_seconds >= 0
    assert report.history == () and report.history_limit == 0
    assert result.stop_reason in VOCABULARY and result.n_workers >= 1
    assert result.engine == engine and result.n_variables == len(PROFITS)
    assert [name for name, _ in result.versions][:1] == ["rko-solver"]
