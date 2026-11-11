"""Regression checks for public-instance identity and decoder parity.

Objective:
- Prevent cross-instance objective reuse and discrepancies on tied keys.

Technical approach:
- Exercise the real loader with local synthetic download responses.
- Compare native and Python objectives on tied and generated key vectors.

Flow (pseudocode):
1. Load two catalog entries sharing a basename.
2. Require distinct problem identifiers.
3. Compare the native tour cost with the Python cost.

Important criteria:
- No network requests or long solver runs are needed.

Inputs/Outputs:
- Input: local fixtures and deterministically generated keys.
- Output: identity and objective-equivalence assertions.
"""

import json
import numpy as np
import pytest
from benchmarks.public.problems import TspObjective
from benchmarks.public.sources import PublicSources
from benchmarks.public.sources import Source
from benchmarks.public.study import PublicLoader
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from pathlib import Path
from types import SimpleNamespace


def test_kplib_loader_keeps_directory_in_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same basename in different KPLIB families must remain distinct."""
    sources = (
        Source("kplib/family-a/s092.kp", "unused", "unused"),
        Source("kplib/family-b/s092.kp", "unused", "unused"),
    )
    monkeypatch.setattr(PublicSources, "kplib", lambda: sources)
    monkeypatch.setattr(PublicSources, "fetch", lambda source, cache: cache / source.relative)
    monkeypatch.setattr(
        "benchmarks.public.study.read_kp_instance",
        lambda path: SimpleNamespace(benefits=[3], weights=[2], capacity=2),
    )
    monkeypatch.setattr("benchmarks.public.study.exact_profit", lambda data: 3)
    problems = PublicLoader._kplib(Path("unused"))
    assert [problem.name for problem in problems] == [source.relative for source in sources]


def test_native_tour_matches_python_with_all_keys_tied() -> None:
    """Equal keys retain city index order in both engines."""
    rng = np.random.default_rng(156)
    points = rng.integers(0, 100, size=(40, 2))
    distances = np.floor(np.linalg.norm(points[:, None] - points[None, :], axis=2) + 0.5)
    objective = TspObjective(distances)
    keys = np.zeros(40)
    assert objective.numba_function()(keys) == objective.cost(keys)


_SMALL_DISTANCES = np.array([[0, 2, 7, 4], [2, 0, 3, 8], [7, 3, 0, 5], [4, 8, 5, 0]], dtype=float)
_SMALL_TSP = TspObjective(_SMALL_DISTANCES)
_NATIVE_TSP = _SMALL_TSP.numba_function()


@settings(max_examples=100, derandomize=True, deadline=None)
@given(st.lists(st.integers(0, 3), min_size=4, max_size=4))
def test_native_tour_matches_python_for_generated_ties(values: list[int]) -> None:
    """Every generated key vector has the same objective in both engines."""
    keys = np.asarray(values, dtype=float) / 3
    assert _NATIVE_TSP(keys) == _SMALL_TSP.cost(keys)


def test_run_prepares_one_native_objective_before_solving(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each instance gets one warmed objective, never a post-solve replacement."""
    from benchmarks.public import study
    from benchmarks.public.problems import PublicProblem

    events: list[str] = []
    objective = TspObjective(_SMALL_DISTANCES)

    def cost(keys: np.ndarray) -> float:
        """Record the warm-up call and return the reference cost."""
        events.append("warm")
        return objective.cost(keys)

    def compile_cost() -> object:
        """Record one objective construction."""
        events.append("compile")
        return cost

    def solve(*args: object) -> dict[str, object]:
        """Require warm-up before returning a minimal recorded row."""
        assert events == ["compile", "warm"]
        return {"instance": "tiny"}

    monkeypatch.setattr(objective, "numba_function", compile_cost)
    monkeypatch.setattr(study.PublicStudy, "solve", solve)
    monkeypatch.setattr(study, "ROW_FIELDS", ("instance",))
    problems = [PublicProblem("tiny", "tsplib", objective, None)]
    study._run(problems, range(1), (1.0,), tmp_path / "rows.jsonl")
    assert events == ["compile", "warm"]


def test_resume_rejects_foreign_run_metadata(tmp_path: Path) -> None:
    """Rows from another revision must not silently suppress current runs."""
    from benchmarks.public.study import PublicStudy

    row = {
        "instance": "tiny",
        "arm": "rust-default",
        "seed": 0,
        "seconds": 1.0,
        "host": "foreign-host",
        "commit": "foreign-commit",
        "benchmark_sha256": "foreign-code",
        "schema_version": 1,
    }
    output = tmp_path / "rows.jsonl"
    output.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        PublicStudy.existing(output)


def test_infeasible_solution_has_no_reference_gap() -> None:
    """An over-capacity profit is not a meaningful reference comparison."""
    from benchmarks.public.problems import KnapsackObjective
    from benchmarks.public.problems import PublicProblem

    objective = KnapsackObjective(np.array([10.0]), np.array([[2.0]]), np.array([1.0]))
    problem = PublicProblem("tiny", "kplib", objective, 0.0)
    checked = problem.check(np.array([1.0]))
    assert not checked.feasible
    assert checked.gap is None


def test_solve_rejects_disagreement_between_engine_and_checker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A solver score for another tour must fail rather than enter the report."""
    from benchmarks.public import study
    from benchmarks.public.problems import PublicProblem

    result = SimpleNamespace(keys=np.zeros(4), cost=-1.0, n_evaluations=1, stop_reason="time_limit")
    monkeypatch.setattr(
        study, "RKO", lambda *args, **kwargs: SimpleNamespace(solve=lambda s: result)
    )
    problem = PublicProblem("tiny", "tsplib", _SMALL_TSP, None)
    with pytest.raises(ValueError, match="cost"):
        study.PublicStudy.solve(problem, "python-process-default", 0, 1.0, None)


def test_report_rejects_mixed_provenance() -> None:
    """Different machines cannot silently contribute to one family median."""
    from benchmarks.public.report import summarize

    base = {
        "family": "tsplib",
        "arm": "rust-default",
        "seconds": 2.0,
        "feasible": True,
        "gap": 0.1,
        "evaluations": 100,
        "wall_seconds": 2.0,
    }
    rows = [{**base, "host": "first"}, {**base, "host": "second"}]
    with pytest.raises(ValueError, match="provenance"):
        summarize(rows)
