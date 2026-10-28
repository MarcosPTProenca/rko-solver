"""Contract tests for automatic engine selection.

Objective:
- Verify that `engine="auto"` builds the right candidates, decides noise-aware, tolerates
  failing candidates and reports its choice without changing the other engines.

Technical approach:
- Test the pure decision rule with worked numbers, then run short real solves on a small
  Knapsack through native problems and Python callbacks.

Flow (pseudocode):
1. Check candidate lists for each problem kind and the preference order.
2. Check the decision rule inside and outside the noise margin.
3. Run short auto solves and inspect the recorded selection.

Important criteria:
- Tests use short budgets and never assert timing-dependent winners.

Inputs/Outputs:
- Input: small synthetic problems and sub-second budgets.
- Output: exact assertions on the selection record.
"""

import numpy as np
import pytest
from rko_solver import RKO
from rko_solver.orchestration.auto import AutoEngine

native = pytest.importorskip("rko_solver._native")
PROFITS = [10 + (index * 37) % 91 for index in range(40)]
WEIGHTS = [5 + (index * 53) % 47 for index in range(40)]
CAPACITY = 500


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


def names(kind: str, cpp: bool = True, requested: list[str] | None = None) -> list[str]:
    """List candidate names for a problem kind.

    Args:
        kind (str): Problem kind.
        cpp (bool): Whether the C++ reference is installed.
        requested (list[str] | None): Optional restriction.

    Returns:
        list[str]: Candidate names in preference order.
    """
    return [item.name for item in AutoEngine.candidates(kind, 8, cpp, requested)]


def test_candidates_depend_on_how_the_problem_is_defined() -> None:
    """Only callbacks can use processes; the all-threads variant needs the lock to matter."""
    assert names("callbacks") == ["rust", "rust-threads", "process", "cpp"]
    assert names("batch") == ["rust", "rust-threads", "cpp"]
    assert names("native") == ["rust", "cpp"]
    assert names("native", cpp=False) == ["rust"]
    assert names("callbacks", requested=["cpp", "process"]) == ["process", "cpp"]
    assert AutoEngine.build("rust-threads", 8).n_jobs == 8
    assert AutoEngine.build("rust-threads", 1).n_jobs == 2
    with pytest.raises(ValueError, match="subset"):
        names("native", requested=["thread"])
    with pytest.raises(ValueError, match="No requested"):
        names("native", cpp=False, requested=["cpp"])


def test_decision_prefers_the_evidence_inside_the_noise_and_the_best_outside() -> None:
    """A tiny edge over the preferred candidate is noise; a large one wins."""
    prior = ["process", "rust"]
    chosen, margin, reason, means = AutoEngine.decide(
        {"process": [-100.0, -90.0], "rust": [-97.0, -99.0]}, prior
    )
    assert (chosen, margin) == ("process", 5.0)
    assert "no clear advantage" in reason and means["rust"] == -98.0
    chosen, margin, _, _ = AutoEngine.decide(
        {"rust": [-1000.0, -1000.0], "process": [-1030.0, -1030.0]}, ["rust", "process"]
    )
    assert (chosen, margin) == ("rust", 50.0)
    chosen, _, _, _ = AutoEngine.decide(
        {"rust": [-1000.0, -1000.0], "process": [-1100.0, -1100.0]}, ["rust", "process"]
    )
    assert chosen == "process"
    chosen, _, reason, _ = AutoEngine.decide(
        {"process": [-100.0, -98.0], "rust": [-200.0, -198.0]}, prior
    )
    assert (chosen, reason) == ("rust", "lowest mean probe cost")
    assert AutoEngine.decide({"process": [-1.0, -1.0], "rust": [-1.0, -1.0]}, prior)[0] == (
        "process"
    )
    with pytest.raises(ValueError, match="usable"):
        AutoEngine.decide({"process": []}, prior)


def test_native_problem_auto_runs_and_reports_the_selection() -> None:
    """A built-in problem selects among compiled-capable engines and returns the best cost."""
    problem = native.KnapsackProblem(PROFITS, WEIGHTS, CAPACITY)
    solver = RKO(problem, engine="auto", auto_time=0.4, auto_candidates=["rust"], random_state=3)
    result = solver.solve(0.4)
    assert result.engine == "rust" and result.auto is not None
    assert result.auto.chosen == "rust" and result.auto.candidates == ("rust",)
    assert len(result.auto.probes) == AutoEngine.REPEATS
    assert all(probe.error is None and probe.cost is not None for probe in result.auto.probes)
    best_probe = min(probe.cost for probe in result.auto.probes)
    assert result.cost <= best_probe
    assert result.auto.probe_seconds == pytest.approx(0.4 / AutoEngine.REPEATS)
    assert result.elapsed_seconds >= 0.4 + 0.4 - 0.05
    assert solver.result_ is result


def test_callback_auto_excludes_a_failing_candidate_and_names_the_error() -> None:
    """Unpicklable callbacks break the process candidate, which is dropped, not fatal."""
    solver = RKO(
        lambda keys: np.asarray(keys) > 0.5,
        cost,
        len(PROFITS),
        engine="auto",
        auto_time=0.6,
        auto_candidates=["process", "rust"],
        algorithms=["SA", "ILS"],
        random_state=1,
    )
    result = solver.solve(0.3)
    assert result.auto is not None and result.engine == "rust"
    failed = [probe for probe in result.auto.probes if probe.error]
    assert failed and failed[0].candidate == "process"
    assert [name for name, _ in result.auto.scores] == ["rust"]


def test_every_candidate_failing_raises_with_the_first_error() -> None:
    """If nothing can run the problem, the failure is explicit."""
    solver = RKO(
        lambda keys: np.asarray(keys) > 0.5,
        cost,
        len(PROFITS),
        engine="auto",
        auto_time=0.2,
        auto_candidates=["process"],
        algorithms=["SA"],
    )
    with pytest.raises(RuntimeError, match="every candidate"):
        solver.solve(0.2)


def test_auto_validates_its_settings_without_touching_other_engines() -> None:
    """Reject n_jobs and bad budgets for auto while process keeps its own rules."""
    problem = native.KnapsackProblem(PROFITS, WEIGHTS, CAPACITY)
    with pytest.raises(ValueError, match="n_jobs"):
        RKO(problem, engine="auto", n_jobs=2).solve(0.2)
    with pytest.raises(ValueError, match="auto_time"):
        RKO(problem, engine="auto", auto_time=0.0).solve(0.2)
    with pytest.raises(ValueError, match="engine"):
        RKO(problem, engine="automatic").solve(0.2)
    result = RKO(problem, engine="rust", random_state=1).solve(0.2)
    assert result.auto is None and result.engine == "rust"
