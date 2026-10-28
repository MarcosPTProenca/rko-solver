"""Contracts for bounded run histories and offline interactive HTML reports.

Objective:
- Verify scalar traces, optional dependencies and safe post-run serialization.

Technical approach:
- Exercise all execution modes and deterministic callback-boundary histories.
- Inspect generated HTML/data rather than comparing renderer-specific image bytes.

Flow (pseudocode):
1. Run deterministic searches with capture enabled and disabled.
2. Check compaction bounds, counters and immutable result metadata.
3. Render offline artifacts and verify escaping, privacy and failure isolation.

Important criteria:
- Rendering never invokes problem callbacks or requires a browser.
- Plotly-specific checks skip when the optional extra is not installed.

Inputs/Outputs:
- Input: bounded searches, synthetic incumbent sequences and output paths.
- Output: assertions about metadata, JSON and HTML artifacts.
"""

import importlib.util
import json
import numpy as np
import pickle
import pytest
import subprocess
import sys
from dataclasses import replace
from html.parser import HTMLParser
from pathlib import Path
from rko_solver import RKO
from rko_solver import ValidationCheck
from rko_solver import ValidationReport
from rko_solver import assess_result
from rko_solver.components import SolutionPool
from rko_solver.explain import DecoderDiagram
from rko_solver.explain import DiagramEdge
from rko_solver.explain import DiagramNode
from rko_solver.metaheuristics import MultiStart
from rko_solver.metaheuristics.multi_start import MultiStart as MultiStartKernel
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.types import HistoryPoint
from rko_solver.reporting import OptimizationReport
from rko_solver.solver import RKOResult
from unittest.mock import Mock

HAS_PLOTLY = importlib.util.find_spec("plotly") is not None


def test_optional_diagram_is_snapshotted_without_callback_replay() -> None:
    """Export optional static structure without changing a completed numerical result."""
    result = RKOResult((0.1,), 1.0, 0.0, "python", ("SA",), None, execution="sequential")
    diagram = DecoderDiagram.from_source("def decode(keys):\n    return keys\n")
    report = OptimizationReport(result, decoder_diagram=diagram)
    assert report.to_dict()["decoder_diagram"] == diagram.to_dict()
    data = report.to_dict()
    data["decoder_diagram"]["nodes"][0]["label"] = "changed"
    assert report.to_dict()["decoder_diagram"]["nodes"][0]["label"] == "Entry"
    assert result.cost == 1.0
    assert OptimizationReport(result).to_dict()["decoder_diagram"] is None
    with pytest.raises(ValueError, match="diagram"):
        OptimizationReport(result, decoder_diagram={"nodes": []})


@pytest.mark.skipif(not HAS_PLOTLY, reason="Install the optional reports extra")
@pytest.mark.parametrize("source_derived", [False, True])
def test_diagram_html_is_offline_accessible_and_explicitly_static(
    source_derived: bool,
    tmp_path: Path,
) -> None:
    """Escape diagram content and retain provenance plus complete text alternatives.

    Args:
        source_derived (bool): Whether to derive a graph from explicitly supplied source.
        tmp_path (Path): Isolated HTML output directory.
    """
    attack = '</script><img src=x onerror="alert(1)">'
    if source_derived:
        diagram = DecoderDiagram.from_source(
            f"def decode(keys):\n    value = {attack!r}\n    return value\n",
            title=attack,
            include_expressions=True,
        )
    else:
        diagram = DecoderDiagram(
            [
                DiagramNode("start", attack, "start"),
                DiagramNode("step", "Capacity check"),
                DiagramNode("end", "Solution", "end"),
            ],
            [
                DiagramEdge("start", "step", attack),
                DiagramEdge("step", "step", "repeat"),
                DiagramEdge("step", "end", "done"),
            ],
            title=attack,
            description=attack,
        )
    result = RKOResult((0.1,), 1.0, 0.0, "python", ("SA",), None, execution="sequential")
    report = OptimizationReport(result, decoder_diagram=diagram)
    text = report.write_html(tmp_path / "diagram.html").read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(text)
    assert not parser.external_scripts
    assert attack not in text
    assert text.count("plotly.js v") == 1
    assert text.count('id="rko-chart-') == 1
    assert "not an execution trace" in text
    assert "Diagram steps (text alternative)" in text
    assert "Diagram transitions" in text
    assert "source-derived" in text if source_derived else "author-declared" in text
    assert json.loads(parser.scripts[-1]) == report.to_dict()


def decode(keys: np.ndarray) -> np.ndarray:
    """Return an identity-decoded numerical candidate.

    Args:
        keys (np.ndarray): Detached candidate.

    Returns:
        np.ndarray: Unchanged candidate.
    """
    return keys


def objective(solution: np.ndarray) -> float:
    """Compute a finite sphere objective.

    Args:
        solution (np.ndarray): Numerical vector.

    Returns:
        float: Squared norm to minimize.
    """
    return float(np.dot(solution, solution))


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
def test_history_crosses_all_execution_modes(execution: str) -> None:
    """Retain bounded immutable histories and consistent final incumbents.

    Args:
        execution (str): Execution strategy.
    """
    result = RKO(
        decode,
        objective,
        4,
        algorithms=["MultiStart", "SA"],
        adaptation="fixed",
        execution=execution,
        random_state=42,
        history_limit=8,
    ).solve(0.1)
    assert result.searches
    local_bests = []
    for search in result.searches:
        assert search.history_limit == 8
        if search.stop_reason == "not_started":
            assert not search.history
            continue
        assert 1 <= len(search.history) <= 8
        assert search.history_observations >= len(search.history)
        assert search.history[-1].n_evaluations == search.n_evaluations
        assert search.history[-1].elapsed_seconds == search.elapsed_seconds
        assert all(
            a.cost >= b.cost for a, b in zip(search.history, search.history[1:], strict=False)
        )
        assert all(
            a.elapsed_seconds <= b.elapsed_seconds
            for a, b in zip(search.history, search.history[1:], strict=False)
        )
        local_bests.append(search.history[-1].cost)
    assert result.cost == min(local_bests)


def test_compaction_preserves_first_and_terminal_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact a fixed improving sequence without performing extra objective calls.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the algorithm loop with bounded work.
    """
    algorithm = MultiStartKernel(SolutionPool(), decode, objective, 3)

    def fixed_work(limit_time: float) -> tuple[list[object], list[float], float]:
        """Evaluate exactly one hundred strictly improving vectors.

        Args:
            limit_time (float): Unused search allowance for this deterministic test.

        Returns:
            tuple[list[object], list[float], float]: Final vector and its observed objective.
        """
        keys = np.zeros(3)
        value = 0.0
        for index in range(1, 101):
            keys = np.full(3, 1.0 / index)
            value = algorithm.cost(algorithm.decoder(keys))
        return [], keys.tolist(), value

    monkeypatch.setattr(algorithm, "solve", fixed_work)
    session = SearchSession(decode, objective)
    session.solve(algorithm, 30, history_limit=5)
    report = session.metadata
    assert report is not None
    assert report.n_evaluations == 100
    assert report.history_observations == 101
    assert len(report.history) <= 5
    assert report.history[0].n_evaluations == 1
    assert report.history[-1].n_evaluations == 100
    assert report.history[-1].cost == pytest.approx(3 / 10000)
    session.solve(algorithm, 30)
    assert session.metadata is not None
    assert session.metadata.history == ()
    assert session.metadata.history_observations == 0
    assert session.metadata.n_evaluations == 100


@pytest.mark.parametrize("limit", [-1, 1, True, 2.5, 10001, "100"])
def test_invalid_history_limits_fail_before_callbacks(limit: object) -> None:
    """Reject unsafe or ambiguous public recording limits.

    Args:
        limit (object): Invalid requested cap.
    """
    cost = Mock(side_effect=AssertionError("callback must not execute"))
    with pytest.raises(ValueError, match="history_limit"):
        RKO(decode, cost, 3, history_limit=limit).solve(0.01)
    cost.assert_not_called()


def test_standalone_history_and_data_privacy(tmp_path: Path) -> None:
    """Expose history for class-based solves without exporting candidate coordinates.

    Args:
        tmp_path (Path): Isolated export directory.
    """
    solver = MultiStart(
        decode,
        objective,
        3,
        adaptation="fixed",
        execution="sequential",
        random_state=42,
        history_limit=5,
    )
    result = solver.solve(time_limit=0.01)
    assert result.searches[0].history
    report = OptimizationReport(result, metrics={"feasible": True, "profit": 10})
    target = report.write_json(tmp_path / "run.json")
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["summary"]["cost"] == result.cost
    assert data["domain_metrics"] == {"feasible": True, "profit": 10}
    assert "keys" not in data["summary"]
    assert "keys" not in data
    assert set(data["searches"][0]["history"][0]) == {"elapsed_seconds", "n_evaluations", "cost"}
    detached = report.to_dict()
    detached["summary"]["cost"] = 999
    assert report.to_dict()["summary"]["cost"] == result.cost


def test_import_and_json_do_not_load_plotly(tmp_path: Path) -> None:
    """Keep core reports data export independent of the optional rendering stack.

    Args:
        tmp_path (Path): Isolated output directory.
    """
    command = (
        "import sys; from rko_solver.reporting import OptimizationReport; "
        "from rko_solver.solver import RKOResult; "
        "r = RKOResult((0.1,), 1., 0., 'serial', ('SA',), 1); "
        f"OptimizationReport(r).write_json({str(tmp_path / 'isolated.json')!r}); "
        "assert 'plotly' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", command], check=True, timeout=30)


@pytest.mark.parametrize("metrics", [{"bad": float("nan")}, {"bad": []}, {1: "bad"}])
def test_invalid_domain_metrics_are_rejected(metrics: dict[object, object]) -> None:
    """Reject nonfinite or nonscalar metadata before embedding it.

    Args:
        metrics (dict[object, object]): Invalid caller-supplied values.
    """
    result = RKOResult((0.1,), 1.0, 0.0, "python", ("SA",), None, execution="sequential")
    with pytest.raises(ValueError):
        OptimizationReport(result, metrics=metrics)


def test_missing_optional_dependency_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explain the optional extra and preserve prior output on rendering failure.

    Args:
        monkeypatch (pytest.MonkeyPatch): Simulates an installation without Plotly.
        tmp_path (Path): Existing output location.
    """
    result = RKOResult((0.1,), 1.0, 0.0, "python", ("SA",), None, execution="sequential")
    path = tmp_path / "run.html"
    path.write_text("previous", encoding="utf-8")
    monkeypatch.setattr(
        "rko_solver.reporting.importlib.import_module", Mock(side_effect=ImportError)
    )
    with pytest.raises(ImportError, match="reports.*extra"):
        OptimizationReport(result).write_html(path)
    assert path.read_text() == "previous"
    assert result.cost == 1.0


class ReportParser(HTMLParser):
    """Collect report resources and embedded scalar data for offline/security checks.

    Responsibilities:
    - Detect external scripts and identify the report's application/json block.

    Attributes:
        external_scripts (list[str]): Script resources requiring a separate file or network.
        scripts (list[str]): Contents of all inline script blocks.
        _inside_script (bool): Whether the parser is collecting inline script text.
    """

    def __init__(self) -> None:
        """Initialize an empty resource and script-text collector."""
        super().__init__()
        self.external_scripts: list[str] = []
        self.scripts: list[str] = []
        self._inside_script = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Inspect resource attributes and start collecting inline scripts.

        Args:
            tag (str): HTML element name.
            attrs (list[tuple[str, str | None]]): Element attributes.
        """
        if tag == "script":
            source = dict(attrs).get("src")
            if source:
                self.external_scripts.append(source)
            self._inside_script = True
            self.scripts.append("")

    def handle_data(self, data: str) -> None:
        """Append text inside the current script element.

        Args:
            data (str): Parsed element text.
        """
        if self._inside_script:
            self.scripts[-1] += data

    def handle_endtag(self, tag: str) -> None:
        """Finish collecting an inline script.

        Args:
            tag (str): Closed element name.
        """
        if tag == "script":
            self._inside_script = False


@pytest.mark.skipif(not HAS_PLOTLY, reason="Install the optional reports extra")
def test_html_is_offline_escaped_and_data_consistent(tmp_path: Path) -> None:
    """Render multiple charts with one embedded bundle and escaped user metadata.

    Args:
        tmp_path (Path): Isolated HTML output directory.
    """
    solver = MultiStart(
        decode, objective, 3, adaptation="fixed", execution="sequential", history_limit=5
    )
    result = solver.solve(time_limit=0.01)
    attack = '</script><script>alert("unsafe")</script>'
    report = OptimizationReport(result, title=attack, metrics={attack: attack})
    target = report.write_html(tmp_path / "offline.html")
    text = target.read_text(encoding="utf-8")
    parser = ReportParser()
    parser.feed(text)
    assert parser.external_scripts == []
    assert attack not in text
    assert "&lt;/script&gt;" in text
    assert text.count("plotly.js v") == 1
    assert text.count('id="rko-chart-') == 4
    assert json.loads(parser.scripts[-1]) == report.to_dict()
    assert "not a synchronized portfolio timeline" in text
    assert "not independently verified" in text
    assert "No exact time-to-target" in text


@pytest.mark.skipif(not HAS_PLOTLY, reason="Install the optional reports extra")
def test_html_without_history_is_honest(tmp_path: Path) -> None:
    """Keep missing convergence data unavailable instead of inventing observations.

    Args:
        tmp_path (Path): Output directory.
    """
    solver = MultiStart(decode, objective, 3, adaptation="fixed", execution="sequential")
    result = solver.solve(time_limit=0.01)
    report = OptimizationReport(result)
    text = report.write_html(tmp_path / "summary.html").read_text(encoding="utf-8")
    assert "Histories unavailable for: MultiStart" in text
    assert text.count('id="rko-chart-') == 2
    assert report.to_dict()["searches"][0]["history"] == []


def test_atomic_output_failure_preserves_existing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preserve prior output and clean temporary files when rename fails.

    Args:
        monkeypatch (pytest.MonkeyPatch): Forces an output replacement error.
        tmp_path (Path): Existing output directory.
    """
    path = tmp_path / "run.json"
    path.write_text("previous", encoding="utf-8")
    report = OptimizationReport(
        RKOResult((0.1,), 1.0, 0.0, "python", ("SA",), None, execution="sequential")
    )
    monkeypatch.setattr("rko_solver.reporting.os.replace", Mock(side_effect=OSError("read-only")))
    with pytest.raises(OSError, match="read-only"):
        report.write_json(path)
    assert path.read_text() == "previous"
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(ValueError, match="PDF"):
        report.write_html(tmp_path / "run.pdf")


def test_history_metadata_remains_pickleable() -> None:
    """Keep bounded history compatible with normal immutable result serialization."""
    solver = MultiStart(
        decode, objective, 3, adaptation="fixed", execution="sequential", history_limit=5
    )
    result = solver.solve(time_limit=0.01)
    copied = pickle.loads(pickle.dumps(result))
    assert copied == result
    legacy = replace(result, searches=())
    assert OptimizationReport(legacy).to_dict()["searches"] == []
    assert HistoryPoint(0.1, 1, 2.0).cost == 2.0


@pytest.mark.skipif(not HAS_PLOTLY, reason="Install the optional reports extra")
def test_assessment_report_escapes_diagnostics_without_callbacks(tmp_path: Path) -> None:
    """Expose failed checks truthfully without serializing payloads or re-running callbacks.

    Args:
        tmp_path (Path): Isolated report directory.
    """
    solver = MultiStart(decode, objective, 3, adaptation="fixed", execution="sequential")
    result = solver.solve(time_limit=0.01)
    attack = '</script><script>alert("unsafe")</script>'
    payload = object()
    decoder = Mock(return_value=payload)
    validator = Mock(
        return_value=ValidationReport(
            checks=(ValidationCheck(attack, "failed", message=attack, value=8, limit=5),),
            feasible=False,
            checker=attack,
        )
    )
    assessment = assess_result(result, decoder=decoder, validator=validator)
    report = OptimizationReport(result, assessment=assessment)
    html = report.write_html(tmp_path / "checked.html").read_text(encoding="utf-8")
    assert attack not in html
    assert "&lt;/script&gt;" in html
    assert "Optional solution validation" in html
    assert "post_run_decode" in html
    assert "constraint coverage or global optimality" in html
    parser = ReportParser()
    parser.feed(html)
    assert parser.external_scripts == []
    embedded = json.loads(parser.scripts[-1])
    assert embedded["assessment"] == assessment.to_dict()
    assert embedded["assessment"]["validation"]["status"] == "failed"
    assert embedded["assessment"]["validation"]["feasible"] is False
    assert "solution" not in embedded["assessment"]
    decoder.assert_called_once()
    validator.assert_called_once_with(payload)
