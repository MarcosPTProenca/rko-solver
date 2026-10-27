"""Contract tests for the mutation score report.

Objective:
- Verify parsing of both tools' result formats, the score definition and the floor rule.

Technical approach:
- Feed the report small result texts shaped exactly like `mutmut results --all true` and
  `cargo-mutants` `outcomes.json`, then call the entry point with temporary files.

Flow (pseudocode):
1. Parse synthetic results and check the per-module counts.
2. Check scores, derived floors and violations.
3. Run the command line to write a baseline, pass against it and fail below it.

Important criteria:
- Timeouts count as killed; unviable and untested mutants never count.
- A module missing from the results fails.

Inputs/Outputs:
- Input: synthetic mutation results and temporary baselines.
- Output: exact counts, scores, messages and exit statuses.
"""

import json
import pytest
from pathlib import Path
from tools.mutation_report import MutationReport

MUTMUT = """\
    rko_solver.solver.xǁRKOǁ__init____mutmut_1: survived
    rko_solver.solver.xǁRKOǁ__init____mutmut_2: killed
    rko_solver.solver.xǁRKOǁsolve__mutmut_3: timeout
    rko_solver.solver.xǁRKOǁsolve__mutmut_4: no tests
    rko_solver.components.solution_pool.xǁSolutionPoolǁinsert__mutmut_1: killed
    rko_solver.components.solution_pool.xǁSolutionPoolǁinsert__mutmut_2: skipped
"""


def outcome(name: str, summary: str) -> dict[str, object]:
    """Build one `cargo-mutants` outcome entry.

    Args:
        name (str): Source file of the mutant.
        summary (str): Outcome summary such as `CaughtMutant`.

    Returns:
        dict[str, object]: Entry shaped like those in `outcomes.json`.
    """
    return {"scenario": {"Mutant": {"file": name}}, "summary": summary}


CARGO = {
    "outcomes": [
        {"scenario": "Baseline", "summary": "Success"},
        outcome("a.rs", "CaughtMutant"),
        outcome("a.rs", "Timeout"),
        outcome("a.rs", "MissedMutant"),
        outcome("a.rs", "Unviable"),
        outcome("b.rs", "MissedMutant"),
    ]
}


def test_mutmut_lines_are_grouped_by_module_and_ignore_untested_mutants() -> None:
    """Killed and timed out count as killed, survived as survived, everything else is ignored."""
    counts = MutationReport.from_mutmut(MUTMUT)
    assert dict(counts["rko_solver.solver"]) == {"survived": 1, "killed": 2}
    assert dict(counts["rko_solver.components.solution_pool"]) == {"killed": 1}


def test_cargo_outcomes_are_grouped_by_file_and_skip_baseline_and_unviable() -> None:
    """Caught and timed-out mutants are killed; the baseline and unviable ones are not counted."""
    counts = MutationReport.from_cargo(CARGO)
    assert dict(counts["a.rs"]) == {"killed": 2, "survived": 1}
    assert dict(counts["b.rs"]) == {"survived": 1}


def test_score_floors_and_failures_follow_the_definition() -> None:
    """The score is killed over viable, floors sit five points lower and missing modules fail."""
    sections = {
        "python": MutationReport.from_mutmut(MUTMUT),
        "rust": MutationReport.from_cargo(CARGO),
    }
    assert MutationReport.score(sections["rust"]["a.rs"]) == pytest.approx(66.666, abs=0.01)
    assert MutationReport.score(sections["rust"]["b.rs"]) == 0.0
    assert MutationReport.score(MutationReport.from_mutmut("")["x"]) == 100.0
    floors = MutationReport.floors(sections)
    assert floors["python"]["rko_solver.solver"] == 61
    assert floors["rust"]["a.rs"] == 61 and floors["rust"]["b.rs"] == 0
    assert MutationReport.failures(sections, floors) == []
    stricter = {"python": {"rko_solver.solver": 90, "gone": 10}, "rust": {"a.rs": 61}}
    messages = MutationReport.failures(sections, stricter)
    assert any("rko_solver.solver: score 66.7% is below its floor of 90%" in m for m in messages)
    assert "python gone: missing from the results" in messages
    table = MutationReport.table(sections)
    assert "| python | rko_solver.solver | 2 | 1 | 66.7% |" in table and "**total**" in table


def test_command_line_writes_a_baseline_then_passes_and_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Write floors from results, pass against them, then fail when a score drops.

    Args:
        tmp_path (Path): Directory for the result and baseline files.
        capsys (pytest.CaptureFixture[str]): Captured output.
    """
    results, outcomes, baseline = tmp_path / "m.txt", tmp_path / "o.json", tmp_path / "b.json"
    results.write_text(MUTMUT, encoding="utf-8")
    outcomes.write_text(json.dumps(CARGO), encoding="utf-8")
    arguments = ["--mutmut-results", str(results), "--cargo-outcomes", str(outcomes)]
    arguments += ["--baseline", str(baseline)]
    assert MutationReport.main([*arguments, "--write-baseline"]) == 0
    assert json.loads(baseline.read_text())["python"]["rko_solver.solver"] == 61
    assert MutationReport.main(arguments) == 0
    worse = MUTMUT.replace("killed", "survived")
    results.write_text(worse, encoding="utf-8")
    capsys.readouterr()
    assert MutationReport.main(arguments) == 1
    assert "FLOOR python rko_solver.solver" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        MutationReport.main([])
