"""Statistical and format contracts of the non-inferiority gate.

Objective:
- Prove the verdict rule, the minimum instance count, the split disjointness and the JSONL
  fields with synthetic data, without running a solve.

Technical approach:
- Build synthetic trial rows with known differences and call `GateReport.analyze`.

Flow (pseudocode):
1. Build rows for a clearly equal, a clearly worse and an undersized cell.
2. Check verdicts, Holm adjustment and the refusal without a native hash.

Important criteria:
- A small sample is `inconclusive`, never `pass`.

Inputs/Outputs:
- Input: synthetic dictionaries.
- Output: assertions.
"""

import json
import pytest
import random
import tomllib
from benchmarks.engines import gate
from benchmarks.engines import gate_instances
from benchmarks.engines import gate_support
from benchmarks.engines.gate import ConfigOverrides
from benchmarks.engines.gate import Gate
from benchmarks.engines.gate import GateConfig
from benchmarks.engines.gate import GateInstance
from benchmarks.engines.gate import GateInstances
from benchmarks.engines.gate import GateReport
from benchmarks.engines.gate import SideSpec
from benchmarks.engines.gate import TrialSpec
from benchmarks.engines.gate import parse_budgets
from pathlib import Path
from typing import Any


def rows(family: str, count: int, offset: float) -> list[dict[str, Any]]:
    """Build paired rows whose candidate gap is the baseline gap plus an offset.

    Args:
        family (str): Family label.
        count (int): Instances.
        offset (float): Candidate minus baseline gap.

    Returns:
        list[dict[str, Any]]: Candidate and baseline rows.
    """
    result = []
    for index in range(count):
        base = 10.0 + index
        for side, gap in (("baseline", base), ("candidate", base + offset)):
            result.append(
                {
                    "family": family,
                    "budget": "2s",
                    "max_evaluations": None,
                    "instance": f"i{index}",
                    "side": side,
                    "status": "ok",
                    "normalized_gap": gap,
                }
            )
    return result


def test_verdicts_follow_margin_count_and_holm() -> None:
    """Pass an improving cell, fail a worsening one and flag an undersized one."""
    cells = GateReport().analyze(
        rows("good", 15, -1.0) + rows("bad", 15, 3.0) + rows("small", 5, -1.0)
    )
    verdict = {cell["family"]: cell for cell in cells}
    assert verdict["good"]["verdict"] == "pass"
    assert verdict["bad"]["verdict"] == "fail"
    assert verdict["small"]["verdict"] == "inconclusive"
    assert verdict["small"]["p_holm"] is None
    assert verdict["good"]["p_holm"] >= verdict["good"]["p_one_sided"]


def test_equal_candidate_is_not_declared_non_inferior_by_silence() -> None:
    """Reject a candidate whose interval is wider than the margin even if not significant."""
    noisy = rows("noisy", 15, 0.0)
    for index, row in enumerate(r for r in noisy if r["side"] == "candidate"):
        row["normalized_gap"] += 5.0 if index % 2 else -5.0
    assert GateReport().analyze(noisy)[0]["verdict"] == "fail"


def test_failed_trials_are_counted_not_dropped() -> None:
    """Keep error rows visible as failure counts."""
    data = rows("f", 15, 0.0)
    data[0]["status"] = "error"
    assert (
        GateReport().analyze(data)[0]["candidate_failures"]
        + GateReport().analyze(data)[0]["baseline_failures"]
        == 1
    )


def test_calibration_and_test_instances_are_disjoint() -> None:
    """Never share a KPLIB id between splits."""
    test = GateInstances(Path("."), "test", 15).knapsack_ids()
    calibration = GateInstances(Path("."), "calibration", 15).knapsack_ids()
    assert len(test) == len(calibration) == 15
    assert not set(test) & set(calibration)


def test_run_refuses_without_native_hash(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Raise before any trial when the extension cannot be hashed.

    Args:
        monkeypatch (pytest.MonkeyPatch): Hides the extension.
        tmp_path (Path): Output directory.
    """
    monkeypatch.setattr(gate.importlib.util, "find_spec", lambda name: None)
    runner = Gate(
        GateConfig("rust", None, ["knapsack"], parse_budgets("1", 10), 1, tmp_path / "out.jsonl"),
        GateInstances(tmp_path, "test", 1),
    )
    with pytest.raises(RuntimeError, match="no recordable hash"):
        runner.run()
    assert not (tmp_path / "out.jsonl").exists()


def test_trial_row_carries_required_fields(tmp_path: Path) -> None:
    """Write sha256, commit, host, budget and gap for each trial, and reuse them as a baseline.

    Args:
        tmp_path (Path): Output directory.
    """
    instance = GateInstance("rastrigin-3", "x", "h", 0.0, None, lambda keys: 1.5)
    budget = parse_budgets("2", 10)[0]
    options = ConfigOverrides.parse([])
    row = Gate._trial(
        TrialSpec(instance, budget, 0, 7), SideSpec("candidate", "rust", None, options)
    )
    assert row["status"] == "error" and row["gap"] is None
    path = tmp_path / "base.jsonl"
    path.write_text(json.dumps({**row, "schema": gate.SCHEMA, "side": "candidate"}) + "\n")
    assert Gate.load_baseline(path)[0]["side"] == "baseline"
    for field in ("budget", "reference_value", "gap", "seed", "instance_sha256"):
        assert field in row


def test_budget_parsing() -> None:
    """Map tokens to time and evaluation budgets."""
    budgets = parse_budgets("2,10,evals", 500)
    assert [item.label for item in budgets] == ["2s", "10s", "evals"]
    assert budgets[2].max_evaluations == 500
    with pytest.raises(ValueError):
        parse_budgets("0", 1)


def test_evals_budget_is_sequential_with_high_time_cap() -> None:
    """Run the evals budget on one Rust worker with a time cap that never binds."""
    budgets = parse_budgets("2,evals", 500, 900.0)
    timed, evals = budgets
    assert (evals.n_jobs, evals.time_limit, evals.max_evaluations) == (1, 900.0, 500)
    assert parse_budgets("evals", 5)[0].time_limit == gate.EVALUATION_TIME_CAP == 600.0
    assert timed.n_jobs is None
    assert evals.solver_options("rust") == {"max_evaluations": 500, "n_jobs": 1}
    assert evals.solver_options("cpp") == {"max_evaluations": 500}
    assert timed.solver_options("rust") == {"max_evaluations": None}


def test_trial_row_records_jobs_and_time_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write n_jobs and time_cap on every trial row, failures included."""
    seen: dict[str, object] = {}

    class FakeRKO:
        """Capture constructor options and fail the solve."""

        def __init__(self, *args: object, **options: object) -> None:
            """Store the options.

            Args:
                *args (object): Positional arguments, ignored.
                **options (object): Keyword options.
            """
            seen.update(options)

        def solve(self, time_limit: float) -> None:
            """Record the time limit and fail.

            Args:
                time_limit (float): Requested seconds.

            Raises:
                RuntimeError: Always.
            """
            seen["time_limit"] = time_limit
            raise RuntimeError("stop")

    monkeypatch.setattr(gate, "RKO", FakeRKO)
    instance = GateInstance("knapsack", "k", "h", 10.0, None, lambda keys: 0.0)
    evals = parse_budgets("evals", 500)[0]
    options = ConfigOverrides.parse([])
    row = Gate._trial(
        TrialSpec(instance, evals, 0, 1), SideSpec("candidate", "rust", None, options)
    )
    assert (row["n_jobs"], row["time_cap"], row["status"]) == (1, 600.0, "error")
    assert seen["n_jobs"] == 1 and seen["time_limit"] == 600.0
    timed = Gate._trial(
        TrialSpec(instance, parse_budgets("2", 1)[0], 0, 1),
        SideSpec("candidate", "rust", None, options),
    )
    assert (timed["n_jobs"], timed["time_cap"]) == (None, 2.0)


def test_margin_uses_floor_when_baseline_gap_is_zero() -> None:
    """Take the family floor when the fractional margin collapses to zero."""
    data = rows("knapsack", 15, 0.0)
    for row in data:
        row["normalized_gap"] = 0.0
    cell = GateReport().analyze(data)[0]
    assert cell["margin"] == gate_support.KNAPSACK_FLOOR
    assert cell["margin_term"] == "floor"


def test_margin_uses_fraction_when_it_exceeds_floor() -> None:
    """Take the fractional margin when it is larger than the floor."""
    cell = GateReport(floors={"rastrigin": 0.01}).analyze(rows("rastrigin-100", 15, 0.0))[0]
    assert cell["margin_term"] == "fraction"
    assert cell["margin"] > 0.01


def test_floor_override_parsing() -> None:
    """Parse and validate the `--floor` syntax."""
    assert GateReport.parse_floors("knapsack=0.1,rastrigin=0.2") == {
        "knapsack": 0.1,
        "rastrigin": 0.2,
    }
    with pytest.raises(ValueError):
        GateReport.parse_floors("tsp=1")
    assert GateReport(floors={"knapsack": 0.3}).floors["knapsack"] == 0.3


def test_equal_engines_with_small_noise_pass_and_worse_candidate_fails() -> None:
    """Pass a near-zero-gap A/A at n >= 15, fail a shift beyond the margin, hold n < 15."""
    generator = random.Random(1)

    def noisy(family: str, count: int, shift: float) -> list[dict[str, Any]]:
        """Build rows with baseline gap near zero and symmetric small noise.

        Args:
            family (str): Family label.
            count (int): Instances.
            shift (float): Added to every candidate gap.

        Returns:
            list[dict[str, Any]]: Candidate and baseline rows.
        """
        data = rows(family, count, 0.0)
        for row in data:
            row["normalized_gap"] = 0.03 + generator.uniform(-0.005, 0.005)
            if row["side"] == "candidate":
                row["normalized_gap"] += shift
        return data

    cells = GateReport().analyze(
        noisy("knapsack", 20, 0.0)
        + noisy("rastrigin-100", 20, 0.5)
        + noisy("rastrigin-300", 5, 0.0)
    )
    verdict = {cell["family"]: cell["verdict"] for cell in cells}
    assert verdict == {
        "knapsack": "pass",
        "rastrigin-100": "fail",
        "rastrigin-300": "inconclusive",
    }


def test_rastrigin_dimensions_and_family_expansion() -> None:
    """Use 100 and 300 dimensions and drop the saturated 30."""
    assert gate_instances.RASTRIGIN_DIMENSIONS == (100, 300)
    assert gate.expand_families(["knapsack", "rastrigin"]) == (
        "knapsack",
        "rastrigin-100",
        "rastrigin-300",
    )


SYNTHETIC_CONFIG = """# header
[run]
control = 1 # inline note
pool_size = 10
prefill_pool = false
legacy_schedule = false
ratio = 0.5
label = "a"

[other]
pool_size = 3

[active_metaheuristics]
list = ["SA", "ILS"]
"""


def synthetic_params(tmp_path: Path) -> Path:
    """Write a small parameter directory with a `config.toml` and one algorithm file.

    Args:
        tmp_path (Path): Parent directory.

    Returns:
        Path: The new parameter directory.
    """
    source = tmp_path / "params"
    source.mkdir()
    (source / "config.toml").write_text(SYNTHETIC_CONFIG, encoding="utf-8")
    (source / "sa.toml").write_text("[offline]\nalpha = 0.9\n", encoding="utf-8")
    return source


def test_option_parsing_types_and_records(tmp_path: Path) -> None:
    """Parse bool, int, float and str values with the type of the default and record them."""
    source = synthetic_params(tmp_path)
    parsed = ConfigOverrides.parse(
        ["legacy_schedule=TRUE", "run.pool_size=25", "ratio=0.25", "label=zz"], source
    )
    assert parsed.as_record() == {
        "run.label": "zz",
        "run.legacy_schedule": True,
        "run.pool_size": 25,
        "run.ratio": 0.25,
    }
    assert ConfigOverrides.parse([], source).as_record() == {}


def test_option_parsing_rejects_bad_input(tmp_path: Path) -> None:
    """Refuse unknown, ambiguous, malformed, repeated and mistyped options."""
    source = synthetic_params(tmp_path)
    with pytest.raises(ValueError, match="Unknown config.toml key 'nope'.*run.pool_size"):
        ConfigOverrides.parse(["nope=1"], source)
    with pytest.raises(ValueError, match="ambiguous"):
        ConfigOverrides.parse(["pool_size=1"], source)
    with pytest.raises(ValueError, match="KEY=VALUE"):
        ConfigOverrides.parse(["legacy_schedule"], source)
    with pytest.raises(ValueError, match="more than once"):
        ConfigOverrides.parse(["run.ratio=1.0", "ratio=2.0"], source)
    with pytest.raises(ValueError, match="expected true or false"):
        ConfigOverrides.parse(["legacy_schedule=1"], source)
    with pytest.raises(ValueError, match="Invalid value 'x'"):
        ConfigOverrides.parse(["run.pool_size=x"], source)
    with pytest.raises(ValueError, match="finite"):
        ConfigOverrides.parse(["ratio=nan"], source)


def test_overrides_write_temporary_directory(tmp_path: Path) -> None:
    """Copy every parameter file, rewrite only the requested keys and keep the rest intact."""
    source = synthetic_params(tmp_path)
    overrides = ConfigOverrides.parse(
        ["legacy_schedule=true", "run.pool_size=25", "label=b"], source
    )
    target = tmp_path / "out"
    target.mkdir()
    assert overrides.write(target, source) == target
    written = tomllib.loads((target / "config.toml").read_text(encoding="utf-8"))
    assert written["run"] == {
        "control": 1,
        "pool_size": 25,
        "prefill_pool": False,
        "legacy_schedule": True,
        "ratio": 0.5,
        "label": "b",
    }
    assert written["other"] == {"pool_size": 3}
    assert written["active_metaheuristics"] == {"list": ["SA", "ILS"]}
    assert (target / "sa.toml").read_text(encoding="utf-8") == "[offline]\nalpha = 0.9\n"
    assert "pool_size = 25" in (target / "config.toml").read_text(encoding="utf-8")
    assert "legacy_schedule = false" in (source / "config.toml").read_text(encoding="utf-8")


def test_default_config_overrides_parse_and_write(tmp_path: Path) -> None:
    """Accept existing keys of the installed config.toml and refuse a missing one."""
    overrides = ConfigOverrides.parse(["progress_schedule=true", "pool_size=7"])
    assert overrides.as_record() == {"run.pool_size": 7, "run.progress_schedule": True}
    overrides.write(tmp_path)
    written = tomllib.loads((tmp_path / "config.toml").read_text())["run"]
    assert written["progress_schedule"] is True
    assert written["pool_size"] == 7
    with pytest.raises(ValueError, match="Unknown config.toml key.*run.progress_schedule"):
        ConfigOverrides.parse(["nome_inexistente=1"])


def test_gate_rejects_baseline_options_without_live_baseline(tmp_path: Path) -> None:
    """Refuse baseline options when the baseline is a recorded file or absent."""
    recorded = tmp_path / "base.jsonl"
    recorded.write_text("{}\n")
    for baseline in (str(recorded), None):
        with pytest.raises(ValueError, match="live --baseline"):
            Gate(
                GateConfig(
                    "rust",
                    baseline,
                    ["knapsack"],
                    parse_budgets("1", 10),
                    1,
                    tmp_path / "out.jsonl",
                    baseline_options=["pool_size=7"],
                ),
                GateInstances(tmp_path, "test", 1),
            )


def test_trial_passes_params_dir_to_solver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hand the side's parameter directory to the `RKO` constructor."""
    seen: dict[str, object] = {}

    class FakeRKO:
        """Capture constructor options and fail the solve."""

        def __init__(self, *args: object, **options: object) -> None:
            """Store the options.

            Args:
                *args (object): Positional arguments, ignored.
                **options (object): Keyword options.
            """
            seen.update(options)

        def solve(self, time_limit: float) -> None:
            """Fail without solving.

            Args:
                time_limit (float): Requested seconds, ignored.

            Raises:
                RuntimeError: Always.
            """
            raise RuntimeError("stop")

    monkeypatch.setattr(gate, "RKO", FakeRKO)
    instance = GateInstance("knapsack", "k", "h", 10.0, None, lambda keys: 0.0)
    options = ConfigOverrides.parse([])
    Gate._trial(
        TrialSpec(instance, parse_budgets("1", 1)[0], 0, 1),
        SideSpec("candidate", "rust", None, options, tmp_path),
    )
    assert seen["params_dir"] == tmp_path
