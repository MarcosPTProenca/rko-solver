"""Validate compact paired study reports without running the expanded experiment.

Objective:
- Reject incomplete, overlapping or incomparable evidence before publishing summaries.

Technical approach:
- Use the identified pilot as a valid fixture and corrupt individual controls.
- Recompute outcomes and medians independently from its archived trial records.

Flow (pseudocode):
1. Compile known paired rows into compact observations and grouped summaries.
2. Mutate accounting, provenance or pairing fields one at a time.
3. Require validation failures instead of silently merging bad observations.

Important criteria:
- Existing pilot pairs must not be counted twice in an expanded study.
- No timing or statistical-superiority conclusions are inferred by aggregation.

Inputs/Outputs:
- Input: archived JSONL and isolated corrupted copies.
- Output: validated groups or explicit data-validation failures.
"""

import csv
import importlib
import json
import numpy as np
import pytest
import statistics
import sys
from pathlib import Path
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.grasp import GRASP
from rko_solver.parameters.loader import load_offline
from types import ModuleType


@pytest.fixture
def reporter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import optional reporting code from the repository benchmark directory.

    Args:
        monkeypatch (pytest.MonkeyPatch): Expose benchmark tools during the test.

    Returns:
        ModuleType: Report compiler.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    return importlib.import_module("prefill_report")


def test_report_recomputes_pilot_groups(reporter: ModuleType) -> None:
    """Check compact evidence against the independent identified raw pilot.

    Args:
        reporter (ModuleType): Report compiler.
    """
    source = Path(reporter.__file__).with_name("prefill-work-reference.jsonl")
    report, pairs = reporter.compile_report([source])
    assert report["pairs"] == len(pairs) == 60 and len(report["groups"]) == 12
    for group in report["groups"]:
        selected = [
            row
            for row in pairs
            if (row["algorithm"], row["workload"]) == (group["algorithm"], group["workload"])
        ]
        assert group["pairs"] == len(selected) == 5
        assert group["better"] + group["tied"] + group["worse"] == 5
        assert group["default_median"] == statistics.median(row["default_cost"] for row in selected)
        assert group["prefilled_median"] == statistics.median(
            row["prefilled_cost"] for row in selected
        )
    assert report["timing"] is False


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "missing_pair",
        "accounting",
        "seed",
        "prefix",
        "nonfinite",
        "cost_type",
        "timing",
        "unknown",
        "missing_field",
        "missing_family",
    ],
)
def test_report_rejects_corrupted_shards(
    reporter: ModuleType,
    tmp_path: Path,
    damage: str,
) -> None:
    """Reject a single scientific-control mutation instead of emitting plausible totals.

    Args:
        reporter (ModuleType): Report compiler.
        tmp_path (Path): Isolated corrupted shard directory.
        damage (str): Control or schema field to corrupt.
    """
    source = Path(reporter.__file__).with_name("prefill-work-reference.jsonl")
    records = [json.loads(line) for line in source.read_text().splitlines()]
    index = next(
        i for i, row in enumerate(records) if row.get("kind") == "trial" and row["prefill"]
    )
    if damage == "duplicate":
        records.append(records[index].copy())
    elif damage == "missing_pair":
        records.pop(index)
    elif damage == "accounting":
        records[index]["search_evaluations"] -= 1
    elif damage == "seed":
        records[index]["search_seed"] += 1
    elif damage == "prefix":
        records[index]["startup_seed_prefix_sha256"] = "bad"
    elif damage == "nonfinite":
        records[index]["cost"] = float("nan")
    elif damage == "cost_type":
        records[index]["cost"] = "invalid"
    elif damage == "timing":
        records[0]["timing"] = True
    elif damage == "unknown":
        records.append({"kind": "unknown"})
    elif damage == "missing_field":
        del records[index]["cost"]
    else:
        records = [row for row in records if row.get("workload") != "flat"]
    target = tmp_path / "corrupt.jsonl"
    target.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    with pytest.raises(ValueError):
        reporter.compile_report([target])


def test_expanded_reference_matches_compact_observations(reporter: ModuleType) -> None:
    """Recompute all expanded outcomes and medians after raw workflow artifacts expire.

    Args:
        reporter (ModuleType): Report compiler location.
    """
    directory = Path(reporter.__file__).parent
    report = json.loads((directory / "prefill-expanded-summary.json").read_text())
    with (directory / "prefill-expanded-pairs.csv").open(newline="") as stream:
        pairs = list(csv.DictReader(stream))
    assert report["pairs"] == len(pairs) == 1440 and len(report["groups"]) == 72
    keys = [
        (row["dimension"], row["evaluations"], row["algorithm"], row["workload"], row["seed"])
        for row in pairs
    ]
    assert len(set(keys)) == len(keys)
    assert {(int(row["dimension"]), int(row["evaluations"])) for row in pairs} == {
        (dimension, budget) for dimension in (20, 100, 500) for budget in (256, 1024)
    }
    for group in report["groups"]:
        selected = [
            row
            for row in pairs
            if (int(row["dimension"]), int(row["evaluations"]), row["algorithm"], row["workload"])
            == (group["dimension"], group["evaluations"], group["algorithm"], group["workload"])
        ]
        assert group["pairs"] == len(selected) == 20
        assert {int(row["seed"]) for row in selected} == set(range(200, 220))
        outcomes = {"better": 0, "tied": 0, "worse": 0}
        for row in selected:
            before, after = float(row["default_cost"]), float(row["prefilled_cost"])
            outcomes["better" if after < before else "worse" if after > before else "tied"] += 1
            assert 10 <= int(row["startup_evaluations"]) <= 19
            assert after <= float(row["startup_best_cost"])
        assert all(group[key] == count for key, count in outcomes.items())
        assert group["default_median"] == statistics.median(
            float(row["default_cost"]) for row in selected
        )
        assert group["prefilled_median"] == statistics.median(
            float(row["prefilled_cost"]) for row in selected
        )
        assert group["search_improved_startup"] == sum(
            row["search_improved_startup"] == "True" for row in selected
        )
    winners = [
        row
        for row in pairs
        if row["algorithm"] == "GRASP"
        and row["dimension"] == "500"
        and float(row["prefilled_cost"]) < float(row["default_cost"])
    ]
    assert len(winners) == 102
    assert all(row["search_improved_startup"] == "False" for row in winners)


@pytest.mark.parametrize(
    "dimension, searches, lower_bound", [(20, 87, 350), (100, 243, 974), (500, 6315, 25262)]
)
def test_grasp_first_construction_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
    dimension: int,
    searches: int,
    lower_bound: int,
) -> None:
    """Check documented minimum-stage work through actual construction and line-search seams.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fix minimum intensity and deterministic order.
        dimension (int): Studied chromosome length.
        searches (int): Required coordinate line searches at the minimum intensity.
        lower_bound (int): Objectives through construction, including initial/final scores.
    """
    visits: list[int] = []
    costs: list[float] = []

    def score(keys: np.ndarray) -> float:
        """Count actual line-search or final construction objective observations.

        Args:
            keys (np.ndarray): Candidate chromosome.

        Returns:
            float: Flat finite objective.
        """
        costs.append(0.0)
        return 0.0

    def search(pos: int, step: float, keys: np.ndarray) -> tuple[float, float]:
        """Count coordinate requests without spending full candidate callbacks.

        Args:
            pos (int): Candidate coordinate.
            step (float): Reference offline step.
            keys (np.ndarray): Current partial solution.

        Returns:
            tuple[float, float]: Unchanged coordinate and tied score.
        """
        visits.append(pos)
        return float(keys[pos]), 0.0

    algorithm = GRASP(SolutionPool(), lambda keys: keys, score, dimension)
    streams = RandomStreams(42)
    monkeypatch.setattr(streams.python, "uniform", lambda lower, upper: 0.3)
    monkeypatch.setattr(streams.python, "shuffle", lambda values: None)
    monkeypatch.setattr(algorithm, "_line_search", search)
    step = load_offline("GRASP")["hs"]
    assert step == 0.125
    with streams.activate():
        algorithm._constructive(step, np.full(dimension, 0.5))
        assert len(visits) == searches and len(costs) == 1
        GRASP._line_search(algorithm, 0, step, np.full(dimension, 0.5))
    quota = len(costs) - 1
    assert quota == 4 and 2 + quota * searches == lower_bound


def test_report_cli_writes_compact_evidence(
    reporter: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Exercise workflow output filenames and retained pair evidence end to end.

    Args:
        reporter (ModuleType): Report compiler/CLI.
        monkeypatch (pytest.MonkeyPatch): Supply validated report arguments.
        tmp_path (Path): Isolated input and output directories.
    """
    source = Path(reporter.__file__).with_name("prefill-work-reference.jsonl")
    directory = tmp_path / "raw"
    directory.mkdir()
    (directory / "pilot.jsonl").write_bytes(source.read_bytes())
    prefix = tmp_path / "report" / "prefill-expanded"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prefill_report",
            "--input-dir",
            str(directory),
            "--output-prefix",
            str(prefix),
            "--run-url",
            "https://example.test/study",
        ],
    )
    reporter.main()
    summary = json.loads(prefix.with_name(prefix.name + "-summary.json").read_text())
    assert summary["pairs"] == 60 and summary["workflow_run_url"] == "https://example.test/study"
    assert b"\r" not in prefix.with_name(prefix.name + "-pairs.csv").read_bytes()
    with prefix.with_name(prefix.name + "-pairs.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 60 and all(row["evaluations"] == "256" for row in rows)


def test_report_rejects_overlap_and_mixed_provenance(reporter: ModuleType, tmp_path: Path) -> None:
    """Keep repeated pilot seeds and incompatible producer revisions out of combined totals.

    Args:
        reporter (ModuleType): Report compiler.
        tmp_path (Path): Isolated mixed-provenance shard directory.
    """
    source = Path(reporter.__file__).with_name("prefill-work-reference.jsonl")
    with pytest.raises(ValueError, match="duplicate"):
        reporter.compile_report([source, source])
    records = [json.loads(line) for line in source.read_text().splitlines()]
    records[0]["source_revision"] = "different"
    target = tmp_path / "different.jsonl"
    target.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    with pytest.raises(ValueError, match="mixed study provenance"):
        reporter.compile_report([source, target])
    with pytest.raises(ValueError, match="no study shards"):
        reporter.compile_report([])
