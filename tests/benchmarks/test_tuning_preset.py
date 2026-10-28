"""Contracts for training-only selection and sealed confirmation of one fixed preset.

Objective:
- Prevent selecting a general preset from test scores or silently reusing instance sizes.

Technical approach:
- Check explicit-size banks, family-balanced synthetic selection records and the frozen
  selection boundary before running a small real spawned-process study.

Flow (pseudocode):
1. Exercise independent size overrides and synthetic training-only decisions.
2. Reject incomplete or incompatible evidence and attempts to alter locked selections.
3. Run a small complete selection and confirmation through actual solver calls.

Important criteria:
- The confirmation set is never used to choose the candidate.
- Failed noninferiority checks reject a candidate rather than changing the margin.

Inputs/Outputs:
- Input: generated instances and synthetic or small measured score records.
- Output: assertions about data isolation, selection and acceptance gates.
"""

import json
import pytest
from benchmarks.tuning.instances import InstanceBank
from benchmarks.tuning.preset_analysis import PresetAnalysis
from benchmarks.tuning.preset_analysis import PresetDesign
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from pathlib import Path
from typing import Any


def test_explicit_instance_sizes_are_isolated_and_do_not_mutate_defaults(tmp_path: Path) -> None:
    """An explicit size produces separate instance and calibration paths.

    Args:
        tmp_path (Path): Parent directory for all banks.
    """
    default = InstanceBank(tmp_path)
    small = InstanceBank(tmp_path, size=10)
    large = InstanceBank(tmp_path, size=60)
    for family in ("kp", "tsp", "rastrigin"):
        first = small.generate(family, "train", 1)[0]
        second = large.generate(family, "train", 1)[0]
        assert first.path != second.path
        assert int(first.path.read_text().split()[0]) == 10
        assert int(second.path.read_text().split()[0]) == 60
        item = default.generate(family, "train", 1)[0]
        assert int(item.path.read_text().split()[0]) == InstanceBank.SIZES[family]
    assert small.generate("kp", "train", 1)[0].path == first.path.parent / "kp-train-0.txt"
    with pytest.raises(ValueError, match="size"):
        InstanceBank(tmp_path, size=0)


@pytest.fixture
def design() -> PresetDesign:
    """Provide unequal family context counts and cheap deterministic bootstrap draws.

    Returns:
        PresetDesign: Synthetic design whose equal family weights can be checked exactly.
    """
    return PresetDesign(
        sizes=(("kp", (10,)), ("tsp", (5, 10))),
        budgets=(200, 400),
        train=3,
        test=4,
        train_seeds=1,
        test_seeds=1,
        resamples=1000,
    )


def evidence(
    design: PresetDesign, phase: str, values: dict[str, dict[str, float]]
) -> list[dict[str, Any]]:
    """Generate complete constant-score records for a synthetic decision.

    Args:
        design (PresetDesign): Counts and contexts.
        phase (str): Training or test phase.
        values (dict): Candidate, then family, to constant score.

    Returns:
        list[dict[str, Any]]: One record per candidate and cell.
    """
    count = design.train if phase == "train" else design.test
    return [
        {
            "phase": phase,
            "family": family,
            "size": size,
            "evaluations": budget,
            "candidate": candidate,
            "scores": [scores[family]] * count,
        }
        for candidate, scores in values.items()
        for family, size, budget in design.cases
    ]


def test_selection_weights_families_equally_and_rejects_local_regressions(
    design: PresetDesign,
) -> None:
    """A good pooled average cannot hide an excessive loss in one training cell.

    Args:
        design (PresetDesign): Unequal contexts per family.
    """
    analysis = PresetAnalysis(design)
    values = {
        "default": {"kp": 1.0, "tsp": 1.0},
        "greedy": {"kp": 0.5, "tsp": 1.1},
        "balanced": {"kp": 0.95, "tsp": 0.95},
    }
    ranked = analysis.ranking(evidence(design, "train", values), set(values))
    assert ranked[0]["candidate"] == "balanced"
    rejected = next(row for row in ranked if row["candidate"] == "greedy")
    assert rejected["score"] == pytest.approx(0.8)
    assert not rejected["eligible"] and rejected["worst_delta"] == pytest.approx(0.1)
    assert (
        analysis.balanced_mean({("kp", 10, 200): 10, ("tsp", 5, 200): -10, ("tsp", 10, 200): -10})
        == 0
    )


def test_partial_duplicate_nonfinite_and_wrong_phase_evidence_fails_closed(
    design: PresetDesign,
) -> None:
    """Missing or incompatible rows cannot silently change family weights.

    Args:
        design (PresetDesign): Synthetic study design.
    """
    analysis = PresetAnalysis(design)
    rows = evidence(design, "train", {"default": {"kp": 1.0, "tsp": 1.0}})
    for broken in (
        rows[:-1],
        [*rows, rows[0]],
        [dict(rows[0], phase="test"), *rows[1:]],
        [dict(rows[0], scores=[float("nan")] * design.train), *rows[1:]],
    ):
        with pytest.raises(ValueError):
            analysis.ranking(broken, {"default"})
    with pytest.raises(ValueError, match="default"):
        analysis.ranking([], {"candidate"})


def test_confirmation_requires_global_gain_and_every_cell_within_margin(
    design: PresetDesign,
) -> None:
    """The fixed confirmation gate rejects local harm even when the overall mean improves.

    Args:
        design (PresetDesign): Synthetic confirmation design.
    """
    analysis = PresetAnalysis(design)
    good = evidence(
        design, "test", {"default": {"kp": 1, "tsp": 1}, "candidate": {"kp": 0.8, "tsp": 0.8}}
    )
    result = analysis.confirmation(good, "candidate")
    assert result["passed"] and result["delta"] == pytest.approx(-0.2)
    bad = evidence(
        design, "test", {"default": {"kp": 1, "tsp": 1}, "candidate": {"kp": 1.03, "tsp": 0.5}}
    )
    result = analysis.confirmation(bad, "candidate")
    assert result["delta"] < 0 and not result["passed"]
    assert all(not cell["passed"] for cell in result["cells"] if cell["family"] == "kp")
    same = evidence(design, "test", {"default": {"kp": 1, "tsp": 1}})
    assert not analysis.confirmation(same, "default")["passed"]


def test_bootstrap_preserves_paired_instances_across_budgets(design: PresetDesign) -> None:
    """Opposite per-instance differences at two budgets cancel in every overall bootstrap draw.

    Args:
        design (PresetDesign): Two-budget confirmation design.
    """
    rows = evidence(
        design, "test", {"default": {"kp": 0, "tsp": 0}, "candidate": {"kp": 0, "tsp": 0}}
    )
    for row in rows:
        if row["candidate"] == "candidate":
            sign = 1 if row["evaluations"] == 200 else -1
            row["scores"] = [sign * value for value in (-1, -2, 1, 2)]
    result = PresetAnalysis(design).confirmation(rows, "candidate")
    assert result["delta"] == 0 and result["upper"] == 0


@pytest.fixture
def candidate_source(tmp_path: Path) -> Path:
    """Write a source containing configurations and deliberately unusable historical test scores.

    Args:
        tmp_path (Path): Directory for the source file.

    Returns:
        Path: Candidate source whose test-score fields must never be inspected.
    """
    from benchmarks.tuning.spaces import SPACES

    path = tmp_path / "candidates.jsonl"
    configs = [dict(SPACES["SA"].default(), t0=0.1), dict(SPACES["SA"].default(), t0=10.0)]
    rows = [
        {
            "kind": "session",
            "algorithm": "SA",
            "tuner": "racing",
            "config": config,
            "test": "not selection evidence",
        }
        for config in configs
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def test_selection_is_training_only_and_cannot_be_changed_after_lock(
    tmp_path: Path, candidate_source: Path, design: PresetDesign
) -> None:
    """Test records cannot influence selection, and changed training data invalidate its lock.

    Args:
        tmp_path (Path): Study directory parent.
        candidate_source (Path): Fixed candidate source.
        design (PresetDesign): Synthetic design.
    """
    from benchmarks.tuning.preset import PresetStudy

    study = PresetStudy(tmp_path / "study", candidate_source, design)
    study.freeze()
    pool = study.pool()
    values = {
        name: {family: 0.8 if config["t0"] == 0.1 else 1.0 for family, _ in design.sizes}
        for name, config in pool.items()
    }
    rows = evidence(design, "train", values)
    training = study.directory / "train.jsonl"
    training.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    locked = study.select()
    assert locked["config"]["t0"] == 0.1
    assert not list(study.directory.rglob("*-test-*.txt"))
    (study.directory / "test.jsonl").write_text("test results must not change selection\n")
    assert study.select() == locked
    training.write_text("\n".join(json.dumps(row) for row in rows[:-1]) + "\n")
    with pytest.raises(ValueError):
        study.select()


def test_confirmation_cannot_generate_tests_without_complete_locked_training(
    tmp_path: Path, candidate_source: Path, design: PresetDesign
) -> None:
    """A missing training phase fails before any test instance is generated.

    Args:
        tmp_path (Path): Study directory parent.
        candidate_source (Path): Fixed candidate source.
        design (PresetDesign): Synthetic study design.
    """
    from benchmarks.tuning.preset import PresetStudy

    study = PresetStudy(tmp_path / "study", candidate_source, design)
    with pytest.raises((ValueError, FileNotFoundError)):
        study.run("confirm", workers=1)
    assert not list(study.directory.rglob("*-test-*.txt"))


def test_real_study_locks_a_candidate_before_confirmation_and_resumes(
    tmp_path: Path, candidate_source: Path
) -> None:
    """The actual spawned-worker path selects, seals and confirms without changing parameters.

    Args:
        tmp_path (Path): Parent directory for the complete small study.
        candidate_source (Path): Fixed source configurations.
    """
    from benchmarks.tuning.preset import PresetStudy

    design = PresetDesign(
        sizes=(("rastrigin", (4, 8)),),
        budgets=(200, 400),
        train=2,
        test=2,
        train_seeds=1,
        test_seeds=1,
        resamples=200,
    )
    study = PresetStudy(tmp_path / "actual", candidate_source, design)
    study.run("select", workers=1)
    locked = study.select()
    assert not list(study.directory.rglob("*-test-*.txt"))
    study.run("confirm", workers=1)
    before = (study.directory / "test.jsonl").read_bytes()
    study.run("confirm", workers=1)
    assert (study.directory / "test.jsonl").read_bytes() == before
    report = study.report()
    assert isinstance(report["passed"], bool) and len(report["cells"]) == 4
    assert study.select() == locked
    from unittest.mock import patch

    with patch("benchmarks.tuning.preset.subprocess.check_output", return_value="new-docs-commit"):
        assert study.report() == report
        with pytest.raises(ValueError, match="Frozen"):
            study.select()


@given(
    st.integers(1, 8),
    st.integers(1, 8),
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
    st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, derandomize=True)
def test_family_weight_is_independent_of_the_number_of_cells(
    first_count: int, second_count: int, first: float, second: float
) -> None:
    """Repeating a family's constant score cannot increase its influence on the overall mean.

    Args:
        first_count (int): Number of contexts in the first family.
        second_count (int): Number of contexts in the second family.
        first (float): Constant score of the first family.
        second (float): Constant score of the second family.
    """
    values = {("kp", i, 200): first for i in range(first_count)}
    values.update({("tsp", i, 200): second for i in range(second_count)})
    assert PresetAnalysis.balanced_mean(values) == pytest.approx((first + second) / 2, abs=1e-8)


def test_freeze_rejects_changed_design_or_candidate_source(
    tmp_path: Path, candidate_source: Path, design: PresetDesign
) -> None:
    """An existing study cannot resume with relaxed limits or a different candidate pool.

    Args:
        tmp_path (Path): Parent folder for the frozen study.
        candidate_source (Path): Input configurations.
        design (PresetDesign): Original scientific design.
    """
    from benchmarks.tuning.preset import PresetStudy
    from dataclasses import replace

    root = tmp_path / "locked"
    study = PresetStudy(root, candidate_source, design)
    study.freeze()
    with pytest.raises(ValueError, match="Frozen"):
        PresetStudy(root, candidate_source, replace(design, margin=0.05)).freeze()
    candidate_source.write_text(candidate_source.read_text().replace('"t0": 0.1', '"t0": 0.2'))
    with pytest.raises(ValueError, match="Frozen"):
        study.freeze()


def test_sealed_report_rejects_changed_selection_or_training(
    tmp_path: Path, candidate_source: Path, design: PresetDesign
) -> None:
    """Read-only reporting cannot hide changes to the selected configuration or training bytes.

    Args:
        tmp_path (Path): Study directory parent.
        candidate_source (Path): Fixed source configurations.
        design (PresetDesign): Synthetic study design.
    """
    from benchmarks.tuning.preset import PresetStudy

    study = PresetStudy(tmp_path / "sealed", candidate_source, design)
    study.freeze()
    pool = study.pool()
    rows = evidence(
        design, "train", {name: {family: 1.0 for family, _ in design.sizes} for name in pool}
    )
    training = study.directory / "train.jsonl"
    training.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    selection = study.select()
    path = study.directory / "selection.json"
    changed = dict(selection, config=dict(selection["config"], alpha=0.7))
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="configuration"):
        study.report()
    path.write_text(json.dumps(selection))
    training.write_text(training.read_text() + "\n")
    with pytest.raises(ValueError, match="hashes"):
        study.report()


def test_general_preset_reference_reconstructs_selection_scales_and_gates() -> None:
    """The committed reference reconstructs its locked selection, normalization and decision."""
    import hashlib

    path = Path(__file__).resolve().parents[2] / "benchmarks" / "general-sa-preset-reference.json"
    data = json.loads(path.read_text())
    design = PresetDesign(**data["protocol"]["design"])
    analysis = PresetAnalysis(design)
    selected = data["selection"]
    assert len(data["training"]) == 372 and len(data["confirmation"]) == 24
    assert len(data["instances"]) == len(set(data["instances"].values())) == 108
    assert data["protocol"]["source_sha"].startswith("aa9f29e")
    ranking = analysis.ranking(data["training"], set(data["protocol"]["candidates"]))
    assert ranking[0]["candidate"] == selected["candidate"]
    protocol = json.dumps(data["protocol"], sort_keys=True, allow_nan=False)
    training = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in data["training"]
    )
    assert hashlib.sha256(protocol.encode()).hexdigest() == selected["protocol_sha256"]
    assert hashlib.sha256(training.encode()).hexdigest() == selected["training_sha256"]
    result = analysis.confirmation(data["confirmation"], selected["candidate"])
    assert result["passed"] == data["summary"]["passed"]
    assert result["upper"] == pytest.approx(data["summary"]["upper"], abs=1e-12)
    assert result["delta"] == pytest.approx(data["summary"]["delta"], abs=1e-12)
    for row in data["training"] + data["confirmation"]:
        cohort = design.offset(row["family"], row["size"])
        name = f"cohort-{cohort}/size-{row['size']}/calibration-{row['family']}.json"
        calibration = data["calibration"][name]
        for index, (cost, score) in enumerate(zip(row["costs"], row["scores"], strict=True)):
            point = calibration[f"{row['family']}-{row['phase']}-{index}"]
            assert score == pytest.approx(
                (cost - point["best"]) / (point["anchor"] - point["best"])
            )
