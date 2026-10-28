"""Cross-engine contracts for confirmation of the already locked fixed preset.

Objective:
- Validate Python and C++ scoring without changing the locked candidate or reference scales.

Technical approach:
- Exercise real engine calls, verify deterministic instance hashes before scoring and reject
  altered protocols or records from another engine.

Flow (pseudocode):
1. Check engine routing and objective reuse on a small real instance.
2. Replay sealed reference instances without calibration or candidate selection.
3. Run small actual confirmation studies through spawned workers.

Important criteria:
- C++ tests skip only when the optional engine is absent.
- Paired comparisons are within an engine; seeded trajectories need not match across engines.

Inputs/Outputs:
- Input: isolated work directories and sealed reference measurements.
- Output: assertions about engine routing, replay identity and sealed confirmation gates.
"""

import json
import numpy as np
import pytest
from benchmarks.tuning.instances import InstanceBank
from benchmarks.tuning.spaces import SPACES
from pathlib import Path
from rko_solver import _native

ENGINES = [
    "python",
    "rust",
    pytest.param(
        "cpp",
        marks=pytest.mark.skipif(
            not _native.cpp_available(), reason="Optional C++ engine is absent."
        ),
    ),
]


@pytest.mark.parametrize("engine", ENGINES)
def test_instance_bank_scores_on_the_requested_engine(tmp_path: Path, engine: str) -> None:
    """A real fixed-parameter solve uses the selected backend and retains its objective cache.

    Args:
        tmp_path (Path): Isolated instance directory.
        engine (str): Engine to run.
    """
    bank = InstanceBank(tmp_path, evaluations=200, size=8)
    item = bank.generate("rastrigin", "test", 1)[0]
    config = SPACES["SA"].default()
    first = bank.cost(item, "SA", config, 7, engine=engine)
    second = bank.cost(item, "SA", config, 8, engine=engine)
    assert np.isfinite(first) and np.isfinite(second)
    assert len(bank._compiled) == 1
    assert len(bank._problems) == (0 if engine == "python" else 1)


def test_scoring_rejects_a_search_that_did_not_reach_its_budget(tmp_path: Path) -> None:
    """A stopped or truncated search must fail instead of entering confirmation evidence.

    Args:
        tmp_path (Path): Isolated instance directory.
    """
    from types import SimpleNamespace
    from unittest.mock import Mock
    from unittest.mock import patch

    bank = InstanceBank(tmp_path, evaluations=200, size=8)
    item = bank.generate("rastrigin", "test", 1)[0]
    solver = Mock()
    solver.return_value.solve.return_value = SimpleNamespace(
        cost=0.0, searches=(SimpleNamespace(n_evaluations=199),)
    )
    with patch("benchmarks.tuning.instances.RKO", solver):
        with pytest.raises(RuntimeError, match="evaluation budget"):
            bank.cost(item, "SA", SPACES["SA"].default(), 7, engine="python")


@pytest.fixture
def reference(tmp_path: Path) -> Path:
    """Copy the immutable full measurement into an isolated input directory.

    Args:
        tmp_path (Path): Input directory.

    Returns:
        Path: Copied reference, never the maintained source artifact.
    """
    source = Path(__file__).resolve().parents[4] / "benchmarks" / "general-sa-preset-reference.json"
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())
    return path


def test_replay_preserves_instances_scales_and_the_locked_configuration(
    tmp_path: Path, reference: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay does not calibrate again or use confirmation scores to select a candidate.

    Args:
        tmp_path (Path): Work directory parent.
        reference (Path): Copied sealed reference.
        monkeypatch (pytest.MonkeyPatch): Guard forbidden operations.
    """
    import hashlib
    from benchmarks.tuning.engine_confirmation import EngineConfirmation
    from benchmarks.tuning.preset_analysis import PresetAnalysis

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail if replay attempts calibration or candidate ranking.

        Args:
            args (object): Unused positional inputs.
            kwargs (object): Unused named inputs.

        Raises:
            AssertionError: Always, because these operations are forbidden in replay.
        """
        raise AssertionError("Replay must not recalibrate or reselect.")

    monkeypatch.setattr(InstanceBank, "calibrate", forbidden)
    monkeypatch.setattr(PresetAnalysis, "ranking", forbidden)
    run = EngineConfirmation(tmp_path / "replay", reference)
    tasks = run.prepare("python")
    data = json.loads(reference.read_text())
    assert len(tasks) == 24
    assert {task.candidate for task in tasks} == {"default", data["selection"]["candidate"]}
    assert all(task.engine == "python" and task.phase == "test" for task in tasks)
    for task in tasks:
        assert task.config == data["protocol"]["candidates"][task.candidate]
    root = run.directory / "python"
    files = list(root.rglob("*.txt"))
    assert len(files) == 72 and not list(root.rglob("*-train-*.txt"))
    for path in files:
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest()
            == data["instances"][path.relative_to(root).as_posix()]
        )
    first = files[0]
    first.write_text("changed instance\n")
    with pytest.raises(ValueError, match="instance"):
        run.prepare("python")


def test_replay_rejects_changed_selection_and_source(tmp_path: Path, reference: Path) -> None:
    """Modified configurations cannot become an alternative candidate through resume.

    Args:
        tmp_path (Path): Work directory parent.
        reference (Path): Copied sealed reference.
    """
    from benchmarks.tuning.engine_confirmation import EngineConfirmation

    EngineConfirmation(tmp_path / "replay", reference).prepare("python")
    data = json.loads(reference.read_text())
    data["selection"]["config"]["alpha"] = 0.7
    reference.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="selection"):
        EngineConfirmation(tmp_path / "changed", reference).prepare("python")
    data["selection"]["config"] = data["protocol"]["candidates"][data["selection"]["candidate"]]
    data["runtime"]["python"] = "another source"
    reference.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Frozen"):
        EngineConfirmation(tmp_path / "replay", reference).prepare("python")


def test_engine_report_rejects_rows_from_another_engine(tmp_path: Path, reference: Path) -> None:
    """Rust results cannot masquerade as unmeasured Python confirmation evidence.

    Args:
        tmp_path (Path): Work directory parent.
        reference (Path): Copied sealed reference.
    """
    from benchmarks.tuning.engine_confirmation import EngineConfirmation

    run = EngineConfirmation(tmp_path / "replay", reference)
    run.prepare("python")
    records = json.loads(reference.read_text())["confirmation"]
    (run.directory / "python" / "test.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    with pytest.raises(ValueError, match="engine"):
        run.report("python")


@pytest.fixture
def tiny_reference(tmp_path: Path) -> Path:
    """Build a small sealed Rust study using real training, calibration and confirmation.

    Args:
        tmp_path (Path): Isolated work directory parent.

    Returns:
        Path: Complete portable reference for actual cross-engine replay.
    """
    import hashlib
    from benchmarks.tuning.preset import PresetStudy
    from benchmarks.tuning.preset_analysis import PresetDesign

    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {"algorithm": "SA", "tuner": "racing", "config": dict(SPACES["SA"].default(), t0=0.1)}
        )
        + "\n"
    )
    design = PresetDesign(
        sizes=(("rastrigin", (4, 8)),),
        budgets=(200, 400),
        train=2,
        test=2,
        train_seeds=1,
        test_seeds=1,
        resamples=200,
    )
    root = tmp_path / "original"
    study = PresetStudy(root, source, design)
    study.run("select", workers=1)
    study.run("confirm", workers=1)
    data = {
        "protocol": json.loads((root / "protocol.json").read_text()),
        "selection": json.loads((root / "selection.json").read_text()),
        "training": [json.loads(line) for line in (root / "train.jsonl").read_text().splitlines()],
        "calibration": {
            p.relative_to(root).as_posix(): json.loads(p.read_text())
            for p in root.rglob("calibration-*.json")
        },
        "instances": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*.txt")
        },
    }
    path = tmp_path / "tiny-reference.json"
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("engine", [ENGINES[0], ENGINES[2]])
def test_real_engine_confirmation_resumes_without_new_selection(
    tmp_path: Path, tiny_reference: Path, engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Actual spawned scoring preserves the candidate and completed rows across a resume.

    Args:
        tmp_path (Path): Work directory parent.
        tiny_reference (Path): Sealed small Rust study.
        engine (str): New engine to score.
        monkeypatch (pytest.MonkeyPatch): Replace the current reporting revision.
    """
    from benchmarks.tuning.engine_confirmation import EngineConfirmation

    run = EngineConfirmation(tmp_path / "replay", tiny_reference)
    selected = json.loads(tiny_reference.read_text())["selection"]
    first = run.run(engine, 1)
    path = run.directory / engine / "test.jsonl"
    before = path.read_bytes()
    assert first["selection"] == selected and first["engine"] == engine
    assert all(json.loads(line)["engine"] == engine for line in before.splitlines())
    assert first == run.run(engine, 1) and path.read_bytes() == before
    monkeypatch.setattr("subprocess.check_output", lambda *args, **kwargs: "another revision")
    assert run.report(engine) == first
    with pytest.raises(ValueError, match="Frozen"):
        run.run(engine, 1)


def test_engine_reference_reconstructs_scales_locks_and_all_acceptance_gates() -> None:
    """The identified engine reference preserves original inputs and reproduces both decisions."""
    import hashlib
    from benchmarks.tuning.preset_analysis import PresetAnalysis
    from benchmarks.tuning.preset_analysis import PresetDesign

    root = Path(__file__).resolve().parents[4] / "benchmarks"
    reference = root / "general-sa-preset-reference.json"
    original = json.loads(reference.read_text())
    data = json.loads((root / "general-sa-preset-engines-reference.json").read_text())
    digest = hashlib.sha256(reference.read_text().encode()).hexdigest()
    assert data["reference_sha256"] == digest
    assert data["selection"] == original["selection"]
    design = PresetDesign(**original["protocol"]["design"])
    for engine in ("python", "cpp"):
        protocol = data["protocols"][engine]
        assert protocol["engine"] == engine and protocol["source_sha"].startswith("1a544ca")
        assert protocol["reference_sha256"] == digest
        assert protocol["design"] == original["protocol"]["design"]
        assert protocol["default"] == original["protocol"]["candidates"]["default"]
        assert protocol["selection"] == original["selection"]
        rows = data["measurements"][engine]
        assert len(rows) == 24 and all(row["engine"] == engine for row in rows)
        result = PresetAnalysis(design).confirmation(rows, data["selection"]["candidate"])
        expected = data["summaries"][engine]
        assert result["passed"] is expected["passed"] is True
        assert result["delta"] == pytest.approx(expected["delta"], abs=1e-12)
        assert result["upper"] == pytest.approx(expected["upper"], abs=1e-12)
        for actual, stored in zip(result["cells"], expected["cells"], strict=True):
            assert actual["passed"] is stored["passed"] is True
            assert actual["upper"] == pytest.approx(stored["upper"], abs=1e-12)
        for row in rows:
            offset = design.offset(row["family"], row["size"])
            path = f"cohort-{offset}/size-{row['size']}/calibration-{row['family']}.json"
            scales = original["calibration"][path]
            for index, (cost, score) in enumerate(zip(row["costs"], row["scores"], strict=True)):
                scale = scales[f"{row['family']}-test-{index}"]
                assert score == pytest.approx(
                    (cost - scale["best"]) / (scale["anchor"] - scale["best"])
                )
