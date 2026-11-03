"""Bounded training lifecycle, manifest and checkpoint qualification checks.

Objective:
- Stop exhausted runs without inventing a successful experiment or extra search allowance.

Technical approach:
- Exercise the actual tiny training lifecycle and replace only the external processor clock.

Flow (pseudocode):
1. Freeze a new run's data and numerical settings.
2. Execute a tiny training and held-out qualification cycle.
3. Check exhaustion and existing-output rejection.

Important criteria:
- These are lifecycle checks, not evidence of useful optimizer performance.

Inputs/Outputs:
- Input: small explicit settings and temporary output directories.
- Output: completed records, valid model artifacts and explicit budget status.
"""

import json
import pytest
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.training import run
from pathlib import Path


def test_tiny_training_run_freezes_inputs_and_qualifies_export(tmp_path: Path) -> None:
    """The real pipeline produces separate tuning/qualification panels and a valid model.

    Args:
        tmp_path (Path): Isolated run-output root.
    """
    output = tmp_path / "pilot"
    settings = run.TrainingSettings("tour", size=4, steps=1, cpu_seconds=60.0, seed=11)
    report = run.TrainingRun(settings, output).run()
    assert report["status"] == "complete"
    assert report["completed_updates"] == 1
    assert isinstance(report["advance"], bool)
    scorer = FrozenScorer.load(output / "best-model.json")
    assert scorer.fingerprint == report["model_fingerprint"]
    manifest = json.loads((output / "manifest.json").read_bytes())
    tuning = {row["instance_hash"] for row in manifest["panels"]["tune"]}
    qualification = {row["instance_hash"] for row in manifest["panels"]["qualification"]}
    assert tuning.isdisjoint(qualification)
    assert report["export_parity"]["checked_states"] > 0
    assert report["export_parity"]["decision_mismatches"] == 0
    with pytest.raises(FileExistsError):
        run.TrainingRun(settings, output).run()


def test_exhausted_processor_budget_is_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exhausted processor clock cannot become an apparently complete training result.

    Args:
        tmp_path (Path): Isolated run-output root.
        monkeypatch (pytest.MonkeyPatch): Clock-boundary replacement helper.
    """
    monkeypatch.setattr(run.time, "process_time", lambda: 10.0)
    settings = run.TrainingSettings("cover", size=4, steps=1, cpu_seconds=1.0)
    report = run.TrainingRun(settings, tmp_path / "exhausted").run()
    assert report["status"] == "budget_exhausted"
    assert report["completed_updates"] == 0
    assert report["advance"] is False


def test_comparisons_reject_duplicated_candidate_instances() -> None:
    """A repeated candidate row cannot masquerade as another paired instance."""
    row = run.Evaluation(1, "one-instance", 1.0, 1.0, 1.0, 1)
    with pytest.raises(ValueError, match="distinct"):
        run.TrainingRun.compare([row, row], [row])
