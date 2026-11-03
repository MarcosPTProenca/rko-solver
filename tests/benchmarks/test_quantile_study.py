"""End-to-end F5 screen lifecycle checks at a tiny, fast-executing scale.

Objective:
- Verify headroom gating, frozen evidence and the paired exit-gate rule, without
  running the full declared matrix.

Technical approach:
- Monkeypatch the module's size/budget/panel constants to tiny values and use
  tiny deterministic frozen scorers, so the real RKO solver still runs but
  completes in a fraction of a second per trial.

Flow (pseudocode):
1. Check that a scorer failing validation headroom stops before freezing a panel.
2. Run the full tiny lifecycle and check the manifest, trials and summary shapes.
3. Check the paired exit-gate rule directly on synthetic comparison tables.

Important criteria:
- A fresh output directory is required; rerunning the same directory must fail.
- The summary's `advance` flag matches the declared per-cell win rule exactly.

Inputs/Outputs:
- Input: tiny monkeypatched settings and temporary output directories.
- Output: assertions on manifest/trial/summary contents and gate decisions.
"""

import json
import numpy as np
import pytest
from benchmarks.quantile import study
from benchmarks.quantile.model import FrozenScorer
from pathlib import Path


def _scorer(family: str, seed: int) -> FrozenScorer:
    """Build a tiny deterministic frozen scorer for one feature family.

    Args:
        family (str): 'tour' or 'cover'.
        seed (int): Weight-initialization seed.

    Returns:
        FrozenScorer: Tiny two-hidden-unit scorer.
    """
    rng = np.random.default_rng(seed)
    return FrozenScorer(
        family, rng.normal(size=(4, 2)), np.zeros(2), rng.normal(size=2), np.array(0.0)
    )


def test_headroom_failure_stops_before_any_panel_is_touched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A scorer worse than the analytic reference never reaches test-panel freezing.

    Args:
        tmp_path (Path): Isolated output root.
        monkeypatch (pytest.MonkeyPatch): Scale-constant override helper.
    """
    monkeypatch.setattr(study, "SIZE", 4)
    bad = FrozenScorer(
        "tour",
        np.array([[0.1, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
        np.zeros(2),
        np.array([50.0, 0.0]),
        np.array(0.0),
    )
    with pytest.raises(study.HeadroomFailure, match="tour"):
        study.HeadroomCheck.run("tour", bad, instances=4)


def test_tiny_screen_runs_end_to_end_and_rejects_reusing_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The full lifecycle produces a manifest, raw trials and a well-shaped summary.

    Args:
        tmp_path (Path): Isolated output root.
        monkeypatch (pytest.MonkeyPatch): Scale-constant override helper.
    """
    monkeypatch.setattr(study, "SIZE", 4)
    monkeypatch.setattr(study, "BUDGETS", (0.3,))
    monkeypatch.setattr(study, "RANDOMIZATIONS", 1)
    monkeypatch.setattr(study, "TEST_INSTANCES", 2)
    monkeypatch.setattr(study, "VALIDATION_SEED_START", 900000)
    monkeypatch.setattr(study, "TEST_SEED_START", 900100)
    monkeypatch.setattr(
        study.HeadroomCheck, "run", staticmethod(lambda family, scorer, instances: -0.01)
    )
    checkpoints = {"tour": _scorer("tour", 11), "cover": _scorer("cover", 13)}
    output = tmp_path / "screen"
    summary = study.Screen(output, checkpoints).run()
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["test_panel"]) == {"tour", "cover"}
    assert len(manifest["test_panel"]["tour"]) == 2
    trials = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    assert len(trials) == 2 * 2 * len(study.ARMS) * 1 * 1
    assert set(summary["cells"]) == {"tour@0.3s", "cover@0.3s"}
    assert isinstance(summary["advance"], bool)
    with pytest.raises(FileExistsError):
        study.Screen(output, checkpoints).run()


def test_summarize_advances_only_when_every_cell_wins() -> None:
    """A clean structured-search win in every cell, and only then, yields `advance: true`.

    Uses directly fabricated trial rows so the exit-gate rule is checked in
    isolation from the RKO solver and scorer fitting.
    """
    rows = []
    for family in study.FAMILIES:
        for budget in study.BUDGETS:
            for seed in range(3):
                costs = {
                    "classical": 10.0,
                    "independent": 10.0,
                    "genetic": 5.0,
                    "portfolio": 4.0,
                }
                for arm, base in costs.items():
                    rows.append(
                        {
                            "family": family,
                            "seed_group": seed,
                            "arm": arm,
                            "budget": budget,
                            "cost": base + 0.01 * seed,
                        }
                    )
    summary = study.Screen.summarize(rows)
    assert summary["advance"] is True
    assert len(summary["wins"]) == len(study.FAMILIES) * len(study.BUDGETS)


def test_summarize_does_not_advance_when_classical_beats_structured_search() -> None:
    """A classical construction win over genetic/portfolio blocks the advance decision."""
    rows = []
    for family in study.FAMILIES:
        for budget in study.BUDGETS:
            for seed in range(3):
                costs = {
                    "classical": 1.0,
                    "independent": 10.0,
                    "genetic": 5.0,
                    "portfolio": 4.0,
                }
                for arm, base in costs.items():
                    rows.append(
                        {
                            "family": family,
                            "seed_group": seed,
                            "arm": arm,
                            "budget": budget,
                            "cost": base + 0.01 * seed,
                        }
                    )
    summary = study.Screen.summarize(rows)
    assert summary["advance"] is False
    assert summary["wins"] == []
