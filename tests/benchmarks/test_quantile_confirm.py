"""F5b Tour-only confirmation checks against the strongest known 2-opt seeded baseline.

Objective:
- Verify the 2-opt seeded baseline arm, the pre-registered decision rule and the frozen
  evidence lifecycle at a tiny, fast scale.

Technical approach:
- Run the real RKO solver on very small tours and fabricate trial rows to test the rule alone.

Flow (pseudocode):
1. Solve one tiny instance with the 2-opt seeded arm and check its independent cost.
2. Apply the decision rule to fabricated rows for win, loss and inconclusive cases.
3. Run the tiny lifecycle and check the frozen manifest, raw trials and output isolation.

Important criteria:
- Seeding with the exact 2-opt tour means the baseline never ends worse than that tour.
- The new test seeds are disjoint from the validation and F5 test seeds.

Inputs/Outputs:
- Input: tiny monkeypatched settings and temporary output directories.
- Output: assertions on baseline quality, gate decisions and persisted evidence.
"""

import json
import numpy as np
import pytest
from benchmarks.engines.tsp_problems import TspFamily
from benchmarks.engines.tsp_problems import TspInstance
from benchmarks.quantile import confirm
from benchmarks.quantile import study
from benchmarks.quantile.instances import Instances
from benchmarks.quantile.model import FrozenScorer
from pathlib import Path


def _scorer(seed: int) -> FrozenScorer:
    """Build a tiny deterministic frozen tour scorer.

    Args:
        seed (int): Weight-initialization seed.

    Returns:
        FrozenScorer: Tiny two-hidden-unit scorer.
    """
    rng = np.random.default_rng(seed)
    return FrozenScorer(
        "tour", rng.normal(size=(4, 2)), np.zeros(2), rng.normal(size=2), np.array(0.0)
    )


def _rows(neural: float, baseline: float) -> list[dict[str, object]]:
    """Fabricate trial rows where each neural arm shares one cost and each baseline another.

    Args:
        neural (float): Base cost of both neural arms.
        baseline (float): Base cost of both 2-opt seeded arms.

    Returns:
        list[dict[str, object]]: One row per arm and instance.
    """
    rows: list[dict[str, object]] = []
    for seed in range(4):
        for arm in confirm.ARMS:
            base = neural if arm in ("genetic", "portfolio") else baseline
            rows.append(
                {"seed_group": seed, "arm": arm, "budget": 60.0, "cost": base + 0.01 * seed}
            )
    return rows


def test_two_opt_arm_never_ends_worse_than_its_own_seed_tour() -> None:
    """Seeding the exact 2-opt tour keeps the baseline at or below that tour's length."""
    task = Instances.make("tour", 12, 5)
    instance = TspInstance("tour-12-5", task.coordinates)
    prior = TspFamily.two_opt(instance, TspFamily.nearest_neighbour(instance))
    prior_cost = TspFamily.length(instance, prior)
    cost, setup, evaluations = confirm.TwoOptArm(task, noise=0.01).run(("BRKGA",), 3, 1.0)
    assert cost <= prior_cost + 1e-9
    assert setup >= 0.0
    assert evaluations > 0


def test_rule_advances_only_when_a_neural_arm_beats_its_baseline_with_a_negative_interval() -> None:
    """A clear neural win advances; a loss or a tie does not."""
    assert confirm.Confirmation.summarize(_rows(5.0, 10.0))["advance"] is True
    assert confirm.Confirmation.summarize(_rows(10.0, 5.0))["advance"] is False
    assert confirm.Confirmation.summarize(_rows(10.0, 10.0))["advance"] is False


def test_summary_reports_both_declared_comparisons() -> None:
    """The summary carries each neural arm against the baseline with the same algorithms."""
    summary = confirm.Confirmation.summarize(_rows(5.0, 10.0))
    assert set(summary["comparisons"]) == {
        "genetic_vs_twoopt_genetic",
        "portfolio_vs_twoopt_portfolio",
    }
    assert summary["comparisons"]["genetic_vs_twoopt_genetic"]["ci_high"] < 0.0


def test_new_test_seeds_are_disjoint_from_every_earlier_range() -> None:
    """No confirmation instance repeats a validation or F5 test seed."""
    used = set(range(study.VALIDATION_SEED_START, study.VALIDATION_SEED_START + 4)) | set(
        range(study.TEST_SEED_START, study.TEST_SEED_START + study.TEST_INSTANCES)
    )
    fresh = set(range(confirm.TEST_SEED_START, confirm.TEST_SEED_START + confirm.TEST_INSTANCES))
    assert not used & fresh


def test_tiny_confirmation_freezes_evidence_and_rejects_reusing_its_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The lifecycle tunes the noise, freezes a manifest, records every trial and stays isolated.

    Args:
        tmp_path (Path): Isolated output root.
        monkeypatch (pytest.MonkeyPatch): Scale-constant override helper.
    """
    monkeypatch.setattr(study, "SIZE", 8)
    monkeypatch.setattr(study, "VALIDATION_SEED_START", 900000)
    monkeypatch.setattr(confirm, "TEST_SEED_START", 900100)
    monkeypatch.setattr(confirm, "TEST_INSTANCES", 2)
    monkeypatch.setattr(confirm, "BUDGET", 0.4)
    monkeypatch.setattr(confirm, "TUNING_BUDGET", 0.3)
    monkeypatch.setattr(confirm, "NOISES", (0.01, 0.1))
    output = tmp_path / "confirm"
    summary = confirm.Confirmation(output, _scorer(11)).run()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["noise"] in (0.01, 0.1)
    assert len(manifest["test_panel"]) == 2
    trials = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
    assert len(trials) == 2 * len(confirm.ARMS)
    assert isinstance(summary["advance"], bool)
    with pytest.raises(FileExistsError):
        confirm.Confirmation(output, _scorer(11)).run()
