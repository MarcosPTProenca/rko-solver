"""Behavioral checks for seed validation, portable artifacts and trusted problem semantics.

Objective:
- Verify artifact integrity, source validation and independently checked problem semantics.
Technical approach:
- Round-trip source without importing it and evaluate hand-worked tiny problem instances.
Flow (pseudocode):
1. Export and reload an artifact, then alter its source bytes.
2. Submit prohibited source forms to static validation.
3. Compare trusted objectives, feasibility and optima with known answers.
Important criteria:
- Generated source is not executed; tampering and invalid witnesses must be rejected.
Inputs/Outputs:
- Input: temporary artifact paths, decoder source strings and tiny problem data.
- Output: assertions on integrity failures and exact trusted evaluation results.
"""

import pytest
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem


def test_artifact_roundtrip_does_not_execute_source(tmp_path: Path) -> None:
    """Export/load source without importing it and reject modified bytes.

    Args:
        tmp_path (Path): Temporary parent for the exported decoder artifact.
    """
    source = 'def decode(keys, instance):\n    """Select items."""\n    return [0]\n'
    artifact = DecoderArtifact.create("knapsack", source)
    destination = tmp_path / "decoder"
    artifact.export(destination)
    loaded = DecoderArtifact.load(destination)
    assert loaded.source == source
    assert loaded.digest == artifact.digest
    assert loaded.status == "experimental"
    (destination / "decoder.py").write_text(source + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        DecoderArtifact.load(destination)


def test_trusted_evaluator_recomputes_knapsack_profit() -> None:
    """Check a hand-calculated feasible witness and reject invalid indices."""
    problem = Problem("knapsack", {"weights": [2, 3, 4], "profits": [3, 5, 6], "capacity": 5})
    assert problem.evaluate([0, 1])["objective"] == 8
    assert problem.optimum() == 8
    assert problem.evaluate([1, 2])["feasible"] is False
    with pytest.raises(ValueError):
        problem.evaluate([0, 0])
    with pytest.raises(ValueError):
        problem.evaluate([True])


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef decode(keys, instance):\n    return []\n",
        "def decode(keys, instance):\n    import os\n    return []\n",
        "def decode(keys):\n    return []\n",
        "def decode(keys, instance):\n    return open('secret').read()\n",
        "def decode(keys, instance):\n    return keys.__class__\n",
        "def decode(keys, instance):\n    decode.state = []\n    return []\n",
        "raise RuntimeError('module executed')\n",
    ],
)
def test_unapproved_source_fails_without_execution(source: str) -> None:
    """Reject prohibited imports, interfaces, introspection and module-scope execution.

    Args:
        source (str): Parametrized decoder source containing a prohibited construct.
    """
    with pytest.raises(ValueError):
        DecoderArtifact.create("knapsack", source)


def test_scheduling_and_cover_have_independent_known_answers() -> None:
    """Check exact references and reconstructed objectives against hand-worked tiny cases."""
    scheduling = Problem("scheduling", {"processing_times": [3, 1], "due_dates": [4, 1]})
    assert scheduling.evaluate([0, 1])["objective"] == 3
    assert scheduling.evaluate([1, 0])["objective"] == 0
    assert scheduling.optimum() == 0
    assert scheduling.constructive() == [1, 0]
    with pytest.raises(ValueError):
        scheduling.evaluate([0])
    cover = Problem(
        "set_cover",
        {"subsets": [[0, 1], [1, 2], [0], [2]], "costs": [3, 3, 1, 1], "universe_size": 3},
    )
    assert cover.optimum() == 4
    assert cover.evaluate([2, 3])["feasible"] is False
    assert cover.evaluate([0, 3])["objective"] == 4
    assert cover.evaluate(cover.constructive())["feasible"] is True
