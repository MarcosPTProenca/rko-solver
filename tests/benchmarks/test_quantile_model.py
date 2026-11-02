"""Safe, frozen numerical-model artifacts for contextual action scoring.

Objective:
- Qualify data-only weights without importing a neural framework into the solver environment.

Technical approach:
- Compare a literal four-array network with known scalar calculations and reload its artifact.

Flow (pseudocode):
1. Freeze validated network arrays and evaluate legal-action features.
2. Export a bounded numerical artifact.
3. Reload and verify its identity, shape and output.

Important criteria:
- Mutating caller arrays must not alter a frozen model.

Inputs/Outputs:
- Input: small floating-point arrays and feature rows.
- Output: deterministic scores and data-only artifact checks.
"""

import json
import numpy as np
import pickle
import pytest
from benchmarks.quantile import model
from benchmarks.quantile.codec import TrajectoryCodec
from benchmarks.quantile.tasks import Tour
from pathlib import Path


def test_known_network_exports_without_a_neural_runtime(tmp_path: Path) -> None:
    """A frozen identity hidden layer computes tanh and preserves its artifact identity.

    Args:
        tmp_path (Path): Isolated artifact directory.
    """
    w1 = np.eye(4)
    scorer = model.FrozenScorer(
        "tour", w1, np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0]), np.array(0.0)
    )
    features = np.array([[2.0, 0.0, 0.0, 0.0]])
    np.testing.assert_allclose(scorer.scores(features), [np.tanh(2.0)])
    w1[:] = 0.0
    np.testing.assert_allclose(scorer.scores(features), [np.tanh(2.0)])
    path = tmp_path / "scorer.json"
    scorer.save(path)
    restored = model.FrozenScorer.load(path)
    assert restored.fingerprint == scorer.fingerprint
    np.testing.assert_array_equal(restored.scores(features), scorer.scores(features))


@pytest.mark.parametrize(
    "field,value",
    [
        ("w1", np.ones((3, 2))),
        ("w1", np.ones((4, 65))),
        ("b1", np.ones(3)),
        ("w2", np.array([float("nan"), 1.0])),
        ("b2", np.zeros(1)),
    ],
)
def test_invalid_model_arrays_are_rejected(field: str, value: np.ndarray) -> None:
    """Invalid dimensions and nonfinite weights never enter a frozen scoring model.

    Args:
        field (str): Parameter to corrupt.
        value (np.ndarray): Invalid replacement parameter.
    """
    weights = {"w1": np.ones((4, 2)), "b1": np.zeros(2), "w2": np.ones(2), "b2": np.array(0.0)}
    weights[field] = value
    with pytest.raises(ValueError):
        model.FrozenScorer("tour", **weights)


def test_unknown_family_is_rejected() -> None:
    """A weight artifact cannot silently claim an unsupported feature family."""
    with pytest.raises(ValueError, match="family"):
        model.FrozenScorer("unknown", np.ones((4, 1)), np.zeros(1), np.ones(1), np.array(0.0))


def test_tampered_or_oversized_artifacts_are_rejected(tmp_path: Path) -> None:
    """Fingerprint, schema and bounded-file checks precede use of exported weights.

    Args:
        tmp_path (Path): Isolated artifact directory.
    """
    scorer = model.FrozenScorer("tour", np.ones((4, 1)), np.zeros(1), np.ones(1), np.array(0.0))
    path = tmp_path / "weights.json"
    scorer.save(path)
    record = json.loads(path.read_bytes())
    record["weights"]["b2"] = 3.0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="fingerprint"):
        model.FrozenScorer.load(path)
    path.write_text('{"format": "not-supported"}')
    with pytest.raises(ValueError, match="schema"):
        model.FrozenScorer.load(path)
    path.write_bytes(b" " * 1000001)
    with pytest.raises(ValueError, match="size"):
        model.FrozenScorer.load(path)


def test_feature_validation_clipping_and_serialized_immutability() -> None:
    """Qualified weights stay immutable across worker serialization and bound extreme logits."""
    scorer = model.FrozenScorer(
        "cover", np.ones((4, 1)), np.zeros(1), np.array([100.0]), np.array(0.0)
    )
    np.testing.assert_array_equal(scorer.scores(np.array([[1.0] * 4, [-1.0] * 4])), [10.0, -10.0])
    with pytest.raises(ValueError, match="features"):
        scorer.scores(np.full((1, 4), float("nan")))
    restored = pickle.loads(pickle.dumps(scorer))
    with pytest.raises(ValueError):
        restored.w1.setflags(write=True)
    assert restored.fingerprint == scorer.fingerprint


def test_frozen_contextual_scores_drive_construction_and_its_inverse() -> None:
    """A learned distance preference changes selection and still replays every trace exactly."""
    task = Tour(np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]]))
    scorer = model.FrozenScorer(
        "tour",
        np.array([[1.0], [0.0], [0.0], [0.0]]),
        np.zeros(1),
        np.array([-10.0]),
        np.array(0.0),
    )
    mapping = TrajectoryCodec(task, scorer=scorer)
    assert mapping.decode(np.array([0.6, 0.5])).trace == (1, 2)
    for trace in ((1, 2), (2, 1)):
        assert mapping.decode(mapping.encode(trace)).trace == trace
    with pytest.raises(ValueError, match="family"):
        TrajectoryCodec(
            task, scorer=model.FrozenScorer("cover", scorer.w1, scorer.b1, scorer.w2, scorer.b2)
        )


def test_linear_same_feature_control_is_explicit() -> None:
    """The cheap control uses a declared four-feature linear rule, not a neural model."""
    scorer = model.LinearRule("tour", (-10.0, 0.0, 0.0, 0.0))
    np.testing.assert_array_equal(scorer.scores(np.array([[0.2, 1.0, 1.0, 1.0]])), [-2.0])
