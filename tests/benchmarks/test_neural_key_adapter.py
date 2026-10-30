"""Behavioral properties of the learned key-distribution adapter.

Objective:
- Verify bounded deterministic key generation and preservation of the original decoder contract.

Technical approach:
- Exercise composition, a protected-route round trip and pickle transport on generated keys.
- Check learning against a non-neural control without running optimization searches.

Flow (pseudocode):
1. Generate an elite archive and fit the key distribution.
2. Transform keys and verify bounds, repeatability and finite outputs.
3. Encode original solutions through the protected route and check decoder equivalence.

Important criteria:
- The model generates keys, not objective predictions.
- Every original solution remains reachable through the protected route.

Inputs/Outputs:
- Input: synthetic elite archives and random-key vectors.
- Output: assertions on generator and decoder behavior.
"""

import numpy as np
import pickle
import pytest
from benchmarks.engines.neural_adapter_study import AdapterDesign
from benchmarks.engines.neural_adapter_study import AdapterStudy
from benchmarks.engines.neural_adapter_study import ArchiveCost
from benchmarks.engines.neural_key_adapter import KeyDistribution
from benchmarks.engines.neural_key_adapter import NeuralKeyAdapter
from benchmarks.engines.surrogate_seed_study import IdentityDecoder
from benchmarks.engines.surrogate_seed_study import ProblemFamily
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from unittest.mock import Mock


def archive(seed: int, width: int = 12) -> np.ndarray:
    """Create a correlated archive within the unit cube.

    Args:
        seed (int): Sampling seed.
        width (int): Number of keys per vector.

    Returns:
        np.ndarray: Forty correlated key vectors.
    """
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(40, 2))
    logits = latent @ rng.normal(size=(2, width)) + rng.normal(size=width)
    return 1.0 / (1.0 + np.exp(-logits))


def test_distribution_learns_bounded_reproducible_keys() -> None:
    """Equal seeds and archives produce equal bounded, nonconstant generated keys."""
    elite = archive(1)
    first = KeyDistribution.fit(elite, 2)
    second = KeyDistribution.fit(elite, 2)
    probes = np.random.default_rng(3).random((20, 12))
    generated = np.array([first.transform(key) for key in probes])
    again = np.array([second.transform(key) for key in probes])
    np.testing.assert_array_equal(generated, again)
    assert generated.shape == probes.shape
    assert np.isfinite(generated).all()
    assert np.all(generated >= 0.0) and np.all(generated < 1.0)
    assert np.max(np.std(generated, axis=0)) > 0.01


@settings(max_examples=50, derandomize=True, deadline=None)
@given(st.lists(st.floats(min_value=0.0, max_value=0.999999), min_size=12, max_size=12))
def test_protected_route_preserves_every_original_key(keys: list[float]) -> None:
    """Encoding through the protected branch reconstructs arbitrary original keys.

    Args:
        keys (list[float]): Original-space key vector.
    """
    adapter = NeuralKeyAdapter(IdentityDecoder(), KeyDistribution.marginal(archive(0)))
    original = np.array(keys)
    np.testing.assert_allclose(adapter(adapter.protected(original)), original, atol=1e-15)


def test_adapter_preserves_the_callers_decoded_type_and_pickle_behavior() -> None:
    """Composition calls the original decoder and is stable across process serialization."""
    adapter = NeuralKeyAdapter(tuple, KeyDistribution.fit(archive(1), 2))
    keys = np.random.default_rng(3).random(12)
    decoded = adapter(keys)
    assert isinstance(decoded, tuple) and len(decoded) == 12
    assert pickle.loads(pickle.dumps(adapter))(keys) == decoded


@pytest.mark.parametrize("uniform", [0.0, 1.0, -0.1, float("nan")])
def test_invalid_uniform_probabilities_are_rejected(uniform: float) -> None:
    """The mixture must retain both the uniform and learned routes.

    Args:
        uniform (float): Invalid mixture probability.
    """
    with pytest.raises(ValueError, match="uniform"):
        NeuralKeyAdapter(tuple, KeyDistribution.marginal(archive(1)), uniform)


@pytest.mark.parametrize("bad", [np.ones((5, 4)), np.full((5, 4), np.nan), np.ones(5)])
def test_invalid_archives_are_rejected(bad: np.ndarray) -> None:
    """Training accepts only finite matrices of keys within the unit cube.

    Args:
        bad (np.ndarray): Malformed elite archive.
    """
    with pytest.raises(ValueError, match="elite"):
        KeyDistribution.fit(bad, 1)


def test_expired_training_deadline_uses_the_non_neural_fallback() -> None:
    """An exhausted training allowance never forces an epoch of neural work."""
    fitted = KeyDistribution.fit(archive(0), 1, deadline=0.0)
    assert fitted.epochs == 0 and fitted.weights is None
    keys = np.random.default_rng(2).random(12)
    np.testing.assert_array_equal(
        fitted.transform(keys), KeyDistribution.marginal(archive(0)).transform(keys)
    )


def test_training_reduces_reconstruction_error_on_correlated_keys() -> None:
    """The neural decoder reconstructs a structured archive better than its mean alone."""
    fitted = KeyDistribution.fit(archive(0), 1)
    assert fitted.epochs == 40 and fitted.weights is not None
    assert fitted.error < 0.8


def test_archive_keeps_best_costs_and_detaches_original_keys() -> None:
    """The bounded recorder keeps its own keys without retaining decoded values."""
    problem = ProblemFamily.make("sphere", 12, 1)
    recorder = ArchiveCost(problem, 5)
    keys = np.random.default_rng(3).random((20, 12))
    values = [recorder(key) for key in keys]
    expected = np.argsort(values)[:5]
    np.testing.assert_array_equal(recorder.rows(), keys[expected])
    keys[:] = 0.0
    assert np.any(recorder.rows() != 0.0) and len(recorder.heap) == 5


def test_real_solver_accepts_the_composed_decoder_and_retains_a_warm_start() -> None:
    """Exercise decoder composition through the public solve entry point with bounded work."""
    problem = ProblemFamily.make("sphere", 12, 1)
    rows = archive(3)
    adapter = NeuralKeyAdapter(problem.decoder, KeyDistribution.fit(rows, 2))
    initial = adapter.protected(rows[0])
    solver = AdapterStudy.solver(problem, adapter, 3, [initial.tolist()])
    solver.max_evaluations = 10
    result = solver.solve(1.0)
    assert result.initial_keys is not None and result.initial_keys.n_evaluations == 1
    assert result.cost <= problem.cost(problem.decoder(rows[0])) + 1e-12
    assert result.cost == pytest.approx(problem.cost(adapter(np.asarray(result.keys))))


def test_exhausted_total_allowance_preserves_warmup_without_extra_search() -> None:
    """The study has no forced minimum search slice after preparation exhausts its budget."""
    problem = ProblemFamily.make("sphere", 12, 1)
    elite = archive(2)
    incumbent = float(problem.cost(problem.decoder(elite[0])))
    row = AdapterStudy(AdapterDesign(budget=0.1)).trial(
        problem, (elite, incumbent, 1.0), "seeded", 1
    )
    assert row["cost"] == incumbent and row["search_allowance"] == 0.0


def test_seed_deployment_returns_to_the_original_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generate proposals once but never run learned inference inside the final search.

    Args:
        monkeypatch (pytest.MonkeyPatch): Installs a forwarding spy on the real solver factory.
    """
    problem = ProblemFamily.make("sphere", 12, 1)
    elite = archive(2)
    costs = [problem.cost(problem.decoder(row)) for row in elite]
    elite = elite[np.argsort(costs)]
    design = AdapterDesign(size=12, budget=0.1, generator="coupling", deployment="seeds")
    study = AdapterStudy(design)
    factory = Mock(wraps=AdapterStudy.solver)
    monkeypatch.setattr(study, "solver", factory)
    row = study.trial(problem, (elite, float(min(costs)), 0.001), "neural", 2)
    assert row["verified"] and row["cost"] <= min(costs)
    assert factory.call_args.args[1] is problem.decoder


def test_explicit_preparation_keeps_the_post_preparation_search_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit preparation adds to the total charge rather than erasing the final allowance.

    Args:
        monkeypatch (pytest.MonkeyPatch): Installs a forwarding spy on real solver construction.
    """
    problem = ProblemFamily.make("sphere", 12, 1)
    elite = archive(1)
    incumbent = float(problem.cost(problem.decoder(elite[0])))
    design = AdapterDesign(size=12, budget=0.05, preparation_seconds=1.0)
    study = AdapterStudy(design)
    factory = Mock(wraps=AdapterStudy.solver)
    monkeypatch.setattr(study, "solver", factory)
    row = study.trial(problem, (elite, incumbent, 0.9), "seeded", 1)
    assert row["verified"] and row["search_evaluations"] > 0
    assert row["search_allowance"] > 0.05
    assert row["charged_seconds"] >= 0.9


def test_archive_never_fills_with_copies_of_a_stalled_incumbent() -> None:
    """Re-evaluating the same good vector must not evict distinct elite vectors."""
    problem = ProblemFamily.make("sphere", 12, 1)
    recorder = ArchiveCost(problem, 8)
    keys = np.random.default_rng(4).random((8, 12))
    for key in keys:
        recorder(key)
    best = keys[np.argmin([problem.cost(problem.decoder(key)) for key in keys])]
    for _ in range(50):
        recorder(best.copy())
    assert len(recorder.heap) == 8
    assert len(np.unique(recorder.rows(), axis=0)) == 8


def test_warmup_search_uses_an_independent_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The warm-up must not replay the final search's own trajectory.

    Args:
        monkeypatch (pytest.MonkeyPatch): Installs a forwarding spy on solver construction.
    """
    problem = ProblemFamily.make("sphere", 12, 1)
    study = AdapterStudy(AdapterDesign(size=12, budget=0.1, preparation_seconds=0.1))
    factory = Mock(wraps=AdapterStudy.solver)
    monkeypatch.setattr(study, "solver", factory)
    study.warmup(problem, 5)
    assert factory.call_args.args[2] != 5
