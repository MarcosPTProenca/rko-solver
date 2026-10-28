"""Verify fixed-work IPR diagnostic accounting and isolated diversity controls.

Objective:
- Keep initialization, path observations and budget cuts reproducible and honest.

Technical approach:
- Exercise the maintained probe through real SearchSession and production IPR.
- Test raw and exact-vector-controlled pools without changing production policies.

Flow (pseudocode):
1. Load the diagnostic module and vary workload, initialization and evaluation cap.
2. Check every completed objective is charged once and seed prefixes match.
3. Verify repeatability, tied-vector admission and the one-differing-block trace.

Important criteria:
- Controls are not production fixes or measurements of Knapsack feasibility.
- A cut during initialization or a partial path still returns a finite incumbent.

Inputs/Outputs:
- Input: tiny deterministic matrices and controlled pool entries.
- Output: accounting, ownership and independent known-count assertions.
"""

import importlib
import numpy as np
import pickle
import pytest
import random
from pathlib import Path
from types import ModuleType


@pytest.fixture
def diagnosis(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the maintained benchmark module without a runtime package dependency.

    Args:
        monkeypatch (pytest.MonkeyPatch): Temporarily expose the benchmark directory.

    Returns:
        ModuleType: Fixed-work IPR diagnostic definitions.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    return importlib.import_module("ipr_diagnosis")


@pytest.mark.parametrize("workload", ["sphere", "threshold", "flat"])
@pytest.mark.parametrize("limit", [2, 3, 11, 64])
@pytest.mark.parametrize("prefill", [0, 10])
@pytest.mark.parametrize("distinct", [False, True])
def test_probe_counts_every_evaluation(
    diagnosis: ModuleType, workload: str, limit: int, prefill: int, distinct: bool
) -> None:
    """Charge seeds and partial paths to the same exact objective cap.

    Args:
        diagnosis (ModuleType): Probe implementation.
        workload (str): Numerical or discrete objective.
        limit (int): Budget, including cuts inside external initialization.
        prefill (int): Normal bootstrap or ten initial elites.
        distinct (bool): Isolated duplicate-admission control.
    """
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    row = diagnosis.IPRProbe(workload, 200, 20, limit, prefill, distinct_pool=distinct).run()
    assert row["evaluations"] == limit
    assert row["seed_evaluations"] == min(limit, 11 if prefill else 2)
    assert row["seed_evaluations"] + row["path_evaluations"] == limit
    assert row["endpoint_evaluations"] + row["intermediate_evaluations"] == row["path_evaluations"]
    assert row["best_cost"] <= row["best_seed_cost"]
    assert row["unique_pool_vectors"] <= row["pool_size"] <= 10
    assert row["unique_pool_decoded"] <= row["unique_pool_vectors"]
    assert row["path_outside_seed_alleles"] == 0
    assert row["equal_parent_pairs"] == 0
    assert row["unique_pool_vectors"] == row["pool_size"]
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


def test_replay_and_seed_prefix(diagnosis: ModuleType) -> None:
    """Repeat traces exactly and compare initialization prefixes across all four arms.

    Args:
        diagnosis (ModuleType): Probe implementation.
    """
    prefixes = set()
    for prefill in (0, 10):
        for distinct in (False, True):
            rows = [
                diagnosis.IPRProbe("sphere", 200, 32, 1024, prefill, distinct_pool=distinct).run()
                for _ in range(2)
            ]
            assert rows[0] == rows[1]
            assert rows[0]["unique_pool_vectors"] == rows[0]["pool_size"]
            assert rows[0]["equal_parent_pairs"] == 0
            prefixes.add(rows[0]["seed_prefix_sha256"])
    assert len(prefixes) == 1


def test_distinct_control_keeps_different_tied_vectors(diagnosis: ModuleType) -> None:
    """Reject exact clones, not different chromosomes with equal objective values.

    Args:
        diagnosis (ModuleType): Probe implementation.
    """
    pool = diagnosis.DistinctVectorPool(10)
    pool.insert((0.0, [0.1, 0.2]))
    pool.insert((0.0, [0.1, 0.2]))
    pool.insert((0.0, [0.2, 0.1]))
    assert pool.size == 2


def test_single_differing_block_has_only_two_unique_vectors(diagnosis: ModuleType) -> None:
    """Check the actual pruned kernel, not merely an analytical callback estimate.

    Args:
        diagnosis (ModuleType): Probe implementation.
    """
    row = diagnosis.single_block_path()
    assert row["evaluations"] == 2
    assert row["unique_vectors"] == 2
    assert row["start_evaluations"] == 1
    assert row["guide_evaluations"] == 1
    assert row["returned_guide"]


def test_pruned_path_stops_at_next_changed_callback(diagnosis: ModuleType) -> None:
    """Preserve the callback deadline boundary after virtual no-op decisions.

    Args:
        diagnosis (ModuleType): Production session and algorithm definitions.
    """

    def score(keys: np.ndarray) -> float:
        """Expire after the first finite result without allowing another objective call.

        Args:
            keys (np.ndarray): Initial observed chromosome.

        Returns:
            float: Squared norm.
        """
        session.deadline = -1.0
        return float(np.dot(keys, keys))

    session = diagnosis.SearchSession(lambda keys: keys, score)
    algorithm = diagnosis.IPR(diagnosis.SolutionPool(), session.decode, session.evaluate, 20)
    start = np.full(20, 0.8)
    guide = start.copy()
    guide[-2:] = 0.2
    with pytest.raises(diagnosis._BudgetExpired):
        algorithm._relink(start, guide)
    assert session.n_evaluations == 1 and session.best is not None
    assert np.array_equal(session.best[0], start)
    assert np.array_equal(start, np.full(20, 0.8))


def test_unpruned_ipr_is_not_an_rng_only_baseline(diagnosis: ModuleType) -> None:
    """Classify intentional callback-budget changes separately from RNG isolation.

    Args:
        diagnosis (ModuleType): Current production constructor.
    """

    class Unpruned:
        """Represent an old constructor without the pruned-path compatibility token.

        Responsibilities:
        - Supply historical diagnostic identity, not an executable solver.
        """

    replay = importlib.import_module("random_streams")
    assert replay.replay_modes(diagnosis.IPR, Unpruned) == ()
    assert replay.replay_modes(diagnosis.IPR, diagnosis.IPR) == (0,)


def test_scalar_copy_control_uses_same_pruned_policy(diagnosis: ModuleType) -> None:
    """Keep copy-kernel comparisons separate from changes to evaluated candidates.

    Args:
        diagnosis (ModuleType): Probe module with the production algorithm.
    """
    scalar = importlib.import_module("ipr").ScalarIPR
    traces: list[list[bytes]] = []
    results = []
    start = np.full(20, 0.8)
    guide = start.copy()
    guide[:2] = 0.2
    for constructor in (diagnosis.IPR, scalar):
        trace: list[bytes] = []

        def score(keys: np.ndarray, captured: list[bytes] = trace) -> float:
            """Capture callback representations while evaluating a deterministic norm.

            Args:
                keys (np.ndarray): Actual copy-kernel candidate.
                captured (list[bytes]): Snapshot list bound to this copy-kernel run.

            Returns:
                float: Squared norm.
            """
            captured.append(keys.tobytes())
            return float(np.dot(keys, keys))

        algorithm = constructor(diagnosis.SolutionPool(), lambda keys: keys, score, 20)
        keys, cost = algorithm._relink(start, guide)
        results.append((keys.tobytes(), cost))
        traces.append(trace)
    assert results[0] == results[1]
    assert traces[0] == traces[1] and len(traces[0]) == 2


def test_probe_detects_stalled_callback_progress(
    diagnosis: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail after bounded outer checks if relinking ceases to consume its work cap.

    Args:
        diagnosis (ModuleType): Probe implementation.
        monkeypatch (pytest.MonkeyPatch): Simulate a path that never invokes callbacks.
    """
    probe = diagnosis.IPRProbe("sphere", 200, 20, 4, 0)
    monkeypatch.setattr(probe, "_relink", lambda start, guide: (start, 0.0))
    with pytest.raises(AssertionError, match="stalled"):
        probe.run()


@pytest.mark.parametrize("dimension,limit,prefill", [(0, 2, 0), (1, 1, 0), (1, 2, 1), (1, 2, 11)])
def test_invalid_probe_bounds(
    diagnosis: ModuleType, dimension: int, limit: int, prefill: int
) -> None:
    """Reject unsupported caps and initialization shapes before search begins.

    Args:
        diagnosis (ModuleType): Probe implementation.
        dimension (int): Requested shape.
        limit (int): Requested evaluation cap.
        prefill (int): Requested initialization count.
    """
    with pytest.raises(ValueError):
        diagnosis.IPRProbe("sphere", 200, dimension, limit, prefill)


@pytest.mark.parametrize("dimension", [1, 8, 21])
def test_small_and_partial_dimensions(diagnosis: ModuleType, dimension: int) -> None:
    """Account for one-gene and partial-tail paths under the same evaluation cap.

    Args:
        diagnosis (ModuleType): Probe implementation.
        dimension (int): Positive shape exercising relinking edge cases.
    """
    assert diagnosis.IPRProbe("sphere", 200, dimension, 32, 0).run()["evaluations"] == 32
