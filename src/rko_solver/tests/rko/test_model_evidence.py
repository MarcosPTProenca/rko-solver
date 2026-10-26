"""Completed generated-model evidence across search and execution boundaries.

Objective:
- Preserve feasible original-objective witnesses that score-based incumbents discard.

Technical approach:
- Drive deterministic rejected candidates through SearchSession's normal completion gate.
- Exercise owned snapshots through serial, thread and real spawn process execution.

Flow (pseudocode):
1. Generate candidates whose original-objective and score orderings disagree.
2. Retain only successfully completed search and feasible incumbents.
3. Merge bounded startup/worker evidence without invoking the decoder again.

Important criteria:
- Search scores and feasible witnesses keep separate invocation identities.
- These are private integration tests; frontend/public-API qualification is separate.

Inputs/Outputs:
- Input: tiny owned models, deterministic candidates and normal execution settings.
- Output: assertions for bounded witness transport, counting and failure propagation.
"""

import math
import numpy as np
import pytest
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import LinearRow
from rko_solver.modeling.linear import LinearVariable
from rko_solver.modeling.linear import ModelEvidence
from rko_solver.orchestration.initialization import initialize_pool
from rko_solver.orchestration.search import SearchSession
from rko_solver.orchestration.search import _BudgetExpired
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn
from rko_solver.orchestration.types import PoolInitializationMetadata
from rko_solver.orchestration.types import RunConfig
from rko_solver.orchestration.types import RunResult
from rko_solver.orchestration.types import SearchMetadata
from rko_solver.solver import RKO


class RejectedCandidates:
    """Evaluate scripted chromosomes but report the first one as algorithm incumbent.

    Responsibilities:
    - Simulate rejection by an algorithm without publishing candidates into an elite pool.

    Attributes:
        name (str): Test-only algorithm label.
        decoder (DecoderFn): Session-replaced generated decoder.
        cost (CostFn): Session-replaced cost evaluator.
        keys (tuple[float, ...]): One-dimensional chromosomes to evaluate in order.
    """

    name = "rejected"
    _cancelled = None

    def __init__(self, model: CompiledLinearModel, keys: tuple[float, ...]) -> None:
        """Retain generated callbacks and the exact evaluation sequence.

        Args:
            model (CompiledLinearModel): Owned model snapshot.
            keys (tuple[float, ...]): Scalar keys evaluated in order.
        """
        self.decoder: DecoderFn = model.decode
        self.cost: CostFn = model.cost
        self.keys = keys

    def solve(self, duration: float) -> RunResult:
        """Evaluate every scripted candidate but intentionally keep only the first.

        Args:
            duration (float): Unused cooperative allowance supplied by SearchSession.

        Returns:
            RunResult: First candidate, regardless of later score/objective improvements.
        """
        costs = [self.cost(self.decoder(np.array([key]))) for key in self.keys]
        return [], [self.keys[0]], costs[0]


@pytest.mark.parametrize("maximize", [False, True])
def test_feasible_witness_survives_rejection_and_score_disagreement(maximize: bool) -> None:
    """Archive a rejected feasible candidate even when its penalty score is worse.

    Args:
        maximize (bool): Original objective direction.
    """
    coefficient = -10 if maximize else 10
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1),),
        (coefficient,),
        (LinearRow("feasible_only_at_one", ((0, 1),), 1, 1),),
        scoring="weighted_penalty",
        penalty_weight=0.1,
        maximize=maximize,
    )
    session = SearchSession(model.decode, model.cost)
    result = session.solve(RejectedCandidates(model, (0, 1)), 10)
    evidence = session.model_evidence
    assert session.n_evaluations == 2
    assert evidence is not None
    assert evidence.incumbent.keys == (0,)
    assert not evidence.incumbent.feasible
    assert evidence.best_feasible is not None
    assert evidence.best_feasible.keys == (1,)
    assert evidence.best_feasible.objective == coefficient
    assert evidence.best_feasible.score > evidence.incumbent.score
    merged = model.merge_evidence((evidence,), result[1], result[2])
    assert merged == evidence


@pytest.mark.parametrize("maximize", [False, True])
def test_raw_objective_archive_improves_despite_normalization_tie(maximize: bool) -> None:
    """Compare feasible original objectives directly when float-normalized scores tie.

    Args:
        maximize (bool): Original objective direction.
    """
    model = CompiledLinearModel(
        (LinearVariable("wide", "wide", -1e90, 1e90), LinearVariable("tiny", "tiny", 0, 1)),
        (1, 1),
        maximize=maximize,
    )
    session = SearchSession(model.decode, model.cost)
    first, improved = (0.1, 0.2) if maximize else (0.2, 0.1)
    for key in (first, improved):
        session.evaluate(session.decode(np.array([0.5, key])))
    evidence = session.model_evidence
    assert evidence is not None
    assert evidence.best_feasible is not None
    assert evidence.incumbent.score == evidence.best_feasible.score
    assert evidence.incumbent.keys == (0.5, first)
    assert evidence.best_feasible.keys == (0.5, improved)
    assert evidence.best_feasible.objective == improved


def test_only_completed_finite_evaluations_are_eligible() -> None:
    """Exclude decoded-only, nonfinite and deadline-interrupted candidates from archives."""
    model = CompiledLinearModel((LinearVariable("x", "x", 0, 1),), (1,))
    session = SearchSession(model.decode, model.cost)
    decoded = session.decode(np.array([0]))
    assert session.model_evidence is None
    session.cost = lambda _: math.nan
    with pytest.raises(ValueError, match="finite costs"):
        session.evaluate(decoded)
    assert session.n_evaluations == 0
    assert session.model_evidence is None
    session.cost = model.cost
    session.evaluate(session.decode(np.array([1])))
    pending = session.decode(np.array([0.5]))
    session.deadline = 0
    with pytest.raises(_BudgetExpired):
        session.evaluate(pending)
    assert session.n_evaluations == 1
    assert session.model_evidence.incumbent.keys == (1,)


def test_evidence_resets_and_ordinary_callbacks_have_no_model_witnesses() -> None:
    """Reset archives across fresh searches and keep ordinary callback payloads unchanged."""
    model = CompiledLinearModel((LinearVariable("x", "x", 0, 1),), (1,))
    session = SearchSession(model.decode, model.cost)
    session.solve(RejectedCandidates(model, (0.1,)), 10)
    session.solve(RejectedCandidates(model, (0.9,)), 10)
    assert session.model_evidence.incumbent.keys == (0.9,)
    plain = SearchSession(lambda keys: keys, lambda values: float(values[0]))
    assert plain.evaluate(plain.decode(np.array([0.3]))) == 0.3
    assert plain.model_evidence is None
    assert plain.n_evaluations == 1


def test_merge_requires_actual_identity_and_never_replays(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject missing evidence instead of recreating it from returned search keys.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replace decoding after completed work.
    """
    model = CompiledLinearModel((LinearVariable("x", "x", 0, 1),), (1,))
    first = SearchSession(model.decode, model.cost)
    second = SearchSession(model.decode, model.cost)
    first.evaluate(first.decode(np.array([0.8])))
    second.evaluate(second.decode(np.array([0.2])))

    def forbidden(keys: np.ndarray) -> None:
        """Fail if merging attempts to recreate a discarded witness.

        Args:
            keys (np.ndarray): Unexpected replay chromosome.

        Raises:
            AssertionError: Always, because replay violates the completion contract.
        """
        raise AssertionError("Decoder replay is forbidden")

    monkeypatch.setattr(CompiledLinearModel, "decode", forbidden)
    archives = (first.model_evidence, second.model_evidence)
    merged = model.merge_evidence(archives, [0.8], 0.4)
    assert merged.incumbent.keys == (0.8,)
    assert merged.best_feasible.keys == (0.2,)
    assert merged.first_feasible_at == first.model_evidence.first_feasible_at
    assert merged.first_feasible_at <= second.model_evidence.first_feasible_at
    with pytest.raises(ValueError, match="no matching completed"):
        model.merge_evidence(archives, [0.3], 0.15)


@pytest.mark.parametrize("execution", ["sequential", "thread", "process"])
@pytest.mark.parametrize("prefill", [False, True])
def test_bounded_evidence_survives_execution_backends(execution: str, prefill: bool) -> None:
    """Collect independent bounded records through startup, threads and spawn workers.

    Args:
        execution (str): Existing execution execution.
        prefill (bool): Whether to include separately accounted startup evaluation evidence.
    """
    model = CompiledLinearModel(
        (LinearVariable("x", "x", -2, 2),),
        (1,),
        objective_constant=-3,
    )
    solver = RKO(model.decode, model.cost, 1, execution=execution, n_jobs=1, random_state=42)
    evidence: list[ModelEvidence] = []
    searches: list[SearchMetadata] = []
    initialization: list[PoolInitializationMetadata] = []
    config = RunConfig(0, 2, ("MultiStart", "SA"), prefill_pool=prefill)
    keys, score = solver._execute(
        config,
        0.05,
        started_at=0.0,
        search_metadata=searches,
        initialization_metadata=initialization,
        model_evidence=evidence,
    )
    assert 1 <= len(evidence) <= 2 + int(prefill)
    assert all(archive.best_feasible is not None for archive in evidence)
    assert all(search.n_evaluations > 0 for search in searches)
    assert len(initialization) == int(prefill)
    merged = model.merge_evidence(tuple(evidence), keys, score)
    assert merged.incumbent.keys == tuple(keys)
    assert merged.incumbent.score == score
    assert merged.best_feasible.objective <= merged.incumbent.objective
    for archive in evidence:
        for candidate in (archive.incumbent, archive.best_feasible):
            assert candidate.objective == candidate.values[0] - 3
            assert candidate.values == (candidate.keys[0] * 4 - 2,)


def test_startup_witness_survives_pool_eviction() -> None:
    """Keep a worse-scoring feasible startup assignment after a capacity-one pool evicts it."""
    seed = int(np.random.SeedSequence(42, spawn_key=(0, 0)).generate_state(1)[0])
    samples = RandomStreams(seed).numpy.random(2)
    target = float(max(samples))
    model = CompiledLinearModel(
        (LinearVariable("x", "x", 0, 1),),
        (10,),
        (LinearRow("startup_feasible", ((0, 1),), target, target),),
        scoring="weighted_penalty",
        penalty_weight=0.001,
        absolute_tolerance=0,
    )
    pool = SolutionPool(1)
    archives: list[ModelEvidence] = []
    report = initialize_pool(pool, model.decode, model.cost, 1, 2, 10, 42, model_evidence=archives)
    assert report.n_evaluations == 2
    assert len(archives) == 1
    assert archives[0].best_feasible is not None
    assert archives[0].best_feasible.values == (target,)
    assert pool.best[1] != [target]
    search = SearchSession(model.decode, model.cost)
    value = search.evaluate(search.decode(np.array([0])))
    pool.insert((value, [0]), "search")
    merged = model.merge_evidence(tuple(archives) + (search.model_evidence,), [0], value)
    assert merged.incumbent.feasible is False
    assert merged.best_feasible.values == (target,)
    assert merged.first_feasible_at == archives[0].first_feasible_at
    assert merged.first_feasible_at is not None
    assert pool.best == (value, [0])


@pytest.mark.parametrize(
    "algorithm",
    [
        "MultiStart",
        "SA",
        "ILS",
        "VNS",
        "LNS",
        "BRKGA",
        "BRKGA_CS",
        "PSO",
        "GA",
        "GRASP",
        "IPR",
    ],
)
@pytest.mark.parametrize("control", [0, 1])
def test_all_existing_algorithms_use_completed_model_boundary(algorithm: str, control: int) -> None:
    """Retain identity-consistent finite model evidence across every existing algorithm.

    Args:
        algorithm (str): Existing algorithm factory name.
        control (int): Fixed or adaptive existing execution mode.
    """
    model = CompiledLinearModel((LinearVariable("x", "x", 0, 1),), (1,))
    solver = RKO(model.decode, model.cost, 1, execution="sequential", random_state=42)
    evidence: list[ModelEvidence] = []
    reports: list[SearchMetadata] = []
    keys, score = solver._execute(
        RunConfig(control, 2, (algorithm,)),
        0.01,
        started_at=0,
        search_metadata=reports,
        model_evidence=evidence,
    )
    assert len(evidence) == 1
    assert len(reports) == 1
    assert reports[0].n_evaluations > 0
    assert model.merge_evidence(tuple(evidence), keys, score).incumbent.feasible
