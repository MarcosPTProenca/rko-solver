"""Bounded owned-model transport and original-source identity contracts.

Objective:
- Carry model mathematics without native objects, untrusted pickle or candidate execution.
Technical approach:
- Round-trip real native extraction and independently check supplied assignment evidence.
Flow (pseudocode):
1. Extract the original GAP model through an existing qualified frontend.
2. Serialize only bounded owned data and effective policy as strict JSON.
3. Reconstruct protected numerical evaluation without importing the native model.
Important criteria:
- Reference answers are parent-only declarations, never candidate construction input.
Inputs/Outputs:
- Input: original model records, policy and optional reference metadata.
- Output: identity-preserving pure data or explicit preflight failures.
"""

import copy
import json
import numpy as np
import pytest
from benchmarks.modeling.cases import PracticalModels
from benchmarks.self_evolve.gap import GeneralizedAssignment
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.experimental.self_evolve.model_data import ModelProblem
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox import SandboxError
from rko_solver.modeling.adapters import ModelAdapter
from rko_solver.modeling.construction import ModelConstruction
from rko_solver.modeling.linear import CompiledLinearModel


def test_owned_model_roundtrip_preserves_original_identity_and_evaluation() -> None:
    """JSON restoration reproduces original source semantics, not native handles."""
    pytest.importorskip("pyomo")
    case = GeneralizedAssignment.fixture().formulation()
    source = ModelAdapter.extract(PracticalModels.native(case, "pyomo"))
    problem = ModelProblem.from_source(source)
    restored = ModelProblem(json.loads(json.dumps(problem.data, allow_nan=False)))
    assert restored.source.fingerprint == source.fingerprint
    assert restored.dimension == 18
    evaluation = restored.snapshot.evaluate_assignment(
        np.zeros(18), (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0)
    )
    assert evaluation.objective == 23 and evaluation.feasible
    assert restored.snapshot.repair_max_sweeps == 1024
    assert restored.snapshot.repair == "continuous"
    with pytest.raises(ValueError, match="reference"):
        restored.optimum()


def test_model_artifact_describes_assignments_not_selected_indices() -> None:
    """A model constructor uses a distinct contract without changing pilot index artifacts."""
    source = "def decode(keys, instance):\n    return [0.0 for v in instance.variables]\n"
    model = DecoderArtifact.create("linear_model", source)
    assert model.metadata()["model_contract"] == 1
    assert model.metadata()["solution"] == "complete original-order numerical assignments"
    legacy = DecoderArtifact.create("knapsack", source)
    assert "model_contract" not in legacy.metadata()
    assert legacy.metadata()["solution"].startswith("unique zero-based")


@pytest.mark.parametrize("frontend", ["pyomo", "pulp", "mathopt"])
def test_owned_model_transport_preserves_original_native_source(frontend: str) -> None:
    """All three real extractors preserve original IDs/domains/arithmetic across JSON.

    Args:
        frontend (str): Installed qualified native frontend.
    """
    pytest.importorskip("ortools" if frontend == "mathopt" else frontend)
    fixture = GeneralizedAssignment.fixture()
    native = PracticalModels.native(fixture.formulation(), frontend)
    source = ModelAdapter.extract(native)
    problem = ModelProblem.from_source(source)
    restored = ModelProblem(json.loads(json.dumps(problem.data, allow_nan=False)))
    assert restored.source == source
    assert restored.source.fingerprint == source.fingerprint
    assert restored.snapshot == problem.snapshot


@pytest.mark.parametrize(
    "mutation", ["schema", "extra", "fingerprint", "boolean", "policy", "runtime"]
)
def test_model_transport_rejects_unknown_or_drifting_original_semantics(mutation: str) -> None:
    """A malformed mathematical envelope never reaches constructor/runtime setup.

    Args:
        mutation (str): Strict source/policy boundary violation.
    """
    pytest.importorskip("pyomo")
    native = PracticalModels.native(GeneralizedAssignment.fixture().formulation(), "pyomo")
    original = ModelProblem.from_source(ModelAdapter.extract(native))
    data = copy.deepcopy(original.data)
    if mutation == "schema":
        data["schema"] = True
    elif mutation == "extra":
        data["optimum"] = 16
    elif mutation == "fingerprint":
        data["source"]["fingerprint"] = "0" * 64
    elif mutation == "boolean":
        data["source"]["objective"][0] = True
    elif mutation == "policy":
        data["policy"]["repair_sweeps"] = 0
    else:
        data["implementation"] = {}
    with pytest.raises(ValueError):
        ModelProblem(data)


def test_parent_checks_both_original_witnesses_without_decode_or_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verification never invokes source construction, default mapping or restoration.

    Args:
        monkeypatch (pytest.MonkeyPatch): Trap any forbidden witness replay.
    """
    pytest.importorskip("pyomo")
    native = PracticalModels.native(GeneralizedAssignment.fixture().formulation(), "pyomo")
    problem = ModelProblem.from_source(ModelAdapter.extract(native))
    values = (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0)
    point = problem.snapshot.evaluate_assignment(np.zeros(problem.dimension), values)
    witness = {
        "keys": list(point.keys),
        "values": list(point.values),
        "objective": point.objective,
        "score": point.score,
        "feasible": point.feasible,
    }
    record = {
        **witness,
        "solution": witness["values"],
        "search_incumbent": dict(witness),
        "best_feasible": dict(witness),
        "model_source_fingerprint": problem.source.fingerprint,
        "model_policy_fingerprint": problem.policy_fingerprint,
        "no_search": False,
        "construction_used": True,
        "evaluations": 1,
        "infeasible_evaluations": 0,
    }

    def forbidden(*args: object, **kwargs: object) -> None:
        """Fail on source/default decoding or a protected restoration replay.

        Args:
            args (object): Unexpected replay arguments.
            kwargs (object): Unexpected replay keywords.

        Raises:
            AssertionError: Always, because this is forbidden during parent verification.
        """
        raise AssertionError("Verification must not construct or restore a witness.")

    monkeypatch.setattr(ModelConstruction, "decode", forbidden)
    monkeypatch.setattr(CompiledLinearModel, "decode", forbidden)
    monkeypatch.setattr(CompiledLinearModel, "_restore_continuous", forbidden)
    assert problem.verify_record(record)["objective"] == 23
    changed = copy.deepcopy(record)
    changed["search_incumbent"]["objective"] = 16
    with pytest.raises(ValueError, match="mismatch"):
        problem.verify_record(changed)
    changed = copy.deepcopy(record)
    changed["model_policy_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        problem.verify_record(changed)
    changed = copy.deepcopy(record)
    changed["infeasible_evaluations"] = 1
    with pytest.raises(ValueError, match="counts"):
        problem.verify_record(changed)
    changed = {
        **record,
        "infeasible_evaluations": 999,
        "distinct_outputs_capped": 1,
        "wall_seconds": 0.01,
        "cpu_seconds": 0.01,
        "peak_memory_bytes": 1,
    }
    runner = object.__new__(PodmanRunner)
    runner.qualified = True
    with pytest.raises(SandboxError, match="counters"):
        runner._verify_record(changed, problem, 0.1)
