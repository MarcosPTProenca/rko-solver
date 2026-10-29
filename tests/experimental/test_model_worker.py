"""Opt-in live rootless model construction and independent witness verification.

Objective:
- Qualify actual mathematical model jobs, not only process configuration or fake actors.
Technical approach:
- Use a fixed original feasible construction inside the existing rootless worker.
Flow (pseudocode):
1. Extract original GAP data through a qualified native builder.
2. Qualify the explicitly opted-in worker image.
3. Run construction/control jobs and recheck both model witnesses on the parent.
Important criteria:
- No provider call, hidden optimizer, new study or host candidate execution.
Inputs/Outputs:
- Input: explicit RKO_EVOLVE_IMAGE and original bounded model fixture.
- Output: independently verified complete original-coordinate model evidence.
"""

import dataclasses
import numpy as np
import os
import pytest
from benchmarks.modeling.cases import PracticalModels
from benchmarks.self_evolve.gap import GeneralizedAssignment
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.experimental.self_evolve.model_data import ModelProblem
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox import SandboxError
from rko_solver.modeling.adapters import ModelAdapter
from rko_solver.modeling.records import LinearRow
from rko_solver.modeling.records import LinearVariable
from rko_solver.modeling.source import ExtractedLinearModel

SOURCE = """def decode(keys, instance):
    return (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0)
"""


@pytest.mark.parametrize("mode", ["check", "rko", "random", "fixed"])
def test_rootless_model_worker_retains_actual_feasible_witnesses(mode: str) -> None:
    """The protected model hook preserves evidence inside the actual isolated job.

    Args:
        mode (str): Frozen optimizer or matched control mode.
    """
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE for actual rootless model execution.")
    pytest.importorskip("pyomo")
    native = PracticalModels.native(GeneralizedAssignment.fixture().formulation(), "pyomo")
    problem = ModelProblem.from_source(ModelAdapter.extract(native))
    artifact = DecoderArtifact.create("linear_model", SOURCE)
    runner = PodmanRunner(image)
    runner.qualify()
    result = runner.run(artifact, problem, mode=mode, seed=17, seconds=1.0)
    assert result["status"] == "ok" and result["feasible"] is True
    assert result["objective"] == 23
    assert result["model_status"] == "feasible"
    assert result["search_incumbent"]["keys"] == result["keys"]
    assert result["best_feasible"]["values"] == result["solution"]
    assert result["evaluations"] > 0 and result["infeasible_evaluations"] == 0
    assert result["distinct_outputs_capped"] == 1


@pytest.mark.parametrize("unseen", [False, True])
def test_generic_mapper_seed_matches_existing_mapping_on_unseen_dimensions(unseen: bool) -> None:
    """The explicit seed is portable across free-variable counts without source modification.

    Args:
        unseen (bool): Whether to use the different-size seven-job fixture.
    """
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE for actual rootless model execution.")
    native = PracticalModels.native(
        GeneralizedAssignment.fixture(unseen=unseen).formulation(), "pyomo"
    )
    problem = ModelProblem.from_source(ModelAdapter.extract(native))
    runner = PodmanRunner(image)
    runner.qualify()
    record = runner.run(ModelProblem.seed_artifact(), problem, mode="check", seed=19, seconds=0.2)
    expected = problem.snapshot.decode(np.asarray(record["keys"]))
    assert record["solution"] == list(expected.values)
    assert record["score"] == expected.score and record["objective"] == expected.objective
    assert problem.dimension == (21 if unseen else 18)


def test_model_worker_separates_row_infeasibility_from_invalid_domains() -> None:
    """Original-row failures remain diagnostic; domain-invalid construction fails the job."""
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE for actual rootless model execution.")
    native = PracticalModels.native(GeneralizedAssignment.fixture().formulation(), "pyomo")
    problem = ModelProblem.from_source(ModelAdapter.extract(native))
    runner = PodmanRunner(image)
    runner.qualify()
    zeros = DecoderArtifact.create(
        "linear_model",
        "def decode(keys, instance):\n    return [0.0 for v in instance.variables]\n",
    )
    record = runner.run(zeros, problem, mode="check", seed=19, seconds=0.2)
    assert record["model_status"] == "no_feasible_found" and record["best_feasible"] is None
    assert record["infeasible_evaluations"] == record["evaluations"] == 5
    invalid = DecoderArtifact.create(
        "linear_model",
        "def decode(keys, instance):\n    return [2.0 for v in instance.variables]\n",
    )
    with pytest.raises(SandboxError, match="Worker rejected"):
        runner.run(invalid, problem, mode="check", seed=19, seconds=0.2)


@pytest.mark.parametrize("mode", ["check", "rko"])
def test_fully_fixed_model_worker_bypasses_source_construction(mode: str) -> None:
    """No-search workers retain original evidence without calling a failing constructor.

    Args:
        mode (str): Declared endpoint or ordinary model solve path.
    """
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE for actual rootless model execution.")
    source = ModelAdapter.extract(
        PracticalModels.native(GeneralizedAssignment.fixture().formulation(), "pyomo")
    )
    values = (0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0)
    fixed = dataclasses.replace(
        source,
        variables=tuple(
            dataclasses.replace(variable, fixed=float(value))
            for variable, value in zip(source.variables, values, strict=True)
        ),
    )
    problem = ModelProblem.from_source(fixed)
    artifact = DecoderArtifact.create(
        "linear_model", "def decode(keys, instance):\n    return 1 / 0\n"
    )
    runner = PodmanRunner(image)
    runner.qualify()
    record = runner.run(artifact, problem, mode=mode, seed=19, seconds=0.2)
    assert problem.dimension == 0 and record["evaluations"] == 1
    assert record["objective"] == 23 and record["feasible"] is True
    assert record["no_search"] is True and record["construction_used"] is False


def test_generic_seed_matches_mixed_affine_original_model_with_restricted_bounds() -> None:
    """A GAP-only binary mapper cannot pass continuous/integer/fixed/affine repair equivalence."""
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE for actual rootless model execution.")
    source = ExtractedLinearModel(
        "pyomo",
        "owned-control",
        "mixed-affine-control",
        (
            LinearVariable("x", "x", -float("inf"), float("inf"), "continuous"),
            LinearVariable("y", "y", 0.0, 3.0, "integer"),
            LinearVariable("z", "z", 0.0, 1.0, "binary"),
            LinearVariable("f", "f", 1.0, 1.0, "continuous", 1.0),
        ),
        (3.0, -2.0, 1.0, 0.0),
        5.0,
        True,
        (
            LinearRow("ranged", ((0, 1.0), (1, 1.0)), 2.0, 4.0, 0.25),
            LinearRow("equality", ((0, 1.0), (3, 1.0)), 2.0, 2.0),
        ),
    )
    problem = ModelProblem.from_source(
        source, policy={"search_bounds": {"x": (-2.0, 2.0), "y": (0.0, 2.0)}}
    )
    runner = PodmanRunner(image)
    runner.qualify()
    record = runner.run(ModelProblem.seed_artifact(), problem, mode="check", seed=23, seconds=1.0)
    expected = problem.snapshot.decode(np.asarray(record["keys"]))
    assert record["solution"] == list(expected.values)
    assert record["objective"] == expected.objective and record["score"] == expected.score
    assert record["search_incumbent"]["repair"]["policy"] == "continuous"
    assert problem.dimension == 3 and problem.snapshot.repair_max_sweeps == 1024
