"""Opt-in live rootless-runtime tests, including hard callback timeout and offline replay.

Objective:
- Verify offline artifact replay and fail-closed execution in a live rootless runtime.
Technical approach:
- Qualify an explicitly selected image and execute seed or nonreturning decoder fixtures.
Flow (pseudocode):
1. Require the live image opt-in and qualify the shared runner.
2. Replay exported artifacts on unseen instances and trigger a callback timeout.
3. Assert trusted objectives, external deadlines and rejection before qualification.
Important criteria:
- Skip live checks without RKO_EVOLVE_IMAGE; callbacks cannot enforce their own deadlines.
Inputs/Outputs:
- Input: live image environment setting, problem family names and temporary artifact paths.
- Output: assertions on replay evidence and sandbox qualification or deadline errors.
"""

import os
import pytest
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve.sandbox import PodmanRunner
from rko_solver.experimental.self_evolve.sandbox import SandboxError


@pytest.fixture(scope="module")
def runner() -> PodmanRunner:
    """Qualify the explicitly opted-in runtime image before executing test decoders.

    Returns:
        PodmanRunner: Live qualified runner, or a pytest skip when not opted in.
    """
    image = os.environ.get("RKO_EVOLVE_IMAGE")
    if not image:
        pytest.skip("Set RKO_EVOLVE_IMAGE to run live rootless isolation tests.")
    result = PodmanRunner(image)
    result.qualify()
    return result


@pytest.mark.parametrize("family", ["knapsack", "scheduling", "set_cover"])
def test_offline_replay_on_unseen_size(runner: PodmanRunner, family: str, tmp_path: Path) -> None:
    """Replay exported explicit seeds on a new instance without provider access.

    Args:
        runner (PodmanRunner): Qualified live rootless runner for artifact replay.
        family (str): Parametrized problem family for seed creation and instance generation.
        tmp_path (Path): Temporary parent for the exported artifact directory.
    """
    original = DecoderArtifact.create(family, Problem.seed_source(family))
    output = tmp_path / "artifact"
    original.export(output)
    restored = DecoderArtifact.load(output)
    problem = Problem.generate(family, 9281, 14)
    evidence = runner.run(restored, problem, seconds=0.05)
    assert evidence["objective"] == problem.evaluate(evidence["solution"])["objective"]
    assert evidence["evaluations"] > 0
    assert evidence["image"] == runner.image


def test_deadline_stops_a_nonreturning_decoder(runner: PodmanRunner) -> None:
    """An intentionally unbounded callback is killed externally, never trusted to cooperate.

    Args:
        runner (PodmanRunner): Qualified live runner enforcing the callback deadline.
    """
    source = (
        'def decode(keys, instance):\n    """Adversarial timeout fixture."""\n'
        "    while True:\n        pass\n"
    )
    artifact = DecoderArtifact.create("knapsack", source)
    with pytest.raises(SandboxError, match="deadline"):
        runner.run(artifact, Problem.generate("knapsack", 3, 4), seconds=0.01)


def test_unqualified_runner_fails_closed() -> None:
    """Qualification cannot be skipped by requesting an ordinary evaluation job."""
    if not os.environ.get("RKO_EVOLVE_IMAGE"):
        pytest.skip("Live local runtime is not opted in.")
    runner = PodmanRunner(os.environ["RKO_EVOLVE_IMAGE"])
    artifact = DecoderArtifact.create("knapsack", Problem.seed_source("knapsack"))
    with pytest.raises(SandboxError, match="qualification"):
        runner.run(artifact, Problem.generate("knapsack", 3, 4))
