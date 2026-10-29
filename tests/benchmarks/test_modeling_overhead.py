"""Correctness and measurement-integrity tests for model feature overhead diagnostics.

Objective:
- Ensure timing controls do identical work and transport only bounded completed evidence.

Technical approach:
- Exercise the exact Git baseline and current boundary with shared callback objects.
- Verify generated archive-off/on controls, declared statistical findings and a real spawned pipe.

Flow (pseudocode):
1. Check all callback regimes preserve coordinates, completed counts and costs.
2. Inspect distinct bounded score/feasible payloads and pickle identity semantics.
3. Run a small real send/unpickle/ack exchange without timing acceptance assertions.
4. Reject invalid settings or attempts to overwrite existing measurements.

Important criteria:
- Tests never demand a wall-clock speedup or turn inconclusive evidence into acceptance.
- Native/scientific/modeling optional dependencies are unnecessary.

Inputs/Outputs:
- Input: controlled workloads and local source/transport resources.
- Output: assertions for semantic parity, proper findings and diagnostic safety.
"""

import multiprocessing as mp
import numpy as np
import pickle
import pytest
from benchmarks.modeling.overhead import ModelOverheadProbe
from pathlib import Path
from rko_solver.orchestration.search import SearchSession


def test_fixed_work_controls_preserve_unchanged_callbacks_and_costs() -> None:
    """Run all unchanged callback regimes through exact original/current boundaries."""
    baseline, digest = ModelOverheadProbe.baseline("bdee82a")
    assert len(digest) == 64
    for workload in ModelOverheadProbe.workloads():
        before = workload.keys.copy()
        control = SearchSession if workload.generated else baseline
        assert ModelOverheadProbe.sample(control, workload, 4, archive=not workload.generated) > 0
        assert ModelOverheadProbe.sample(SearchSession, workload, 4) > 0
        assert np.array_equal(workload.keys, before)
        assert not workload.keys.flags.writeable


@pytest.mark.parametrize(
    "ratio, finding", [(1.0, "within_boundary_threshold"), (1.1, "regression")]
)
def test_predeclared_boundary_findings(ratio: float, finding: str) -> None:
    """Keep the protection decision tied to the declared paired interval threshold.

    Args:
        ratio (float): Synthetic constant candidate/control timing ratio.
        finding (str): Expected descriptive protection finding.
    """
    pairs = [{"control_seconds": 1.0, "candidate_seconds": ratio} for _ in range(7)]
    result = ModelOverheadProbe.paired_summary(pairs, protection=True)
    assert result["finding"] == finding
    assert result["paired_ratio_median"] == ratio
    assert result["bootstrap95_ratio"] == [ratio, ratio]


def test_overlapping_interval_is_inconclusive_not_a_pass() -> None:
    """Preserve uncertainty rather than advertising a noisy ratio as non-regression."""
    pairs = [
        {"control_seconds": 1.0, "candidate_seconds": ratio}
        for ratio in (0.9, 1.2, 0.9, 1.2, 0.9, 1.2)
    ]
    assert ModelOverheadProbe.paired_summary(pairs, protection=True)["finding"] == "inconclusive"
    assert ModelOverheadProbe.paired_summary(pairs)["finding"] == "measured_only"


def test_worker_payloads_are_bounded_distinct_completed_witnesses() -> None:
    """Prevent repeated-object memoization from disguising multi-session archive payload costs."""
    control, candidate = ModelOverheadProbe.messages(8, 4, 4)
    assert len(control) == 1 and len(candidate) == 2
    assert control[0] == candidate[1]
    kind, archives = candidate[0]
    assert kind == "model_evidence"
    assert len(archives) == 4 and len({id(archive) for archive in archives}) == 4
    for archive in archives:
        assert not archive.incumbent.feasible
        assert archive.best_feasible.feasible
        assert archive.incumbent is not archive.best_feasible
        assert archive.incumbent.score < archive.best_feasible.score
        assert archive.first_feasible_at is not None
    assert pickle.loads(pickle.dumps(candidate)) == candidate


def test_spawned_transport_probe_exchanges_actual_bounded_messages() -> None:
    """Exercise the real child unpickler and acknowledgement path without speed assertions."""
    context = mp.get_context("spawn")
    parent, child = context.Pipe()
    worker = context.Process(target=ModelOverheadProbe.echo, args=(child,))
    worker.start()
    child.close()
    try:
        assert parent.poll(30) and parent.recv() == "ready"
        for messages in ModelOverheadProbe.messages(8, 4, 1):
            assert ModelOverheadProbe.transport_sample(parent, messages, 2) > 0
        parent.send(None)
        worker.join(10)
        assert worker.exitcode == 0
    finally:
        parent.close()
        if worker.is_alive():
            worker.terminate()
            worker.join()
        worker.close()


@pytest.mark.parametrize(
    "repeats, target", [(2, 0.1), (True, 0.1), (3, 0.0), (3, True), (3, float("nan"))]
)
def test_invalid_diagnostic_settings_fail_before_timing(
    tmp_path: Path,
    repeats: int,
    target: float,
) -> None:
    """Reject invalid controls without creating an output or changing process affinity.

    Args:
        tmp_path (Path): Temporary unused output directory.
        repeats (int): Requested paired trial count.
        target (float): Requested calibration target.
    """
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError):
        ModelOverheadProbe.run(output, repeats=repeats, target_seconds=target)
    assert not output.exists()


def test_diagnostic_output_cannot_overwrite_an_existing_measurement(tmp_path: Path) -> None:
    """Protect prior evidence rather than silently replacing an inconvenient measurement.

    Args:
        tmp_path (Path): Temporary existing artifact directory.
    """
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ModelOverheadProbe.run(output, repeats=3)
    assert output.read_text(encoding="utf-8") == "keep"
