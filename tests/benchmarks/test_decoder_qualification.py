"""Integrity tests for general-model decoder qualification.

Objective:
- Independently check generated witnesses, fixed-key diversity and total-time accounting.

Technical approach:
- Property-test generated witnesses and verify small optima separately with HiGHS.
- Check diversity identities and exercise the public model factory for both decoder arms.

Flow (pseudocode):
1. Generate signed binary, ranged integer and mixed models.
2. Check planted witnesses against original rows and certified reference objectives.
3. Exercise diversity and timing records, without asserting a wall-clock performance gain.

Important criteria:
- Tests require semantic agreement, not a heuristic finding an optimum in a short budget.

Inputs/Outputs:
- Input: generated models, uniform keys and controlled clocks.
- Output: assertions on independent validity and measurement integrity.
"""

import json
import numpy as np
import pytest
from benchmarks.modeling.cases import PracticalModels
from benchmarks.modeling.decoder_qualification import DecoderQualification
from benchmarks.modeling.qualification_cases import QualificationCases
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from pathlib import Path


@given(
    family=st.sampled_from(("binary_chain", "ranged_integer", "mixed")),
    blocks=st.integers(1, 10),
    seed=st.integers(0, 10000),
)
@settings(max_examples=100, derandomize=True)
def test_generated_witnesses_are_feasible_and_attain_the_proved_optimum(
    family: str, blocks: int, seed: int
) -> None:
    """Every generator output owns a feasible witness attaining its analytic lower bound."""
    case, witness = QualificationCases.build(family, 6 * blocks, seed)
    values = dict(zip((v.name for v in case.variables), witness, strict=True))
    checked = PracticalModels.check(case, values)
    assert checked["feasible"]
    assert checked["objective"] == pytest.approx(case.expected)
    assert (case, witness) == QualificationCases.build(family, 6 * blocks, seed)


@pytest.mark.parametrize("family", ("binary_chain", "ranged_integer", "mixed"))
def test_independent_reference_confirms_generated_optima(family: str) -> None:
    """HiGHS certifies the same optimum without feeding any witness into the decoder."""
    pytest.importorskip("scipy.optimize")
    case, _ = QualificationCases.build(family, 48, 20261107)
    assert PracticalModels.reference(case)["objective"] == pytest.approx(case.expected)


def test_fixed_key_probe_reports_collisions_and_validated_outputs() -> None:
    """A probe reports both all-output and feasible-output diversity, never conflating them."""
    case, _ = QualificationCases.build("ranged_integer", 6, 7)
    keys = np.zeros((12, 6))
    record = DecoderQualification.probe(case, "default", keys)
    assert record["samples"] == 12 and record["unique_assignments"] == 1
    assert record["collision_fraction"] == pytest.approx(11 / 12)
    assert record["mean_adjacent_distance"] == 0
    assert record["feasible_samples"] == 0 and record["unique_feasible_assignments"] == 0


def test_no_binary_rows_preserve_fixed_key_assignments() -> None:
    """The general-integer control remains numerically identical through either decoder."""
    case, _ = QualificationCases.build("ranged_integer", 12, 11)
    keys = np.random.default_rng(17).random((32, 12))
    default = DecoderQualification.probe(case, "default", keys)
    structured = DecoderQualification.probe(case, "structure", keys)
    for field in ("assignment_digest", "feasible_samples", "unique_assignments"):
        assert default[field] == structured[field]


def test_decoder_pair_order_is_balanced() -> None:
    """Each arm runs first equally often across an even number of seed positions."""
    assert DecoderQualification.arm_order(0) == ("default", "structure")
    assert DecoderQualification.arm_order(1) == ("structure", "default")


def test_setup_exhaustion_does_not_start_a_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup consumes the total budget; the harness must not grant a new full search budget."""
    from benchmarks.modeling import decoder_qualification as module

    case, _ = QualificationCases.build("ranged_integer", 6, 0)
    clock = iter((10.0, 12.0))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(module.PracticalModels, "native", lambda *args: object())
    monkeypatch.setattr(module.RKO, "from_model", lambda *args, **kwargs: object())
    record = DecoderQualification.timed(case, "SA", "default", 0, 1.0)
    assert record["status"] == "setup_exhausted"
    assert record["setup_seconds"] == record["total_seconds"] == 2.0
    assert record["solve_seconds"] == 0.0 and record["overrun_seconds"] == 1.0
    assert record["evaluations"] == 0


@pytest.mark.parametrize("decoder", ("default", "structure"))
def test_real_public_factory_has_reconciled_timing(decoder: str) -> None:
    """Run the public conversion/search boundary and verify recorded timing identities."""
    pytest.importorskip("ortools.math_opt.python.mathopt")
    case, _ = QualificationCases.build("mixed", 6, 9)
    record = DecoderQualification.timed(case, "SA", decoder, 1, 0.05)
    assert record["total_seconds"] == pytest.approx(
        record["setup_seconds"] + record["solve_seconds"]
    )
    assert record["overrun_seconds"] == pytest.approx(max(0, record["total_seconds"] - 0.05))
    assert record["status"] in ("completed", "setup_exhausted")


def test_campaign_is_append_only_and_complete(tmp_path: Path) -> None:
    """Output is reserved before work and contains metadata, both probes and both timed arms."""
    pytest.importorskip("ortools.math_opt.python.mathopt")
    output = tmp_path / "study.jsonl"
    case, _ = QualificationCases.build("ranged_integer", 6, 0)
    DecoderQualification.run(output, (case,), seconds=0.02, seeds=(0,), samples=4)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["type"] == "metadata"
    assert sum(r["type"] == "probe" for r in records) == 2
    assert sum(r["type"] == "search" for r in records) == 4
    assert records[-1]["type"] == "complete"
    with pytest.raises(FileExistsError):
        DecoderQualification.run(output, (case,), seconds=0.02, seeds=(0,), samples=4)


@pytest.mark.parametrize("size", (0, 5, -6))
def test_generator_rejects_invalid_sizes(size: int) -> None:
    """All families require positive multiples of six so analytic optima remain valid."""
    with pytest.raises(ValueError):
        QualificationCases.build("mixed", size, 0)


def test_setup_is_subtracted_before_solve(monkeypatch: pytest.MonkeyPatch) -> None:
    """A controlled clock proves solve gets the remainder, with overruns retained honestly."""
    from benchmarks.modeling import decoder_qualification as module
    from types import SimpleNamespace

    case, _ = QualificationCases.build("ranged_integer", 6, 0)
    evaluation = module.KeyFeasibilityStudy._snapshot(case, "continuous").decode(np.zeros(6))
    received = []

    def solve(limit: float) -> object:
        """Retain the granted limit and return independently checkable incomplete evidence."""
        received.append(limit)
        return SimpleNamespace(
            searches=(SimpleNamespace(n_evaluations=2),),
            model=SimpleNamespace(
                search_incumbent=SimpleNamespace(evaluation=evaluation),
                best_feasible=None,
            ),
        )

    clock = iter((10.0, 10.25, 11.1))
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(module.PracticalModels, "native", lambda *args: object())
    monkeypatch.setattr(
        module.RKO, "from_model", lambda *args, **kwargs: SimpleNamespace(solve=solve)
    )
    record = DecoderQualification.timed(case, "SA", "default", 0, 1.0)
    assert received == [0.75]
    assert record["setup_seconds"] == 0.25
    assert record["solve_seconds"] == pytest.approx(0.85)
    assert record["total_seconds"] == pytest.approx(1.1)
    assert record["overrun_seconds"] == pytest.approx(0.1)


def test_errors_leave_partial_evidence_and_no_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed reference stops the campaign and is not relabelled as an infeasible search."""

    def fail(case: object) -> dict:
        """Raise a controlled reference error before any optimization begins."""
        raise RuntimeError("reference failed")

    case, _ = QualificationCases.build("ranged_integer", 6, 0)
    output = tmp_path / "failed.jsonl"
    monkeypatch.setattr(DecoderQualification, "warmup", lambda: None)
    monkeypatch.setattr(DecoderQualification, "reference", fail)
    with pytest.raises(RuntimeError, match="reference failed"):
        DecoderQualification.run(output, (case,), seconds=0.02, seeds=(0,), samples=4)
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [r["type"] for r in records] == ["metadata", "error"]
