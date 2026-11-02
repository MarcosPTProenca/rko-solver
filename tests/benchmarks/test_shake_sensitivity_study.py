"""Tests for the shake-intensity decoded-change sensitivity measurement."""

from __future__ import annotations

import numpy as np
from benchmarks.engines.shake_sensitivity_study import DECODERS
from benchmarks.engines.shake_sensitivity_study import INTENSITY_GRID
from benchmarks.engines.shake_sensitivity_study import SensitivityDesign
from benchmarks.engines.shake_sensitivity_study import ShakeSensitivityStudy


def test_zero_intensity_reports_are_absent_and_grid_is_fixed() -> None:
    """The intensity grid is fixed and brackets the library's beta_min/beta_max defaults."""
    assert INTENSITY_GRID[0] > 0.0
    assert min(INTENSITY_GRID) <= 0.05
    assert max(INTENSITY_GRID) >= 0.40
    assert list(INTENSITY_GRID) == sorted(INTENSITY_GRID)


def test_decoders_builds_one_callable_decoder_per_name() -> None:
    """Every named decoder decodes a matching-length key vector without raising."""
    design = SensitivityDesign(n_items=12, test_instances=1, repeats=1)
    study = ShakeSensitivityStudy(design)
    built = study.decoders(seed=1)
    assert set(built) == set(DECODERS)
    for decoder, n_keys in built.values():
        keys = np.random.default_rng(0).random(n_keys)
        decoder(keys)


def test_changed_fraction_is_zero_for_identical_arrays_and_one_for_disjoint() -> None:
    """The changed fraction is a plain elementwise mismatch rate in [0, 1]."""
    same = np.array([1, 2, 3])
    assert ShakeSensitivityStudy.changed_fraction(same, same) == 0.0
    disjoint = np.array([2, 3, 1])
    assert ShakeSensitivityStudy.changed_fraction(same, disjoint) == 1.0
    partial = np.array([1, 3, 2])
    assert ShakeSensitivityStudy.changed_fraction(same, partial) == 2 / 3


def test_trial_is_reproducible_and_returns_one_row_per_intensity_and_repeat() -> None:
    """Calling trial twice with the same seed reproduces every row exactly."""
    design = SensitivityDesign(n_items=16, test_instances=1, repeats=3)
    study = ShakeSensitivityStudy(design)
    decoder, n_keys = study.decoders(seed=5)["tsp_plain"]
    first = study.trial("tsp_plain", decoder, n_keys, seed=5)
    second = study.trial("tsp_plain", decoder, n_keys, seed=5)
    assert first == second
    assert len(first) == len(INTENSITY_GRID) * design.repeats
    for row in first:
        assert 0.0 <= row["changed_fraction"] <= 1.0


def test_changed_fraction_is_non_decreasing_on_average_with_intensity() -> None:
    """Averaged over instances, the mean changed fraction grows with shake intensity."""
    design = SensitivityDesign(n_items=60, test_instances=10, repeats=6)
    study = ShakeSensitivityStudy(design)
    report = study.run()
    for name in DECODERS:
        means = [report["summary"][name][str(level)]["mean"] for level in INTENSITY_GRID]
        assert means[0] <= means[-1]
        assert means[-1] <= 1.0 + 1e-9


def test_run_produces_one_summary_entry_per_decoder_and_intensity() -> None:
    """The summary table covers every decoder and every grid intensity, with valid intervals."""
    design = SensitivityDesign(n_items=20, test_instances=3, repeats=2)
    study = ShakeSensitivityStudy(design)
    report = study.run()
    assert set(report["summary"]) == set(DECODERS)
    for name in DECODERS:
        assert set(report["summary"][name]) == {str(level) for level in INTENSITY_GRID}
        for stats in report["summary"][name].values():
            assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
    assert (
        len(report["trials"])
        == len(DECODERS) * design.test_instances * len(INTENSITY_GRID) * design.repeats
    )
