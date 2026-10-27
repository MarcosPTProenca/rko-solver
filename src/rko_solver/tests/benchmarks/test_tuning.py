"""Tests of the tuning bake-off harness.

Objective:
- Pin that the budget is enforced, the search spaces always decode to configurations the solver
  accepts, each tuner improves on random configurations and respects its budget, and the
  protocol and analysis behave deterministically and resume.

Technical approach:
- Tuners run on a synthetic noisy bowl with a known optimum, so no solver time is spent and the
  outcome is a deterministic function of the seeds.
- The protocol and the instance code run with shrunken budgets against the real solver.

Flow (pseudocode):
1. Build the synthetic runner and the metered objective.
2. Run each tuner and compare the true value of its choice with random configurations.
3. Run a tiny real session, resume it and summarize synthetic records.

Important criteria:
- The module is not collected unless `optuna` and `scipy` are installed (see `conftest.py`).
- No test depends on machine speed: runs are limited by evaluations, not by time.

Inputs/Outputs:
- Input: synthetic runners, small settings and synthetic records.
- Output: assertions on budgets, validity, determinism and report content.
"""

import json
import numpy as np
import pytest
import random
from benchmarks.tuning import instances as instance_module
from benchmarks.tuning.analysis import Analysis
from benchmarks.tuning.instances import FAMILIES
from benchmarks.tuning.instances import InstanceBank
from benchmarks.tuning.objective import BudgetExhausted
from benchmarks.tuning.objective import TuningObjective
from benchmarks.tuning.spaces import SPACES
from benchmarks.tuning.spaces import ParameterSpec
from benchmarks.tuning.tuners import IteratedRacing
from benchmarks.tuning.tuners import RandomHalving
from pathlib import Path
from rko_solver import RKO
from rko_solver import Numba

TARGET = np.array([0.3, 0.7, 0.2, 0.6])
SPACE = SPACES["BRKGA"]


def unit_of(config: dict[str, float]) -> np.ndarray:
    """Map a BRKGA configuration back to the unit cube.

    Args:
        config (dict[str, float]): Decoded configuration.

    Returns:
        np.ndarray: Four unit-cube coordinates.
    """
    log = np.log
    return np.array(
        [
            (log(config["p"]) - log(10)) / (log(500) - log(10)),
            (config["pe"] - 0.05) / 0.30,
            (config["pm"] - 0.01) / 0.29,
            (config["rhoe"] - 0.5) / 0.4,
        ]
    )


def true_value(config: dict[str, float]) -> float:
    """Return the noise-free distance of a configuration to the synthetic optimum.

    Args:
        config (dict[str, float]): BRKGA configuration.

    Returns:
        float: Squared distance in the unit cube.
    """
    return float(np.sum((unit_of(config) - TARGET) ** 2))


def noisy(config: dict[str, float], instance: int, seed: int) -> float:
    """Score a configuration on a synthetic instance with seeded noise.

    Args:
        config (dict[str, float]): BRKGA configuration.
        instance (int): Instance index.
        seed (int): Run seed.

    Returns:
        float: True value scaled by the instance plus Gaussian noise.
    """
    noise = np.random.default_rng(seed * 131 + 7).normal(0, 0.15)
    return true_value(config) * (1 + 0.3 * np.sin(instance)) + noise


def test_decoded_points_are_valid_for_the_solver_including_the_corners() -> None:
    """Every corner, the center and random points of every space run in the solver."""
    compiled = pytest.importorskip("numba").njit(lambda keys: float(np.sum((keys - 0.3) ** 2)))
    rng = random.Random(3)
    for name, space in SPACES.items():
        points = [[0.0] * space.dimension, [1.0] * space.dimension, [0.5] * space.dimension]
        points += [[rng.choice([0.0, 1.0, rng.random()]) for _ in space.specs] for _ in range(25)]
        for point in points:
            config = space.decode(point)
            RKO(
                Numba(compiled, 8),
                engine="rust",
                algorithms=[name],
                adaptation="fixed",
                hyperparameters={name: config},
                max_evaluations=200,
                random_state=1,
                n_jobs=1,
            ).solve(30.0)
        assert all(spec.low <= space.default()[spec.name] <= spec.high for spec in space.specs), (
            name
        )


def test_decode_is_pure_clips_and_checks_its_input() -> None:
    """Equal points decode equally, out-of-range values clip and a wrong size is rejected."""
    point = [0.2, 0.4, 0.6, 0.8]
    assert SPACE.decode(point) == SPACE.decode(point)
    assert SPACE.decode([-5, 9, 0.5, 0.5]) == SPACE.decode([0, 1, 0.5, 0.5])
    with pytest.raises(ValueError, match="coordinates"):
        SPACE.decode([0.5])
    with pytest.raises(ValueError, match="scale"):
        ParameterSpec("x", 1, 2, "cubic").decode(0.5)


def test_objective_enforces_the_budget_and_shares_seeds() -> None:
    """The run after the budget raises, instances are checked and seeds depend on the instance."""
    seen: list[tuple[int, int]] = []
    objective = TuningObjective(
        lambda config, instance, seed: seen.append((instance, seed)) or 1.0, 3, 4, base_seed=10
    )
    assert objective.mean({}, [0, 1, 2]) == 1.0 and objective.remaining == 1
    objective.run({}, 1)
    with pytest.raises(BudgetExhausted):
        objective.run({}, 0)
    with pytest.raises(IndexError):
        TuningObjective(lambda *_: 0.0, 3, 4).run({}, 3)
    with pytest.raises(ValueError, match="positive"):
        TuningObjective(lambda *_: 0.0, 0, 4)
    assert seen == [(0, 10), (1, 11), (2, 12), (1, 11)]


@pytest.mark.parametrize("tuner", [RandomHalving(), IteratedRacing()])
def test_builtin_tuners_beat_random_configurations_within_budget(tuner: object) -> None:
    """Each method uses at most its budget and lands far closer to the optimum than random.

    Args:
        tuner (object): Tuner under test.
    """
    randoms = [true_value(SPACE.decode(np.random.default_rng(i).random(4))) for i in range(100)]
    chosen = []
    for seed in range(4):
        objective = TuningObjective(noisy, 12, 600, base_seed=seed * 100)
        config = tuner.tune(objective, SPACE, seed)
        assert objective.runs_used <= 600 and objective.runs_used >= 400
        assert set(config) == {"p", "pe", "pm", "rhoe"}
        chosen.append(true_value(config))
    assert np.mean(chosen) < np.median(randoms) / 5
    again = tuner.tune(TuningObjective(noisy, 12, 600, base_seed=0), SPACE, 0)
    assert true_value(again) == chosen[0]


def test_external_tuners_beat_random_configurations_within_budget() -> None:
    """Optuna's TPE and RKO itself also stay within budget and beat random configurations."""
    from benchmarks.tuning.external import OptunaTPE
    from benchmarks.tuning.external import RKOTuner

    randoms = [true_value(SPACE.decode(np.random.default_rng(i).random(4))) for i in range(100)]
    for tuner in (OptunaTPE(), RKOTuner()):
        objective = TuningObjective(noisy, 12, 400, base_seed=5)
        config = tuner.tune(objective, SPACE, 2)
        assert objective.runs_used <= 400
        assert true_value(config) < np.median(randoms) / 3


def test_tuner_arguments_are_validated() -> None:
    """Out-of-range method settings are rejected."""
    with pytest.raises(ValueError):
        RandomHalving(eta=1)
    with pytest.raises(ValueError):
        IteratedRacing(alpha=2.0)
    from benchmarks.tuning.external import OptunaTPE
    from benchmarks.tuning.external import RKOTuner

    with pytest.raises(ValueError):
        OptunaTPE(startup=0)
    with pytest.raises(ValueError):
        RKOTuner(instances=0)


@pytest.fixture
def small_bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> InstanceBank:
    """Create a bank whose calibration is cheap.

    Args:
        tmp_path (Path): Work directory.
        monkeypatch (pytest.MonkeyPatch): Fixture used to shrink the calibration.

    Returns:
        InstanceBank: Bank with small calibration budgets.
    """
    monkeypatch.setattr(instance_module, "CALIBRATION_EVALUATIONS", 3000)
    monkeypatch.setattr(instance_module, "CALIBRATION_SEEDS", 2)
    return InstanceBank(tmp_path, evaluations=2000)


@pytest.mark.parametrize("family", FAMILIES)
def test_instances_are_deterministic_calibrated_and_scored(
    small_bank: InstanceBank, family: str
) -> None:
    """Files are reproducible, calibration persists and the score matches its definition.

    Args:
        small_bank (InstanceBank): Bank with cheap calibration.
        family (str): Problem family.
    """
    first = small_bank.generate(family, "train", 2)
    content = first[0].path.read_text(encoding="utf-8")
    assert small_bank.generate(family, "train", 2)[0].path.read_text(encoding="utf-8") == content
    assert small_bank.generate(family, "test", 1)[0].path.read_text(encoding="utf-8") != content
    calibrated = small_bank.calibrate(first)
    assert all(item.anchor > item.best for item in calibrated)
    reloaded = InstanceBank(small_bank.directory, 2000).generate(family, "train", 2)
    assert [(i.best, i.anchor) for i in reloaded] == [(i.best, i.anchor) for i in calibrated]
    instance = calibrated[0]
    assert instance.normalize(instance.best) == 0.0
    assert instance.normalize(instance.anchor) == pytest.approx(1.0)
    config = SPACES["SA"].default()
    assert small_bank.score(instance, "SA", config, 1) == small_bank.score(
        instance, "SA", config, 1
    )
    with pytest.raises(ValueError, match="Unknown"):
        small_bank.generate("nope", "train", 1)


def synthetic_records() -> list[dict[str, object]]:
    """Build records in which `good` beats the default, `bad` loses and `same` ties.

    Returns:
        list[dict[str, object]]: Meta line and session records.
    """
    rng = np.random.default_rng(0)
    base = rng.uniform(0.4, 0.6, 8)
    records: list[dict[str, object]] = [{"kind": "meta", "settings": {}}]
    shifts = {"default": 0.0, "random": 0.2, "good": -0.2, "bad": 0.2, "same": 0.0}
    for tuner, shift in shifts.items():
        for rep in range(1 if tuner == "default" else 4):
            noise = rng.normal(0, 0.02, 8)
            records.append(
                {
                    "kind": "session",
                    "algorithm": "SA",
                    "family": "kp",
                    "tuner": tuner,
                    "rep": rep,
                    "runs_used": 0 if tuner in ("default", "random") else 100,
                    "test": list(
                        base + shift + (0 * noise if tuner in ("default", "same") else noise)
                    ),
                }
            )
    return records


def test_analysis_classifies_better_worse_and_tie(tmp_path: Path) -> None:
    """Verdicts follow the bootstrap interval of the difference to the default.

    Args:
        tmp_path (Path): Directory for the results file.
    """
    path = tmp_path / "results.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in synthetic_records()) + "\n", encoding="utf-8")
    analysis = Analysis(path, resamples=500)
    data = analysis.matrices("SA", "kp")
    default = data["default"][0]
    assert analysis.verdict(data["good"], default) == "better"
    assert analysis.verdict(data["bad"], default) == "worse"
    assert analysis.verdict(data["same"], default) == "tie"
    assert analysis.interval(data["good"]) == analysis.interval(data["good"])
    report = analysis.report()
    assert "### SA on kp" in report and "| good |" in report and "better" in report
    assert analysis.cell("SA", "tsp") == ""


def test_protocol_runs_resumes_and_scores_held_out_instances(
    small_bank: InstanceBank, tmp_path: Path
) -> None:
    """A tiny bake-off writes its sessions once, resumes without rerunning and reports them.

    Args:
        small_bank (InstanceBank): Bank used to calibrate the instances in-process.
        tmp_path (Path): Directory for the results file.
    """
    from benchmarks.tuning.protocol import BASELINES
    from benchmarks.tuning.protocol import BakeOff
    from benchmarks.tuning.protocol import Settings
    from benchmarks.tuning.protocol import session_seed

    settings = Settings(
        str(small_bank.directory),
        budget=30,
        train=3,
        test=3,
        test_seeds=2,
        evaluations=2000,
        workers=1,
    )
    small_bank.calibrate(small_bank.generate("rastrigin", "train", 3))
    small_bank.calibrate(small_bank.generate("rastrigin", "test", 3))
    bake = BakeOff(settings, tmp_path / "results.jsonl")
    count = bake.run(["GRASP"], ["rastrigin"], ["random_halving"], reps=2)
    assert count == 1 + 2 + 2
    assert bake.run(["GRASP"], ["rastrigin"], ["random_halving"], reps=2) == 0
    lines = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert lines[0]["kind"] == "meta" and lines[0]["settings"]["budget"] == 30
    sessions = [x for x in lines if x["kind"] == "session"]
    assert {s["tuner"] for s in sessions} == {*BASELINES, "random_halving"}
    tuned = [s for s in sessions if s["tuner"] == "random_halving"]
    assert all(0 < s["runs_used"] <= 30 and len(s["test"]) == 3 for s in tuned)
    assert all(s["runs_used"] == 0 for s in sessions if s["tuner"] in BASELINES)
    assert session_seed("SA", "kp", "racing", 1) == session_seed("SA", "kp", "racing", 1)
    assert session_seed("SA", "kp", "racing", 1) != session_seed("SA", "kp", "racing", 2)
