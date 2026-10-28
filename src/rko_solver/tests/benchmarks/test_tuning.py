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
from hypothesis import given
from hypothesis import settings as hypothesis_settings
from hypothesis import strategies as st
from pathlib import Path
from rko_solver import RKO
from rko_solver import Numba
from rko_solver.components.validation import ParameterValidator

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


@pytest.mark.parametrize("profile", ["original", "expanded"])
def test_decoded_points_are_valid_for_the_solver_including_the_corners(profile: str) -> None:
    """Run boundary and random configurations of both profiles through the actual solver.

    Args:
        profile (str): Search-space profile to exercise.
    """
    from benchmarks.tuning.spaces import SPACE_PROFILES

    compiled = pytest.importorskip("numba").njit(lambda keys: float(np.sum((keys - 0.3) ** 2)))
    rng = random.Random(3)
    for name, space in SPACE_PROFILES[profile].items():
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


@pytest.mark.parametrize("profile", ["original", "expanded"])
def test_brkga_points_always_decode_with_a_mutant_and_keep_valid_pm(profile: str) -> None:
    """Repair `pm` upward only when the population would round to zero mutants.

    Args:
        profile (str): Search-space profile to exercise.
    """
    from benchmarks.tuning.spaces import SPACE_PROFILES

    space = SPACE_PROFILES[profile]["BRKGA"]
    rng = random.Random(11)
    points = [[0.0] * 4, [1.0] * 4] + [[rng.random() for _ in range(4)] for _ in range(500)]
    repaired = 0
    for point in points:
        raw = {spec.name: spec.decode(unit) for spec, unit in zip(space.specs, point, strict=True)}
        config = space.decode(point)
        assert int(config["p"] * config["pm"]) >= 1, point
        if (
            int(raw["p"] * raw["pm"]) >= 1
            and int(raw["p"] * raw["pm"]) < raw["p"] * (1 - raw["pe"]) - 1
        ):
            assert config["pm"] == raw["pm"], point
        repaired += int(raw["p"] * raw["pm"]) < 1
    assert repaired > 0


@pytest.mark.parametrize("name", list(SPACES))
def test_decoded_values_are_plain_numbers_even_for_numpy_points(name: str) -> None:
    """The solver rejects numpy scalars, so decoding must return exact `int` or `float`.

    Args:
        name (str): Algorithm name.
    """
    space = SPACES[name]
    for point in (np.random.default_rng(1).random(space.dimension), [0.3] * space.dimension):
        assert all(type(value) in (int, float) for value in space.decode(point).values())


def test_a_failing_session_stops_the_bake_off_instead_of_being_swallowed(
    small_bank: InstanceBank, tmp_path: Path
) -> None:
    """The error of a session that fails inside a worker propagates out of `BakeOff.run`.

    Args:
        small_bank (InstanceBank): Bank used for its directory.
        tmp_path (Path): Directory for the results file.
    """
    from benchmarks.tuning.protocol import BakeOff
    from benchmarks.tuning.protocol import Settings

    small_bank.calibrate(small_bank.generate("rastrigin", "train", 1))
    small_bank.calibrate(small_bank.generate("rastrigin", "test", 1))
    settings = Settings(str(small_bank.directory), budget=10, train=1, test=1, workers=1)
    bake = BakeOff(settings, tmp_path / "results.jsonl")
    with pytest.raises(KeyError, match="NOPE"):
        bake.run(["NOPE"], ["rastrigin"], ["random_halving"], reps=1)
    assert [x for x in (tmp_path / "results.jsonl").read_text().splitlines()][1:] == []


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


def test_the_bank_keeps_compiled_objectives_of_one_family_only(small_bank: InstanceBank) -> None:
    """Switching family drops the compiled objectives of the previous one.

    Args:
        small_bank (InstanceBank): Bank with cheap calibration.
    """
    tsp = small_bank.generate("tsp", "train", 2)
    rastrigin = small_bank.generate("rastrigin", "train", 1)
    small_bank.objective(tsp[0])
    small_bank.objective(tsp[1])
    assert set(small_bank._compiled) == {tsp[0].name, tsp[1].name}
    small_bank.objective(rastrigin[0])
    assert set(small_bank._compiled) == {rastrigin[0].name}


def test_the_bank_builds_one_numba_problem_per_instance(small_bank: InstanceBank) -> None:
    """Repeated runs on one instance reuse its `Numba` problem object.

    Args:
        small_bank (InstanceBank): Bank with cheap calibration.
    """
    instance = small_bank.calibrate(small_bank.generate("rastrigin", "train", 1))[0]
    config = SPACES["GRASP"].default()
    for seed in range(3):
        small_bank.cost(instance, "GRASP", config, seed)
    assert set(small_bank._problems) == {instance.name}
    problem = small_bank._problems[instance.name]
    small_bank.cost(instance, "GRASP", config, 9)
    assert small_bank._problems[instance.name] is problem


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
                    "tune_seconds": 1.0,
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
    assert "| SA on kp | good (" in report and "Best tuner per cell" in report
    low_good = analysis.pair_interval(data["good"], data["bad"])
    assert low_good[2] < 0 and low_good[0] == pytest.approx(-0.4, abs=0.05)
    assert (
        analysis.pair_interval(data["same"], data["same"])[1]
        <= 0
        <= analysis.pair_interval(data["same"], data["same"])[2]
    )
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


def test_confirmation_scores_the_default_and_chosen_configurations_and_resumes(
    small_bank: InstanceBank, tmp_path: Path
) -> None:
    """The longer-budget check scores each configuration once, resumes and summarizes.

    Args:
        small_bank (InstanceBank): Bank used to calibrate the instances in-process.
        tmp_path (Path): Directory for the result files.
    """
    from benchmarks.tuning.confirm import Confirmation
    from benchmarks.tuning.protocol import BakeOff
    from benchmarks.tuning.protocol import Settings

    small_bank.calibrate(small_bank.generate("rastrigin", "train", 3))
    small_bank.calibrate(small_bank.generate("rastrigin", "test", 3))
    settings = Settings(
        str(small_bank.directory),
        budget=30,
        train=3,
        test=3,
        test_seeds=1,
        evaluations=2000,
        workers=1,
    )
    results = tmp_path / "results.jsonl"
    BakeOff(settings, results).run(["GRASP"], ["rastrigin"], ["random_halving"], reps=2)
    check = Confirmation(
        results, tmp_path / "confirm.jsonl", small_bank.directory, "random_halving"
    )
    assert check.run(4000, 2, 1, 1) == 3
    assert check.run(4000, 2, 1, 1) == 0
    lines = [json.loads(x) for x in (tmp_path / "confirm.jsonl").read_text().splitlines()]
    assert sorted(x["label"] for x in lines) == ["default", "random_halving-0", "random_halving-1"]
    assert all(len(x["scores"]) == 2 and x["evaluations"] == 4000 for x in lines)
    table = check.report(4000, resamples=200)
    assert "| GRASP on rastrigin |" in table and "Verdict" in table


@pytest.mark.parametrize("algorithm", ["SA", "BRKGA"])
def test_expanded_spaces_contain_the_original_and_widen_binding_axes(algorithm: str) -> None:
    """Keep the first study reproducible while widening its binding parameter axes.

    Args:
        algorithm (str): Algorithm whose binding ranges must expand.
    """
    from benchmarks.tuning.spaces import SPACE_PROFILES

    assert SPACE_PROFILES["original"] is SPACES
    original = SPACES[algorithm]
    expanded = SPACE_PROFILES["expanded"][algorithm]
    for old, new in zip(original.specs, expanded.specs, strict=True):
        assert old.name == new.name and new.low <= old.low <= old.high <= new.high
    widened = {
        new.name
        for old, new in zip(original.specs, expanded.specs, strict=True)
        if new.low < old.low or new.high > old.high
    }
    assert widened == (
        {"sa_max", "t0", "alpha", "beta_min", "beta_max"} if algorithm == "SA" else {"p"}
    )
    for value in (0.0, 0.5, 1.0):
        config = expanded.decode([value] * expanded.dimension)
        assert all(type(item) is float for item in config.values())


@pytest.mark.parametrize("family", FAMILIES)
def test_seed_offset_isolates_new_instances_and_calibration(tmp_path: Path, family: str) -> None:
    """A new cohort cannot reuse the first study's instance files or calibration.

    Args:
        tmp_path (Path): Parent directory shared by the two banks.
        family (str): Problem family.
    """
    old = InstanceBank(tmp_path)
    new = InstanceBank(tmp_path, seed_offset=1_000_000)
    old_item = old.generate(family, "test", 1)[0]
    new_item = new.generate(family, "test", 1)[0]
    assert old_item.path != new_item.path
    assert old_item.path.read_bytes() != new_item.path.read_bytes()
    again = InstanceBank(tmp_path, seed_offset=1_000_000).generate(family, "test", 1)[0]
    assert new_item.path.read_bytes() == again.path.read_bytes()
    old_calibration = {old_item.name: {"best": -5.0, "anchor": 5.0}}
    (old.directory / f"calibration-{family}.json").write_text(json.dumps(old_calibration))
    assert InstanceBank(tmp_path).generate(family, "test", 1)[0].best == -5.0
    assert InstanceBank(tmp_path, seed_offset=1_000_000).generate(family, "test", 1)[0].best == 0.0


@pytest.mark.parametrize(
    "changed", [{"space_profile": "expanded"}, {"seed_offset": 1_000_000}, {"evaluations": 100_000}]
)
def test_resume_refuses_a_different_experiment(tmp_path: Path, changed: dict[str, object]) -> None:
    """Reject mixing profiles, cohorts or evaluation budgets in an existing results file.

    Args:
        tmp_path (Path): Directory for synthetic metadata.
        changed (dict[str, object]): Setting changed from the stored experiment.
    """
    from benchmarks.tuning.protocol import BakeOff
    from benchmarks.tuning.protocol import Settings
    from dataclasses import asdict
    from dataclasses import replace

    settings = Settings(str(tmp_path))
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({"kind": "meta", "settings": asdict(settings)}) + "\n")
    with pytest.raises(ValueError, match="settings"):
        BakeOff(replace(settings, **changed), path).done()
    assert BakeOff(replace(settings, workers=2), path).done() == set()


def test_expanded_session_and_confirmation_share_the_fresh_cohort(
    small_bank: InstanceBank, tmp_path: Path
) -> None:
    """Spawned sessions use the selected ranges and confirmation retains the cohort.

    Args:
        small_bank (InstanceBank): Bank whose calibration budgets are reduced by the fixture.
        tmp_path (Path): Parent folder for the fresh cohort and results.
    """
    from benchmarks.tuning.confirm import Confirmation
    from benchmarks.tuning.protocol import BakeOff
    from benchmarks.tuning.protocol import Settings
    from benchmarks.tuning.protocol import session_seed
    from benchmarks.tuning.spaces import SPACE_PROFILES

    bank = InstanceBank(tmp_path, evaluations=2000, seed_offset=1_000_000)
    for role in ("train", "test"):
        bank.calibrate(bank.generate("rastrigin", role, 2))
    settings = Settings(
        str(tmp_path),
        train=2,
        test=2,
        test_seeds=1,
        evaluations=2000,
        workers=1,
        space_profile="expanded",
        seed_offset=1_000_000,
    )
    results = tmp_path / "expanded.jsonl"
    bake = BakeOff(settings, results)
    assert bake.run(["SA"], ["rastrigin"], [], 1) == 2
    records = [json.loads(line) for line in results.read_text().splitlines()]
    chosen = next(row for row in records if row.get("tuner") == "random")
    seed = session_seed("SA", "rastrigin", "random", 0)
    point = np.random.default_rng(seed).random(5)
    assert chosen["config"] == SPACE_PROFILES["expanded"]["SA"].decode(point)
    assert chosen["config"] != SPACES["SA"].decode(point)
    check = Confirmation(results, tmp_path / "confirm.jsonl", tmp_path, "random")
    assert {task[0] for task in check.tasks(4000, 2, 1)} == {str(bank.directory)}


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False), min_size=5, max_size=5))
@hypothesis_settings(max_examples=100, derandomize=True)
def test_expanded_decode_preserves_parameter_domains(point: list[float]) -> None:
    """Generated unit-cube points remain finite, bounded and valid after decoding.

    Args:
        point (list[float]): Five generated unit coordinates, including exact boundaries.
    """
    from benchmarks.tuning.spaces import SPACE_PROFILES

    for space in SPACE_PROFILES["expanded"].values():
        config = space.decode(point[: space.dimension])
        assert all(type(value) is float and np.isfinite(value) for value in config.values())
        for spec in space.specs:
            assert spec.low * (1 - 1e-12) <= config[spec.name] <= spec.high * (1 + 1e-12)
        ParameterValidator._values(space.algorithm, config)


@pytest.mark.parametrize("suffix, count, field", [("", 63, "test"), ("-confirm", 33, "scores")])
def test_expanded_reference_files_are_complete_and_paired(
    suffix: str, count: int, field: str
) -> None:
    """Reference files retain matching cohorts, complete paired records and finite measurements.

    Args:
        suffix (str): Empty for the primary study or the longer-budget confirmation suffix.
        count (int): Expected number of measurement records per profile.
        field (str): Key holding each record's per-instance scores.
    """
    root = Path(__file__).resolve().parents[4] / "benchmarks"
    measurements = {}
    for profile in ("original", "expanded"):
        path = root / f"tuning-wide-{profile}{suffix}-reference.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[0]["source_sha"].startswith("2455be0")
        assert rows[0]["settings"]["seed_offset"] == 1_000_000
        assert rows[0]["settings"]["space_profile"] == profile
        records = rows[1:]
        assert len(records) == count
        keyed = {(r["family"], r.get("label", (r.get("tuner"), r.get("rep")))): r for r in records}
        assert len(keyed) == count
        assert all(r["algorithm"] == "SA" and np.isfinite(r[field]).all() for r in records)
        measurements[profile] = keyed
    assert measurements["original"].keys() == measurements["expanded"].keys()
    for key, first in measurements["original"].items():
        second = measurements["expanded"][key]
        assert len(first[field]) == len(second[field]) == (10 if suffix else 20)
        if first.get("label", first.get("tuner")) == "default":
            assert first[field] == second[field]
