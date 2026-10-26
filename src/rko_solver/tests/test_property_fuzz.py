"""Property-based and fuzz tests across engines, parameters and objective values.

Objective:
- Find inputs that break an engine, by generating many bounded configurations and asserting
  invariants that hold for every valid one, instead of checking hand-picked examples.

Technical approach:
- Hypothesis strategies draw algorithms, adaptation modes, dimensions, seeds, evaluation limits
  and objective scales. Every size is small, so no generated case can allocate much memory.
- The packaged TOML parameter files are mutated one value at a time with wrong types, signed
  and huge numbers, `nan` and `inf` literals, arrays and tables; the run must then either
  succeed or raise `ValueError`, identically on the Python and Rust engines.
- Objective values at the edges of floating point (zero, subnormal, huge, non-finite) are fed
  through all three engines.

Flow (pseudocode):
1. Generate a configuration or a mutated parameter file.
2. Run it on each engine and record the outcome.
3. Assert the invariant: keys in [0, 1), cost recomputable from keys, exact evaluation count,
   reproducibility where promised, and one clean error type for invalid input.

Important criteria:
- `RKO_FUZZ_SCALE=N` multiplies every example count, for deeper scheduled runs.
- Hypothesis runs derandomized with no deadline, so the suite is stable in CI and a failing
  example is reproducible from its printed arguments.
- The C++ engine never takes part in reproducibility checks, which it cannot satisfy.
- Known differences between engines are pinned in their own test so they cannot drift unseen.
- Mutated-parameter runs use a one-second time limit: a configuration that makes an algorithm
  do no work (LNS with `t0` below its 0.01 cooling floor) legitimately runs until the deadline.

Inputs/Outputs:
- Input: generated configurations, parameter files and objective values.
- Output: invariants over results and error types.
"""

import hashlib
import numpy as np
import os
import pytest
import re
import shutil
import tempfile
import tomllib
from hypothesis import HealthCheck
from hypothesis import example
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from pathlib import Path
from rko_solver import RKO
from rko_solver import Callbacks
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.parameters.loader import _NAME_TO_FILE
from rko_solver.parameters.loader import default_params_dir

native = pytest.importorskip("rko_solver._native")
ALGORITHMS = st.sampled_from(sorted(MH_SPECS))
ADAPTATIONS = st.sampled_from(["fixed", "adaptive"])
SEEDS = st.integers(min_value=0, max_value=2**31 - 1)
LIMITS = st.integers(min_value=1, max_value=300)
SETTINGS = {"python": {"execution": "sequential"}, "rust": {"n_jobs": 1}, "cpp": {}}
needs_cpp = pytest.mark.skipif(not native.cpp_available(), reason="built without the C++ engine")
ENGINES = ["python", "rust", pytest.param("cpp", marks=needs_cpp)]
FILE_TO_NAME = {stem: name for name, stem in _NAME_TO_FILE.items()}
REPEATABLE = ["python", "rust"]
SCALE = max(1, int(os.environ.get("RKO_FUZZ_SCALE", "1")))
FUZZ = {"deadline": None, "derandomize": True, "suppress_health_check": list(HealthCheck)}


def decode(keys: np.ndarray) -> np.ndarray:
    """Return the keys unchanged.

    Args:
        keys (np.ndarray): Random keys.

    Returns:
        np.ndarray: The same keys.
    """
    return np.asarray(keys)


def cost(point: np.ndarray) -> float:
    """Return the squared distance to 0.3, a smooth objective with an interior optimum.

    Args:
        point (np.ndarray): Decoded keys.

    Returns:
        float: Objective to minimize.
    """
    return float(np.sum((np.asarray(point) - 0.3) ** 2))


def solve(engine: str, dimension: int, **options: object) -> object:
    """Run the default decoder and objective on one engine.

    Args:
        engine (str): Key of `SETTINGS`.
        dimension (int): Number of random keys.
        **options (object): Additional `RKO` settings.

    Returns:
        object: The `RKOResult`.
    """
    return RKO(
        Callbacks(decode, cost, dimension),
        engine=engine,
        **SETTINGS[engine],
        **options,
    ).solve(60.0)


@pytest.mark.parametrize("engine", ENGINES)
@settings(max_examples=12 * SCALE, **FUZZ)
@given(
    algorithm=ALGORITHMS,
    adaptation=ADAPTATIONS,
    dimension=st.integers(min_value=10, max_value=40),
    seed=SEEDS,
    limit=LIMITS,
)
def test_every_key_given_to_the_decoder_is_finite_and_in_the_unit_interval(
    engine: str, algorithm: str, adaptation: str, dimension: int, seed: int, limit: int
) -> None:
    """No algorithm on any engine ever hands the decoder a key outside `[0, 1)`.

    Args:
        engine (str): Engine under test.
        algorithm (str): Registered algorithm.
        adaptation (str): `fixed` or `adaptive`.
        dimension (int): Number of keys, at least ten so the C++ IPR is valid.
        seed (int): Random seed.
        limit (int): Evaluation limit.
    """
    seen: list[np.ndarray] = []

    def recording(keys: np.ndarray) -> np.ndarray:
        """Store a copy of the keys and return them.

        Args:
            keys (np.ndarray): Random keys.

        Returns:
            np.ndarray: The same keys.
        """
        seen.append(np.asarray(keys).copy())
        return np.asarray(keys)

    result = RKO(
        Callbacks(recording, cost, dimension),
        algorithms=[algorithm],
        adaptation=adaptation,
        max_evaluations=limit,
        random_state=seed,
        engine=engine,
        **SETTINGS[engine],
    ).solve(60.0)
    keys = np.concatenate(seen)
    assert np.all(np.isfinite(keys)) and keys.min() >= 0.0 and keys.max() < 1.0
    assert all(0.0 <= key < 1.0 for key in result.keys)


@pytest.mark.parametrize("engine", REPEATABLE)
@settings(max_examples=40 * SCALE, **FUZZ)
@given(
    algorithm=ALGORITHMS,
    adaptation=ADAPTATIONS,
    dimension=st.integers(min_value=1, max_value=40),
    seed=SEEDS,
    limit=LIMITS,
)
def test_results_are_consistent_for_any_dimension_algorithm_and_seed(
    engine: str, algorithm: str, adaptation: str, dimension: int, seed: int, limit: int
) -> None:
    """The cost is recomputable from the keys and the evaluation count is exact.

    Args:
        engine (str): Engine under test.
        algorithm (str): Registered algorithm.
        adaptation (str): `fixed` or `adaptive`.
        dimension (int): Number of keys, down to one.
        seed (int): Random seed.
        limit (int): Evaluation limit.
    """
    result = solve(
        engine,
        dimension,
        algorithms=[algorithm],
        adaptation=adaptation,
        max_evaluations=limit,
        random_state=seed,
    )
    assert len(result.keys) == dimension
    assert result.cost == cost(decode(np.asarray(result.keys)))
    assert [item.n_evaluations for item in result.searches] == [limit]
    assert result.searches[0].stop_reason == "max_evaluations"
    assert result.searches[0].random_state is not None


@pytest.mark.parametrize("engine", REPEATABLE)
@settings(max_examples=25 * SCALE, **FUZZ)
@given(
    algorithm=ALGORITHMS,
    adaptation=ADAPTATIONS,
    dimension=st.integers(min_value=1, max_value=40),
    seed=SEEDS,
    limit=LIMITS,
)
def test_repeating_a_seeded_run_gives_the_same_result(
    engine: str, algorithm: str, adaptation: str, dimension: int, seed: int, limit: int
) -> None:
    """Equal seeds and limits reproduce keys, cost and counts on the repeatable engines.

    Args:
        engine (str): Python sequential or one-worker Rust.
        algorithm (str): Registered algorithm.
        adaptation (str): `fixed` or `adaptive`.
        dimension (int): Number of keys.
        seed (int): Random seed.
        limit (int): Evaluation limit.
    """
    options = {
        "algorithms": [algorithm],
        "adaptation": adaptation,
        "max_evaluations": limit,
        "random_state": seed,
    }
    runs = [solve(engine, dimension, **options) for _ in range(2)]
    prints = [
        (run.cost, hashlib.sha256(repr(run.keys).encode()).digest(), run.n_evaluations)
        for run in runs
    ]
    assert prints[0] == prints[1]


@pytest.mark.parametrize("engine", ENGINES)
@settings(max_examples=20 * SCALE, **FUZZ)
@given(
    scale=st.floats(min_value=-1e300, max_value=1e300, allow_nan=False, allow_infinity=False),
    seed=SEEDS,
)
@example(scale=0.0, seed=1)
@example(scale=-0.0, seed=1)
@example(scale=5e-324, seed=1)
@example(scale=1e300, seed=1)
@example(scale=-1e300, seed=1)
def test_any_finite_objective_scale_gives_a_matching_finite_result(
    engine: str, scale: float, seed: int
) -> None:
    """Extreme but finite objective values, including subnormals and zeros, are handled.

    Args:
        engine (str): Engine under test.
        scale (float): Multiplier of the key sum, from subnormal to 1e300.
        seed (int): Random seed.
    """

    def scaled(point: np.ndarray) -> float:
        """Return the scaled key sum.

        Args:
            point (np.ndarray): Decoded keys.

        Returns:
            float: Finite objective of any sign.
        """
        return float(scale * np.sum(point))

    result = RKO(
        Callbacks(decode, scaled, 10),
        algorithms=["SA", "BRKGA"],
        max_evaluations=40,
        random_state=seed,
        engine=engine,
        **SETTINGS[engine],
    ).solve(60.0)
    assert np.isfinite(result.cost)
    assert result.cost == scaled(np.asarray(result.keys))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_objectives_fail_cleanly_and_leave_the_engine_usable(
    engine: str, value: float
) -> None:
    """A non-finite cost raises an error naming the problem, and the next solve still works.

    Args:
        engine (str): Engine under test.
        value (float): Non-finite objective value.
    """
    solver = RKO(
        Callbacks(decode, lambda point: value, 10),
        algorithms=["SA"],
        max_evaluations=20,
        random_state=1,
        engine=engine,
        **SETTINGS[engine],
    )
    with pytest.raises(Exception, match="finite"):
        solver.solve(5.0)
    assert solver.result_ is None
    assert solve(engine, 10, algorithms=["SA"], max_evaluations=20, random_state=1).n_evaluations


def test_known_difference_non_finite_cost_exception_type() -> None:
    """Pin the current difference so a future parity fix has to update this expectation.

    The Python engine raises `ValueError` for a non-finite cost while the Rust and C++ engines
    raise `RuntimeError`; both messages name the finite-cost requirement.
    """
    kinds = {}
    for engine in [name for name in SETTINGS if name != "cpp" or native.cpp_available()]:
        with pytest.raises(Exception) as caught:
            RKO(
                Callbacks(decode, lambda point: float("nan"), 10),
                algorithms=["SA"],
                max_evaluations=20,
                engine=engine,
                **SETTINGS[engine],
            ).solve(5.0)
        kinds[engine] = type(caught.value).__name__
    assert kinds["python"] == "ValueError"
    assert {kind for name, kind in kinds.items() if name != "python"} == {"RuntimeError"}


def parameter_files() -> dict[str, list[str]]:
    """Map each packaged TOML file to the keys that hold a value.

    Returns:
        dict[str, list[str]]: Parameter file mapped to its keys; files without keys are left out.
    """
    found = {}
    for path in sorted(default_params_dir().glob("*.toml")):
        text = path.read_text(encoding="utf-8")
        keys = re.findall(r"^([A-Za-z_]\w*)\s*=", text, flags=re.MULTILINE)
        if keys:
            found[path.name] = keys
    return found


FILES = parameter_files()
LITERALS = st.one_of(
    st.sampled_from(
        [
            "0",
            "-1",
            "2",
            "1e300",
            "-1e300",
            "1e-300",
            "nan",
            "inf",
            "-inf",
            "true",
            "false",
            '"text"',
            '""',
            "[]",
            "[1, 2]",
            "[[1], [2]]",
            "{a = 1}",
            "0x10",
            "1000000000000",
        ]
    ),
    st.integers(min_value=-50, max_value=300).map(str),
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False).map(repr),
)


def outcome(engine: str, directory: Path, stem: str, adaptation: str) -> str:
    """Solve with a parameter directory and classify what happened.

    Args:
        engine (str): `python` or `rust`.
        directory (Path): Parameter directory to use.
        stem (str): File stem, which selects the algorithm (all defaults for `config`).
        adaptation (str): `fixed` or `adaptive`.

    Returns:
        str: `ok`, `ValueError`, or the unexpected exception type.
    """
    try:
        RKO(
            Callbacks(decode, cost, 12),
            algorithms=[FILE_TO_NAME[stem]] if stem in FILE_TO_NAME else None,
            params_dir=directory,
            adaptation=adaptation,
            max_evaluations=15,
            random_state=1,
            engine=engine,
            **SETTINGS[engine],
        ).solve(1.0)
    except ValueError:
        return "ValueError"
    except BaseException as error:
        return type(error).__name__
    return "ok"


@settings(max_examples=150 * SCALE, **FUZZ)
@given(
    name=st.sampled_from(sorted(FILES)),
    chooser=st.data(),
    literal=LITERALS,
    adaptation=ADAPTATIONS,
)
def test_mutated_parameter_files_either_run_or_raise_value_error_on_every_engine(
    name: str, chooser: st.DataObject, literal: str, adaptation: str
) -> None:
    """One wrong value in a packaged TOML file never causes anything but a clean `ValueError`.

    The Python and Rust engines must also agree on whether the file is acceptable, which proves
    that both validate the same parameters before running.

    Args:
        name (str): Parameter file to mutate.
        chooser (st.DataObject): Draws the key to overwrite.
        literal (str): TOML literal to write in place of its value.
        adaptation (str): `fixed` or `adaptive`.
    """
    key = chooser.draw(st.sampled_from(FILES[name]))
    source = default_params_dir() / name
    pattern = rf"^({re.escape(key)}\s*=\s*)(.+?)((?:\s+#.*)?)$"
    mutated = re.sub(
        pattern,
        lambda m: m.group(1) + literal + m.group(3),
        source.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    try:
        tomllib.loads(mutated)
    except tomllib.TOMLDecodeError:
        return
    with tempfile.TemporaryDirectory(prefix="rko-fuzz-") as temporary:
        directory = Path(temporary) / "params"
        shutil.copytree(default_params_dir(), directory)
        (directory / name).write_text(mutated, encoding="utf-8")
        outcomes = {
            engine: outcome(engine, directory, source.stem, adaptation) for engine in REPEATABLE
        }
    assert set(outcomes.values()) <= {"ok", "ValueError"}, (key, literal, outcomes)
    assert len(set(outcomes.values())) == 1, (key, literal, outcomes)
