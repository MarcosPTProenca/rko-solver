"""Contract tests for the upper bounds on durations, counts and allocation sizes.

Objective:
- Verify that absurd settings are rejected with a clear `ValueError` before any engine starts,
  identically on every engine, and that the documented boundaries are still accepted.

Technical approach:
- Every case that would allocate or wait absurdly runs in one child interpreter whose address
  space is capped where the platform allows it (Linux; macOS refuses the cap and Windows has none), so a regression fails a test with a `MemoryError` instead of
  exhausting the machine. The child prints one JSON line mapping each case to its outcome.
- Direct calls into the native extension check the defense-in-depth validation of the Rust
  `Config`, which must raise instead of panicking.

Flow (pseudocode):
1. Build the cases: oversized times, evaluation limits, populations and pools.
2. Run them in the capped child and collect `ValueError` messages or `ok`.
3. Assert the exact outcome of every case and the unchanged boundaries.

Important criteria:
- A rejected case must not take noticeable time or memory.
- Boundary values are accepted when a small `max_evaluations` ends the run quickly.

Inputs/Outputs:
- Input: oversized and boundary settings for `RKO`.
- Output: a message naming the setting for every rejected case, `ok` for every boundary case.
"""

import json
import pytest
import subprocess
import sys
import textwrap

native = pytest.importorskip("rko_solver._native")
CHILD = """
import json
import sys

if sys.platform != "win32":
    import resource

    limit = 3 * 1024**3
    try:
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except (ValueError, OSError):
        pass

import numpy as np
from rko_solver import RKO, Callbacks, _native


def decode(keys):
    return np.asarray(keys)


def cost(point):
    return float(np.sum(np.asarray(point) ** 2))


def fixed(algorithm, key, value):
    return {
        "algorithms": [algorithm],
        "adaptation": "fixed",
        "hyperparameters": {algorithm: {key: value}},
    }


ENGINES = {
    "python": {"engine": "python", "execution": "sequential"},
    "rust": {"engine": "rust", "n_jobs": 1},
    "cpp": {"engine": "cpp"},
}
CASES = {}
for engine in ENGINES:
    CASES[f"{engine}/time 1e300"] = (engine, {}, 1e300, 10)
    CASES[f"{engine}/time over a year"] = (engine, {}, 31_536_001.0, 10)
    CASES[f"{engine}/time a year"] = (engine, {}, 31_536_000.0, 10)
    CASES[f"{engine}/evaluations 2**63"] = (engine, {"max_evaluations": 2**63}, 0.2, 10)
    CASES[f"{engine}/evaluations 2**63-1"] = (engine, {"max_evaluations": 2**63 - 1}, 0.2, 10)
    CASES[f"{engine}/evaluations 10**30"] = (engine, {"max_evaluations": 10**30}, 0.2, 10)
    CASES[f"{engine}/pool 10**12"] = (engine, {"pool_size": 10**12}, 0.2, 10)
    CASES[f"{engine}/pool cells"] = (engine, {"pool_size": 500_000}, 0.2, 1000)
    CASES[f"{engine}/auto_time"] = (engine, {"auto_time": 1e300}, 0.2, 10)
    for algorithm, key in (("BRKGA", "p"), ("GA", "size_pop"), ("PSO", "p_size")):
        huge = fixed(algorithm, key, 10**12)
        CASES[f"{engine}/{algorithm}.{key} 10**12"] = (engine, huge, 0.2, 10)
    CASES[f"{engine}/BRKGA.p cells"] = (engine, fixed("BRKGA", "p", 500_000), 0.2, 1000)
    CASES[f"{engine}/BRKGA.p boundary"] = (engine, fixed("BRKGA", "p", 1000), 0.2, 1000)
outcomes = {}
for label, (engine, options, time_limit, size) in CASES.items():
    settings = {"algorithms": ["SA"], "max_evaluations": 20, "random_state": 1}
    settings.update(options)
    try:
        RKO(Callbacks(decode, cost, size), **ENGINES[engine], **settings).solve(time_limit)
        outcomes[label] = "ok"
    except ValueError as error:
        outcomes[label] = "ValueError: " + str(error)
    except BaseException as error:
        outcomes[label] = type(error).__name__ + ": " + str(error)[:80]
problem = _native.KnapsackProblem([3, 4], [2, 3], 4)
spec = [("SA", [100.0, 1e5, 0.98, 0.01, 0.05], [], False)]
for engine in ("rust", "cpp"):
    for label, args in (
        ("time 1e300", (1e300, 1, 10, None, 0, 1, False, engine, None)),
        ("pool over bound", (1.0, 1, 1_000_001, None, 0, 1, False, engine, None)),
    ):
        try:
            _native.run(problem, spec, *args)
            outcomes[f"native-{engine}/{label}"] = "ok"
        except ValueError as error:
            outcomes[f"native-{engine}/{label}"] = "ValueError: " + str(error)
        except BaseException as error:
            outcomes[f"native-{engine}/{label}"] = type(error).__name__ + ": " + str(error)[:80]
print(json.dumps(outcomes))
"""


@pytest.fixture(scope="module")
def outcomes() -> dict[str, str]:
    """Run every oversized and boundary case once in a memory-capped child interpreter.

    Returns:
        dict[str, str]: Case label mapped to `ok` or the error type and message.
    """
    done = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(CHILD)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


ENGINE_LABELS = [
    "python",
    "rust",
    pytest.param(
        "cpp",
        marks=pytest.mark.skipif(not native.cpp_available(), reason="built without the C++ engine"),
    ),
]


@pytest.mark.parametrize("engine", ENGINE_LABELS)
@pytest.mark.parametrize(
    "case, setting",
    [
        ("time 1e300", "time_limit"),
        ("time over a year", "time_limit"),
        ("evaluations 2**63", "max_evaluations"),
        ("evaluations 10**30", "max_evaluations"),
        ("pool 10**12", "pool_size"),
        ("pool cells", "pool_size"),
        ("auto_time", "auto_time"),
        ("BRKGA.p 10**12", "BRKGA.p"),
        ("GA.size_pop 10**12", "GA.size_pop"),
        ("PSO.p_size 10**12", "PSO.p_size"),
        ("BRKGA.p cells", "BRKGA.p"),
    ],
)
def test_oversized_settings_are_rejected_before_any_engine_starts(
    outcomes: dict[str, str], engine: str, case: str, setting: str
) -> None:
    """Each oversized value raises a `ValueError` that names the offending setting.

    Args:
        outcomes (dict[str, str]): Results of the capped child run.
        engine (str): Engine the case was dispatched to.
        case (str): Case label suffix.
        setting (str): Setting name that must appear in the message.
    """
    result = outcomes[f"{engine}/{case}"]
    assert result.startswith("ValueError:") and setting in result, result


@pytest.mark.parametrize("engine", ENGINE_LABELS)
@pytest.mark.parametrize("case", ["time a year", "evaluations 2**63-1", "BRKGA.p boundary"])
def test_documented_boundaries_are_still_accepted(
    outcomes: dict[str, str], engine: str, case: str
) -> None:
    """The largest accepted values run normally when `max_evaluations` ends the search.

    Args:
        outcomes (dict[str, str]): Results of the capped child run.
        engine (str): Engine the case was dispatched to.
        case (str): Boundary case label suffix.
    """
    assert outcomes[f"{engine}/{case}"] == "ok"


@pytest.mark.parametrize(
    "engine",
    [
        "rust",
        pytest.param(
            "cpp",
            marks=pytest.mark.skipif(
                not native.cpp_available(), reason="built without the C++ engine"
            ),
        ),
    ],
)
@pytest.mark.parametrize("case", ["time 1e300", "pool over bound"])
def test_native_engines_reject_bad_configurations_instead_of_panicking(
    outcomes: dict[str, str], engine: str, case: str
) -> None:
    """Calling the extension directly with an unbounded value raises `ValueError`, not a panic.

    Args:
        outcomes (dict[str, str]): Results of the capped child run.
        engine (str): Native engine called directly.
        case (str): Direct-call case label.
    """
    result = outcomes[f"native-{engine}/{case}"]
    assert result.startswith("ValueError:"), result
    assert "Panic" not in result


@pytest.mark.parametrize("value", [True, 1.5, "10", 0, -1])
def test_non_integer_evaluation_limits_still_fail_cleanly(value: object) -> None:
    """Types and signs the bound check does not cover keep their error.

    Args:
        value (object): Invalid evaluation limit.
    """
    from rko_solver import RKO

    with pytest.raises(ValueError, match="max_evaluations"):
        RKO(lambda keys: keys, lambda point: 0.0, 3, max_evaluations=value).solve(0.1)
