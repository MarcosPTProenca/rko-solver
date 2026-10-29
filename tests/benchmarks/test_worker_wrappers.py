"""Spawn-worker wrappers used by the startup benchmarks.

Objective:
- Keep the benchmark wrappers around the process worker compatible with its keyword contract.

Technical approach:
- Bind every keyword declared in `WorkerCaptureOptions` to each wrapper signature, so a keyword
  added to the production worker that a wrapper drops fails here instead of in a scheduled job.

Flow (pseudocode):
1. Read the keyword names the process runner forwards to its worker.
2. Bind a placeholder positional call plus those keywords to each wrapper's signature.
3. Fail when a wrapper rejects a keyword the production worker accepts.

Important criteria:
- No process is spawned; the check is purely on the callable signatures.

Inputs/Outputs:
- Input: the wrapper callables of `benchmarks.runtime.startup` and
  `benchmarks.runtime.startup_stages`.
- Output: assertions on signature compatibility.
"""

import importlib
import inspect
import pytest
from collections.abc import Callable
from pathlib import Path
from rko_solver.orchestration.types import WorkerCaptureOptions


def wrappers(monkeypatch: pytest.MonkeyPatch) -> dict[str, Callable[..., None]]:
    """Import the benchmark modules that replace the production process worker.

    Args:
        monkeypatch (pytest.MonkeyPatch): Adds the benchmark directory to the import path.

    Returns:
        dict[str, Callable[..., None]]: Wrapper callable by benchmark module name.
    """
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "benchmarks"))
    startup = importlib.import_module("benchmarks.runtime.startup")
    stages = importlib.import_module("benchmarks.runtime.startup_stages")
    return {
        "startup": startup._timed_worker,
        "startup_stages": stages.StartupProbe.worker,
    }


@pytest.mark.parametrize("module", ["startup", "startup_stages"])
def test_wrapper_accepts_every_worker_keyword(module: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the full set of worker keywords to the benchmark wrapper's signature.

    Args:
        module (str): Benchmark module whose wrapper is checked.
        monkeypatch (pytest.MonkeyPatch): Adds the benchmark directory to the import path.

    Raises:
        AssertionError: If the wrapper rejects a keyword the production worker accepts.
    """
    keywords = dict.fromkeys(WorkerCaptureOptions.__annotations__)
    assert keywords
    inspect.signature(wrappers(monkeypatch)[module]).bind(*range(4), **keywords)
