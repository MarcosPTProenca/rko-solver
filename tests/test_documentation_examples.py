"""Execute every Python example in the README and the documentation pages.

Objective:
- Keep the published examples runnable: a snippet that raises, or that silently stopped
  matching the API, fails the suite instead of misleading a reader.

Technical approach:
- Extract each fenced Python block with its line number and run it in a fresh subprocess inside
  a temporary directory, so generated files never touch the repository.
- Blocks with an `if __name__ == "__main__":` guard are complete programs and run as their own
  script, which also makes spawned worker processes able to import their callbacks. Other
  blocks are fragments that run in order in one namespace per page, seeded with the page
  conventions (`decode`, `objective`, `cost`, `decoder`, `n_variables`, `dimension`, Knapsack
  data) from a small importable prelude module.
- `RKO.solve` is capped at half a second inside the examples, so budgets such as `solve(60.0)`
  exercise the code path without making the suite slow; quality is not asserted here.
- A block can opt out with an invisible HTML comment right before its fence:
  `<!-- doc-test: skip reason -->` or `<!-- doc-test: requires cpp -->` (also `numba`,
  `plotly`), which skips only when that capability is missing. Every page also finds sample
  `instance.json` and `my_instance.json` files in its working directory.

Flow (pseudocode):
1. Parse the Markdown files and collect python blocks with their markers.
2. For each page, run the complete programs, then the fragments in order.
3. Fail with the page, the line of the block and the traceback when one raises.

Important criteria:
- Every skip carries a reason that a reviewer can read in the Markdown source.
- Fragments may only rely on earlier blocks of the same page and on the prelude names.

Inputs/Outputs:
- Input: `README.md` and `docs/*.md`.
- Output: one test per page, failing on the first block that does not run.
"""

import dataclasses
import importlib.util
import json
import os
import pytest
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FENCE = re.compile(r"^```(?:python|py)[^\n]*\n(.*?)^```", re.S | re.M)
MARKER = re.compile(r"<!--\s*doc-test:\s*(skip|requires)\s+(.*?)\s*-->")
GUARD = re.compile(r'if __name__ == "__main__":')
CAPABILITIES = {
    "cpp": lambda: _cpp(),
    "numba": lambda: importlib.util.find_spec("numba") is not None,
    "plotly": lambda: importlib.util.find_spec("plotly") is not None,
}
PRELUDE = '''
import numpy as np


def decode(keys: np.ndarray) -> np.ndarray:
    """Map unit-interval keys to a numerical solution in [-5, 5]."""
    return 10.0 * np.asarray(keys) - 5.0


def objective(solution: np.ndarray) -> float:
    """Return the squared norm to minimize."""
    return float(np.dot(solution, solution))


decoder = decode
cost = objective
n_variables = dimension = 20
profits = np.array([10 + (index * 37) % 91 for index in range(n_variables)], dtype=np.int64)
weights = np.array([5 + (index * 53) % 47 for index in range(n_variables)], dtype=np.int64)
capacity = int(weights.sum() // 2)
'''
DRIVER = """
import json
import sys
import traceback

sys.path.insert(0, ".")
from doc_prelude import *

import rko_solver

_solve = rko_solver.RKO.solve


def _capped(self, time_limit):
    if self.max_evaluations is not None:
        return _solve(self, time_limit)
    return _solve(self, min(time_limit, 0.5))


rko_solver.RKO.solve = _capped


def run():
    namespace = dict(globals())
    namespace["__name__"] = "__main__"
    for block in json.load(open("blocks.json")):
        try:
            exec(compile(block["code"], f"<{block['where']}>", "exec"), namespace)
        except BaseException:
            print(f"FAILED {block['where']}", file=sys.stderr)
            traceback.print_exc()
            raise SystemExit(1)


if __name__ == "__main__":
    run()
"""
SCRIPT_HEAD = """
import sys

sys.path.insert(0, ".")
import rko_solver

_solve = rko_solver.RKO.solve
rko_solver.RKO.solve = lambda self, time_limit: _solve(
    self, time_limit if self.max_evaluations is not None else min(time_limit, 0.5)
)
"""


def _cpp() -> bool:
    """Tell whether the compiled extension includes the C++ engine.

    Returns:
        bool: True when `rko_solver._native.cpp_available()` is true.
    """
    try:
        from rko_solver import _native
    except ImportError:
        return False
    return bool(_native.cpp_available())


@dataclasses.dataclass(frozen=True)
class Block:
    """Describe one Python example found in a Markdown file.

    Attributes:
        where (str): `file:line` of the opening fence.
        code (str): Example source.
        marker (tuple[str, str] | None): Opt-out directive and its argument, if any.
    """

    where: str
    code: str
    marker: tuple[str, str] | None

    @property
    def skipped(self) -> str | None:
        """Explain why the block must not run in this environment.

        Returns:
            str | None: The reason, or None when the block runs.
        """
        if self.marker is None:
            return None
        kind, argument = self.marker
        if kind == "skip":
            return argument
        return None if CAPABILITIES[argument]() else f"requires {argument}"


def blocks(path: Path) -> list[Block]:
    """Extract the Python blocks and their markers from a Markdown file.

    Args:
        path (Path): Markdown file.

    Returns:
        list[Block]: Blocks in file order.
    """
    text = path.read_text(encoding="utf-8")
    found = []
    for match in FENCE.finditer(text):
        before = text[: match.start()].rstrip().splitlines()
        marker = MARKER.fullmatch(before[-1].strip()) if before else None
        line = text[: match.start()].count("\n") + 1
        found.append(
            Block(
                f"{path.relative_to(ROOT).as_posix()}:{line}",
                match.group(1),
                (marker.group(1), marker.group(2)) if marker else None,
            )
        )
    return found


PAGES = sorted(page for page in [ROOT / "README.md", *(ROOT / "docs").glob("*.md")] if blocks(page))


FIXTURES = {
    "instance.json": json.dumps({"lower": [-5, -5, -5], "upper": [5, 5, 5], "target": [1, 2, 3]}),
    "my_instance.json": (ROOT / "src/rko_solver/examples/instances/scheduling.json").read_text(
        encoding="utf-8"
    ),
}


def execute(command: list[str], directory: Path) -> subprocess.CompletedProcess[str]:
    """Run a Python subprocess in a clean directory with the package importable.

    Args:
        command (list[str]): Arguments after the interpreter.
        directory (Path): Working directory.

    Returns:
        subprocess.CompletedProcess[str]: Finished process with captured output.
    """
    environment = {**os.environ, "PYTHONPATH": str(directory), "MPLBACKEND": "Agg"}
    return subprocess.run(
        [sys.executable, *command],
        cwd=directory,
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.relative_to(ROOT).as_posix())
def test_every_python_example_on_the_page_runs(page: Path, tmp_path: Path) -> None:
    """Complete programs run as scripts and the fragments run in order after the prelude.

    Args:
        page (Path): Markdown file under test.
        tmp_path (Path): Clean working directory for generated files.
    """
    (tmp_path / "doc_prelude.py").write_text(textwrap.dedent(PRELUDE), encoding="utf-8")
    (tmp_path / "driver.py").write_text(textwrap.dedent(DRIVER), encoding="utf-8")
    for name, content in FIXTURES.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    fragments: list[dict[str, str]] = []
    ran = 0
    for block in blocks(page):
        if block.skipped:
            continue
        ran += 1
        if GUARD.search(block.code):
            script = tmp_path / "program.py"
            script.write_text(textwrap.dedent(SCRIPT_HEAD) + "\n" + block.code, encoding="utf-8")
            done = execute([str(script)], tmp_path)
            assert done.returncode == 0, f"{block.where} failed:\n{done.stdout}\n{done.stderr}"
        else:
            fragments.append({"where": block.where, "code": block.code})
    if fragments:
        (tmp_path / "blocks.json").write_text(json.dumps(fragments), encoding="utf-8")
        done = execute(["driver.py"], tmp_path)
        assert done.returncode == 0, f"{page.name} fragments failed:\n{done.stdout}\n{done.stderr}"
    if not ran:
        pytest.skip("every block on this page is skipped with a reason")


def test_every_skip_has_a_reason_and_requirements_are_known() -> None:
    """Opt-outs must explain themselves and name a known capability."""
    for page in PAGES:
        for block in blocks(page):
            if block.marker is None:
                continue
            kind, argument = block.marker
            assert kind in ("skip", "requires"), block.where
            if kind == "skip":
                assert len(argument.split()) >= 3, f"{block.where}: give a reason"
            else:
                assert argument in CAPABILITIES, f"{block.where}: unknown capability"
