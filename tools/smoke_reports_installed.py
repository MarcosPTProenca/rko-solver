"""Exercise optional reports from a clean installed prerelease outside the checkout.

Objective:
- Verify that the reports extra produces offline artifacts from an ordinary managed solve.

Technical approach:
- Use importable opaque numerical callbacks and inspect installed package provenance.
- Render bounded history and explicitly requested static source structure after solving.

Flow (pseudocode):
1. Reject source/editable optimizer imports and run a small sequential solve.
2. Attach static decoder structure without replaying the callback for the diagram.
3. Write/read temporary HTML and JSON and check retained numerical metadata.

Important criteria:
- This is an installation/feature smoke, not performance or feasibility evidence.
- Generated reports are temporary and never part of a distribution.

Inputs/Outputs:
- Input: installed rko-solver[reports] in an isolated interpreter.
- Output: assertions and temporary artifacts deleted at completion.
"""

import json
import numpy as np
import rko_solver
import tempfile
from pathlib import Path
from rko_solver import RKO
from rko_solver.explain import DecoderDiagram
from rko_solver.reporting import OptimizationReport


def decode(keys: np.ndarray) -> np.ndarray:
    """Scale a detached random-key vector into numerical candidate coordinates.

    Args:
        keys (np.ndarray): Finite unit-interval random keys.

    Returns:
        np.ndarray: Independently computed bounded coordinates.
    """
    return 2.0 * keys - 1.0


def cost(solution: np.ndarray) -> float:
    """Compute a deterministic finite squared norm.

    Args:
        solution (np.ndarray): Candidate numerical coordinates.

    Returns:
        float: Nonnegative minimization objective.
    """
    return float(np.dot(solution, solution))


def main() -> None:
    """Generate and inspect installed optional reports without opening a browser.

    Raises:
        AssertionError: If installed provenance or exported result metadata is wrong.
        OSError: If temporary artifacts cannot be written/read.
        ImportError: If the requested reports extra is missing.
        ValueError: If solver or report validation fails.
    """
    root = Path(__file__).resolve().parents[1]
    assert not Path(rko_solver.__file__).resolve().is_relative_to(root / "src")
    result = RKO(
        decode,
        cost,
        4,
        algorithms=["SA"],
        adaptation="fixed",
        execution="sequential",
        random_state=42,
        history_limit=8,
    ).solve(0.05)
    assert result.cost == cost(decode(np.asarray(result.keys)))
    report = OptimizationReport(result, decoder_diagram=DecoderDiagram.from_callable(decode))
    with tempfile.TemporaryDirectory(prefix="rko-report-smoke-") as directory:
        html = report.write_html(Path(directory) / "report.html").read_text(encoding="utf-8")
        exported = json.loads(report.write_json(Path(directory) / "report.json").read_text())
        assert "plotly.js v" in html and "not an execution trace" in html
        assert exported["summary"]["cost"] == result.cost
        assert exported["decoder_diagram"]["origin"] == "source_derived"
    print("Installed reports extra passed.")


if __name__ == "__main__":
    main()
