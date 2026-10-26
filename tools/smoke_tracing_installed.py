"""Exercise original-decision evidence from installed core artifacts on all Python execution modes.

Objective:
- Catch missing package resources, identity transport or optional-dependency regressions.

Technical approach:
- Run the installed scheduling callbacks with real sequential/thread/spawn portfolios and prefill.
- Verify public immutable evidence and report JSON without replay or source-tree imports.

Flow (pseudocode):
1. Reject accidental source-package imports.
2. Run fixed small traced portfolios, including genuine manager/spawn transport.
3. Check original bindings, primitive public costs and immutable pickle round trips.
4. Snapshot report JSON; optionally exercise offline HTML with the reports extra.

Important criteria:
- Core witness/report-JSON behavior does not require Plotly or HTTP dependencies.
- Events are explicit authored observations, not semantic extraction or execution certificates.

Inputs/Outputs:
- Input: isolated installed SDK, optionally reports extra and --reports flag.
- Output: assertions and a short nonprivate success marker.
"""

import argparse
import pickle
import rko_solver
from pathlib import Path
from rko_solver.examples.instance_pipeline import SchedulingInstance
from rko_solver.examples.instance_pipeline import SchedulingProblem
from rko_solver.reporting import OptimizationReport
from rko_solver.solver import RKO
from rko_solver.tracing import TraceConfig
from tempfile import TemporaryDirectory


class InstalledTraceSmoke:
    """Check installed API and execution modes without importing source tests or snapshots.

    Responsibilities:
    - Verify actual owned process transport and independent reports dependencies.

    Attributes:
        None: All data is local to one bounded temporary invocation.
    """

    @staticmethod
    def main() -> None:
        """Run small real installed portfolios and verify their immutable observation bindings.

        Raises:
            AssertionError: If installed evidence, public numeric types or transport is incorrect.
        """
        parser = argparse.ArgumentParser(description="Check installed original-decision evidence.")
        parser.add_argument("--reports", action="store_true")
        args = parser.parse_args()
        assert "/src/rko_solver/" not in str(Path(rko_solver.__file__).resolve()).replace("\\", "/")
        problem = SchedulingProblem(SchedulingInstance("installed-trace", (3, 2, 5), (3, 6, 9)))
        with TemporaryDirectory(prefix="rko-installed-trace-") as temporary:
            for execution in ("sequential", "thread", "process"):
                result = RKO(
                    problem.decode,
                    problem.cost,
                    problem.n_variables,
                    algorithms=["SA", "PSO"],
                    execution=execution,
                    n_jobs=1 if execution == "process" else None,
                    prefill_pool=True,
                    pool_size=3,
                    adaptation="fixed",
                    random_state=42,
                    history_limit=4,
                    trace=TraceConfig(),
                ).solve(time_limit=0.04)
                assert result.trace is not None
                assert result.trace.state == "recorded", (
                    execution,
                    result.trace.state,
                    result.trace.reason,
                )
                assert result.trace.matches(result.keys, result.cost)
                assert result.trace.events[0].action == "priority_order"
                assert type(result.cost) is float
                assert all(
                    type(point.cost) is float
                    for search in result.searches
                    for point in search.history
                )
                copy = pickle.loads(pickle.dumps(result))
                assert copy.trace == result.trace
                report = OptimizationReport(result)
                assert (
                    report.to_dict()["decision_trace"]["evaluation_id"]
                    == result.trace.evaluation_id
                )
                report.write_json(Path(temporary) / f"{execution}.json")
                if args.reports:
                    report.write_html(Path(temporary) / f"{execution}.html")
        print(
            "Installed original-decision evidence: sequential/thread/process/prefill/JSON passed."
        )


if __name__ == "__main__":
    InstalledTraceSmoke.main()
