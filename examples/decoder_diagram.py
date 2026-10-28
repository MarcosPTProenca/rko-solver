"""Post-run scheduling report with an explicitly requested static decoder diagram.

Objective:
- Demonstrate instance loading, ordinary solving and optional source structure reporting.

Technical approach:
- Reuse the packaged scheduling adapter and the standard RKO object interface.
- Inspect available Python source after solving; never re-run callbacks for the diagram.
- Attach independently requested post-run validation and static graph metadata to a report.

Flow (pseudocode):
1. Read packaged instance data and bind it through SchedulingProblem.
2. Configure RKO, solve and optionally reconstruct/check the returned schedule once.
3. Derive supported Python source structure without executing the decoder.
4. Write self-contained HTML/JSON into the local ignored results directory.

Important criteria:
- Source structure is not the original winning evaluation or full business semantics.
- Validation reconstruction is separately timed and explicitly distinguished from a trace.
- Diagram generation/rendering occurs outside optimization and changes no evaluation counts.

Inputs/Outputs:
- Input: packaged scheduling instance and fixed small example solver settings.
- Output: results/decoder-diagram.html and results/decoder-diagram.json.
"""

from examples import instance_pipeline
from pathlib import Path
from rko_solver.assessment import assess_result
from rko_solver.explain import DecoderDiagram
from rko_solver.reporting import OptimizationReport
from rko_solver.solver import RKO


def main() -> None:
    """Save a complete scheduling report with an opt-in static source diagram.

    Raises:
        ImportError: If optional HTML-report dependencies are unavailable.
        OSError: If instance/artifact files cannot be read or written.
        ValueError: If instance data, optimization settings or source structure are invalid.
    """
    path = Path(instance_pipeline.__file__).with_name("instances") / "scheduling.json"
    instance = instance_pipeline.read_data(path)
    problem = instance_pipeline.SchedulingProblem(instance)
    solver = RKO(
        problem.decode,
        problem.cost,
        problem.n_variables,
        algorithms=["SA", "PSO"],
        adaptation="fixed",
        execution="sequential",
        random_state=42,
        history_limit=1000,
    )
    result = solver.solve(time_limit=1.0)
    assessment = assess_result(result, decoder=problem.decode, validator=problem.validate)
    diagram = DecoderDiagram.from_callable(
        problem.decode, title="Scheduling decoder: static structure"
    )
    report = OptimizationReport(
        result,
        title="Scheduling with optional static decoder structure",
        assessment=assessment,
        decoder_diagram=diagram,
    )
    directory = Path("results")
    directory.mkdir(exist_ok=True)
    print(report.write_html(directory / "decoder-diagram.html"))
    print(report.write_json(directory / "decoder-diagram.json"))


if __name__ == "__main__":
    main()
