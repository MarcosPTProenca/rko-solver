"""Offline interactive analysis of completed random-key optimization runs.

Objective:
- Export honest single-run summaries and bounded convergence traces as standalone HTML.
- Preserve data-only provenance without leaking chromosomes or calling problem callbacks.

Technical approach:
- Snapshot immutable result metadata into a versioned JSON-compatible document.
- Import optional Plotly only for HTML rendering and embed its JavaScript once.
- Render independent search histories on explicitly local clocks, never a synthetic portfolio curve.
- Reuse Plotly for explicitly attached bounded static diagrams and accessible text alternatives.
- Atomically replace output only after successful serialization/rendering.

Flow (pseudocode):
1. Validate user labels/metrics and snapshot a completed result without its keys.
2. Build configuration tables, work/budget charts and available incumbent curves.
3. Embed all chart assets and escaped metadata in one offline HTML document.
4. Write HTML or data-only JSON through an adjacent temporary file and rename.

Important criteria:
- Reporting never reevaluates objectives, inspects source, starts browsers or downloads assets.
- Declared/source-derived diagrams remain static descriptions, never original evaluation traces.
- Missing histories remain unavailable; compacted histories are explicitly approximate.
- Portfolio members cooperate and must not be presented as independent competitors.
- File generation is outside the measured search lifecycle and can raise independently.

Inputs/Outputs:
- Input: RKOResult, optional labels, precomputed diagnostics and a validated static diagram.
- Output: self-contained HTML or versioned JSON, excluding candidate coordinates.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from .assessment import SolutionAssessment
from .assessment import ValidationReport
from .explain import DecoderDiagram
from .solver import RKOResult
from .tracing import DecisionTrace
from collections import deque
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import asdict
from html import escape
from pathlib import Path
from tempfile import NamedTemporaryFile
from textwrap import wrap
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


class OptimizationReport:
    """Snapshot and export completed optimization observations without rerunning search.

    Responsibilities:
    - Keep summary, configuration, bounded histories and user metrics together.
    - Render offline interactive charts and write outputs atomically.

    Attributes:
        title (str): User-facing report heading, escaped before HTML insertion.
    """

    def __init__(
        self,
        result: RKOResult,
        *,
        title: str = "RKO optimization report",
        metrics: Mapping[str, str | int | float | bool] | None = None,
        assessment: SolutionAssessment[object] | None = None,
        decoder_diagram: DecoderDiagram | None = None,
    ) -> None:
        """Snapshot finite result metadata and optional already-computed domain metrics.

        Args:
            result (RKOResult): Completed managed run or legacy result without histories.
            title (str): Report title, not raw HTML.
            metrics (Mapping[str, str | int | float | bool] | None): User-supplied
                domain values such as feasibility or profit; never verified automatically.
            assessment (SolutionAssessment[object] | None): Optional completed post-run
                assessment for this result; only diagnostics, not its payload, are exported.
            decoder_diagram (DecoderDiagram | None): Explicit static graph; reporting never
                inspects source or executes a callback automatically.

        Raises:
            ValueError: If labels/metric types are invalid, numbers are nonfinite,
                or the supplied assessment belongs to a different result.
        """
        if not isinstance(title, str):
            raise ValueError("Report title must be a string.")
        domain = dict(metrics) if metrics is not None else {}
        for key, value in domain.items():
            if not isinstance(key, str) or type(value) not in (str, int, float, bool):
                raise ValueError("Domain metrics require string labels and scalar values.")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Domain metric numbers must be finite.")
        if assessment is not None:
            if not isinstance(assessment, SolutionAssessment) or assessment.result != result:
                raise ValueError("Assessment belongs to a different result or has an invalid type.")
        if decoder_diagram is not None and not isinstance(decoder_diagram, DecoderDiagram):
            raise ValueError("decoder_diagram must be a validated DecoderDiagram or None.")
        if result.trace is not None:
            if not isinstance(result.trace, DecisionTrace):
                raise ValueError("Result decision evidence has an invalid type.")
            if result.trace.evaluation_id is not None and not result.trace.matches(
                result.keys, result.cost
            ):
                raise ValueError("Decision evidence belongs to a different numerical observation.")
        self.title = title
        data = {
            "schema_version": 1,
            "title": title,
            "summary": {
                "cost": result.cost,
                "n_variables": result.n_variables,
                "algorithms": result.algorithms,
                "engine": result.engine,
                "execution": result.execution,
                "adaptation": result.adaptation,
                "random_state": result.random_state,
                "time_limit": result.time_limit,
                "elapsed_seconds": result.elapsed_seconds,
                "n_evaluations": result.n_evaluations,
                "stop_reason": result.stop_reason,
                "max_evaluations": result.max_evaluations,
                "pool_size": result.pool_size,
                "n_jobs": result.n_jobs,
                "n_workers": result.n_workers,
                "available_cpus": result.available_cpus,
            },
            "versions": dict(result.versions),
            "initialization": asdict(result.initialization) if result.initialization else None,
            "searches": [asdict(search) for search in result.searches],
            "domain_metrics": domain,
            "decoder_diagram": decoder_diagram.to_dict() if decoder_diagram is not None else None,
            "decision_trace": result.trace.to_dict() if result.trace is not None else None,
            "assessment": assessment.to_dict()
            if assessment is not None
            else {
                "origin": None,
                "elapsed_seconds": None,
                "validation": ValidationReport().to_dict(),
            },
        }
        self._data = json.loads(json.dumps(data, allow_nan=False))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible snapshot, without solution coordinates.

        Returns:
            dict[str, Any]: Versioned summary, histories, settings and supplied domain values.
        """
        return json.loads(json.dumps(self._data, allow_nan=False))

    @staticmethod
    def _table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
        """Escape plain values into an accessible scrollable HTML table.

        Args:
            headers (Sequence[str]): Column labels.
            rows (Sequence[Sequence[object]]): Plain-text cell values, never trusted markup.

        Returns:
            str: Escaped HTML table with explicit column headers.
        """
        headings = "".join(f'<th scope="col">{escape(label)}</th>' for label in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
            for row in rows
        )
        return (
            f'<div class="table"><table><thead><tr>{headings}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>"
        )

    @staticmethod
    def _decoder_figure(go: ModuleType, diagram: dict[str, Any]) -> Figure:
        """Place a bounded directed graph in simple depth bands using existing Plotly.

        Args:
            go (ModuleType): Lazily loaded optional Plotly graph objects module.
            diagram (dict[str, Any]): Previously validated graph snapshot.

        Returns:
            Figure: Plotly figure with directed transitions and escaped plain labels.
        """
        nodes, edges = diagram["nodes"], diagram["edges"]
        adjacency = {node["id"]: [] for node in nodes}
        for edge in edges:
            adjacency[edge["source"]].append(edge["target"])
        start = next(node["id"] for node in nodes if node["kind"] == "start")
        depths = {start: 0}
        pending = deque([start])
        while pending:
            current = pending.popleft()
            for target in adjacency[current]:
                if target not in depths:
                    depths[target] = depths[current] + 1
                    pending.append(target)
        final_depth = max(depths.values()) + 1
        for node in nodes:
            if node["kind"] == "end":
                depths[node["id"]] = final_depth
        bands: dict[int, list[str]] = {}
        for node in nodes:
            bands.setdefault(depths[node["id"]], []).append(node["id"])
        positions = {
            identifier: ((index - (len(band) - 1) / 2) * 2.7, -depth * 1.8)
            for depth, band in bands.items()
            for index, identifier in enumerate(band)
        }
        figure = go.Figure()
        for edge in edges:
            sx, sy = positions[edge["source"]]
            tx, ty = positions[edge["target"]]
            if edge["source"] == edge["target"]:
                route = [(sx, sy), (sx + 0.8, sy + 0.45), (sx + 0.8, sy - 0.45), (tx, ty)]
            elif ty >= sy:
                lane = max(sx, tx) + 0.8
                route = [(sx, sy), (lane, sy), (lane, ty), (tx, ty)]
            else:
                route = [(sx, sy), (tx, ty)]
            figure.add_trace(
                go.Scatter(
                    x=[point[0] for point in route],
                    y=[point[1] for point in route],
                    mode="lines",
                    line={"color": "#68778d", "width": 1.5},
                    hoverinfo="skip",
                )
            )
            figure.add_annotation(
                x=tx,
                y=ty,
                ax=route[-2][0],
                ay=route[-2][1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                text="",
                showarrow=True,
                arrowhead=3,
                arrowcolor="#68778d",
                arrowwidth=1.5,
                standoff=18,
            )
            if edge["label"]:
                figure.add_annotation(
                    x=(route[0][0] + route[1][0]) / 2,
                    y=(route[0][1] + route[1][1]) / 2 + 0.22,
                    text=escape(edge["label"]),
                    showarrow=False,
                    bgcolor="white",
                )
        symbols = {
            "start": "circle",
            "end": "circle",
            "error": "x",
            "decision": "diamond",
            "operation": "square",
            "return": "square",
        }
        figure.add_trace(
            go.Scatter(
                x=[positions[node["id"]][0] for node in nodes],
                y=[positions[node["id"]][1] for node in nodes],
                mode="markers+text",
                text=[
                    "<br>".join(escape(part) for part in wrap(node["label"], 28)) for node in nodes
                ],
                textposition="bottom center",
                hovertext=[
                    escape(f"{node['id']}: {node['label']} ({node['kind']})") for node in nodes
                ],
                hoverinfo="text",
                marker={
                    "size": 30,
                    "symbol": [symbols[node["kind"]] for node in nodes],
                    "color": "#dceafa",
                    "line": {"color": "#285b8c", "width": 2},
                },
            )
        )
        figure.update_layout(
            title=escape(diagram["title"]),
            showlegend=False,
            hovermode="closest",
            height=max(500, min(1800, (max(depths.values()) + 1) * 110)),
            xaxis={"visible": False},
            yaxis={"visible": False},
            margin={"l": 60, "r": 100, "t": 70, "b": 100},
        )
        return figure

    def _decoder_section(self, chart: str) -> str:
        """Render explicit graph provenance with complete accessible text alternatives.

        Args:
            chart (str): Already-rendered optional diagram figure, not user markup.

        Returns:
            str: Escaped optional-decoder HTML section, including absence disclosure.
        """
        diagram = self._data["decoder_diagram"]
        if diagram is None:
            return (
                '<section id="decoder"><h2>Optional decoder logic</h2>'
                "<p>No diagram attached. "
                "Reporting never inspects source automatically.</p></section>"
            )
        origin = (
            "author-declared logic"
            if diagram["origin"] == "author_declared"
            else "source-derived structure"
        )
        metadata = self._table(
            ("Diagram field", "Value"),
            [
                ("Origin", origin),
                ("Function", diagram["function_name"]),
                ("Source SHA-256", diagram["source_sha256"]),
                ("Expression/literal disclosure opted in", diagram["includes_expressions"]),
            ],
        )
        steps = self._table(
            ("ID", "Kind", "Plain step", "Source snippet line"),
            [
                (node["id"], node["kind"], node["label"], node["source_line"])
                for node in diagram["nodes"]
            ],
        )
        transitions = self._table(
            ("From", "To", "Branch"),
            [(edge["source"], edge["target"], edge["label"]) for edge in diagram["edges"]],
        )
        return (
            '<section id="decoder"><h2>Optional decoder logic</h2>'
            f'<h3>{escape(diagram["title"])}</h3><p id="decoder-description">'
            f"{escape(diagram['description'])}</p>{metadata}"
            '<p class="notice">This is not an execution trace. Declared logic is not checked '
            "against code. Source structure does not model implicit exceptions, subexpression "
            "control or called-library internals. Neither explains the original winning evaluation "
            "or certifies completeness, feasibility or optimality. "
            "Diagram helpers/rendering do not invoke the decoder/objective. "
            "Source inspection and rendering are outside solve time; review identifiers/labels "
            "before sharing. Expression opt-in can disclose literals or business secrets.</p>"
            f'<div role="img" aria-label="{escape(diagram["title"], quote=True)}" '
            f'aria-describedby="decoder-description">{chart}</div>'
            f"<h3>Diagram steps (text alternative)</h3>{steps}"
            f"<h3>Diagram transitions</h3>{transitions}</section>"
        )

    def _charts(self) -> tuple[str, str]:
        """Render interactive figures with one inline Plotly bundle and no external assets.

        Returns:
            tuple[str, str]: Analysis fragments and a separate optional decoder chart.

        Raises:
            ImportError: If the optional reports dependency is not installed.
        """
        try:
            go = importlib.import_module("plotly.graph_objects")
            pio = importlib.import_module("plotly.io")
        except ImportError as exc:
            raise ImportError(
                "Interactive HTML reports require the 'reports' extra: "
                "uv sync --extra reports (source checkout) or install 'rko-solver[reports]'."
            ) from exc
        searches = self._data["searches"]
        figures = []
        traced = [search for search in searches if search["history"]]
        for axis, label in (
            ("elapsed_seconds", "Elapsed seconds since each search started"),
            ("n_evaluations", "Completed objective evaluations in each search"),
        ):
            if not traced:
                break
            figure = go.Figure()
            for search in traced:
                points = search["history"]
                compacted = search["history_observations"] > len(points)
                name = escape(search["algorithm"]) + (" (compacted)" if compacted else "")
                figure.add_trace(
                    go.Scatter(
                        x=[point[axis] for point in points],
                        y=[point["cost"] for point in points],
                        mode="lines+markers",
                        name=name,
                        line={"shape": "hv"},
                    )
                )
            figure.update_layout(
                title="Best observed objective — independent local search axes",
                xaxis_title=label,
                yaxis_title="Objective (minimize)",
            )
            figures.append(figure)
        if searches:
            labels = [escape(search["algorithm"]) for search in searches]
            work = go.Figure(
                go.Bar(
                    x=labels,
                    y=[search["n_evaluations"] for search in searches],
                    name="Completed evaluations",
                )
            )
            work.update_layout(title="Observed search work", yaxis_title="Finite objective calls")
            figures.append(work)
            timing = go.Figure()
            for field, label in (
                ("time_limit", "Assigned search seconds"),
                ("elapsed_seconds", "Measured session seconds"),
            ):
                timing.add_trace(
                    go.Bar(
                        x=labels,
                        y=[search[field] for search in searches],
                        name=label,
                    )
                )
            timing.update_layout(
                title="Assigned budget versus session duration",
                yaxis_title="Seconds",
                barmode="group",
            )
            figures.append(timing)
        diagram = self._data["decoder_diagram"]
        if diagram is not None:
            figures.append(self._decoder_figure(go, diagram))
        fragments = []
        for index, figure in enumerate(figures):
            figure.update_layout(template="plotly_white", font={"family": "Arial, sans-serif"})
            fragments.append(
                pio.to_html(
                    figure,
                    include_plotlyjs=index == 0,
                    include_mathjax=False,
                    full_html=False,
                    div_id=f"rko-chart-{index}",
                    config={"responsive": True, "displaylogo": False},
                )
            )
        if diagram is not None:
            return "\n".join(fragments[:-1]), fragments[-1]
        return "\n".join(fragments), ""

    @staticmethod
    def _write(path: Path, content: str, suffix: str) -> Path:
        """Atomically write UTF-8 output after rendering has completed.

        Args:
            path (Path): Target with the required suffix and an existing parent directory.
            content (str): Already-rendered complete document.
            suffix (str): Required lower-case suffix, including the dot.

        Returns:
            Path: Target path after successful replacement.

        Raises:
            ValueError: If the target suffix is unsupported.
            OSError: If temporary creation, writing or atomic replacement fails.
        """
        if path.suffix.lower() != suffix:
            raise ValueError(f"Expected a {suffix} output path.")
        temporary = None
        try:
            with NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
            ) as stream:
                temporary = Path(stream.name)
                stream.write(content)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return path

    def write_json(self, path: str | Path) -> Path:
        """Export versioned data without requiring Plotly or including candidate keys.

        Args:
            path (str | Path): JSON target; existing files are atomically replaced.

        Returns:
            Path: Written file.

        Raises:
            ValueError: If the file suffix is not .json.
            OSError: If output cannot be written.
        """
        return self._write(Path(path), json.dumps(self._data, indent=2, allow_nan=False), ".json")

    def write_html(self, path: str | Path) -> Path:
        """Write one offline HTML file with hover, zoom and legend-based trace selection.

        Args:
            path (str | Path): HTML target; existing files are atomically replaced.

        Returns:
            Path: Written self-contained report.

        Raises:
            ImportError: If the optional reports dependency is unavailable.
            ValueError: If the target suffix is not .html.
            OSError: If output cannot be written.
        """
        target = Path(path)
        if target.suffix.lower() != ".html":
            raise ValueError("Expected a .html output path; automatic PDF export is not supported.")
        data = self._data
        summary = self._table(("Run metric", "Value"), list(data["summary"].items()))
        searches = self._table(
            (
                "Algorithm",
                "Adaptation",
                "Seed",
                "Stop reason",
                "Evaluations",
                "History kept/observed",
            ),
            [
                (
                    search["algorithm"],
                    "adaptive" if search["control"] else "fixed",
                    search["random_state"],
                    search["stop_reason"],
                    search["n_evaluations"],
                    f"{len(search['history'])}/{search['history_observations']}",
                )
                for search in data["searches"]
            ],
        )
        configuration = self._table(
            ("Algorithm", "Initial fixed values", "Adaptive axes (not visited states)"),
            [
                (search["algorithm"], search["parameters"], search["parameter_grid"])
                for search in data["searches"]
            ],
        )
        domain = self._table(
            ("User metric", "User-supplied value"), list(data["domain_metrics"].items())
        )
        validation = data["assessment"]["validation"]
        validation_summary = self._table(
            ("Assessment field", "Value"),
            [
                ("Check status", validation["status"]),
                ("Checker-declared feasibility", validation["feasible"]),
                ("Checker", validation["checker"]),
                ("Checker version", validation["checker_version"]),
                ("Origin", data["assessment"]["origin"]),
                ("Separate postprocessing seconds", data["assessment"]["elapsed_seconds"]),
            ],
        )
        validation_checks = self._table(
            ("Constraint", "Status", "Diagnostic", "Value", "Limit", "Unit"),
            [
                (
                    check["name"],
                    check["status"],
                    check["message"],
                    check["value"],
                    check["limit"],
                    check["unit"],
                )
                for check in validation["checks"]
            ],
        )
        provenance = self._table(("Runtime", "Version"), list(data["versions"].items()))
        initialization = self._table(
            ("Startup metric", "Value"), list((data["initialization"] or {}).items())
        )
        missing = [search["algorithm"] for search in data["searches"] if not search["history"]]
        unavailable = (
            (
                "No search histories recorded. Enable history_limit=1000 before solving."
                if not data["searches"]
                else "Histories unavailable for: " + ", ".join(missing)
            )
            if missing or not data["searches"]
            else "All search histories were recorded."
        )
        embedded = (
            json.dumps(data, allow_nan=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )
        charts, decoder_chart = self._charts()
        decoder_section = self._decoder_section(decoder_chart)
        trace = data["decision_trace"]
        decision_summary = self._table(
            ("Evidence field", "Value"),
            [
                (name, value)
                for name, value in (trace or {"state": "disabled"}).items()
                if name != "events"
            ],
        )
        decision_events = self._table(
            ("Step", "Authored action", "Authored reason"),
            [
                (index, event["action"], event["reason"])
                for index, event in enumerate(trace["events"] if trace else [], start=1)
            ],
        )
        heading = escape(self.title)
        document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{heading}</title><style>
body{{font:16px/1.5 Arial,sans-serif;max-width:1150px;margin:auto;padding:24px;color:#172b4d}}
h1,h2{{line-height:1.2}}.notice{{background:#fff3cd;padding:16px;border-left:4px solid #bf8700}}
.table{{overflow:auto}}table{{border-collapse:collapse;width:100%;margin:16px 0}}
th,td{{text-align:left;padding:8px;border:1px solid #dfe1e6;vertical-align:top}}
th{{background:#f4f5f7}}nav a{{margin-right:16px}}section{{margin-top:32px}}
@media print{{body{{max-width:none}}nav{{display:none}}.table{{overflow:visible}}}}
</style></head><body><header><h1>{heading}</h1><p>Single-run observations · minimized objective</p>
<nav><a href="#summary">Summary</a><a href="#analysis">Charts</a>
<a href="#validation">Validation</a><a href="#decoder">Decoder</a>
<a href="#decisions">Observed decisions</a>
<a href="#configuration">Configuration</a>
<a href="#provenance">Provenance</a></nav></header>
<p class="notice">No feasibility or optimality certificate is inferred from finite costs.
Portfolio members cooperate: these are not independent benchmark comparisons.
Completed does not mean converged. Timed runs depend on scheduling.</p>
<section id="summary"><h2>Run summary</h2>{summary}<h3>User-supplied domain metrics</h3>
<p>Values below are supplied by the caller, not independently verified by this report.</p>
{domain}</section>
<section id="validation"><h2>Optional solution validation</h2>{validation_summary}
{validation_checks}<p>Checks are supplied by the application, not a proof of complete
constraint coverage or global optimality. None means unknown: neither feasible nor infeasible
has been declared.
Missing/incomplete checks remain not_checked. Passed subsets do not declare feasibility.
Post-run decoding is reconstruction: stateful/stochastic decoders may produce a different
solution from the original evaluation. No objective consistency check, repair or reranking
is performed. Assessment time is separate from solve time; decoded payloads are not exported.</p>
</section>
<section id="analysis"><h2>Interactive analysis</h2><p>{escape(unavailable)}</p>
<p>Hover for observations, zoom to inspect, and click legend labels to select curves.
Each convergence curve starts at its own search start, excluding queueing and shared prefill;
it is not a synchronized portfolio timeline. No global shared-pool trajectory is recorded.
Compacted step curves are approximate: intermediate improvements may be omitted.
Terminal observations retain actual timestamps, including cooperative deadline overruns.
No exact time-to-target is inferred from compacted curves.</p>{charts}
<h3>Search accounting</h3>{searches}<p>Session durations can overlap; their sum is not wall time.
Evaluation counts are not CPU utilization or equivalent work across objectives.</p></section>
{decoder_section}
<section id="decisions"><h2>Observed decoder decisions</h2>{decision_summary}{decision_events}
<p>These are explicit author-emitted events from the selected original evaluation, when
its identity was preserved; no callback was replayed for this report. They are not a
static flowchart, post-run reconstruction, complete execution log or feasibility proof.
No-events does not prove the decoder made no decisions. Truncated recording omits events;
unavailable evidence must not be inferred from matching keys/cost. Labels can disclose
values only if the author explicitly includes them. Binding checksums are not execution
authenticity certificates.</p></section>
<section id="configuration"><h2>Effective initial search settings</h2>{configuration}
<h3>Separate shared initialization</h3>{initialization}
<p>Shared initialization is outside search histories; its result can be better than a search's
local incumbent. Final cost and search curves need not originate from the same publication.</p>
</section>
<section id="provenance"><h2>Runtime provenance</h2>{provenance}
<p>Seeds and settings are not a complete replay artifact. No callbacks, candidate keys or
input datasets are included by default. Export JSON for a separate data-only artifact.</p></section>
<script type="application/json" id="rko-report-data">{embedded}</script>
</body></html>"""
        return self._write(target, document, ".html")
