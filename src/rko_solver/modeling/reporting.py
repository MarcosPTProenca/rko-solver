"""Generated-model report section and score-label policy.

Objective:
- Keep original model objectives distinct from minimized search scores in reports.

Technical approach:
- Export coordinate-free model provenance and reuse the report's escaped table renderer.

Flow (pseudocode):
1. Attach coordinate-free completed model diagnostics when present.
2. Select explicit search-score labels for generated-model charts.
3. Render a separate model witness section without replaying callbacks.

Important criteria:
- No assignment coordinates, feasibility certificate or optimality claim is invented.

Inputs/Outputs:
- Input: validated portable report data and escaped table rendering callback.
- Output: coordinate-free model data and explicit HTML/axis labels.
"""

from __future__ import annotations

from .results import ModelResult
from collections.abc import Callable
from typing import Any


class ModelReportSection:
    """Keep generated-model presentation separate from general portfolio reporting.

    Responsibilities:
    - Label search scores separately from original model objectives.
    - Render completed coordinate-free model evidence without callback replay.
    """

    @staticmethod
    def attach(data: dict[str, Any], model: ModelResult | None) -> None:
        """Attach an optional coordinate-free model section.

        Args:
            data (dict[str, Any]): Invocation-owned report data.
            model (ModelResult | None): Completed portable model evidence.
        """
        if model is not None:
            data["model"] = model.to_dict(include_assignments=False)
            data["summary"]["cost_kind"] = "search_score"

    @staticmethod
    def axes(data: dict[str, Any]) -> tuple[str, str]:
        """Choose explicit generated-score or ordinary-objective chart labels.

        Args:
            data (dict[str, Any]): Coordinate-free validated report data.

        Returns:
            tuple[str, str]: History title and minimized vertical-axis label.
        """
        if "model" in data:
            return (
                "Best observed search score — independent local search axes",
                "Search score (minimize)",
            )
        return ("Best observed objective — independent local search axes", "Objective (minimize)")

    @staticmethod
    def html(data: dict[str, Any], table: Callable[..., str]) -> tuple[str, str]:
        """Render the optional model witness section with the existing escaped renderer.

        Args:
            data (dict[str, Any]): Coordinate-free validated report data.
            table (Callable[..., str]): Existing HTML-escaping table renderer.

        Returns:
            tuple[str, str]: Optional model section and accurate summary subtitle.
        """
        model_section = ""
        subtitle = "Single-run observations · minimized objective"
        if "model" in data:
            model = data["model"]
            feasible = model["best_feasible"]
            details = table(
                ("Model field", "Value"),
                [
                    ("Status", model["status"]),
                    (
                        "Best feasible original objective",
                        feasible["objective"] if feasible else None,
                    ),
                    ("Original direction", model["search_incumbent"]["direction"]),
                    ("Search-incumbent feasible", model["search_incumbent"]["feasible"]),
                    (
                        "Search-incumbent diagnostic objective",
                        model["search_incumbent"]["objective"],
                    ),
                    ("Frontend", model["frontend"]),
                    ("Frontend version", model["frontend_version"]),
                    ("Scoring", model["scoring"]),
                    ("Continuous repair", model["repair"]),
                    ("Repair sweep cap", model["repair_max_sweeps"]),
                    ("Penalty weight", model["penalty_weight"]),
                    ("Absolute tolerance", model["absolute_tolerance"]),
                    ("Relative tolerance", model["relative_tolerance"]),
                    ("Separate compilation seconds", model["compilation_seconds"]),
                    ("No search", model["no_search"]),
                ],
            )
            model_section = (
                '<section id="model"><h2>Model solution — separate from search score</h2>'
                + details
                + "<p>Feasible means validated under the declared numerical rules, "
                "not optimal or certified by an external solver. No-feasible-found is not "
                "an infeasibility certificate. The feasible witness may differ from the root "
                "search incumbent; histories, optional assessments and decision traces describe "
                "the search incumbent only. No callbacks were replayed "
                "for this report.</p></section>"
            )
            subtitle = "Single-run observations · minimized search score"
        return model_section, subtitle
