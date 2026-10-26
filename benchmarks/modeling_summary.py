"""Complete planned-run model validation aggregation.

Objective:
- Preserve reproducible complete benchmark evidence.

Technical approach:
- Separate observational bookkeeping from timed solver execution.

Flow (pseudocode):
1. Read original runtime or campaign data.
2. Validate all planned evidence before aggregation.
3. Return explicit reproducible provenance or complete summaries.

Important criteria:
- Never omit errors, missing runs or unsupported source identity.

Inputs/Outputs:
- Input: original runtime or campaign observations.
- Output: portable complete benchmark evidence.
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


class CampaignSummary:
    """Aggregate complete campaign evidence without selection bias.

    Responsibilities:
    - Preserve failures and missing seeds in planned-run denominators.
    - Keep original directions and explicit no-feasible outcomes.
    """

    @staticmethod
    def read(path: Path) -> list[dict[str, Any]]:
        """Aggregate every observed and missing planned seed.

        Args:
            path (Path): Complete original campaign observations.

        Returns:
            list[dict[str, Any]]: Validated original campaign data.

        Raises:
            OSError: If observations cannot be read.
            ValueError: If observations or planned seed identities are invalid.
        """
        groups, metadata = CampaignSummary._load(path)
        directions = {case["name"]: case["maximize"] for case in metadata["cases"]}
        CampaignSummary._plan(groups, directions, metadata)
        return [
            CampaignSummary._group(key, runs, metadata, directions)
            for key, runs in sorted(groups.items())
        ]

    @staticmethod
    def _load(path: Path) -> tuple[dict, dict[str, Any]]:
        """Read all observations and require a nonempty planned campaign.

        Args:
            path (Path): Original campaign JSONL observations.

        Returns:
            tuple[dict, dict[str, Any]]: Validated original campaign data.

        Raises:
            OSError: If observations cannot be read.
            ValueError: If JSON or campaign metadata are invalid.
        """
        groups = defaultdict(list)
        metadata = None
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record["kind"] == "metadata":
                    metadata = record
                elif record["kind"] == "run":
                    groups[record["case"], record["frontend"], record["scoring"]].append(record)
        if metadata is None:
            raise ValueError("Campaign metadata are required to detect missing runs.")
        planned = len(metadata["seeds"])
        if planned < 1:
            raise ValueError("Campaign metadata must declare at least one seed.")
        return (groups, metadata)

    @staticmethod
    def _plan(groups: dict, directions: dict[str, bool], metadata: dict[str, Any]) -> None:
        """Include every planned group, including groups with zero observations.

        Args:
            groups (dict): Observed groups.
            directions (dict[str, bool]): Original case directions.
            metadata (dict[str, Any]): Complete campaign plan.

        """
        for case in directions:
            for frontend in metadata["frontends"]:
                for scoring in metadata["scoring_modes"]:
                    groups.setdefault((case, frontend, scoring), [])

    @staticmethod
    def _group(
        key: tuple[str, str, str],
        runs: list[dict[str, Any]],
        metadata: dict[str, Any],
        directions: dict[str, bool],
    ) -> dict[str, Any]:
        """Reject biased seed sets and summarize every planned observation.

        Args:
            key (tuple[str, str, str]): Original case frontend and scoring identity.
            runs (list[dict[str, Any]]): All observations in this group.
            metadata (dict[str, Any]): Full planned campaign.
            directions (dict[str, bool]): Original objective directions.

        Returns:
            dict[str, Any]: Validated original campaign data.

        Raises:
            ValueError: If observed seeds are duplicated or unexpected.
        """
        case, frontend, scoring = key
        planned = len(metadata["seeds"])
        observed_seeds = [run["seed"] for run in runs]
        if len(set(observed_seeds)) != len(runs) or any(
            seed not in metadata["seeds"] for seed in observed_seeds
        ):
            raise ValueError("Duplicate or unexpected campaign seeds would bias the summary.")
        record = {
            "case": case,
            "frontend": frontend,
            "scoring": scoring,
            "runs": len(runs),
            "planned_runs": planned,
            "missing_runs": planned - len(runs),
            "errors": sum(run["status"] == "error" for run in runs),
        }
        record.update(CampaignSummary._statistics(runs, directions[case], planned))
        return record

    @staticmethod
    def _statistics(runs: list[dict[str, Any]], maximize: bool, planned: int) -> dict[str, Any]:
        """Summarize feasible witnesses without removing failures from denominators.

        Args:
            runs (list[dict[str, Any]]): Every observed run.
            maximize (bool): Original objective direction.
            planned (int): Full planned seed denominator.

        Returns:
            dict[str, Any]: Validated original campaign data.
        """
        feasible = [run for run in runs if run["status"] == "feasible"]
        gaps = [run["signed_gap_to_exact_reference_percent"] for run in feasible]
        first_times = [run["time_to_first_feasible_seconds"] for run in feasible]
        return {
            "feasible": len(feasible),
            "feasible_rate_over_planned": len(feasible) / planned,
            "median_gap_percent_when_feasible": statistics.median(gaps) if gaps else None,
            "best_feasible_objective": CampaignSummary._best(feasible, maximize),
            "median_first_feasible_seconds_when_found": statistics.median(first_times)
            if first_times
            else None,
        }

    @staticmethod
    def _best(feasible: list[dict[str, Any]], maximize: bool) -> float | None:
        """Select the best original objective in its native direction.

        Args:
            feasible (list[dict[str, Any]]): Feasible completed runs.
            maximize (bool): Original objective direction.

        Returns:
            float | None: Validated original campaign data.
        """
        if not feasible:
            return None
        if maximize:
            return max(run["best_feasible_objective"] for run in feasible)
        return min(run["best_feasible_objective"] for run in feasible)
