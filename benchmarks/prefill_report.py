"""Validate and compact hosted matched-work study shards without timing claims.

Objective:
- Preserve paired quality evidence without committing every full trace record.

Technical approach:
- Reject mixed provenance, duplicate/incomplete pairs and unequal work controls.
- Recompute per-family outcomes/medians from raw trials rather than trusting summaries.
- Retain compact paired CSV, source hashes and input-artifact fingerprints.

Flow (pseudocode):
1. Read complete JSONL shards from the existing prefill-work runner.
2. Validate each completed trial and pair, including seeds and starting chromosomes.
3. Group paired costs by dimension, budget, algorithm and workload.
4. Write reproducible compact observations and descriptive summaries.

Important criteria:
- No raw-cost pooling across objective families or statistical superiority claim.
- Repeated pilot seeds cannot be merged twice as independent observations.
- Input validation happens before output files are created.

Inputs/Outputs:
- Input: raw JSONL shards and an explicit output prefix.
- Output: provenance/summary JSON and per-pair CSV, or a validation error.
"""

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def compile_report(paths: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate raw study controls and derive independent paired summaries.

    Args:
        paths (list[Path]): Complete, nonoverlapping runner shards.

    Returns:
        tuple[dict, list[dict]]: Provenance/group summaries and compact paired rows.

    Raises:
        ValueError: If provenance, bounds, accounting or pairing is inconsistent.
        OSError: If an input cannot be read.
    """
    if not paths:
        raise ValueError("no study shards")
    producer: dict[str, Any] | None = None
    expected_labels = {
        (algorithm, workload)
        for algorithm in ("IPR", "GRASP", "MultiStart")
        for workload in ("sphere", "rugged", "ordering", "flat")
    }
    inputs: list[dict[str, str]] = []
    paired: dict[tuple[Any, ...], dict[bool, dict[str, Any]]] = defaultdict(dict)
    prefixes: dict[tuple[int, int, int], str] = {}
    for path in sorted(paths):
        content = path.read_bytes()
        records = [json.loads(line) for line in content.decode("utf-8").splitlines()]
        if not records or not all(isinstance(row, dict) for row in records):
            raise ValueError("invalid study records")
        if records[0].get("kind") != "provenance":
            raise ValueError("missing provenance")
        header = records[0]
        if not all(
            header.get(key)
            for key in ("source_revision", "source_sha256", "probe_sha256", "python", "numpy")
        ):
            raise ValueError("incomplete source provenance")
        if (
            header.get("timing") is not False
            or header.get("backend") != "serial"
            or header.get("control") != 0
            or header.get("policy_clock") != "frozen_zero"
        ):
            raise ValueError("incompatible study controls")
        if producer is not None and header != producer:
            raise ValueError("mixed study provenance")
        producer = header
        inputs.append({"file": path.name, "sha256": hashlib.sha256(content).hexdigest()})
        trials = [row for row in records[1:] if row.get("kind") == "trial"]
        summaries = [row for row in records[1:] if row.get("kind") == "summary"]
        if not trials or len(trials) + len(summaries) + 1 != len(records):
            raise ValueError("incomplete or unknown study records")
        required = {
            "dimension",
            "evaluations",
            "capacity",
            "algorithm",
            "workload",
            "seed",
            "prefill",
            "startup_evaluations",
            "search_evaluations",
            "decoder_calls",
            "unscored_decoder_calls",
            "termination",
            "cost",
            "search_seed",
            "first_search_sha256",
            "startup_seed_prefix_sha256",
            "startup_best_cost",
            "search_improved_startup",
        }
        if any(not required.issubset(row) for row in trials) or any(
            not {"algorithm", "workload", "pairs"}.issubset(row) for row in summaries
        ):
            raise ValueError("missing study fields")
        shapes = {(row["dimension"], row["evaluations"], row["capacity"]) for row in trials}
        trial_labels = {(row["algorithm"], row["workload"]) for row in trials}
        summary_labels = {(row["algorithm"], row["workload"]) for row in summaries}
        if (
            len(shapes) != 1
            or len(summaries) != 12
            or trial_labels != expected_labels
            or summary_labels != expected_labels
        ):
            raise ValueError("incomplete shard matrix")
        for row in trials:
            dimension, budget, capacity = row["dimension"], row["evaluations"], row["capacity"]
            prefill = row["prefill"]
            startup, search = row["startup_evaluations"], row["search_evaluations"]
            if (
                any(
                    type(row[key]) is not int
                    for key in (
                        "dimension",
                        "evaluations",
                        "capacity",
                        "seed",
                        "startup_evaluations",
                        "search_evaluations",
                        "decoder_calls",
                        "unscored_decoder_calls",
                    )
                )
                or type(prefill) is not bool
                or row["seed"] < 0
                or dimension < 1
                or capacity < 2
                or budget <= 2 * capacity - 1
                or startup + search != budget
                or row["decoder_calls"] != budget + 1
                or row["unscored_decoder_calls"] != 1
                or row["termination"] != "evaluation_budget"
                or type(row["cost"]) not in (int, float)
                or not math.isfinite(row["cost"])
                or (not prefill and startup != 0)
                or (prefill and not capacity <= startup <= 2 * capacity - 1)
            ):
                raise ValueError("invalid trial work accounting")
            if prefill:
                if (
                    type(row["startup_best_cost"]) not in (int, float)
                    or not math.isfinite(row["startup_best_cost"])
                    or row["cost"] > row["startup_best_cost"]
                    or row["search_improved_startup"] != (row["cost"] < row["startup_best_cost"])
                ):
                    raise ValueError("invalid startup incumbent evidence")
                prefix_key = (dimension, capacity, row["seed"])
                prefix = row["startup_seed_prefix_sha256"]
                if prefixes.setdefault(prefix_key, prefix) != prefix:
                    raise ValueError("different initial seed prefixes")
            key = (dimension, budget, capacity, row["algorithm"], row["workload"], row["seed"])
            if prefill in paired[key]:
                raise ValueError("duplicate trial or overlapping seeds")
            paired[key][prefill] = row
        cohort_seeds: set[int] | None = None
        for summary in summaries:
            selected = [
                row
                for row in trials
                if (row["algorithm"], row["workload"])
                == (summary["algorithm"], summary["workload"])
            ]
            seeds = {row["seed"] for row in selected}
            if (
                len(selected) != 2 * len(seeds)
                or summary["pairs"] != len(seeds)
                or (cohort_seeds is not None and cohort_seeds != seeds)
            ):
                raise ValueError("incomplete shard summary")
            cohort_seeds = seeds
    compact: list[dict[str, Any]] = []
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for key, pair in sorted(paired.items()):
        if set(pair) != {False, True}:
            raise ValueError("incomplete pair")
        before, after = pair[False], pair[True]
        if (
            before["search_seed"] != after["search_seed"]
            or before["first_search_sha256"] != after["first_search_sha256"]
        ):
            raise ValueError("different paired search controls")
        dimension, budget, capacity, algorithm, workload, seed = key
        row = {
            "dimension": dimension,
            "evaluations": budget,
            "capacity": capacity,
            "algorithm": algorithm,
            "workload": workload,
            "seed": seed,
            "default_cost": before["cost"],
            "prefilled_cost": after["cost"],
            "startup_evaluations": after["startup_evaluations"],
            "startup_best_cost": after["startup_best_cost"],
            "search_improved_startup": after["search_improved_startup"],
            "first_search_sha256": before["first_search_sha256"],
            "startup_seed_prefix_sha256": after["startup_seed_prefix_sha256"],
        }
        compact.append(row)
        groups[key[:-1]].append(row)
    summaries = []
    for (dimension, budget, capacity, algorithm, workload), rows in sorted(groups.items()):
        counts = {"better": 0, "tied": 0, "worse": 0}
        for row in rows:
            before, after = row["default_cost"], row["prefilled_cost"]
            counts["better" if after < before else "worse" if after > before else "tied"] += 1
        summaries.append(
            {
                "dimension": dimension,
                "evaluations": budget,
                "capacity": capacity,
                "algorithm": algorithm,
                "workload": workload,
                "pairs": len(rows),
                **counts,
                "default_median": statistics.median(row["default_cost"] for row in rows),
                "prefilled_median": statistics.median(row["prefilled_cost"] for row in rows),
                "search_improved_startup": sum(row["search_improved_startup"] for row in rows),
            }
        )
    return {
        "producer": producer,
        "inputs": inputs,
        "pairs": len(compact),
        "timing": False,
        "groups": summaries,
    }, compact


def main() -> None:
    """Write compact evidence only after validating every downloaded raw shard.

    Raises:
        ValueError: If a shard fails scientific controls or work-accounting checks.
        OSError: If input/output paths are inaccessible.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--run-url", default=None)
    args = parser.parse_args()
    report, rows = compile_report(list(args.input_dir.rglob("*.jsonl")))
    report["workflow_run_url"] = args.run_url
    prefix = args.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_name(prefix.name + "-summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    with prefix.with_name(prefix.name + "-pairs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Validated {report['pairs']} pairs in {len(report['groups'])} family-specific groups.")


if __name__ == "__main__":
    main()
