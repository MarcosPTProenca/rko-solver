"""Validated summaries of paired fixed-work baseline diagnostics.

Objective:
- Preserve measurement scopes and reject mismatched diagnostic evidence.

Technical approach:
- Require one plain/profiled record for each source-identified workload and seed.
- Summarize unprofiled search latency separately from profiled runner costs.
- Rank only functions present in every paired seed's truncated top-function list.

Flow (pseudocode):
1. Validate records, source identity and complete replay-equivalent pairs.
2. Group workloads without pooling different algorithms or dimensions.
3. Emit descriptive medians and diagnostic exclusive-time fractions.

Important criteria:
- Three-seed summaries are screening evidence, not confidence or speedup claims.
- Inclusive function times overlap and are never added.
- Missing truncated functions are not assigned zero cost.

Inputs/Outputs:
- Input: JSONL records from benchmarks.runtime.performance_profile.
- Output: a validated JSON summary; invalid evidence raises ValueError.
"""

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


class PerformanceReport:
    """Validate paired baseline evidence before computing descriptive summaries.

    Responsibilities:
    - Reject incomplete, duplicate, nonfinite, mixed-source or changed replay data.
    - Keep search-only timings distinct from full-runner profiling diagnostics.
    """

    @staticmethod
    def validate(row: dict[str, object]) -> None:
        """Require a complete diagnostic record with valid timing scopes.

        Args:
            row (dict[str, object]): Parsed versioned diagnostic record.

        Raises:
            ValueError: If a required field, scope, timing or fingerprint is invalid.
        """
        try:
            if row["schema_version"] != 1 or row["scope"] != "serial_fixed_work_search_only":
                raise ValueError("Unsupported diagnostic schema or scope.")
            if type(row["profiled"]) is not bool:
                raise ValueError("profiled must be boolean.")
            result = row["result"]
            for field in ("dimension", "evaluations"):
                if type(result[field]) is not int or result[field] < 1:
                    raise ValueError("Invalid diagnostic work.")
            if type(result["seed"]) is not int or result["seed"] < 0:
                raise ValueError("Invalid seed.")
            if not all(isinstance(result[field], str) for field in ("algorithm", "workload")):
                raise ValueError("Invalid workload identity.")
            for field in ("cost", "seconds", "cpu_seconds"):
                value = result[field]
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("Nonfinite diagnostic result.")
                if field != "cost" and value < 0:
                    raise ValueError("Negative diagnostic time.")
            for value in (
                row["provenance"]["source_sha256"],
                *(result[field] for field in ("keys_sha256", "pool_sha256", "rng_sha256")),
            ):
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                ):
                    raise ValueError("Invalid source or replay fingerprint.")
            if type(row["seed_pool_evaluations"]) is not int or row["seed_pool_evaluations"] < 0:
                raise ValueError("Invalid seed-pool accounting.")
            if not isinstance(row["profile"], list):
                raise ValueError("Invalid function profile.")
            if row["profiled"]:
                if row["profile_scope"] != "fixed_work_runner_including_construction_and_seed_pool":
                    raise ValueError("Invalid profile scope.")
                total = row["profile_total_seconds"]
                if type(total) not in (int, float) or not math.isfinite(total) or total <= 0:
                    raise ValueError("Invalid profile total.")
                seen = set()
                for function in row["profile"]:
                    key = (function["file"], function["line"], function["function"])
                    if key in seen:
                        raise ValueError("Duplicate profile function.")
                    seen.add(key)
                    if not isinstance(key[0], str) or not isinstance(key[2], str):
                        raise ValueError("Invalid function identity.")
                    if type(key[1]) is not int or key[1] < 0:
                        raise ValueError("Invalid function line.")
                    for field in ("exclusive_seconds", "inclusive_seconds"):
                        value = function[field]
                        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                            raise ValueError("Invalid function time.")
                if sum(item["exclusive_seconds"] for item in row["profile"]) > total + 1e-9:
                    raise ValueError("Exclusive profile times exceed the profile total.")
            elif (
                row["profile"]
                or row["profile_scope"] is not None
                or row["profile_total_seconds"] is not None
            ):
                raise ValueError("Unprofiled record contains profile evidence.")
            if not all(isinstance(row["provenance"][field], str) for field in ("python", "numpy")):
                raise ValueError("Missing runtime versions.")
        except (KeyError, TypeError) as error:
            raise ValueError("Malformed diagnostic record.") from error

    @classmethod
    def summarize(cls, records: list[dict[str, object]]) -> dict[str, object]:
        """Summarize exact replay pairs without making optimization claims.

        Args:
            records (list[dict[str, object]]): Complete plain/profiled observations.

        Returns:
            dict[str, object]: Source identity, pair counts and per-workload medians.

        Raises:
            ValueError: If evidence is empty, invalid, incomplete or incomparable.
        """
        if not records:
            raise ValueError("No diagnostic records.")
        pairs = defaultdict(dict)
        identities = set()
        for row in records:
            cls.validate(row)
            provenance = row["provenance"]
            identities.add(
                tuple(provenance[field] for field in ("source_sha256", "python", "numpy"))
            )
            result = row["result"]
            key = tuple(
                result[field]
                for field in ("algorithm", "workload", "dimension", "evaluations", "seed")
            )
            if row["profiled"] in pairs[key]:
                raise ValueError("Duplicate diagnostic observation.")
            pairs[key][row["profiled"]] = row
        if len(identities) != 1:
            raise ValueError("Mixed source or runtime identities.")
        grouped = defaultdict(list)
        for key, pair in pairs.items():
            if set(pair) != {False, True}:
                raise ValueError("Missing plain/profiled diagnostic pair.")
            plain, profiled = pair[False], pair[True]
            plain_replay = {
                key: value
                for key, value in plain["result"].items()
                if key not in ("seconds", "cpu_seconds")
            }
            profiled_replay = {
                key: value
                for key, value in profiled["result"].items()
                if key not in ("seconds", "cpu_seconds")
            }
            if (
                plain_replay != profiled_replay
                or plain["seed_pool_evaluations"] != profiled["seed_pool_evaluations"]
            ):
                raise ValueError("Changed replay between plain and profiled records.")
            grouped[key[:-1]].append(pair)
        groups = []
        for (algorithm, workload, dimension, evaluations), observations in sorted(grouped.items()):
            samples = defaultdict(list)
            for pair in observations:
                row = pair[True]
                package = row["provenance"].get("rko_source", "").replace("\\", "/")
                for function in row["profile"]:
                    filename = function["file"].replace("\\", "/")
                    if package and filename.startswith(package + "/"):
                        filename = "rko/" + filename[len(package) + 1 :]
                    key = (filename, function["line"], function["function"])
                    samples[key].append(
                        function["exclusive_seconds"] / row["profile_total_seconds"]
                    )
            functions = [
                {
                    "file": key[0],
                    "line": key[1],
                    "function": key[2],
                    "observations": len(values),
                    "median_exclusive_fraction": statistics.median(values),
                }
                for key, values in samples.items()
                if len(values) == len(observations)
            ]
            functions.sort(
                key=lambda row: (
                    -row["median_exclusive_fraction"],
                    row["file"],
                    row["line"],
                    row["function"],
                )
            )
            groups.append(
                {
                    "algorithm": algorithm,
                    "workload": workload,
                    "dimension": dimension,
                    "evaluations": evaluations,
                    "pairs": len(observations),
                    "seeds": sorted(pair[False]["result"]["seed"] for pair in observations),
                    "unprofiled_search_median_seconds": statistics.median(
                        pair[False]["result"]["seconds"] for pair in observations
                    ),
                    "profile_runner_median_seconds": statistics.median(
                        pair[True]["profile_total_seconds"] for pair in observations
                    ),
                    "top_functions": functions[:5],
                }
            )
        source, python, numpy = next(iter(identities))
        return {
            "schema_version": 1,
            "scope": "serial_fixed_work_baseline_screening",
            "source_sha256": source,
            "python": python,
            "numpy": numpy,
            "trials": len(records),
            "pairs": len(pairs),
            "groups": groups,
        }


def main() -> None:
    """Read diagnostic shards and write a validated descriptive summary.

    Raises:
        OSError: If inputs cannot be read or output cannot be written.
        ValueError: If JSON or diagnostic evidence is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = {path.name: path.read_bytes() for path in sorted(args.input_dir.glob("*.jsonl"))}
    records = [
        json.loads(line)
        for content in inputs.values()
        for line in content.decode("utf-8").splitlines()
        if line.strip()
    ]
    summary = PerformanceReport.summarize(records)
    summary["inputs_sha256"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in inputs.items()
    }
    summary["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
