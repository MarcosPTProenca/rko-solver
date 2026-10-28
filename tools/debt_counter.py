#!/usr/bin/env python3
"""Measure tolerated complexity, duplication, and dead-code debt.

Objective:
- Prevent measured code debt from increasing above its recorded baseline.

Technical approach:
- Reuse QualityGate source discovery and cyclomatic-complexity measurements.
- Use jscpd for duplicate clones and vulture for dead-code candidates.
- Compare four aggregate counts with a versioned JSON baseline.

Flow (pseudocode):
1. Discover measured source files and collect structure metrics.
2. Count complex functions, duplicate clones, duplicate lines, and dead code.
3. Write a baseline on request or report and enforce the existing baseline.

Important criteria:
- A missing baseline allows no nonzero debt until explicitly initialized.
- The baseline can only be refreshed by the explicit --write-baseline option.
- GitHub Actions receives the report through GITHUB_STEP_SUMMARY when available.

Inputs/Outputs:
- Input: repository root and optional baseline-writing command.
- Output: Markdown report, baseline JSON on request, and a process status.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from quality_gate import QualityGate
from typing import Any

COMPLEXITY_THRESHOLD = 8
DEAD_CODE_MIN_CONFIDENCE = "60"
JSCPD_COMMAND = ["npx", "--yes", "jscpd@4.3.0"]
DEFAULT_BASELINE = Path(".quality/debt-baseline.json")
REPORT_LIMIT = 10
LABELS = {
    "complex_functions": "Functions with cyclomatic complexity ≥ 8",
    "duplicate_clones": "Duplicate code clones",
    "duplicated_lines": "Duplicated lines",
    "dead_code_candidates": "Dead-code candidates",
}


def _run(command: list[str], root: Path, *, ok: tuple[int, ...] = (0,)) -> str:
    """Run a measurement command and return its standard output.

    Args:
        command (list[str]): Executable and command-line arguments.
        root (Path): Repository directory used as the working directory.
        ok (tuple[int, ...]): Accepted process exit codes.

    Returns:
        str: Captured standard output.

    Raises:
        RuntimeError: If the executable is missing or exits unexpectedly.
    """
    try:
        process = subprocess.run(command, cwd=root, text=True, capture_output=True, check=False)
    except OSError as error:
        raise RuntimeError(f"{command[0]} unavailable: {error}") from error
    if process.returncode not in ok:
        detail = process.stderr.strip() or process.stdout.strip()
        raise RuntimeError(f"{command[0]} failed: {detail}")
    return process.stdout


def complex_functions(functions: dict[str, dict[str, Any]]) -> list[tuple[str, int]]:
    """Select functions at or above the complexity threshold.

    Args:
        functions (dict[str, dict[str, Any]]): Function metrics from QualityGate.

    Returns:
        list[tuple[str, int]]: Function names and complexity, highest first.
    """
    found = [
        (name, metrics["cyclomatic_complexity"])
        for name, metrics in functions.items()
        if metrics["cyclomatic_complexity"] >= COMPLEXITY_THRESHOLD
    ]
    return sorted(found, key=lambda item: (-item[1], item[0]))


def duplication(root: Path, paths: list[str]) -> dict[str, Any]:
    """Measure duplicate clones and lines for the selected source paths.

    Args:
        root (Path): Repository directory used as the working directory.
        paths (list[str]): Repository-relative source paths for jscpd.

    Returns:
        dict[str, Any]: Clone count, duplicated-line count, and clone locations.

    Raises:
        RuntimeError: If jscpd fails or its report cannot be produced.
        OSError: If the generated report cannot be read.
        json.JSONDecodeError: If the generated report is invalid JSON.
    """
    if not paths:
        return {"clones": 0, "duplicated_lines": 0, "pairs": []}
    with tempfile.TemporaryDirectory() as output:
        _run(
            [
                *JSCPD_COMMAND,
                "--min-lines",
                "5",
                "--min-tokens",
                "50",
                "--reporters",
                "json",
                "--output",
                output,
                *paths,
            ],
            root,
        )
        report = json.loads((Path(output) / "jscpd-report.json").read_text(encoding="utf-8"))
    total = report["statistics"]["total"]
    pairs = [
        f"{clone['firstFile']['name']}:{clone['firstFile']['start']} ~ "
        f"{clone['secondFile']['name']}:{clone['secondFile']['start']}"
        for clone in report["duplicates"]
    ]
    return {
        "clones": total["clones"],
        "duplicated_lines": total["duplicatedLines"],
        "pairs": pairs,
    }


def dead_code(root: Path, paths: list[str]) -> list[str]:
    """Collect vulture findings for the selected source paths.

    Args:
        root (Path): Repository directory used as the working directory.
        paths (list[str]): Repository-relative source paths for vulture.

    Returns:
        list[str]: Nonempty finding lines emitted by vulture.

    Raises:
        RuntimeError: If vulture is missing or exits with an unexpected status.
    """
    if not paths:
        return []
    output = _run(
        [
            sys.executable,
            "-m",
            "vulture",
            "--min-confidence",
            DEAD_CODE_MIN_CONFIDENCE,
            *paths,
        ],
        root,
        ok=(0, 3),
    )
    return [line for line in output.splitlines() if line.strip()]


def snapshot(root: Path) -> dict[str, Any]:
    """Measure all debt counters for one repository root.

    Args:
        root (Path): Repository directory to measure.

    Returns:
        dict[str, Any]: Aggregate counts and bounded detail lists.

    Raises:
        RuntimeError: If a measurement tool fails.
        OSError: If a measured source file cannot be read.
    """
    gate = QualityGate(root)
    paths = [path.as_posix() for path in gate.source_files()]
    structure = gate.collect_structure()
    complex_items = complex_functions(structure["functions"])
    duplicates = duplication(root, paths)
    dead = dead_code(root, paths)
    return {
        "counts": {
            "complex_functions": len(complex_items),
            "duplicate_clones": duplicates["clones"],
            "duplicated_lines": duplicates["duplicated_lines"],
            "dead_code_candidates": len(dead),
        },
        "complex": complex_items,
        "duplicates": duplicates["pairs"],
        "dead": dead,
    }


def violations(counts: dict[str, int], baseline: dict[str, int]) -> list[str]:
    """List counters whose current values exceed their baseline.

    Args:
        counts (dict[str, int]): Current aggregate debt counts.
        baseline (dict[str, int]): Permitted aggregate debt counts.

    Returns:
        list[str]: Sorted human-readable baseline violations.
    """
    return [
        f"{name}: {value} exceeds baseline {baseline.get(name, 0)}"
        for name, value in sorted(counts.items())
        if value > baseline.get(name, 0)
    ]


def render(snap: dict[str, Any], baseline: dict[str, int]) -> str:
    """Render measured debt and bounded findings as Markdown.

    Args:
        snap (dict[str, Any]): Snapshot returned by snapshot().
        baseline (dict[str, int]): Permitted aggregate debt counts.

    Returns:
        str: Markdown summary with counts and up to REPORT_LIMIT findings.
    """
    lines = [
        "## Technical debt counters",
        "",
        "| Counter | Baseline | Current | Δ |",
        "|---|---:|---:|---:|",
    ]
    for name, value in snap["counts"].items():
        base = baseline.get(name, 0)
        lines.append(f"| {LABELS[name]} | {base} | {value} | {value - base:+d} |")

    sections = [
        ("Most complex functions", [f"`{name}`: {score}" for name, score in snap["complex"]]),
        ("Duplicate code", snap["duplicates"]),
        ("Dead-code candidates", snap["dead"]),
    ]
    for title, items in sections:
        if items:
            lines += ["", f"### {title} (top {REPORT_LIMIT})", ""]
            lines += [f"- {item}" for item in items[:REPORT_LIMIT]]
    return "\n".join(lines) + "\n"


def _write_summary(report: str) -> None:
    """Append a Markdown report to the GitHub Actions job summary when configured.

    Args:
        report (str): Markdown report to append.
    """
    target = os.environ.get("GITHUB_STEP_SUMMARY")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            handle.write(report)


def _load_counts(path: Path) -> dict[str, int]:
    """Load aggregate counts from a baseline JSON file.

    Args:
        path (Path): Baseline path to read.

    Returns:
        dict[str, int]: Stored counts, or an empty mapping if the file is absent.

    Raises:
        OSError: If the baseline file cannot be read.
        json.JSONDecodeError: If the baseline file is malformed.
    """
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("counts", {})


def main() -> int:
    """Write a baseline or enforce the current debt baseline.

    Returns:
        int: Zero on success; one when measurement fails or debt increases.

    Raises:
        SystemExit: If command-line arguments are invalid.
        OSError: If the baseline cannot be written or read.
        json.JSONDecodeError: If the baseline JSON is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--write-baseline", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    baseline_path = root / DEFAULT_BASELINE

    try:
        snap = snapshot(root)
    except RuntimeError as error:
        print(f"Debt counter failed: {error}")
        return 1

    if args.write_baseline:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps({"version": 1, "counts": snap["counts"]}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote debt baseline: {baseline_path}")
        return 0

    baseline = _load_counts(baseline_path)
    report = render(snap, baseline)
    print(report)
    _write_summary(report)

    errors = violations(snap["counts"], baseline)
    if errors:
        print("Debt counter failed:\n- " + "\n- ".join(errors))
        return 1
    print("Debt counter: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
