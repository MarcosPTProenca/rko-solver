"""Mutation score report with per-module floors for the Python and Rust code.

Objective:
- Turn the raw results of `mutmut` and `cargo-mutants` into one table of mutation scores per
  module, and fail when a module's score falls below its recorded floor.

Technical approach:
- A mutant is a small deliberate change to the code; the tests "kill" it by failing. The score
  of a module is killed over killed plus survived, so mutants that do not compile, have no
  tests or were skipped never count for or against it.
- `mutmut results --all true` lines (`module.function__mutmut_N: status`) are grouped by
  module; `cargo-mutants` `outcomes.json` entries are grouped by source file.
- Floors live in `.quality/mutation-baseline.json` and are set a fixed margin below the measured
  score, so natural variation never fails the report while a real loss of test strength does.

Flow (pseudocode):
1. Read the results of one or both tools into killed/survived counts per module.
2. Compute scores, print a Markdown table and optionally write it to a file.
3. Compare with the floors (or write them from the current scores) and exit 0 or 1.

Important criteria:
- Timeouts count as killed: a mutant that makes the tests hang was detected.
- A module present in the baseline but missing from the results fails, so a silently shrinking
  run cannot pass.
- This report measures how strongly the tests constrain the code; it is not a coverage metric.

Inputs/Outputs:
- Input: `mutmut` results text and/or a `cargo-mutants` `outcomes.json`, plus the baseline path.
- Output: a Markdown table on standard output and exit status 0 (pass) or 1 (a floor was missed).
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Counts = dict[str, collections.Counter[str]]


class MutationReport:
    """Parse mutation tool output and compare scores with floors.

    Responsibilities:
    - Group `mutmut` and `cargo-mutants` results by module and compute their scores.
    - Render a table and check scores against the recorded floors.

    Attributes:
        MARGIN (int): Points subtracted from a measured score to obtain its floor.
        CAUGHT_CARGO (frozenset[str]): `cargo-mutants` summaries that count as killed.
        MISSED_CARGO (frozenset[str]): `cargo-mutants` summaries that count as survived.
    """

    MARGIN = 5
    CAUGHT_CARGO = frozenset({"CaughtMutant", "Timeout"})
    MISSED_CARGO = frozenset({"MissedMutant"})
    MUTMUT_LINE = re.compile(r"^\s*(?P<module>\S+?)\.[^.\s]*__mutmut_\d+: (?P<status>.+?)\s*$")

    @classmethod
    def from_mutmut(cls, text: str) -> Counts:
        """Group `mutmut results --all true` lines by module.

        Args:
            text (str): Output of `mutmut results --all true`.

        Returns:
            Counts: Module name mapped to a counter of `killed` and `survived`.
        """
        counts: Counts = collections.defaultdict(collections.Counter)
        for line in text.splitlines():
            match = cls.MUTMUT_LINE.match(line)
            if match is None:
                continue
            status = match["status"]
            if status in ("killed", "timeout"):
                counts[match["module"]]["killed"] += 1
            elif status == "survived":
                counts[match["module"]]["survived"] += 1
        return counts

    @classmethod
    def from_cargo(cls, outcomes: dict[str, Any]) -> Counts:
        """Group `cargo-mutants` outcomes by source file.

        Args:
            outcomes (dict[str, Any]): Parsed `outcomes.json`.

        Returns:
            Counts: Repository-relative file mapped to a counter of `killed` and `survived`.
        """
        counts: Counts = collections.defaultdict(collections.Counter)
        for entry in outcomes["outcomes"]:
            scenario = entry["scenario"]
            if scenario == "Baseline":
                continue
            name = scenario["Mutant"]["file"]
            if entry["summary"] in cls.CAUGHT_CARGO:
                counts[name]["killed"] += 1
            elif entry["summary"] in cls.MISSED_CARGO:
                counts[name]["survived"] += 1
        return counts

    @staticmethod
    def score(counter: collections.Counter[str]) -> float:
        """Return the percentage of viable mutants that the tests killed.

        Args:
            counter (collections.Counter[str]): Counts of `killed` and `survived`.

        Returns:
            float: Score from 0 to 100, or 100 when no mutant was viable.
        """
        viable = counter["killed"] + counter["survived"]
        return 100.0 * counter["killed"] / viable if viable else 100.0

    @classmethod
    def table(cls, sections: dict[str, Counts]) -> str:
        """Render the scores as Markdown.

        Args:
            sections (dict[str, Counts]): Section name (`python`, `rust`) mapped to its counts.

        Returns:
            str: A Markdown table with one row per module and a total per section.
        """
        lines = [
            "| Section | Module | Killed | Survived | Score |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for section, counts in sections.items():
            total: collections.Counter[str] = collections.Counter()
            for module, counter in sorted(counts.items()):
                total.update(counter)
                lines.append(
                    f"| {section} | {module} | {counter['killed']} | {counter['survived']} | "
                    f"{cls.score(counter):.1f}% |"
                )
            lines.append(
                f"| {section} | **total** | {total['killed']} | {total['survived']} | "
                f"**{cls.score(total):.1f}%** |"
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def floors(cls, sections: dict[str, Counts]) -> dict[str, dict[str, int]]:
        """Derive floors a fixed margin below the measured scores.

        Args:
            sections (dict[str, Counts]): Section name mapped to its counts.

        Returns:
            dict[str, dict[str, int]]: Section, then module, mapped to an integer floor.
        """
        return {
            section: {
                module: max(0, math.floor(cls.score(counter)) - cls.MARGIN)
                for module, counter in sorted(counts.items())
            }
            for section, counts in sections.items()
        }

    @classmethod
    def failures(
        cls, sections: dict[str, Counts], baseline: dict[str, dict[str, int]]
    ) -> list[str]:
        """List modules below their floor or missing from the results.

        Args:
            sections (dict[str, Counts]): Section name mapped to its counts.
            baseline (dict[str, dict[str, int]]): Recorded floors.

        Returns:
            list[str]: One message per violation, in section and module order.
        """
        problems = []
        for section, counts in sections.items():
            for module, floor in sorted(baseline.get(section, {}).items()):
                if module not in counts:
                    problems.append(f"{section} {module}: missing from the results")
                elif cls.score(counts[module]) < floor:
                    problems.append(
                        f"{section} {module}: score {cls.score(counts[module]):.1f}% is below "
                        f"its floor of {floor}%"
                    )
        return problems

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Read results, print the table and check or write the floors.

        Args:
            argv (Sequence[str] | None): Command-line arguments, or the process arguments.

        Returns:
            int: 0 when every floor is met, 1 otherwise.

        Raises:
            SystemExit: If no results file is given or the arguments are invalid.
            OSError: If a results or baseline file cannot be read or written.
        """
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--mutmut-results", type=Path)
        parser.add_argument("--cargo-outcomes", type=Path)
        parser.add_argument(
            "--baseline", type=Path, default=Path(".quality/mutation-baseline.json")
        )
        parser.add_argument("--write-baseline", action="store_true")
        args = parser.parse_args(argv)
        sections: dict[str, Counts] = {}
        if args.mutmut_results:
            sections["python"] = cls.from_mutmut(args.mutmut_results.read_text(encoding="utf-8"))
        if args.cargo_outcomes:
            outcomes = json.loads(args.cargo_outcomes.read_text(encoding="utf-8"))
            sections["rust"] = cls.from_cargo(outcomes)
        if not sections:
            parser.error("give --mutmut-results and/or --cargo-outcomes")
        print(cls.table(sections))
        if args.write_baseline:
            args.baseline.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(cls.floors(sections), indent=2, sort_keys=True) + "\n"
            args.baseline.write_text(content, encoding="utf-8")
            return 0
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        problems = cls.failures(sections, baseline)
        for message in problems:
            print(f"FLOOR {message}")
        return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(MutationReport.main())
