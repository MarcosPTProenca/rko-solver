"""Config overrides and verdict statistics for the non-inferiority gate.

Objective:
- Hold the config-override data model and the statistics of the non-inferiority gate so the
  orchestrator in `gate.py` stays within the project size and complexity budgets.

Technical approach:
- `ConfigOverrides` resolves `KEY=VALUE` items against the default `config.toml` and writes a
  temporary parameter directory with the overrides applied.
- `GateReport` pairs candidate and baseline per-instance medians, bounds the difference and
  applies the margin rule, the minimum instance count and Holm correction.

Flow (pseudocode):
1. Parse per-side overrides.
2. Collapse replicates to per-instance medians, pair the sides and decide each cell.
3. Holm-adjust the tested cells of each budget and render the verdict table.

Important criteria:
- A cell with fewer than `minimum` paired instances is `inconclusive` and is not tested.
- The margin is max(fraction * median baseline gap, family floor), in the unit of
  `normalized_gap`.
- An override key must exist in the default `config.toml` and its value is parsed with the type
  of the default; anything else is refused before any trial.

Inputs/Outputs:
- Input: override items and trial rows.
- Output: parameter directories and per-family, per-budget verdict cells.
"""

from __future__ import annotations

import json
import numpy as np
import re
import shutil
import statistics
import tomllib
import zlib
from benchmarks.engines.cpp_python_knapsack import KnapsackBenchmark
from benchmarks.engines.native_report import NativeReport
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from rko_solver.parameters.loader import default_params_dir
from typing import Any

GATE_SEED = 20261201
MIN_INSTANCES = 15
ALPHA = 0.05
MARGIN_FRACTION = 0.02
BOOTSTRAP_SAMPLES = 10_000
KNAPSACK_FLOOR = 0.05
RASTRIGIN_FLOOR = 0.05
DEFAULT_FLOORS = {"knapsack": KNAPSACK_FLOOR, "rastrigin": RASTRIGIN_FLOOR}
NOISE_NOTE = (
    "Note: time-budget results are noisy when the machine has fewer CPUs than engine threads "
    "(6 CPUs versus 10 portfolio threads on the study machine); trust the equal-evaluation "
    "cells for mechanism and the time cells only with that caveat."
)


@dataclass(frozen=True, slots=True)
class ConfigOverrides:
    """Parse and materialize scalar overrides of the solver `config.toml`.

    Responsibilities:
    - Resolve each `KEY=VALUE` against the keys of the default `config.toml`.
    - Parse the value with the type of the default value and refuse mismatches.
    - Write a temporary parameter directory with the overrides applied.

    Attributes:
        values (dict[tuple[str | None, str], bool | int | float | str]): Parsed overrides keyed by
            (table, key); the table is None for a top-level key.
    """

    values: dict[tuple[str | None, str], bool | int | float | str]

    @staticmethod
    def _scalars(config: Mapping[str, Any]) -> dict[tuple[str | None, str], Any]:
        """List the scalar keys of a parsed `config.toml`, top level and one table deep.

        Args:
            config (Mapping[str, Any]): Parsed file.

        Returns:
            dict[tuple[str | None, str], Any]: Default value per (table, key).
        """
        scalars: dict[tuple[str | None, str], Any] = {}
        for key, value in config.items():
            if isinstance(value, dict):
                scalars.update(
                    {
                        (key, inner): item
                        for inner, item in value.items()
                        if not isinstance(item, (dict, list))
                    }
                )
            elif not isinstance(value, list):
                scalars[(None, key)] = value
        return scalars

    @staticmethod
    def _matches(
        name: str, scalars: Mapping[tuple[str | None, str], Any]
    ) -> list[tuple[str | None, str]]:
        """List the keys that a bare or qualified name can refer to.

        Args:
            name (str): `key` or `table.key`.
            scalars (Mapping[tuple[str | None, str], Any]): Known keys of the default file.

        Returns:
            list[tuple[str | None, str]]: Zero, one or many matching key pairs.
        """
        table, _, key = name.rpartition(".")
        if table:
            pair = (table, key)
            return [pair] if pair in scalars else []
        top = (None, key)
        if top in scalars:
            return [top]
        return [candidate for candidate in scalars if candidate[1] == key]

    @classmethod
    def _resolve(
        cls, name: str, scalars: Mapping[tuple[str | None, str], Any], names: str
    ) -> tuple[str | None, str]:
        """Resolve a bare or qualified key name to a single `(table, key)` pair.

        Args:
            name (str): `key` or `table.key`.
            scalars (Mapping[tuple[str | None, str], Any]): Known keys of the default file.
            names (str): Comma-separated known names, for the unknown-key message.

        Returns:
            tuple[str | None, str]: The resolved key pair.

        Raises:
            ValueError: If the name matches no key or more than one.
        """
        matches = cls._matches(name, scalars)
        if not matches:
            raise ValueError(f"Unknown config.toml key {name!r}; this build defines: {names}.")
        if len(matches) > 1:
            raise ValueError(f"Key {name!r} is ambiguous; use one of {matches}.")
        return matches[0]

    @classmethod
    def parse(cls, items: Sequence[str], params_dir: Path | None = None) -> ConfigOverrides:
        """Validate `KEY=VALUE` items against the default `config.toml`.

        Args:
            items (Sequence[str]): Raw `KEY=VALUE` or `table.KEY=VALUE` strings.
            params_dir (Path | None): Parameter directory to validate against; the installed
                default when None.

        Returns:
            ConfigOverrides: Parsed overrides.

        Raises:
            ValueError: If an item is malformed, repeated, names an unknown or ambiguous key, or
                carries a value that does not parse as the type of the default value.
        """
        with ((params_dir or default_params_dir()) / "config.toml").open("rb") as stream:
            scalars = cls._scalars(tomllib.load(stream))
        names = ", ".join(sorted(".".join(filter(None, pair)) for pair in scalars))
        values: dict[tuple[str | None, str], bool | int | float | str] = {}
        for item in items:
            name, separator, text = item.partition("=")
            name = name.strip()
            if not separator or not name:
                raise ValueError(f"Option {item!r} must have the form KEY=VALUE.")
            pair = cls._resolve(name, scalars, names)
            if pair in values:
                raise ValueError(f"Key {name!r} is given more than once.")
            values[pair] = cls._convert(name, text.strip(), scalars[pair])
        return cls(values)

    @staticmethod
    def _convert(name: str, text: str, default: object) -> bool | int | float | str:
        """Parse a value with the type of the key's default value.

        Args:
            name (str): Key name, for messages.
            text (str): Raw value.
            default (Any): Default value whose type decides the parse.

        Returns:
            bool | int | float | str: Parsed value.

        Raises:
            ValueError: If the text is not a valid value of that type.
        """
        try:
            if isinstance(default, bool):
                if text.lower() not in ("true", "false"):
                    raise ValueError("expected true or false")
                return text.lower() == "true"
            if isinstance(default, int):
                return int(text)
            if isinstance(default, float):
                number = float(text)
                if not np.isfinite(number):
                    raise ValueError("expected a finite number")
                return number
        except ValueError as error:
            kind = type(default).__name__
            raise ValueError(f"Invalid value {text!r} for {name!r} ({kind}): {error}.") from error
        return text

    def as_record(self) -> dict[str, bool | int | float | str]:
        """Return the overrides as a JSON-serializable mapping.

        Returns:
            dict[str, bool | int | float | str]: `table.key` or `key` mapped to its value.
        """
        return {".".join(filter(None, pair)): value for pair, value in sorted(self.values.items())}

    @staticmethod
    def _assignment(line: str, table: str | None) -> tuple[str | None, str] | None:
        """Return the `(table, key)` of an assignment line, or None when it is not one.

        Args:
            line (str): One `config.toml` line.
            table (str | None): Current table, or None at the top level.

        Returns:
            tuple[str | None, str] | None: The key pair of an assignment, or None.
        """
        assignment = re.match(r"\s*([A-Za-z0-9_-]+)\s*=", line)
        return (table, assignment.group(1)) if assignment else None

    @staticmethod
    def _rewrite_line(line: str, pair: tuple[str | None, str], value: object) -> str:
        """Rewrite one assignment line to carry a new value, preserving its indentation.

        Args:
            line (str): Original `config.toml` line.
            pair (tuple[str | None, str]): Key pair whose value is replaced.
            value (object): New value; bool and str are JSON-encoded, others use `repr`.

        Returns:
            str: Rewritten line with the same leading whitespace.
        """
        literal = json.dumps(value) if isinstance(value, (bool, str)) else repr(value)
        prefix = line[: len(line) - len(line.lstrip())]
        return f"{prefix}{pair[1]} = {literal}"

    def _apply_overrides(self, lines: list[str]) -> set[tuple[str | None, str]]:
        """Rewrite the assignment lines in place for every override found, tracking current table.

        Args:
            lines (list[str]): `config.toml` lines, modified in place.

        Returns:
            set[tuple[str | None, str]]: Key pairs whose assignment line was rewritten.
        """
        table: str | None = None
        found: set[tuple[str | None, str]] = set()
        for index, line in enumerate(lines):
            header = re.fullmatch(r"\s*\[([^\[\]]+)\]\s*(#.*)?", line)
            if header:
                table = header.group(1).strip()
                continue
            pair = self._assignment(line, table)
            if pair in self.values and pair not in found:
                lines[index] = self._rewrite_line(line, pair, self.values[pair])
                found.add(pair)
        return found

    def _rewrite(self, target: Path) -> None:
        """Rewrite in place the assignment lines of `target` that this override touches.

        Args:
            target (Path): `config.toml` already copied into the destination.

        Raises:
            ValueError: If a requested key has no assignment line, or the rewritten file does not
                parse back to the requested values.
            OSError: If the file cannot be read or written.
        """
        lines = target.read_text(encoding="utf-8").splitlines()
        found = self._apply_overrides(lines)
        if found != set(self.values):
            raise ValueError(f"Could not locate {sorted(set(self.values) - found)} in config.toml.")
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written = self._scalars(tomllib.loads(target.read_text(encoding="utf-8")))
        if any(written.get(pair) != value for pair, value in self.values.items()):
            raise ValueError("The rewritten config.toml does not hold the requested values.")

    def write(self, destination: Path, source: Path | None = None) -> Path:
        """Copy the parameter files to `destination` and apply the overrides to `config.toml`.

        Args:
            destination (Path): Existing empty directory that receives the copy.
            source (Path | None): Parameter directory to copy; the installed default when None.

        Returns:
            Path: `destination`, ready to be passed as `params_dir`.

        Raises:
            ValueError: If a key line is not found or the rewritten file does not parse back to
                the requested values.
            OSError: If a file cannot be read or written.
        """
        for toml_file in (source or default_params_dir()).glob("*.toml"):
            shutil.copy2(toml_file, destination / toml_file.name)
        self._rewrite(destination / "config.toml")
        return destination


class GateReport:
    """Compute and render the non-inferiority verdicts.

    Responsibilities:
    - Pair candidate and baseline per-instance medians and bound the difference.
    - Apply the margin rule, the minimum instance count and Holm correction.

    Attributes:
        margin_fraction (float): Margin as a fraction of the median baseline gap.
        minimum (int): Smallest paired instance count that can be tested.
        floors (dict[str, float]): Absolute margin floor per family prefix (`knapsack`,
            `rastrigin`), in the unit of `normalized_gap`.
    """

    def __init__(
        self,
        margin_fraction: float = MARGIN_FRACTION,
        minimum: int = MIN_INSTANCES,
        floors: Mapping[str, float] | None = None,
    ) -> None:
        """Store the decision rule.

        Args:
            margin_fraction (float): Nonnegative margin as a fraction of the baseline gap.
            minimum (int): Positive minimum paired instances.
            floors (Mapping[str, float] | None): Overrides merged over `DEFAULT_FLOORS`.

        Raises:
            ValueError: If an argument is out of range.
        """
        merged = {**DEFAULT_FLOORS, **(floors or {})}
        if margin_fraction < 0 or minimum < 1 or any(value < 0 for value in merged.values()):
            raise ValueError("margin_fraction and floors must be nonnegative, minimum positive.")
        self.margin_fraction = margin_fraction
        self.minimum = minimum
        self.floors = merged

    @staticmethod
    def parse_floors(text: str) -> dict[str, float]:
        """Parse `knapsack=0.05,rastrigin=0.1` into a floor mapping.

        Args:
            text (str): Comma-separated `family=value` pairs.

        Returns:
            dict[str, float]: Floor per family prefix.

        Raises:
            ValueError: If a pair is malformed, the family is unknown or the value is negative.
        """
        floors: dict[str, float] = {}
        for pair in filter(None, text.split(",")):
            name, separator, value = pair.partition("=")
            if not separator or name not in DEFAULT_FLOORS or float(value) < 0:
                raise ValueError(f"Invalid floor {pair!r}; use {sorted(DEFAULT_FLOORS)}=value.")
            floors[name] = float(value)
        return floors

    @staticmethod
    def _collect(
        rows: Sequence[Mapping[str, Any]],
    ) -> tuple[
        dict[tuple[str, str, Any], dict[str, dict[str, list[float]]]],
        dict[tuple[str, str, Any], dict[str, int]],
    ]:
        """Group rows by (family, budget, evaluations), separating gaps from failures.

        Args:
            rows (Sequence[Mapping[str, Any]]): Trial rows with a `side` and `status`.

        Returns:
            tuple: Per-key per-side instance gaps and per-key per-side failure counts.
        """
        cells: dict[tuple[str, str, Any], dict[str, dict[str, list[float]]]] = {}
        failures: dict[tuple[str, str, Any], dict[str, int]] = {}
        for row in rows:
            key = (str(row["family"]), str(row["budget"]), row.get("max_evaluations"))
            side = str(row["side"])
            failures.setdefault(key, {"candidate": 0, "baseline": 0})
            if row["status"] != "ok":
                failures[key][side] += 1
                continue
            cells.setdefault(key, {"candidate": {}, "baseline": {}})[side].setdefault(
                str(row["instance"]), []
            ).append(float(row["normalized_gap"]))
        return cells, failures

    def _margin(self, family: str, baseline: np.ndarray) -> tuple[float, str]:
        """Compute the margin and the term that set it for one family.

        Args:
            family (str): Analysis family, used to look up the floor.
            baseline (np.ndarray): Paired baseline medians; empty when there are no pairs.

        Returns:
            tuple[float, str]: Effective margin and `floor` or `fraction`.
        """
        fraction = self.margin_fraction * float(np.median(baseline)) if baseline.size else 0.0
        floor = self.floors.get(family.split("-")[0], 0.0)
        return max(fraction, floor), "floor" if floor >= fraction else "fraction"

    @staticmethod
    def _medians(
        shared: Sequence[str],
        candidate: np.ndarray,
        baseline: np.ndarray,
        differences: np.ndarray,
    ) -> dict[str, float | None]:
        """Summarize the paired medians, returning None for every field when there are no pairs.

        Args:
            shared (Sequence[str]): Instances present on both sides.
            candidate (np.ndarray): Candidate per-instance medians.
            baseline (np.ndarray): Baseline per-instance medians.
            differences (np.ndarray): Candidate-minus-baseline differences.

        Returns:
            dict[str, float | None]: Candidate, baseline and difference medians.
        """
        if not shared:
            return {"candidate_median": None, "baseline_median": None, "median_difference": None}
        return {
            "candidate_median": float(np.median(candidate)),
            "baseline_median": float(np.median(baseline)),
            "median_difference": float(np.median(differences)),
        }

    def _cell(
        self,
        key: tuple[str, str, Any],
        sides: Mapping[str, Mapping[str, list[float]]],
        failures: Mapping[str, int],
    ) -> dict[str, Any]:
        """Build one verdict cell with medians, bootstrap interval, margin and signed-rank p.

        Args:
            key (tuple[str, str, Any]): Family, budget label and evaluation cap.
            sides (Mapping[str, Mapping[str, list[float]]]): Candidate and baseline gaps.
            failures (Mapping[str, int]): Candidate and baseline failure counts.

        Returns:
            dict[str, Any]: Cell with a verdict of `inconclusive` or `pending`.
        """
        shared = sorted(set(sides["candidate"]) & set(sides["baseline"]))
        candidate = np.array([statistics.median(sides["candidate"][name]) for name in shared])
        baseline = np.array([statistics.median(sides["baseline"][name]) for name in shared])
        differences = candidate - baseline
        margin, term = self._margin(key[0], baseline)
        cell: dict[str, Any] = {
            "family": key[0],
            "budget": key[1],
            "max_evaluations": key[2],
            "instances": len(shared),
            **self._medians(shared, candidate, baseline, differences),
            "ci_low": None,
            "ci_high": None,
            "margin": margin,
            "margin_term": term,
            "p_one_sided": None,
            "p_holm": None,
            "candidate_failures": failures["candidate"],
            "baseline_failures": failures["baseline"],
            "verdict": "inconclusive",
        }
        if shared:
            cell["ci_low"], cell["ci_high"] = KnapsackBenchmark.bootstrap_median_interval(
                [float(value) for value in differences],
                seed=GATE_SEED + zlib.crc32(repr(key).encode()),
                samples=BOOTSTRAP_SAMPLES,
            )
        if len(shared) >= self.minimum:
            p_value, effect = NativeReport.signed_rank_p(differences - margin)
            cell["p_one_sided"] = p_value / 2 if effect < 0 else 1.0
            cell["verdict"] = "pending"
        return cell

    @staticmethod
    def _holm_budget(tested: list[dict[str, Any]]) -> None:
        """Holm-adjust the p-values of one budget's tested cells and set their verdicts.

        Args:
            tested (list[dict[str, Any]]): Cells of one budget, all marked `pending`.
        """
        adjusted = NativeReport.holm([cell["p_one_sided"] for cell in tested])
        for cell, p_holm in zip(tested, adjusted, strict=True):
            cell["p_holm"] = p_holm
            cell["verdict"] = (
                "pass" if cell["ci_high"] <= cell["margin"] and p_holm < ALPHA else "fail"
            )

    @staticmethod
    def _holm(cells: list[dict[str, Any]]) -> None:
        """Holm-adjust the tested cells of each budget in place and set their verdicts.

        Args:
            cells (list[dict[str, Any]]): Cells, some marked `pending`.
        """
        budgets = {(cell["budget"], cell["max_evaluations"]) for cell in cells}
        for budget in budgets:
            tested = [
                cell
                for cell in cells
                if (cell["budget"], cell["max_evaluations"]) == budget
                and cell["verdict"] == "pending"
            ]
            GateReport._holm_budget(tested)

    def analyze(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Return one verdict per family and budget.

        Args:
            rows (Sequence[Mapping[str, Any]]): Trial rows with both `candidate` and `baseline`
                sides.

        Returns:
            list[dict[str, Any]]: Cells with counts, medians, interval, margin, p-values and
            a verdict of `pass`, `fail` or `inconclusive`.
        """
        cells, failures = self._collect(rows)
        results = [
            self._cell(key, cells.get(key, {"candidate": {}, "baseline": {}}), failures[key])
            for key in sorted(failures, key=str)
        ]
        self._holm(results)
        return results

    @staticmethod
    def render(cells: Sequence[Mapping[str, Any]]) -> str:
        """Format the verdicts as a fixed-width table.

        Args:
            cells (Sequence[Mapping[str, Any]]): Output of `analyze`.

        Returns:
            str: Table followed by the noise caveat.
        """

        def number(value: float | None, digits: int = 3) -> str:
            """Format an optional number.

            Args:
                value (float | None): Value or None.
                digits (int): Decimal places.

            Returns:
                str: Formatted value, or a dash for None.
            """
            return "-" if value is None else f"{value:.{digits}f}"

        lines = [
            f"{'family':14} {'budget':7} {'n':>3} {'cand':>9} {'base':>9} {'diff':>9} "
            f"{'ci_high':>9} {'margin':>9} {'term':>8} {'p_holm':>7} {'fail c/b':>8} verdict"
        ]
        for cell in cells:
            lines.append(
                f"{cell['family']:14} {cell['budget']:7} {cell['instances']:3d} "
                f"{number(cell['candidate_median']):>9} {number(cell['baseline_median']):>9} "
                f"{number(cell['median_difference']):>9} {number(cell['ci_high']):>9} "
                f"{number(cell['margin'], 4):>9} {cell['margin_term']:>8} "
                f"{number(cell['p_holm'], 4):>7} "
                f"{cell['candidate_failures']}/{cell['baseline_failures']:<6} {cell['verdict']}"
            )
        return "\n".join([*lines, NOISE_NOTE])
