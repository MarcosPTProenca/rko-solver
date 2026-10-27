"""Summarize held-out tuning results with bootstrap intervals.

Objective:
- Turn the JSON-lines results into per-cell tables that say how much each tuner gains over the
  packaged default, with an interval that reflects both the tuning repetitions and the
  held-out instances.

Technical approach:
- For one (algorithm, family) cell every tuner has a matrix `G[rep, instance]` of held-out
  scores (normalized gaps, lower is better) and the default has a vector `D[instance]`.
- The mean score and the mean difference to the default are bootstrapped by resampling
  repetitions and instances together, which keeps both sources of uncertainty in the interval.
- A tuner is called better, worse or indistinguishable from the default according to whether the
  95 percent interval of the difference lies below, above or around zero.

Flow (pseudocode):
1. Group records by (algorithm, family) and tuner into matrices.
2. Bootstrap the mean and the difference to the default.
3. Render one Markdown table per cell and a tally across cells.

Important criteria:
- Intervals are percentile bootstraps with a fixed seed, so a report is reproducible.
- Losses against the default are reported like wins; nothing is dropped.

Inputs/Outputs:
- Input: a JSON-lines results file.
- Output: Markdown text.
"""

from __future__ import annotations

import collections
import json
import numpy as np
from pathlib import Path

Matrix = np.ndarray


class Analysis:
    """Bootstrap summaries of held-out results.

    Responsibilities:
    - Load records, build matrices and render tables.

    Attributes:
        records (list[dict]): Session records.
        resamples (int): Bootstrap resamples.
        seed (int): Bootstrap seed.
    """

    def __init__(self, results: Path, resamples: int = 4000, seed: int = 0) -> None:
        """Load a results file.

        Args:
            results (Path): JSON-lines file written by the protocol.
            resamples (int): Bootstrap resamples.
            seed (int): Bootstrap seed.

        Raises:
            OSError: If the file cannot be read.
        """
        lines = [json.loads(line) for line in results.read_text(encoding="utf-8").splitlines()]
        self.records = [line for line in lines if line.get("kind") == "session"]
        self.meta = next((line for line in lines if line.get("kind") == "meta"), {})
        self.resamples = resamples
        self.seed = seed

    def matrices(self, algorithm: str, family: str) -> dict[str, Matrix]:
        """Build the held-out score matrix of every tuner in one cell.

        Args:
            algorithm (str): Algorithm name.
            family (str): Problem family.

        Returns:
            dict[str, Matrix]: Tuner name to an array of shape `(repetitions, instances)`.
        """
        rows: dict[str, list[list[float]]] = collections.defaultdict(list)
        for record in sorted(self.records, key=lambda r: r["rep"]):
            if record["algorithm"] == algorithm and record["family"] == family:
                rows[record["tuner"]].append(record["test"])
        return {tuner: np.asarray(values) for tuner, values in rows.items()}

    def interval(
        self, matrix: Matrix, baseline: Matrix | None = None
    ) -> tuple[float, float, float]:
        """Bootstrap the mean score, or its difference to a baseline vector.

        Args:
            matrix (Matrix): Scores of shape `(repetitions, instances)`.
            baseline (Matrix | None): Per-instance baseline scores of shape `(instances,)`.

        Returns:
            tuple[float, float, float]: Estimate, lower and upper 95 percent bounds.
        """
        values = matrix if baseline is None else matrix - baseline[None, :]
        rng = np.random.default_rng(self.seed)
        reps, instances = values.shape
        draws = np.empty(self.resamples)
        for index in range(self.resamples):
            rows = rng.integers(reps, size=reps)
            columns = rng.integers(instances, size=instances)
            draws[index] = values[np.ix_(rows, columns)].mean()
        low, high = np.percentile(draws, [2.5, 97.5])
        return float(values.mean()), float(low), float(high)

    def verdict(self, matrix: Matrix, baseline: Matrix) -> str:
        """Classify a tuner against the default.

        Args:
            matrix (Matrix): Tuner scores of shape `(repetitions, instances)`.
            baseline (Matrix): Default scores of shape `(instances,)`.

        Returns:
            str: `better`, `worse` or `tie`.
        """
        _, low, high = self.interval(matrix, baseline)
        if high < 0:
            return "better"
        if low > 0:
            return "worse"
        return "tie"

    def cell(self, algorithm: str, family: str) -> str:
        """Render the table of one (algorithm, family) cell.

        Args:
            algorithm (str): Algorithm name.
            family (str): Problem family.

        Returns:
            str: Markdown section, or an empty string when the cell has no default.
        """
        data = self.matrices(algorithm, family)
        if "default" not in data:
            return ""
        default = data["default"][0]
        lines = [
            f"### {algorithm} on {family}",
            "",
            "| Tuner | Runs | Held-out gap (95% CI) | Difference to default (95% CI) "
            "| Sessions better than default | Verdict |",
            "| --- | ---: | --- | --- | ---: | --- |",
        ]
        for tuner in ["default", "random", *sorted(set(data) - {"default", "random"})]:
            if tuner not in data:
                continue
            matrix = data[tuner]
            mean, low, high = self.interval(matrix)
            runs = np.mean([r["runs_used"] for r in self._select(algorithm, family, tuner)])
            if tuner == "default":
                lines.append(f"| default | 0 | {mean:.3f} ({low:.3f} to {high:.3f}) | - | - | - |")
                continue
            diff, dlow, dhigh = self.interval(matrix, default)
            better = int(np.sum(matrix.mean(axis=1) < default.mean()))
            lines.append(
                f"| {tuner} | {runs:.0f} | {mean:.3f} ({low:.3f} to {high:.3f}) | "
                f"{diff:+.3f} ({dlow:+.3f} to {dhigh:+.3f}) | {better}/{len(matrix)} | "
                f"{self.verdict(matrix, default)} |"
            )
        return "\n".join(lines) + "\n"

    def _select(self, algorithm: str, family: str, tuner: str) -> list[dict]:
        """Return the records of one tuner in one cell.

        Args:
            algorithm (str): Algorithm name.
            family (str): Problem family.
            tuner (str): Tuner name.

        Returns:
            list[dict]: Matching session records.
        """
        return [
            r
            for r in self.records
            if (r["algorithm"], r["family"], r["tuner"]) == (algorithm, family, tuner)
        ]

    def report(self) -> str:
        """Render every cell and a tally of verdicts per tuner.

        Returns:
            str: Markdown report.
        """
        cells = sorted({(r["algorithm"], r["family"]) for r in self.records})
        tally: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
        sections = []
        for algorithm, family in cells:
            text = self.cell(algorithm, family)
            if not text:
                continue
            sections.append(text)
            data = self.matrices(algorithm, family)
            for tuner, matrix in data.items():
                if tuner != "default":
                    tally[tuner][self.verdict(matrix, data["default"][0])] += 1
        header = ["## Verdicts against the packaged default, per cell", ""]
        header += ["| Tuner | Better | Tie | Worse |", "| --- | ---: | ---: | ---: |"]
        for tuner, counts in sorted(tally.items()):
            header.append(f"| {tuner} | {counts['better']} | {counts['tie']} | {counts['worse']} |")
        return "\n".join(header) + "\n\n" + "\n".join(sections)
