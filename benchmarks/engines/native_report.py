"""Statistical report for the native-engine benchmark rounds.

Objective:
- Turn the per-trial benchmark data into explicit round winners, a ranking, effect sizes
  and figures, so a reader can see who won each round and how strong the evidence is.

Technical approach:
- The unit of analysis is the instance: replicates are collapsed to a per-instance median,
  and every comparison is paired by instance for one solver at a time.
- Each paired comparison uses an exact two-sided signed-rank permutation test (ties and
  zero differences handled exactly), a seeded bootstrap interval for the median difference,
  the rank-biserial correlation as effect size and Holm correction over the solvers of the
  round. Parity claims use an interval-inclusion equivalence test (TOST at 5 percent).
- A round is won by the contender that no other contender beats on net (significantly
  better solvers minus worse ones); several unbeaten contenders tie for first; rounds whose
  instance count cannot reach significance are flagged as indicative instead of decided.

Flow (pseudocode):
1. `ingest` reads the raw study CSVs and writes one compact, gzip-compressed evidence file.
2. `analyze` builds per-instance medians and runs every paired test and ranking.
3. `render` rewrites the generated README block and the SVG figures from the analysis.

Important criteria:
- Lower scores are better: Knapsack gap to the exact optimum (percent), TSP gap to the best
  tour of the round (percent) and Rastrigin log10(max(objective, 0.01)), so that values below
  0.01 count as solved and ratios, not absolute values, drive the comparison.
- Output is deterministic: bootstrap resampling is seeded and no timestamp is emitted.
- A nonsignificant result is reported as a tie, never as a win, and underpowered rounds say so.

Inputs/Outputs:
- Input: raw study CSVs (`ingest`) or `references/native-evidence.csv.gz` (everything else).
- Output: the evidence file, a generated Markdown block inside `README.md` and SVG figures.
"""

import argparse
import csv
import gzip
import io
import math
import numpy as np
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "references" / "native-evidence.csv.gz"
README = ROOT / "README.md"
FIGURES = ROOT / "figures"
START = "<!-- native-report:start -->"
END = "<!-- native-report:end -->"
ALPHA = 0.05
BOOTSTRAP = 10_000
SEED = 20261110
GREEN, GRAY, RED, BLUE = "#2e9e5b", "#b8bec6", "#d64545", "#3b7dd8"
FLOOR = 0.01
LABELS = {
    "cpp64": "C++ reference",
    "rust": "Rust engine",
    "rust-numba": "Rust + Numba objective",
    "rust-batch": "Rust + NumPy block objective",
    "rust-callback": "Rust + Python callbacks",
    "python": "Python engine",
    "cpp-engine": "C++ backend in rko",
}
FAMILY_LABELS = {"kp": "Knapsack", "tsp": "TSP", "rastrigin": "Rastrigin"}
SCORE_LABELS = {
    "kp": "gap to exact optimum, %",
    "tsp": "gap to best tour, %",
    "rastrigin": "log10(max(objective, 0.01))",
}


@dataclass(frozen=True)
class RoundSpec:
    """Describe one benchmark round and where its raw rows come from.

    Attributes:
        key (str): Stable round identifier.
        title (str): Short human title.
        question (str): Question the round answers.
        contenders (tuple[str, ...]): Implementations compared.
        reference (str): Baseline implementation shown against every other contender.
        sources (tuple[tuple[str, str], ...]): Pairs of (family, raw CSV file name).
        parity_margin (dict[str, float] | None): Equivalence margins per family, in score
            units, or None when the round makes no parity claim.
        note (str): One-line context shown with the round.
    """

    key: str
    title: str
    question: str
    contenders: tuple[str, ...]
    reference: str
    sources: tuple[tuple[str, str], ...]
    parity_margin: dict[str, float] | None = None
    note: str = ""


ROUNDS = (
    RoundSpec(
        "engines-2s",
        "Engines at 2 s, one algorithm at a time",
        "Is the Rust engine better than the C++ reference with the same parameters?",
        ("cpp64", "rust"),
        "cpp64",
        (
            ("kp", "native-vs-cpp-2s-reference.csv"),
            ("tsp", "native-problems-vs-cpp-reference.csv"),
            ("rastrigin", "native-problems-vs-cpp-reference.csv"),
        ),
        note="12 solvers, 3 replicates, nominal 2 s.",
    ),
    RoundSpec(
        "equal-wall-time",
        "Equal wall time",
        "Does the advantage survive when Rust gets the time the C++ trial really used?",
        ("cpp64", "rust"),
        "cpp64",
        (
            ("kp", "native-vs-cpp-matched-time-reference.csv"),
            ("tsp", "native-problems-vs-cpp-matched-time-reference.csv"),
        ),
        note="C++ runs first; local-search sweeps in C++ ignore the deadline.",
    ),
    RoundSpec(
        "portfolio-2s",
        "Default portfolio at 2 s",
        "Does the ten-algorithm default portfolio win with more replicates?",
        ("cpp64", "rust"),
        "cpp64",
        (
            ("kp", "native-vs-cpp-portfolio-2s-reference.csv"),
            ("tsp", "native-problems-vs-cpp-portfolio-2s-reference.csv"),
            ("rastrigin", "native-problems-vs-cpp-portfolio-2s-reference.csv"),
        ),
        note="One solver (RKO-default), 5 replicates.",
    ),
    RoundSpec(
        "portfolio-10s",
        "Default portfolio at 10 s",
        "Does the portfolio advantage hold with a five times longer budget?",
        ("cpp64", "rust"),
        "cpp64",
        (
            ("kp", "native-vs-cpp-portfolio-10s-reference.csv"),
            ("tsp", "native-problems-vs-cpp-portfolio-10s-reference.csv"),
            ("rastrigin", "native-problems-vs-cpp-portfolio-10s-reference.csv"),
        ),
        note="One solver (RKO-default), 2 replicates.",
    ),
    RoundSpec(
        "python-objective",
        "Objective written in Python",
        "Can a problem defined in Python (Numba or NumPy blocks) match the C++ reference?",
        ("cpp64", "rust-numba", "rust-batch"),
        "cpp64",
        (
            ("kp", "native-python-defined-kp-reference.csv"),
            ("tsp", "native-python-defined-problems-reference.csv"),
            ("rastrigin", "native-python-defined-problems-reference.csv"),
        ),
        note="Three solvers; compile time is excluded from the budget.",
    ),
    RoundSpec(
        "callback-ladder",
        "Python callbacks against the Python engine",
        "What does each way of defining a problem buy over the pure-Python engine?",
        ("python", "rust-callback", "rust-batch", "rust-numba"),
        "python",
        (("kp", "native-callback-ladder-reference.csv"),),
        note="Only 5 instances: significance is unreachable, read it as indicative.",
    ),
    RoundSpec(
        "cpp-backend-parity",
        "C++ backend against the pinned C++ driver",
        "Does wrapping the reference inside rko leave its results unchanged?",
        ("cpp64", "cpp-engine"),
        "cpp64",
        (
            ("kp", "native-cpp-engine-kp-reference.csv"),
            ("tsp", "native-cpp-engine-tsp-reference.csv"),
            ("rastrigin", "native-cpp-engine-rastrigin-reference.csv"),
        ),
        parity_margin={"kp": 0.25, "tsp": 2.0, "rastrigin": 0.1},
        note="Parity is the goal: a tie is a pass, equivalence is tested explicitly.",
    ),
)
COLUMNS = (
    "round",
    "family",
    "instance",
    "solver",
    "implementation",
    "replicate",
    "budget_seconds",
    "elapsed_seconds",
    "objective",
    "score",
)


@dataclass(frozen=True)
class Comparison:
    """Hold one paired comparison of two implementations for one solver.

    Attributes:
        solver (str): Solver name.
        subject (str): Implementation tested against `reference`.
        reference (str): Baseline implementation.
        n (int): Number of paired instances.
        subject_median (float): Median over instances of the subject score.
        reference_median (float): Median over instances of the reference score.
        delta (float): Median paired difference, subject minus reference.
        low (float): Lower end of the 95 percent bootstrap interval of the difference.
        high (float): Upper end of the 95 percent bootstrap interval of the difference.
        eq_low (float): Lower end of the 90 percent interval used for equivalence.
        eq_high (float): Upper end of the 90 percent interval used for equivalence.
        better (int): Instances where the subject has the lower score.
        worse (int): Instances where the subject has the higher score.
        tied (int): Instances with equal scores.
        p (float): Exact two-sided signed-rank p-value before correction.
        p_holm (float): Holm-adjusted p-value over the solvers of the round and pair.
        effect (float): Rank-biserial correlation, negative when the subject is better.
        verdict (int): Minus one if the subject is better, one if worse, zero if tied.
    """

    solver: str
    subject: str
    reference: str
    n: int
    subject_median: float
    reference_median: float
    delta: float
    low: float
    high: float
    eq_low: float
    eq_high: float
    better: int
    worse: int
    tied: int
    p: float
    p_holm: float
    effect: float
    verdict: int


@dataclass(frozen=True)
class RoundResult:
    """Hold the analysis of one round for one problem family.

    Attributes:
        spec (RoundSpec): Round definition.
        family (str): Problem family key.
        instances (int): Number of instances.
        solvers (tuple[str, ...]): Solvers compared.
        comparisons (tuple[Comparison, ...]): Every paired comparison, all contender pairs.
        mean_rank (dict[str, float]): Mean rank of each contender over instance-solver blocks.
        leaders (tuple[str, ...]): Contenders no other contender significantly beats on net.
        underpowered (bool): Whether significance is unreachable at this instance count.
        elapsed (dict[str, tuple[float, float]]): Median and maximum seconds per contender.
    """

    spec: RoundSpec
    family: str
    instances: int
    solvers: tuple[str, ...]
    comparisons: tuple[Comparison, ...]
    mean_rank: dict[str, float]
    leaders: tuple[str, ...]
    underpowered: bool
    elapsed: dict[str, tuple[float, float]]

    @property
    def winner(self) -> str | None:
        """Return the sole leader of the round.

        Returns:
            str | None: The only unbeaten contender, or None when several tie for first.
        """
        return self.leaders[0] if len(self.leaders) == 1 else None


class NativeReport:
    """Ingest evidence, run the statistics and render the README block and figures.

    Responsibilities:
    - Convert raw study CSVs into one compact evidence file.
    - Run paired exact tests, bootstrap intervals, effect sizes, Holm correction and rankings.
    - Render deterministic Markdown and SVG outputs.

    Attributes:
        rows (list[dict[str, str]]): Evidence rows loaded from disk.
    """

    def __init__(self, rows: Sequence[dict[str, str]]) -> None:
        """Store the evidence rows.

        Args:
            rows (Sequence[dict[str, str]]): Rows with the evidence columns.
        """
        self.rows = list(rows)

    @staticmethod
    def load(path: Path = EVIDENCE) -> NativeReport:
        """Read the compressed evidence file.

        Args:
            path (Path): Gzip-compressed CSV written by `ingest`.

        Returns:
            NativeReport: Report over the stored rows.

        Raises:
            OSError: If the file cannot be read.
        """
        with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
            return NativeReport(list(csv.DictReader(stream)))

    @staticmethod
    def score(family: str, row: dict[str, str]) -> float:
        """Return the lower-is-better score of one raw study row.

        Args:
            family (str): `kp`, `tsp` or `rastrigin`.
            row (dict[str, str]): Raw study row.

        Returns:
            float: Gap percent (Knapsack, TSP; an infeasible Knapsack result scores 100, the
            worst possible gap) or log10(max(objective, 0.01)) (Rastrigin).

        Raises:
            KeyError: If the expected column is missing.
        """
        if family == "kp":
            return float(row["feasible_gap_percent"] or 100.0)
        if family == "tsp":
            return float(row["quality"])
        return math.log10(max(float(row["objective"]), FLOOR))

    @staticmethod
    def ingest(raw_dir: Path, output: Path = EVIDENCE) -> int:
        """Collapse the raw study CSVs into the compact evidence file.

        Args:
            raw_dir (Path): Directory holding the raw study CSV files.
            output (Path): Destination gzip file.

        Returns:
            int: Number of evidence rows written.

        Raises:
            OSError: If a raw file is missing or the output cannot be written.
        """
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(COLUMNS)
        count = 0
        for spec in ROUNDS:
            for family, name in spec.sources:
                with (raw_dir / name).open(encoding="utf-8", newline="") as stream:
                    for row in csv.DictReader(stream):
                        if row.get("problem", family) != family:
                            continue
                        if row["status"] != "ok":
                            raise ValueError(f"{name} contains a failed trial")
                        instance = (
                            f"{row['family']}/{row['instance']}"
                            if family == "kp"
                            else row["instance"]
                        )
                        writer.writerow(
                            [
                                spec.key,
                                family,
                                instance,
                                row["solver"],
                                row["implementation"],
                                row["replicate"],
                                row["budget_seconds"],
                                f"{float(row['elapsed_seconds']):.4f}",
                                row["objective"],
                                f"{NativeReport.score(family, row):.6f}",
                            ]
                        )
                        count += 1
        output.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(output, "wb", mtime=0) as stream:
            stream.write(buffer.getvalue().encode("utf-8"))
        return count

    @staticmethod
    def signed_rank_p(differences: np.ndarray) -> tuple[float, float]:
        """Run the exact two-sided signed-rank permutation test.

        Args:
            differences (np.ndarray): Paired differences; zeros are discarded.

        Returns:
            tuple[float, float]: p-value and rank-biserial correlation (positive when the
            first sample tends to be larger).
        """
        nonzero = differences[differences != 0]
        count = len(nonzero)
        if count == 0:
            return 1.0, 0.0
        magnitude = np.abs(nonzero)
        order = np.argsort(magnitude, kind="stable")
        ranks = np.empty(count)
        sorted_values = magnitude[order]
        start = 0
        while start < count:
            stop = start
            while stop + 1 < count and sorted_values[stop + 1] == sorted_values[start]:
                stop += 1
            ranks[order[start : stop + 1]] = (start + stop) / 2 + 1
            start = stop + 1
        doubled = np.rint(ranks * 2).astype(int)
        total = int(doubled.sum())
        positive = int(doubled[nonzero > 0].sum())
        counts = np.zeros(total + 1, dtype=object)
        counts[0] = 1
        for value in doubled:
            shifted = np.zeros_like(counts)
            shifted[value:] = counts[: total + 1 - value]
            counts = counts + shifted
        centre = total / 2
        offset = abs(positive - centre)
        sums = np.arange(total + 1)
        extreme = sum(counts[np.abs(sums - centre) >= offset - 1e-9])
        p_value = float(min(1.0, extreme / 2**count))
        effect = (positive - (total - positive)) / total
        return p_value, effect

    @staticmethod
    def bootstrap(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, ...]:
        """Bootstrap percentile bounds of the median difference.

        Args:
            differences (np.ndarray): Paired differences per instance.
            rng (np.random.Generator): Seeded generator.

        Returns:
            tuple[float, ...]: 2.5, 97.5, 5 and 95 percent quantiles of the resampled median.
        """
        draws = rng.integers(0, len(differences), size=(BOOTSTRAP, len(differences)))
        medians = np.median(differences[draws], axis=1)
        return tuple(float(np.quantile(medians, q)) for q in (0.025, 0.975, 0.05, 0.95))

    @staticmethod
    def holm(p_values: Sequence[float]) -> list[float]:
        """Apply the Holm step-down correction.

        Args:
            p_values (Sequence[float]): Unadjusted p-values.

        Returns:
            list[float]: Adjusted p-values in the original order.
        """
        order = sorted(range(len(p_values)), key=lambda index: p_values[index])
        adjusted = [0.0] * len(p_values)
        running = 0.0
        for position, index in enumerate(order):
            running = max(running, min(1.0, (len(p_values) - position) * p_values[index]))
            adjusted[index] = running
        return adjusted

    def medians(self, spec: RoundSpec, family: str) -> dict[tuple[str, str, str], float]:
        """Collapse replicates to the per-instance median score.

        Args:
            spec (RoundSpec): Round to select.
            family (str): Problem family to select.

        Returns:
            dict[tuple[str, str, str], float]: Median by (solver, instance, implementation).
        """
        groups: dict[tuple[str, str, str], list[float]] = {}
        for row in self.rows:
            if row["round"] == spec.key and row["family"] == family:
                key = (row["solver"], row["instance"], row["implementation"])
                groups.setdefault(key, []).append(float(row["score"]))
        return {key: statistics.median(values) for key, values in groups.items()}

    def analyze(self, spec: RoundSpec, family: str) -> RoundResult:
        """Run every paired test, the ranking and the winner rule for one round and family.

        Args:
            spec (RoundSpec): Round definition.
            family (str): Problem family key.

        Returns:
            RoundResult: Complete analysis.

        Raises:
            ValueError: If the evidence has no rows for the round and family.
        """
        medians = self.medians(spec, family)
        if not medians:
            raise ValueError(f"No evidence for {spec.key}/{family}")
        solvers = tuple(sorted({key[0] for key in medians}))
        instances = tuple(sorted({key[1] for key in medians}))
        rng = np.random.default_rng(SEED)
        comparisons: list[Comparison] = []
        pairs = [
            (spec.contenders[a], spec.contenders[b])
            for a in range(len(spec.contenders))
            for b in range(a + 1, len(spec.contenders))
        ]
        for first, second in pairs:
            subject, reference = (first, second) if second == spec.reference else (second, first)
            block: list[Comparison] = []
            for solver in solvers:
                a = np.array([medians[(solver, name, subject)] for name in instances])
                b = np.array([medians[(solver, name, reference)] for name in instances])
                differences = a - b
                p_value, effect = self.signed_rank_p(differences)
                low, high, eq_low, eq_high = self.bootstrap(differences, rng)
                block.append(
                    Comparison(
                        solver,
                        subject,
                        reference,
                        len(instances),
                        float(np.median(a)),
                        float(np.median(b)),
                        float(np.median(differences)),
                        low,
                        high,
                        eq_low,
                        eq_high,
                        int((differences < 0).sum()),
                        int((differences > 0).sum()),
                        int((differences == 0).sum()),
                        p_value,
                        1.0,
                        effect,
                        0,
                    )
                )
            adjusted = self.holm([item.p for item in block])
            for item, p_holm in zip(block, adjusted, strict=True):
                verdict = 0
                if p_holm < ALPHA:
                    verdict = -1 if item.delta < 0 or (item.delta == 0 and item.effect < 0) else 1
                comparisons.append(
                    Comparison(**{**item.__dict__, "p_holm": p_holm, "verdict": verdict})
                )
        net: dict[tuple[str, str], int] = {}
        for item in comparisons:
            net[(item.subject, item.reference)] = (
                net.get((item.subject, item.reference), 0) - item.verdict
            )
            net[(item.reference, item.subject)] = (
                net.get((item.reference, item.subject), 0) + item.verdict
            )
        dominated = {
            name
            for name in spec.contenders
            if any(net.get((other, name), 0) > 0 for other in spec.contenders)
        }
        leaders = (
            tuple(name for name in spec.contenders if name not in dominated) or spec.contenders
        )
        ranks = {name: [] for name in spec.contenders}
        for solver in solvers:
            for name in instances:
                scores = [medians[(solver, name, c)] for c in spec.contenders]
                for contender, value in zip(spec.contenders, scores, strict=True):
                    below = sum(1 for other in scores if other < value)
                    equal = sum(1 for other in scores if other == value)
                    ranks[contender].append(below + (equal + 1) / 2)
        mean_rank = {name: statistics.fmean(values) for name, values in ranks.items()}
        underpowered = 2 / 2 ** len(instances) * len(solvers) > ALPHA
        elapsed: dict[str, tuple[float, float]] = {}
        for contender in spec.contenders:
            values = [
                float(row["elapsed_seconds"])
                for row in self.rows
                if row["round"] == spec.key
                and row["family"] == family
                and row["implementation"] == contender
            ]
            elapsed[contender] = (statistics.median(values), max(values))
        return RoundResult(
            spec,
            family,
            len(instances),
            solvers,
            tuple(comparisons),
            mean_rank,
            leaders,
            underpowered,
            elapsed,
        )

    def results(self) -> list[RoundResult]:
        """Analyze every configured round and family present in the evidence.

        Returns:
            list[RoundResult]: One result per round and family, in specification order.
        """
        output: list[RoundResult] = []
        for spec in ROUNDS:
            for family, _ in spec.sources:
                if any(row["round"] == spec.key and row["family"] == family for row in self.rows):
                    output.append(self.analyze(spec, family))
        return output

    @staticmethod
    def name(key: str) -> str:
        """Return the display name of an implementation.

        Args:
            key (str): Implementation key.

        Returns:
            str: Human-readable label.
        """
        return LABELS.get(key, key)

    @staticmethod
    def effect_label(effect: float) -> str:
        """Classify a rank-biserial correlation by magnitude.

        Args:
            effect (float): Correlation in [-1, 1].

        Returns:
            str: negligible, small, medium or large.
        """
        size = abs(effect)
        return (
            "negligible"
            if size < 0.1
            else "small"
            if size < 0.3
            else "medium"
            if size < 0.5
            else "large"
        )

    def verdict_cell(self, item: Comparison) -> str:
        """Render the explicit winner of one paired comparison.

        Args:
            item (Comparison): Paired result.

        Returns:
            str: Colored marker and the winning implementation, or a tie marker.
        """
        if item.verdict < 0:
            return f"🟢 **{self.name(item.subject)}**"
        if item.verdict > 0:
            return f"🔴 **{self.name(item.reference)}**"
        return "⚪ tie"

    def scoreboard(self, results: Sequence[RoundResult]) -> str:
        """Render the headline scoreboard and overall tally.

        Args:
            results (Sequence[RoundResult]): Analyses of every round and family.

        Returns:
            str: Markdown tables.
        """
        lines = [
            "| Round | Family | Contenders | Outcome | 🟢 / ⚪ / 🔴 | Ranking (mean rank) |",
            "| --- | --- | --- | --- | :---: | --- |",
        ]
        tally: dict[str, list[int]] = {}
        parity = [0, 0]
        medals = ("🥇", "🥈", "🥉")
        for result in results:
            reference = result.spec.reference
            shown = [item for item in result.comparisons if item.reference == reference]
            wins = sum(1 for item in shown if item.verdict < 0)
            losses = sum(1 for item in shown if item.verdict > 0)
            if result.spec.parity_margin is not None:
                if losses == 0:
                    outcome = "✅ **parity**: no solver worse"
                    outcome += f", {wins} better" if wins else ""
                    parity[0] += 1
                else:
                    outcome = f"❌ **regression**: {losses} solvers worse"
                    parity[1] += 1
            elif result.underpowered:
                leader = min(result.mean_rank, key=lambda name: result.mean_rank[name])
                outcome = f"⚠️ indicative only: {self.name(leader)}"
            elif result.winner is not None:
                outcome = f"🏆 **{self.name(result.winner)}**"
            elif len(result.leaders) == len(result.spec.contenders):
                outcome = "🤝 **tie**"
            else:
                names = " = ".join(self.name(name) for name in result.leaders)
                outcome = f"🤝 **tie for first**: {names}"
            if result.spec.parity_margin is None and not result.underpowered:
                for name in result.spec.contenders:
                    row = tally.setdefault(name, [0, 0, 0])
                    if result.winner == name:
                        row[0] += 1
                    elif name in result.leaders:
                        row[1] += 1
                    else:
                        row[2] += 1
            ordered = sorted(result.mean_rank, key=lambda name: result.mean_rank[name])
            parts = []
            position = 0
            for index, name in enumerate(ordered):
                if (
                    index
                    and abs(result.mean_rank[name] - result.mean_rank[ordered[index - 1]]) > 1e-9
                ):
                    position = index
                medal = medals[position] if position < len(medals) else f"{position + 1}."
                parts.append(f"{medal} {self.name(name)} ({result.mean_rank[name]:.2f})")
            lines.append(
                f"| {result.spec.title} | {FAMILY_LABELS[result.family]} ({result.instances}) | "
                f"{' vs '.join(self.name(n) for n in result.spec.contenders)} | {outcome} | "
                f"{wins} / {len(shown) - wins - losses} / {losses} | {' · '.join(parts)} |"
            )
        lines += [
            "",
            "Rounds decided (parity and underpowered rounds excluded), by contender:",
            "",
            "| Contender | 🏆 Rounds won | 🤝 Tied | Lost to another contender |",
            "| --- | ---: | ---: | ---: |",
        ]
        for name, (won, tied, lost) in sorted(
            tally.items(), key=lambda item: (-item[1][0], item[1][2], item[0])
        ):
            lines.append(f"| {self.name(name)} | {won} | {tied} | {lost} |")
        lines += [
            "",
            f"C++ backend parity: ✅ {parity[0]} of {sum(parity)} families, "
            f"❌ {parity[1]} regressions.",
        ]
        return "\n".join(lines)

    def detail(self, result: RoundResult) -> str:
        """Render the per-solver table of one round and family.

        Args:
            result (RoundResult): Analysis to render.

        Returns:
            str: Collapsible Markdown section.
        """
        spec = result.spec
        margin = (spec.parity_margin or {}).get(result.family)
        shown = [item for item in result.comparisons if item.reference == spec.reference]
        columns = [
            "Solver",
            "Contender",
            "Median score (contender vs reference)",
            "Δ median [95% CI]",
            "Better / worse / tied",
            "Holm p",
            "Effect r",
            "Winner",
        ]
        alignment = ["---", "---", "---:", "---:", ":---:", "---:", ":---:", "---"]
        if margin is not None:
            columns.append(f"Parity (±{margin:g})")
            alignment.append(":---:")
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join(alignment) + " |"
        lines = [
            f"<details><summary><b>{spec.title} — {FAMILY_LABELS[result.family]}</b> "
            f"({result.instances} instances, score: {SCORE_LABELS[result.family]})</summary>",
            "",
            f"_{spec.question}_ {spec.note}",
            "",
            header,
            divider,
        ]
        for item in sorted(shown, key=lambda entry: (entry.subject, entry.solver)):
            cells = (
                f"| {item.solver} | {self.name(item.subject)} | "
                f"{item.subject_median:.3f} vs {item.reference_median:.3f} | "
                f"{item.delta:+.3f} [{item.low:+.3f}, {item.high:+.3f}] | "
                f"{item.better} / {item.worse} / {item.tied} | {item.p_holm:.3f} | "
                f"{item.effect:+.2f} ({self.effect_label(item.effect)}) | "
                f"{self.verdict_cell(item)} |"
            )
            if margin is not None:
                equivalent = item.eq_low > -margin and item.eq_high < margin
                cells += " ✅ equivalent |" if equivalent else " ❔ inconclusive |"
            lines.append(cells)
        times = " · ".join(
            f"{self.name(name)} {median:.2f} s (max {high:.2f})"
            for name, (median, high) in result.elapsed.items()
        )
        lines += ["", f"Elapsed seconds, median (maximum): {times}.", ""]
        if result.underpowered:
            lines += [
                f"⚠️ With {result.instances} instances and {len(result.solvers)} solvers even an "
                "all-wins column cannot reach Holm p < 0.05; read the effect sizes, "
                "not the verdicts.",
                "",
            ]
        lines += ["</details>", ""]
        return "\n".join(lines)

    def bottom_line(self, results: Sequence[RoundResult]) -> str:
        """Summarize the decided rounds in plain sentences computed from the analysis.

        Args:
            results (Sequence[RoundResult]): Analyses of every round and family.

        Returns:
            str: Markdown bullet list.
        """
        engine = [
            item
            for item in results
            if item.spec.key in ("engines-2s", "equal-wall-time", "portfolio-2s", "portfolio-10s")
        ]
        won = sum(1 for item in engine if item.winner == "rust")
        tied = sum(1 for item in engine if item.winner is None)
        lost = sum(1 for item in engine if item.winner == "cpp64")
        python = [item for item in results if item.spec.key == "python-objective"]

        def net(item: RoundResult, contender: str) -> int:
            """Count significantly better minus worse solvers against the C++ reference."""
            return -sum(
                entry.verdict
                for entry in item.comparisons
                if entry.subject == contender and entry.reference == "cpp64"
            )

        def describe(contender: str) -> str:
            """List the families where a contender is better, on par or worse than C++."""
            groups: dict[str, list[str]] = {"better": [], "on par": [], "worse": []}
            for item in python:
                value = net(item, contender)
                groups["better" if value > 0 else "worse" if value < 0 else "on par"].append(
                    FAMILY_LABELS[item.family]
                )
            return "; ".join(
                f"{name} in {', '.join(items)}" for name, items in groups.items() if items
            )

        parity = [item for item in results if item.spec.parity_margin is not None]
        regressions = sum(1 for item in parity for entry in item.comparisons if entry.verdict > 0)
        equivalent = sum(
            1
            for item in parity
            for entry in item.comparisons
            if entry.eq_low > -item.spec.parity_margin[item.family]
            and entry.eq_high < item.spec.parity_margin[item.family]
        )
        total = sum(len(item.comparisons) for item in parity)
        return "\n".join(
            [
                f"- 🏆 **Rust engine against the C++ reference:** won {won} of {len(engine)} "
                f"rounds, tied {tied}, lost {lost}.",
                f"- 🐍 **Objective written in Python, against the C++ reference:** Numba is "
                f"{describe('rust-numba')}; NumPy block objectives are {describe('rust-batch')}.",
                f"- ✅ **C++ backend parity:** {regressions} solver comparisons worse "
                f"than the pinned driver, {equivalent} of {total} proven equivalent within the "
                "margins below (the rest are inconclusive, not different).",
            ]
        )

    def method_text(self) -> str:
        """Return the static description of the statistical method.

        Returns:
            str: Markdown bullet list.
        """
        return "\n".join(
            [
                "**How to read this.** Lower scores are better everywhere.",
                "",
                "- **Unit of analysis.** Replicates collapse to one median per instance; the "
                "instance, not the trial, is the sample. Each solver is compared with the "
                "reference on the same instances (paired).",
                "- **Test.** Exact two-sided signed-rank permutation test (ties and zero "
                "differences handled exactly), Holm-adjusted over the solvers of the round. "
                "🟢/🔴 mean the adjusted p is below 0.05; anything else is ⚪ a tie, never a win.",
                "- **Size of the difference.** Δ is the median paired difference with a 95 % "
                f"bootstrap interval ({BOOTSTRAP:,} resamples, seed {SEED}); r is the "
                "rank-biserial correlation (−1: the contender is better on every instance, "
                "0: no tendency).",
                "- **Round winner.** The contender no other contender beats on net "
                "(significantly better solvers minus worse ones). Several unbeaten contenders "
                "tie for first. Rounds whose instance count cannot reach significance are "
                "marked ⚠️ and not counted.",
                "- **Parity** (C++ backend round) is tested by interval inclusion: the 90 % "
                "interval of Δ must lie inside the stated margin. That is a claim of practical "
                'equivalence, stronger than "not significantly different". Margins (0.25 gap '
                "points for Knapsack, 2 for TSP, 0.1 decades for Rastrigin) were fixed when this "
                "analysis was written, after the data existed.",
                "- **Scores.** Knapsack: gap to the exact optimum in percent; an infeasible "
                "result (five Python-engine trials in the callback round) scores 100, the worst "
                "possible gap. TSP: gap to the best tour of the round, percent. Rastrigin: "
                "log10(max(objective, 0.01)), so an objective below 0.01 counts as solved.",
            ]
        )

    def block(self, results: Sequence[RoundResult]) -> str:
        """Render the complete generated README block.

        Args:
            results (Sequence[RoundResult]): Analyses of every round and family.

        Returns:
            str: Markdown between the generation markers.
        """
        parts = [
            START,
            f"<!-- Generated by `python -m {__package__}.native_report render`; do not edit. -->",
            "",
            "### Results at a glance",
            "",
            "🟢 contender significantly better · ⚪ no significant difference (a tie) · "
            "🔴 significantly worse · ⚠️ too few instances to decide.",
            "",
            "![Outcome of every round](figures/scoreboard.svg)",
            "",
            self.bottom_line(results),
            "",
            self.scoreboard(results),
            "",
            self.method_text(),
            "",
            "![Engines at 2 s](figures/engines-2s.svg)",
            "",
            "![C++ backend parity](figures/cpp-backend-parity.svg)",
            "",
            "### Per-solver evidence",
            "",
            "Expand a round for every solver's paired result, effect size and winner.",
            "",
        ]
        parts += [self.detail(result) for result in results]
        parts.append(END)
        return "\n".join(parts)

    @staticmethod
    def tag(name: str, text: str | None = None, **attributes: object) -> str:
        """Build one SVG element.

        Args:
            name (str): Element name.
            text (str | None): Escaped text content, or None for an empty element.
            **attributes (object): Attributes; underscores become hyphens.

        Returns:
            str: Serialized element.
        """
        rendered = " ".join(
            f'{key.replace("_", "-")}="{value}"' for key, value in attributes.items()
        )
        if text is None:
            return f"<{name} {rendered}/>"
        return f"<{name} {rendered}>{escape(text)}</{name}>"

    def svg_open(self, width: int, height: int) -> list[str]:
        """Start an SVG document.

        Args:
            width (int): Canvas width.
            height (int): Canvas height.

        Returns:
            list[str]: Opening lines.
        """
        return [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" font-family="Helvetica, Arial, sans-serif" '
            'font-size="12">',
            self.tag("rect", width=width, height=height, fill="white"),
        ]

    def scoreboard_svg(self, results: Sequence[RoundResult]) -> str:
        """Draw stacked outcome bars with the winner of every round.

        Args:
            results (Sequence[RoundResult]): Analyses of every round and family.

        Returns:
            str: SVG document.
        """
        row, left, bar = 26, 380, 300
        out = self.svg_open(left + bar + 440, 70 + row * len(results))
        out.append(
            self.tag("text", "Who won each round", x=10, y=22, font_size=15, font_weight="bold")
        )
        for offset, color, label in (
            (0, GREEN, "contender better"),
            (130, GRAY, "tie"),
            (180, RED, "reference better"),
        ):
            out.append(self.tag("rect", x=left + offset, y=10, width=10, height=10, fill=color))
            out.append(self.tag("text", label, x=left + offset + 14, y=19, font_size=11))
        for index, result in enumerate(results):
            y = 40 + index * row
            pairs = [item for item in result.comparisons if item.reference == result.spec.reference]
            wins = sum(1 for item in pairs if item.verdict < 0)
            losses = sum(1 for item in pairs if item.verdict > 0)
            label = f"{result.spec.title} · {FAMILY_LABELS[result.family]}"
            out.append(self.tag("text", label, x=10, y=y + 13))
            x = float(left)
            for count, color in ((wins, GREEN), (len(pairs) - wins - losses, GRAY), (losses, RED)):
                width = bar * count / len(pairs)
                if width > 0:
                    out.append(
                        self.tag(
                            "rect", x=f"{x:.1f}", y=y, width=f"{width:.1f}", height=18, fill=color
                        )
                    )
                    x += width
            if result.underpowered:
                verdict = "indicative only"
            elif result.winner is not None:
                verdict = f"winner: {self.name(result.winner)}"
            elif len(result.leaders) == len(result.spec.contenders):
                verdict = "tie"
            else:
                verdict = "tie for first: " + " = ".join(self.name(name) for name in result.leaders)
            out.append(self.tag("text", verdict, x=left + bar + 10, y=y + 13, font_weight="bold"))
        out.append("</svg>")
        return "\n".join(out)

    def forest_svg(self, results: Sequence[RoundResult], key: str) -> str:
        """Draw paired median differences with intervals for every family of a round.

        Args:
            results (Sequence[RoundResult]): Analyses of every round and family.
            key (str): Round key to draw.

        Returns:
            str: SVG document.
        """
        selected = [result for result in results if result.spec.key == key]
        panel, row, top = 280, 18, 56
        spec = selected[0].spec
        subject = spec.contenders[-1]
        height = top + max(len(result.solvers) for result in selected) * row + 50
        out = self.svg_open(150 + panel * len(selected), height)
        title = f"{spec.title}: {self.name(subject)} minus {self.name(spec.reference)}"
        out.append(self.tag("text", title, x=10, y=20, font_size=15, font_weight="bold"))
        note = "median difference per solver, 95% bootstrap interval, left of zero favors the first"
        out.append(self.tag("text", note, x=10, y=38, font_size=11, fill="#555"))
        for column, result in enumerate(selected):
            shown = sorted(
                (item for item in result.comparisons if item.subject == subject),
                key=lambda item: item.solver,
            )
            span = max(max(abs(item.low), abs(item.high), abs(item.delta)) for item in shown) or 1.0
            origin = 150 + column * panel + panel / 2 + 20
            scale = (panel / 2 - 20) / span
            bottom = top + row * len(shown)
            out.append(
                self.tag(
                    "text",
                    FAMILY_LABELS[result.family],
                    x=f"{origin:.0f}",
                    y=top - 6,
                    text_anchor="middle",
                    font_weight="bold",
                )
            )
            out.append(
                self.tag(
                    "line", x1=f"{origin:.1f}", y1=top, x2=f"{origin:.1f}", y2=bottom, stroke="#888"
                )
            )
            for index, item in enumerate(shown):
                y = top + index * row + row / 2
                color = GREEN if item.verdict < 0 else RED if item.verdict > 0 else GRAY
                if column == 0:
                    out.append(self.tag("text", item.solver, x=10, y=f"{y + 4:.1f}"))
                out.append(
                    self.tag(
                        "line",
                        x1=f"{origin + item.low * scale:.1f}",
                        y1=f"{y:.1f}",
                        x2=f"{origin + item.high * scale:.1f}",
                        y2=f"{y:.1f}",
                        stroke=color,
                        stroke_width=2,
                    )
                )
                out.append(
                    self.tag(
                        "circle",
                        cx=f"{origin + item.delta * scale:.1f}",
                        cy=f"{y:.1f}",
                        r=4,
                        fill=color,
                    )
                )
            axis = f"Δ {SCORE_LABELS[result.family]} (± {span:.2f})"
            out.append(
                self.tag(
                    "text",
                    axis,
                    x=f"{origin:.0f}",
                    y=bottom + 18,
                    text_anchor="middle",
                    font_size=10,
                    fill="#555",
                )
            )
        out.append("</svg>")
        return "\n".join(out)

    def render(self, readme: Path = README, figures: Path = FIGURES) -> dict[Path, str]:
        """Compute every generated output.

        Args:
            readme (Path): README that holds the generation markers.
            figures (Path): Directory receiving the SVG files.

        Returns:
            dict[Path, str]: Path to new contents for the README and each figure.

        Raises:
            ValueError: If the README lacks the generation markers.
        """
        results = self.results()
        text = readme.read_text(encoding="utf-8")
        if START not in text or END not in text:
            raise ValueError("README has no native-report markers")
        head, rest = text.split(START, 1)
        tail = rest.split(END, 1)[1]
        outputs = {
            readme: head + self.block(results) + tail,
            figures / "scoreboard.svg": self.scoreboard_svg(results) + "\n",
            figures / "engines-2s.svg": self.forest_svg(results, "engines-2s") + "\n",
            figures / "cpp-backend-parity.svg": self.forest_svg(results, "cpp-backend-parity")
            + "\n",
        }
        return outputs


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv (Sequence[str] | None): Optional argument vector.

    Returns:
        int: Zero on success, one when `render --check` finds stale outputs.

    Raises:
        OSError: If inputs cannot be read or outputs written.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Build the compact evidence file from raw study CSVs.")
    ingest.add_argument("--raw", type=Path, required=True)
    ingest.add_argument("--output", type=Path, default=EVIDENCE)
    render = sub.add_parser("render", help="Rewrite the README block and SVG figures.")
    render.add_argument("--evidence", type=Path, default=EVIDENCE)
    render.add_argument("--readme", type=Path, default=README)
    render.add_argument("--figures", type=Path, default=FIGURES)
    render.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.command == "ingest":
        print(f"{NativeReport.ingest(arguments.raw, arguments.output)} rows")
        return 0
    outputs = NativeReport.load(arguments.evidence).render(arguments.readme, arguments.figures)
    stale = [
        path
        for path, text in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != text
    ]
    if arguments.check:
        for path in stale:
            print(f"stale: {path}")
        return 1 if stale else 0
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
