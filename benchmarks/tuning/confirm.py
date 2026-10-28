"""Re-score tuned configurations at a longer evaluation budget than they were tuned for.

Objective:
- Check whether a conclusion about the packaged default, reached at the tuning budget, survives a
  budget ten times larger, by scoring the same configurations on held-out instances.

Technical approach:
- The configurations chosen by one tuner in the bake-off, and the packaged default, are scored on
  the first held-out instances of each cell with a larger evaluation limit and fresh seeds. No
  tuner runs again, so the question is purely whether a configuration tuned for a short budget is
  still better than the default when the budget grows.
- One task scores one configuration on every selected instance; tasks run in a process pool and
  are appended to a JSON-lines file as they finish, so an interrupted check resumes.
- Summaries reuse the bootstrap of `Analysis`: the mean gap of the tuned configurations is
  compared with the default's, resampling both configurations and instances.

Flow (pseudocode):
1. Read the bake-off results and pick each cell's default and tuned configurations.
2. Score every configuration on the held-out instances at the larger budget.
3. Summarize the difference to the default per cell.

Important criteria:
- Held-out instances only; seeds differ from the bake-off's, and nothing is tuned here.
- Gaps use the same per-instance calibration as the bake-off, so they are comparable to it; a
  negative gap beats the calibration reference.

Inputs/Outputs:
- Input: the bake-off results file, a tuner name, an evaluation limit and instance and seed counts.
- Output: a JSON-lines file of scores and a Markdown summary.
"""

from __future__ import annotations

import functools
import json
import multiprocessing
import numpy as np
from benchmarks.tuning.analysis import Analysis
from benchmarks.tuning.instances import InstanceBank
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

CONFIRM_SEED = 31000


@functools.lru_cache(maxsize=1)
def _bank(directory: str, evaluations: int) -> InstanceBank:
    """Return the per-process bank for a directory and evaluation limit.

    Args:
        directory (str): Folder with the instances and calibration of the bake-off.
        evaluations (int): Evaluation limit of every scored run.

    Returns:
        InstanceBank: Bank that keeps one family's compiled objectives.
    """
    return InstanceBank(Path(directory), evaluations)


def score_configuration(
    payload: tuple[str, int, str, str, dict[str, float], int, int, str],
) -> dict:
    """Score one configuration on the first held-out instances of a family.

    Args:
        payload (tuple): Directory, evaluation limit, algorithm, family, configuration, number
            of instances, seeds per instance and a label naming the configuration.

    Returns:
        dict: Record with the label and the mean gap per instance.
    """
    directory, evaluations, algorithm, family, config, count, seeds, label = payload
    bank = _bank(directory, evaluations)
    instances = bank.generate(family, "test", count)
    scores = [
        float(
            np.mean(
                [bank.score(item, algorithm, config, CONFIRM_SEED + seed) for seed in range(seeds)]
            )
        )
        for item in instances
    ]
    return {
        "kind": "confirm",
        "algorithm": algorithm,
        "family": family,
        "label": label,
        "evaluations": evaluations,
        "config": config,
        "scores": scores,
    }


class Confirmation:
    """Run and summarize the longer-budget check.

    Responsibilities:
    - Build the list of configurations to score from the bake-off results.
    - Run the missing ones in a process pool and render the summary.

    Attributes:
        results (Path): Bake-off results file.
        output (Path): JSON-lines file of this check.
        directory (Path): Instance and calibration folder.
        tuner (str): Tuner whose configurations are re-scored.
    """

    def __init__(self, results: Path, output: Path, directory: Path, tuner: str) -> None:
        """Create the check.

        Args:
            results (Path): Bake-off results file.
            output (Path): JSON-lines file to append to.
            directory (Path): Instance and calibration folder of the bake-off.
            tuner (str): Tuner whose chosen configurations are re-scored.
        """
        self.results = results
        self.output = output
        self.directory = directory
        self.tuner = tuner

    def tasks(
        self, evaluations: int, count: int, seeds: int
    ) -> list[tuple[str, int, str, str, dict[str, float], int, int, str]]:
        """List the configurations to score, default first in each cell.

        Args:
            evaluations (int): Evaluation limit of every run.
            count (int): Held-out instances per cell.
            seeds (int): Seeds per instance.

        Returns:
            list[tuple]: Payloads for `score_configuration`.
        """
        lines = [json.loads(x) for x in self.results.read_text(encoding="utf-8").splitlines()]
        metadata = next(x for x in lines if x.get("kind") == "meta")
        directory = InstanceBank(
            self.directory, seed_offset=metadata["settings"].get("seed_offset", 0)
        ).directory
        sessions = [x for x in lines if x.get("kind") == "session"]
        wanted = [x for x in sessions if x["tuner"] in ("default", self.tuner)]
        wanted.sort(key=lambda x: (x["family"], x["algorithm"], x["tuner"] != "default", x["rep"]))
        return [
            (
                str(directory),
                evaluations,
                x["algorithm"],
                x["family"],
                x["config"],
                count,
                seeds,
                "default" if x["tuner"] == "default" else f"{x['tuner']}-{x['rep']}",
            )
            for x in wanted
        ]

    def run(self, evaluations: int, count: int, seeds: int, workers: int) -> int:
        """Score every configuration that is not yet in the output file.

        Args:
            evaluations (int): Evaluation limit of every run.
            count (int): Held-out instances per cell.
            seeds (int): Seeds per instance.
            workers (int): Worker processes.

        Returns:
            int: Number of configurations scored in this call.

        Raises:
            Exception: The first failure of any task, after cancelling the queued ones.
        """
        known = self._known(evaluations)
        todo = [t for t in self.tasks(evaluations, count, seeds) if (t[2], t[3], t[7]) not in known]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            futures = [pool.submit(score_configuration, item) for item in todo]
            for future in as_completed(futures):
                try:
                    record = future.result()
                except BaseException:
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                with self.output.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
        return len(todo)

    def _known(self, evaluations: int) -> set[tuple[str, str, str]]:
        """Return the configurations already scored at an evaluation limit.

        Args:
            evaluations (int): Evaluation limit of interest.

        Returns:
            set[tuple[str, str, str]]: `(algorithm, family, label)` keys.
        """
        if not self.output.is_file():
            return set()
        lines = [json.loads(x) for x in self.output.read_text(encoding="utf-8").splitlines()]
        return {
            (x["algorithm"], x["family"], x["label"])
            for x in lines
            if x.get("evaluations") == evaluations
        }

    def report(self, evaluations: int, resamples: int = 4000) -> str:
        """Summarize the difference to the default per cell at one evaluation limit.

        Args:
            evaluations (int): Evaluation limit to summarize.
            resamples (int): Bootstrap resamples.

        Returns:
            str: Markdown table.
        """
        lines = [json.loads(x) for x in self.output.read_text(encoding="utf-8").splitlines()]
        records = [x for x in lines if x.get("evaluations") == evaluations]
        analysis = Analysis(self.results, resamples=resamples)
        rows = [
            f"| Cell | Default gap | {self.tuner} gap (95% CI) | Difference (95% CI) | Verdict |",
            "| --- | ---: | --- | --- | --- |",
        ]
        for algorithm, family in sorted({(x["algorithm"], x["family"]) for x in records}):
            cell = [x for x in records if (x["algorithm"], x["family"]) == (algorithm, family)]
            rows.append(self._row(analysis, f"{algorithm} on {family}", cell))
        return "\n".join(rows) + "\n"

    @staticmethod
    def _row(analysis: Analysis, name: str, cell: list[dict]) -> str:
        """Render the table row of one cell.

        Args:
            analysis (Analysis): Bootstrap helper.
            name (str): Cell name.
            cell (list[dict]): Records of the cell, including the default's.

        Returns:
            str: Markdown row.
        """
        default = np.asarray(next(x["scores"] for x in cell if x["label"] == "default"))
        tuned = np.asarray([x["scores"] for x in cell if x["label"] != "default"])
        mean, low, high = analysis.interval(tuned)
        diff, dlow, dhigh = analysis.interval(tuned, default)
        return (
            f"| {name} | {default.mean():.3f} | {mean:.3f} ({low:.3f} to {high:.3f}) "
            f"| {diff:+.3f} ({dlow:+.3f} to {dhigh:+.3f}) | {analysis.verdict(tuned, default)} |"
        )
