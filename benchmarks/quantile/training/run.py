"""Resource-capped training and disjoint-panel qualification for contextual quantile models.

Objective:
- Separate training feasibility and construction quality from any claim about solver search.

Technical approach:
- Freeze source/data identities, compare fixed cheap controls and qualify a selected checkpoint.
- Charge teacher generation, model updates, validation and export to the processor allowance.

Flow (pseudocode):
1. Freeze new output, source fingerprints, settings and disjoint validation panels.
2. Train within an explicit processor budget and retain checked checkpoints.
3. Qualify exported probabilities and compare held-out construction costs.

Important criteria:
- No test-instance tuning, output overwrite, hidden budget extension or neural runtime dependency.
- This Linux processor-only pilot does not establish general search or deployment efficacy.

Inputs/Outputs:
- Input: a supported family, size, seed and bounded training settings.
- Output: data-only weights, progress records, phase accounting and a development verdict.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
import platform
import resource
import subprocess
import time
import torch
from benchmarks.quantile.bootstrap import paired_interval
from benchmarks.quantile.codec import TrajectoryCodec
from benchmarks.quantile.instances import Instances
from benchmarks.quantile.model import FrozenScorer
from benchmarks.quantile.model import LinearRule
from benchmarks.quantile.probabilities import Intervals
from benchmarks.quantile.tasks import Cover
from benchmarks.quantile.tasks import Tour
from benchmarks.quantile.training.policy import TrainingPolicy
from benchmarks.quantile.training.types import Evaluation
from benchmarks.quantile.training.types import TrainingBudgetExpired
from benchmarks.quantile.training.types import TrainingSettings
from dataclasses import asdict
from pathlib import Path
from typing import Any


class TrainingRun:
    """Run one bounded, checkpoint-selected constructive-policy training experiment.

    Responsibilities:
    - Freeze declared source/data identities before measuring model quality.
    - Preserve all progress/checkpoint observations and qualify on a disjoint panel once.

    Attributes:
        settings (TrainingSettings): Immutable first-pilot configuration.
        output (Path): New output directory; existing runs are never overwritten.
        network (TrainingPolicy): Owned differentiable training policy.
        completed_updates (int): Completed policy-gradient updates.
        completed_teacher_steps (int): Completed training-only teacher updates.
    """

    def __init__(self, settings: TrainingSettings, output: Path) -> None:
        """Initialize owned training state without touching an output directory.

        Args:
            settings (TrainingSettings): Bounded frozen training settings.
            output (Path): New run-output directory.
        """
        self.settings, self.output = settings, output
        self.network = TrainingPolicy(settings.family, seed=settings.seed)
        self.optimizer = torch.optim.Adam(self.network.parameters.values(), lr=0.01)
        self.rng = np.random.default_rng([settings.seed, 991])
        self.best = self.network.frozen()
        self.best_score = float("inf")
        self.completed_updates = self.completed_teacher_steps = 0
        self.panels: dict[str, list[tuple[int, Tour | Cover, float]]] = {}
        self.history: list[dict[str, Any]] = []
        self.training_hashes: set[str] = set()
        self.training_capped = False
        self.started = time.perf_counter()

    def check_budget(self) -> None:
        """Check actual process consumption, including framework import and validation.

        Raises:
            TrainingBudgetExpired: If total processor time reaches the explicit allowance.
            MemoryError: If peak resident process memory exceeds fifteen hundred MiB.
        """
        if time.process_time() >= self.settings.cpu_seconds:
            raise TrainingBudgetExpired("The total training processor budget is exhausted.")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss > 1536 * 1024:
            raise MemoryError("The measured training resident-memory allowance is exhausted.")

    def event(self, phase: str, **values: object) -> None:
        """Persist and flush bounded progress records for active supervision.

        Args:
            phase (str): Declared training or validation phase.
            **values (object): Small JSON-compatible progress fields for this observation.

        Raises:
            OSError: If progress cannot be persisted.
        """
        record = {"phase": phase, "cpu_seconds": time.process_time(), **values}
        with (self.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)

    def write(self, name: str, record: dict[str, Any]) -> None:
        """Write a bounded explicit run record without nonfinite numeric placeholders.

        Args:
            name (str): Owned output filename.
            record (dict[str, Any]): JSON-compatible run evidence.

        Raises:
            OSError: If the record cannot be persisted.
            ValueError: If the record contains a nonfinite numerical value.
        """
        (self.output / name).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def prepare(self) -> None:
        """Freeze source identity and two disjoint instance panels before any model selection.

        Raises:
            TrainingBudgetExpired: If preparation exhausts the process allowance.
            OSError: If sources or output records cannot be read or persisted.
        """
        self.check_budget()
        root = Path(__file__).resolve().parents[3]
        base = root / "benchmarks/quantile"
        sources = sorted(base.glob("*.py")) + sorted((base / "training").glob("*.py"))
        sources += [base / "training/pyproject.toml", base / "training/uv.lock"]
        hashes = {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
            if not path.name.startswith("test_")
        }
        declared: dict[str, list[dict[str, Any]]] = {}
        for panel, start in (("tune", 600000), ("qualification", 610000)):
            rows = []
            declared[panel] = []
            for seed in range(start, start + 8):
                self.check_budget()
                task = Instances.make(self.settings.family, self.settings.size, seed)
                reference = TrajectoryCodec(task).greedy().cost
                rows.append((seed, task, reference))
                declared[panel].append(
                    {
                        "seed": seed,
                        "instance_hash": Instances.fingerprint(task),
                        "reference": reference,
                    }
                )
            self.panels[panel] = rows
        self.write(
            "manifest.json",
            {
                "settings": asdict(self.settings),
                "source_revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "source_hashes": hashes,
                "generator": Instances.VERSION,
                "panels": declared,
                "teacher": "analytic greedy, training instances only, at most 100 updates",
                "training_seeds": {"teacher_start": 100000, "gradient_start": 200000},
                "selection": "tune panel only; qualification panel used once after selection",
                "development_gate": (
                    "held-out mean cost below analytic and selected linear controls"
                ),
                "claim": "construction pilot, not confirmation or guided-search evidence",
                "versions": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                },
            },
        )
        self.event("prepared", instances_per_panel=8, samples_per_instance=4)

    def evaluate(self, scorer: FrozenScorer | LinearRule | None, panel: str) -> list[Evaluation]:
        """Evaluate common independent keys and aggregate within each fixed instance.

        Args:
            scorer (FrozenScorer | LinearRule | None): Frozen model or cheap control.
            panel (str): Explicit tuning or held-out qualification panel.

        Returns:
            list[Evaluation]: Checked instance-level observations, not independent rollout rows.

        Raises:
            TrainingBudgetExpired: If model evaluation exhausts the shared allowance.
            ValueError: If construction or its inverse fails an independent check.
        """
        rows = []
        for seed, task, reference in self.panels[panel]:
            mapping = TrajectoryCodec(task, scorer=scorer)
            rng = np.random.default_rng([self.settings.seed, seed, 319])
            candidates = []
            for _ in range(4):
                self.check_budget()
                candidate = mapping.decode(rng.random(task.horizon))
                if panel == "qualification":
                    if mapping.decode(mapping.encode(candidate.trace)) != candidate:
                        raise ValueError("A held-out construction failed exact trajectory replay.")
                candidates.append(candidate)
            mean_cost = float(np.mean([candidate.cost for candidate in candidates]))
            rows.append(
                Evaluation(
                    seed,
                    Instances.fingerprint(task),
                    reference,
                    mean_cost,
                    mean_cost / max(reference, 1e-12),
                    len({candidate.solution for candidate in candidates}),
                )
            )
        return rows

    def controls(self) -> LinearRule:
        """Choose one cheap same-feature linear comparator using only the tuning panel.

        Returns:
            LinearRule: Best validation-selected rule from ten fixed candidates.

        Raises:
            TrainingBudgetExpired: If validation exhausts the shared processor allowance.
        """
        coefficients = [(0.0, 0.0, 0.0, 0.0)]
        for scale in (5.0, 10.0, 20.0):
            if self.settings.family == "tour":
                coefficients += [
                    (-scale, 0.0, 0.0, 0.0),
                    (-scale, -scale / 4.0, 0.0, 0.0),
                    (-scale, scale / 4.0, -scale / 4.0, 0.0),
                ]
            else:
                coefficients += [
                    (0.0, 0.0, scale, 0.0),
                    (scale, -scale, 0.0, 0.0),
                    (scale, -scale / 2.0, scale, 0.0),
                ]
        candidates = [LinearRule(self.settings.family, values) for values in coefficients]
        scores = []
        for rule in candidates:
            rows = self.evaluate(rule, "tune")
            score = float(np.mean([row.normalized_cost for row in rows]))
            scores.append(score)
            self.event("linear_control", coefficients=rule.coefficients, normalized_cost=score)
        selected = candidates[int(np.argmin(scores))]
        self.write(
            "linear-selection.json",
            {"scores": scores, "coefficients": coefficients, "selected": selected.coefficients},
        )
        return selected

    def consider(self, label: str) -> None:
        """Retain checkpoint evidence and update the best model using tuning data only.

        Args:
            label (str): Unique checkpoint label.

        Raises:
            TrainingBudgetExpired: If selection exhausts the total allowance.
            OSError: If the checkpoint cannot be saved.
        """
        scorer = self.network.frozen()
        rows = self.evaluate(scorer, "tune")
        score = float(np.mean([row.normalized_cost for row in rows]))
        diversity = float(np.mean([row.unique_solutions for row in rows]))
        scorer.save(self.output / f"checkpoint-{label}.json")
        self.history.append(
            {
                "checkpoint": label,
                "fingerprint": scorer.fingerprint,
                "normalized_cost": score,
                "mean_unique_solutions": diversity,
                "rows": [asdict(row) for row in rows],
            }
        )
        if score < self.best_score:
            self.best, self.best_score = scorer, score
            self.best.save(self.output / "best-model.json")
        self.event(
            "checkpoint",
            checkpoint=label,
            normalized_cost=score,
            mean_unique_solutions=diversity,
        )

    def training_instance(self, seed: int) -> Tour | Cover:
        """Create and identify a training-only task outside both held-out panels.

        Args:
            seed (int): Declared training generator initialization.

        Returns:
            Tour | Cover: Validated training task.
        """
        task = Instances.make(self.settings.family, self.settings.size, seed)
        self.training_hashes.add(Instances.fingerprint(task))
        return task

    def train(self) -> None:
        """Apply bounded teacher and objective-driven updates with an export/validation reserve.

        Raises:
            TrainingBudgetExpired: If a training operation exhausts the shared allowance.
            RuntimeError: If optimization produces nonfinite gradients.
        """
        reserve = min(60.0, self.settings.cpu_seconds / 5.0)
        if not self._warm_start(reserve):
            return
        self.consider("teacher")
        self._gradient_descent(reserve)

    def _reserve_exhausted(self, reserve: float) -> bool:
        """Tell whether starting another training operation would enter the export reserve.

        Args:
            reserve (float): Processor seconds held back for export and validation.

        Returns:
            bool: True once the remaining budget has shrunk to the reserve.
        """
        if time.process_time() >= self.settings.cpu_seconds - reserve:
            self.training_capped = True
            return True
        return False

    def _warm_start(self, reserve: float) -> bool:
        """Apply bounded training-only teacher updates before any policy-gradient step.

        Args:
            reserve (float): Processor seconds held back for export and validation.

        Returns:
            bool: False if the reserve was entered before this phase completed.

        Raises:
            TrainingBudgetExpired: If a teacher update exhausts the shared allowance.
        """
        for step in range(min(100, self.settings.steps)):
            self.check_budget()
            if self._reserve_exhausted(reserve):
                return False
            task = self.training_instance(100000 + step)
            loss = self.network.imitate(task, self.rng, self.optimizer)
            self.completed_teacher_steps += 1
            if (step + 1) % 25 == 0:
                self.event("teacher", completed=self.completed_teacher_steps, loss=loss)
        return True

    def _gradient_descent(self, reserve: float) -> None:
        """Apply bounded objective-driven policy-gradient updates with periodic checkpoints.

        Args:
            reserve (float): Processor seconds held back for export and validation.

        Raises:
            TrainingBudgetExpired: If a gradient update exhausts the shared allowance.
            RuntimeError: If optimization produces nonfinite gradients.
        """
        interval = max(1, self.settings.steps // 8)
        for step in range(self.settings.steps):
            self.check_budget()
            if self._reserve_exhausted(reserve):
                return
            tasks = [self.training_instance(200000 + 2 * step + offset) for offset in range(2)]
            metrics = self.network.update(tasks, self.rng, self.optimizer)
            self.completed_updates += 1
            if (step + 1) % 25 == 0:
                self.event("gradient", completed=self.completed_updates, **metrics)
            if (step + 1) % interval == 0 or step + 1 == self.settings.steps:
                self.consider(f"step-{step + 1:04d}")

    def export_parity(self) -> dict[str, Any]:
        """Check the selected saved weights against the actual training probability implementation.

        Returns:
            dict[str, Any]: Qualified state count, maximum probability error and action mismatches.

        Raises:
            ValueError: If frozen/training probabilities or fixed-key decisions disagree.
            TrainingBudgetExpired: If export qualification exceeds the processor allowance.
        """
        restored = FrozenScorer.load(self.output / "best-model.json")
        replay = TrainingPolicy(self.settings.family, hidden=restored.w1.shape[1], seed=0)
        with torch.no_grad():
            for name in FrozenScorer.NAMES:
                replay.parameters[name].copy_(
                    torch.tensor(np.array(getattr(restored, name)), dtype=torch.float64)
                )
        checked, mismatches, error = 0, 0, 0.0
        rng = np.random.default_rng([self.settings.seed, 881])
        for _, task, _ in self.panels["qualification"]:
            trace = TrajectoryCodec(task).greedy().trace
            for position in range(0, len(trace), max(1, len(trace) // 8)):
                self.check_budget()
                prefix = trace[:position]
                actions = task.legal(prefix)
                features = task.features(prefix, actions)
                actual = replay.probabilities(features).detach().numpy()
                expected = Intervals.from_logits(restored.scores(features))
                error = max(error, float(np.max(np.abs(actual - expected.probabilities))))
                if not np.allclose(actual, expected.probabilities, rtol=1e-12, atol=1e-14):
                    raise ValueError("Training and exported probability implementations disagree.")
                key = float(rng.random())
                cumulative = np.cumsum(actual)
                cumulative[-1] = 1.0
                mismatches += expected.select(key) != int(
                    np.searchsorted(cumulative, key, side="right")
                )
                checked += 1
        if mismatches:
            raise ValueError("Training and exported fixed-key decisions disagree.")
        return {
            "checked_states": checked,
            "max_probability_error": error,
            "decision_mismatches": mismatches,
        }

    @staticmethod
    def compare(candidate: list[Evaluation], control: list[Evaluation]) -> dict[str, Any]:
        """Summarize paired held-out normalized costs without treating rollouts as instances.

        Args:
            candidate (list[Evaluation]): One aggregated observation per candidate instance.
            control (list[Evaluation]): Matching aggregated cheap-control observations.

        Returns:
            dict[str, Any]: Mean difference and exploratory paired bootstrap interval.

        Raises:
            ValueError: If identity sets or row multiplicities do not match.
        """
        indexed = {row.instance_hash: row for row in control}
        if (
            len(candidate) != len(control)
            or len(indexed) != len(control)
            or {row.instance_hash for row in candidate} != set(indexed)
        ):
            raise ValueError("Held-out comparisons require distinct matching instance identities.")
        differences = np.asarray(
            [row.normalized_cost - indexed[row.instance_hash].normalized_cost for row in candidate]
        )
        return {
            **paired_interval(differences),
            "interpretation": "exploratory validation interval, not confirmation",
        }

    def run(self) -> dict[str, Any]:
        """Execute a new bounded run and preserve explicit complete or exhausted evidence.

        Returns:
            dict[str, Any]: Run accounting, checkpoint identity and held-out development verdict.

        Raises:
            FileExistsError: If the output directory already exists.
            ValueError: If an unexpected construction or numerical validation fails.
            OSError: If input or output records cannot be read or persisted.
        """
        self.output.mkdir(parents=True, exist_ok=False)
        self.best.save(self.output / "best-model.json")
        status, qualification, comparisons, parity = self._execute()
        advance = status == "complete" and all(
            row["difference"] < 0.0 for row in comparisons.values()
        )
        report = self._report(status, advance, qualification, comparisons, parity)
        self.write("summary.json", report)
        self.event(
            "finished", status=status, advance=advance, completed_updates=self.completed_updates
        )
        return report

    def _execute(
        self,
    ) -> tuple[str, dict[str, list[Evaluation]], dict[str, dict[str, Any]], dict[str, Any]]:
        """Run the bounded training and qualification lifecycle, catching budget exhaustion.

        Returns:
            tuple[str, dict[str, list[Evaluation]], dict[str, dict[str, Any]], dict[str, Any]]:
                Status, qualification rows per arm, paired comparisons, and export parity.

        Raises:
            ValueError: If an unexpected construction or numerical validation fails.
            OSError: If input or output records cannot be read or persisted.
        """
        qualification: dict[str, list[Evaluation]] = {}
        comparisons: dict[str, dict[str, Any]] = {}
        parity: dict[str, Any] = {}
        try:
            self.prepare()
            linear = self.controls()
            self.consider("initial")
            self.train()
            parity = self.export_parity()
            qualification = {
                name: self.evaluate(scorer, "qualification")
                for name, scorer in (
                    ("analytic", None),
                    ("linear", linear),
                    ("neural", self.best),
                )
            }
            comparisons = {
                name: self.compare(qualification["neural"], qualification[name])
                for name in ("analytic", "linear")
            }
            status = "training_capped" if self.training_capped else "complete"
        except TrainingBudgetExpired:
            status = "budget_exhausted"
        except MemoryError:
            status = "memory_exhausted"
        return status, qualification, comparisons, parity

    def _report(
        self,
        status: str,
        advance: bool,
        qualification: dict[str, list[Evaluation]],
        comparisons: dict[str, dict[str, Any]],
        parity: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble the complete run report from already-computed lifecycle results.

        Args:
            status (str): Final lifecycle status.
            advance (bool): Whether the held-out development gate passed.
            qualification (dict[str, list[Evaluation]]): Qualification rows per arm.
            comparisons (dict[str, dict[str, Any]]): Paired held-out comparisons.
            parity (dict[str, Any]): Export-parity check result.

        Returns:
            dict[str, Any]: Complete run accounting and held-out development verdict.
        """
        return {
            "status": status,
            "advance": advance,
            "rule": (
                "development only: held-out mean normalized construction cost "
                "below both cheap controls"
            ),
            "settings": asdict(self.settings),
            "completed_teacher_steps": self.completed_teacher_steps,
            "completed_updates": self.completed_updates,
            "processor_seconds": time.process_time(),
            "wall_seconds_after_initialization": time.perf_counter() - self.started,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "training_instance_hashes": sorted(self.training_hashes),
            "model_fingerprint": self.best.fingerprint,
            "checkpoints": self.history,
            "qualification": {
                name: [asdict(row) for row in rows] for name, rows in qualification.items()
            },
            "comparisons": comparisons,
            "export_parity": parity,
            "claim_boundary": "No guided-search, runtime-superiority, or release-readiness claim.",
        }
