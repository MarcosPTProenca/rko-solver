"""Screen a learned key-distribution adapter against warm-start and marginal controls.

Objective:
- Test whether a neural key generator adds value beyond using evaluated good solutions directly.

Technical approach:
- Warm up one simulated-annealing search on the original problem for 15 percent of the allowance,
  recording a bounded elite-key archive. Replay the identical archive to all adapted arms, charging
  every arm its measured warm-up cost rather than giving it free evaluations.
- Compare plain solving, warm-start-only, independent-coordinate logit sampling and the neural
  denoising generator. All final solves use the same single algorithm and sequential execution,
  avoiding the earlier multiprocessing startup and per-algorithm budget confounders.
- Both distribution adapters retain a 25 percent uniform route and receive the same three warm
  starts through that route. The network trains for at most 40 epochs or 0.05 seconds.
- Account for every phase, skip search when its allowance is exhausted, and preserve the warm-up
  incumbent. The deadline is cooperative: a callback or one training operation can overrun it.

Flow (pseudocode):
1. Generate an instance and collect its warm-up archive once.
2. Run four arms, charging the replayed warm-up time to each arm that uses the archive.
3. Independently re-evaluate the decoded final solution and retain the better warm-up incumbent.
4. Normalize costs per instance and compare paired differences with all three controls.

Important criteria:
- The bottleneck screen uses four instances per group, 300 keys, two seconds and seeds 3500–3503.
- The subsequent coupling screen fixes the same workload and rule with seeds 3504–3507 before
  measuring that architecture. It is a second development screen, not independent confirmation.
- A third development screen uses coupling only to propose 32 checked starting candidates, then
  restores the original decoder for search: `--deployment seeds --first-seed 3508`. All checking
  time is charged and the non-neural marginal control receives the same proposal count.
- A fourth screen follows the user's decision to allow 60 seconds of one-time preparation:
  `--generator coupling --deployment seeds --preparation 60 --budget 2 --epochs 400
  --train-cap 10 --archive-size 128 --first-seed 3512`. Every arm's total allowance is 62 seconds;
  unused preparation returns to search, and the plain arm receives all 62 seconds.
- Freeze the architecture before screening. Reserve seeds 4000 onward for a later confirmation;
  screening cannot establish release readiness.
- Advance only with three neural wins over plain, no significant loss over plain, better mean
  quality than seeded-only on winning groups, and a significant win over the marginal control
  on at least one winning group. Intervals are unadjusted paired 95 percent bootstrap intervals.
- This tests composition with an existing decoder, not automatic decoder construction.

Inputs/Outputs:
- Input: size, allowance, instance count, first seed and output path.
- Output: JSON phase timings, raw costs, paired comparisons and screening decision.
"""

from __future__ import annotations

import argparse
import heapq
import json
import numpy as np
import sys
import time
from benchmarks.engines.coupled_key_distribution import CoupledDistribution
from benchmarks.engines.neural_key_adapter import KeyDistribution
from benchmarks.engines.neural_key_adapter import NeuralKeyAdapter
from benchmarks.engines.priority_decoder_study import PairedStatistics
from benchmarks.engines.surrogate_seed_study import FAMILIES
from benchmarks.engines.surrogate_seed_study import IdentityDecoder
from benchmarks.engines.surrogate_seed_study import Problem
from benchmarks.engines.surrogate_seed_study import ProblemFamily
from benchmarks.engines.surrogate_seed_study import SeedingStudy
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from rko_solver import RKO
from typing import Any

ARMS = ("plain", "seeded", "marginal", "neural")


@dataclass(slots=True)
class AdapterDesign:
    """Fixed numerical design of the screening experiment.

    Responsibilities:
    - Keep timing and archive limits shared across all problem groups.

    Attributes:
        size (int): Key dimension.
        budget (float): Nominal charged end-to-end allowance.
        instances (int): Instances per benchmark group.
        first_seed (int): First instance and solver seed.
        warm_fraction (float): Fraction of the allowance spent collecting keys.
        train_cap (float): Maximum neural training seconds, checked each epoch.
        archive_size (int): Maximum number of retained elite keys.
        generator (str): Bottleneck reconstruction or full-dimensional coupling candidate.
        deployment (str): Decode every evaluation or generate starting keys once.
        preparation_seconds (float | None): One-time preparation allowance. When set, warm-up uses
            it, every arm's total allowance is `preparation_seconds + budget`, and unused
            preparation time returns to search.
        epochs (int): Maximum neural training steps.
    """

    size: int = 300
    budget: float = 2.0
    instances: int = 4
    first_seed: int = 3500
    warm_fraction: float = 0.15
    train_cap: float = 0.05
    archive_size: int = 32
    generator: str = "bottleneck"
    deployment: str = "decoder"
    preparation_seconds: float | None = None
    epochs: int = 40


class ArchiveCost:
    """Evaluate original problem costs while retaining a bounded elite-key archive.

    Responsibilities:
    - Separate original-space keys from decoded problem values.
    - Keep the best evaluated rows in a bounded heap without retaining decoded payloads.

    Attributes:
        problem (Problem): Original problem callbacks and dimension.
        capacity (int): Maximum retained rows.
        heap (list[tuple[float, int, np.ndarray]]): Negative-cost heap with unique tie breakers.
        evaluations (int): Completed observations.
    """

    def __init__(self, problem: Problem, capacity: int) -> None:
        """Initialize the recorder without invoking callbacks.

        Args:
            problem (Problem): Original callbacks and key dimension.
            capacity (int): Maximum retained rows.
        """
        self.problem = problem
        self.capacity = capacity
        self.heap: list[tuple[float, int, np.ndarray]] = []
        self.evaluations = 0

    def __call__(self, keys: np.ndarray) -> float:
        """Evaluate a key vector and retain it only when it qualifies for the archive.

        Args:
            keys (np.ndarray): Original-space candidate keys.

        Returns:
            float: Original problem cost.

        Raises:
            Exception: Original decoder or objective exceptions propagate.
        """
        value = float(self.problem.cost(self.problem.decoder(np.array(keys, copy=True))))
        self.evaluations += 1
        if len(self.heap) < self.capacity or value < -self.heap[0][0]:
            entry = (-value, self.evaluations, np.array(keys, copy=True))
            heapq.heappush(self.heap, entry)
            if len(self.heap) > self.capacity:
                heapq.heappop(self.heap)
        return value

    def rows(self) -> np.ndarray:
        """Return distinct archived rows in ascending objective order.

        Returns:
            np.ndarray: Detached archive with duplicate key vectors removed.
        """
        ordered = sorted(self.heap, key=lambda item: -item[0])
        rows = np.array([item[2] for item in ordered])
        _, first = np.unique(rows, axis=0, return_index=True)
        return rows[np.sort(first)]


class AdapterStudy:
    """Run the same learned key adapter across all benchmark groups.

    Responsibilities:
    - Charge warm-up replay and training to each adapted arm.
    - Preserve incumbents and report raw and normalized quality with phase timings.

    Attributes:
        design (AdapterDesign): Frozen architecture-independent experimental settings.
    """

    def __init__(self, design: AdapterDesign) -> None:
        """Store the study design.

        Args:
            design (AdapterDesign): Shared timing and workload settings.
        """
        self.design = design

    @staticmethod
    def solver(
        problem: Problem,
        decoder: Callable[..., Any],
        seed: int,
        initial_keys: list[list[float]] | None = None,
    ) -> RKO:
        """Build a single-search solver without hidden portfolio budget multiplication.

        Args:
            problem (Problem): Original objective and key dimension.
            decoder (Callable[..., Any]): Original or adapted decoder.
            seed (int): Search seed.
            initial_keys (list[list[float]] | None): Optional original or protected warm starts.

        Returns:
            RKO: Sequential simulated-annealing solver.
        """
        return RKO(
            decoder,
            problem.cost,
            problem.n_variables,
            algorithms=["SA"],
            adaptation="fixed",
            execution="sequential",
            random_state=seed,
            initial_keys=initial_keys,
        )

    def warmup(self, problem: Problem, seed: int) -> tuple[np.ndarray, float, float]:
        """Collect an archive and measure all warm-up work.

        Args:
            problem (Problem): Original problem.
            seed (int): Warm-up seed.

        Returns:
            tuple[np.ndarray, float, float]: Archive, incumbent cost and elapsed seconds.
        """
        started = time.perf_counter()
        recorder = ArchiveCost(problem, self.design.archive_size)
        wrapped = Problem(problem.label, IdentityDecoder(), recorder, problem.n_variables)
        search = self.solver(wrapped, wrapped.decoder, seed)
        allowance = (
            self.design.preparation_seconds
            if self.design.preparation_seconds is not None
            else self.design.budget * self.design.warm_fraction
        )
        result = search.solve(max(allowance - (time.perf_counter() - started), 1e-6))
        return recorder.rows(), float(result.cost), time.perf_counter() - started

    def decoder(
        self, problem: Problem, elite: np.ndarray, arm: str, seed: int
    ) -> Callable[..., Any]:
        """Build the adapter or select the original decoder for an arm.

        Args:
            problem (Problem): Original problem.
            elite (np.ndarray): Warm-up archive, sorted by cost.
            arm (str): Study arm.
            seed (int): Network seed.

        Returns:
            Callable[..., Any]: Original decoder or the fitted composition.
        """
        if arm in ("plain", "seeded") or len(elite) < 4:
            return problem.decoder
        generator = {"bottleneck": KeyDistribution, "coupling": CoupledDistribution}[
            self.design.generator
        ]
        distribution = (
            generator.fit(
                elite,
                seed,
                self.design.epochs,
                time.perf_counter() + self.design.train_cap,
            )
            if arm == "neural"
            else generator.marginal(elite)
        )
        return NeuralKeyAdapter(problem.decoder, distribution)

    @staticmethod
    def seed_keys(
        decoder: Callable[..., Any], elite: np.ndarray, arm: str
    ) -> list[list[float]] | None:
        """Encode the same warm starts under the selected decoder representation.

        Args:
            decoder (Callable[..., Any]): Original or adapted decoder.
            elite (np.ndarray): Cost-ordered original-space archive.
            arm (str): Study arm; plain receives no warm starts.

        Returns:
            list[list[float]] | None: Three encoded warm starts, or None for plain.
        """
        if arm == "plain":
            return None
        if isinstance(decoder, NeuralKeyAdapter):
            return [decoder.protected(row).tolist() for row in elite[:3]]
        return elite[:3].tolist()

    def proposals(
        self, problem: Problem, decoder: Callable[..., Any], elite: np.ndarray, arm: str, seed: int
    ) -> tuple[list[list[float]] | None, int]:
        """Check generated original-space proposals with the real objective before warm-starting.

        Args:
            problem (Problem): Original problem callbacks.
            decoder (Callable[..., Any]): Fitted composition or the original decoder.
            elite (np.ndarray): Cost-ordered warm-up archive.
            arm (str): Study arm.
            seed (int): Common proposal-sampling seed.

        Returns:
            tuple[list[list[float]] | None, int]: Three best checked keys and evaluation count.

        Raises:
            ValueError: If a proposal has a non-finite objective value.
            Exception: Original problem exceptions propagate.
        """
        if not isinstance(decoder, NeuralKeyAdapter):
            return self.seed_keys(decoder, elite, arm), 0
        latent = np.random.default_rng([seed, 904]).random((32, problem.n_variables))
        generated = np.array([decoder.distribution.transform(key) for key in latent])
        pool = np.vstack((elite[:3], generated))
        scores = np.array([problem.cost(problem.decoder(np.array(key, copy=True))) for key in pool])
        if not np.isfinite(scores).all():
            raise ValueError("Generated proposals must have finite costs.")
        return pool[np.argsort(scores)[:3]].tolist(), len(scores)

    def prepare_search(
        self, problem: Problem, decoder: Callable[..., Any], elite: np.ndarray, arm: str, seed: int
    ) -> tuple[RKO, float, int]:
        """Select deployment and charge proposal checking plus solver setup.

        Args:
            problem (Problem): Original problem callbacks.
            decoder (Callable[..., Any]): Trained composition or original decoder.
            elite (np.ndarray): Warm-up archive.
            arm (str): Study arm.
            seed (int): Search and proposal seed.

        Returns:
            tuple[RKO, float, int]: Prepared solver, setup seconds and proposal evaluations.
        """
        started = time.perf_counter()
        evaluations = 0
        runtime_decoder = decoder
        if self.design.deployment == "seeds":
            keys, evaluations = self.proposals(problem, decoder, elite, arm, seed)
            runtime_decoder = problem.decoder
        else:
            keys = self.seed_keys(decoder, elite, arm)
        search = self.solver(problem, runtime_decoder, seed, keys)
        return search, time.perf_counter() - started, evaluations

    def trial(
        self, problem: Problem, warmup: tuple[np.ndarray, float, float], arm: str, seed: int
    ) -> dict[str, Any]:
        """Run an arm after charging its share of the common warm-up.

        Args:
            problem (Problem): Original problem callbacks.
            warmup (tuple[np.ndarray, float, float]): Common archive, incumbent and measured cost.
            arm (str): One of the four study arms.
            seed (int): Final search seed.

        Returns:
            dict[str, Any]: Quality and phase timings with explicit replay charges.
        """
        started = time.perf_counter()
        elite, incumbent, preparation = warmup
        charge = 0.0 if arm == "plain" else preparation
        before_fit = time.perf_counter()
        decoder = self.decoder(problem, elite, arm, seed)
        fit_seconds = time.perf_counter() - before_fit
        search, proposal_seconds, proposal_evaluations = self.prepare_search(
            problem, decoder, elite, arm, seed
        )
        total = self.design.budget + (self.design.preparation_seconds or 0.0)
        remaining = max(total - charge - (time.perf_counter() - started), 0.0)
        cost = incumbent if arm != "plain" else float("inf")
        verified = True
        search_evaluations = 0
        if remaining > 0.0:
            result = search.solve(remaining)
            search_evaluations = result.n_evaluations
            observed = float(problem.cost(search.decoder(np.asarray(result.keys))))
            verified = bool(np.isclose(observed, result.cost, rtol=1e-10, atol=1e-10))
            cost = min(cost, observed)
        elapsed = time.perf_counter() - started
        return {
            "instance": problem.label,
            "family": problem.label.rsplit("-", 2)[0],
            "arm": arm,
            "cost": cost,
            "verified": verified,
            "warmup_seconds": charge,
            "fit_seconds": fit_seconds,
            "proposal_seconds": proposal_seconds,
            "proposal_evaluations": proposal_evaluations,
            "search_evaluations": search_evaluations,
            "search_allowance": remaining,
            "wall_seconds": elapsed,
            "charged_seconds": elapsed + charge,
            "epochs": decoder.distribution.epochs if isinstance(decoder, NeuralKeyAdapter) else 0,
            "neural_active": getattr(getattr(decoder, "distribution", None), "weights", None)
            is not None,
        }

    def run(self) -> list[dict[str, Any]]:
        """Run groups sequentially and stop at the first failed verification.

        Returns:
            list[dict[str, Any]]: Raw observations with normalized gaps.

        Raises:
            RuntimeError: If an independently decoded cost differs from the solver report.
        """
        rows = []
        for family in FAMILIES:
            for seed in range(
                self.design.first_seed, self.design.first_seed + self.design.instances
            ):
                problem = ProblemFamily.make(family, self.design.size, seed)
                prefix = self.warmup(problem, seed)
                for arm in ARMS:
                    row = self.trial(problem, prefix, arm, seed)
                    if not row["verified"]:
                        raise RuntimeError(f"Cost verification failed for {problem.label}, {arm}.")
                    rows.append(row)
                    print(f"trial {len(rows)} {problem.label} {arm}", file=sys.stderr, flush=True)
        return SeedingStudy.normalize(rows)

    @staticmethod
    def comparisons(rows: list[dict[str, Any]], family: str) -> dict[str, Any]:
        """Compare neural quality with all controls on one group.

        Args:
            rows (list[dict[str, Any]]): Matched normalized observations.
            family (str): Group to select.

        Returns:
            dict[str, Any]: Paired comparisons and per-group screening flags.
        """
        group = [row for row in rows if row["family"] == family]
        gaps = {arm: [r["gap"] for r in group if r["arm"] == arm] for arm in ARMS}
        table = {
            control: PairedStatistics.compare(gaps["neural"], gaps[control], 10000, 0)
            for control in ARMS[:-1]
        }
        return table

    @staticmethod
    def flags(table: dict[str, Any]) -> dict[str, bool]:
        """Classify one group without changing the predeclared screening thresholds.

        Args:
            table (dict[str, Any]): Comparisons with plain, seeded and marginal controls.

        Returns:
            dict[str, bool]: Win, loss and neural-added-value flags.
        """
        win = table["plain"]["ci_high"] < 0 and table["seeded"]["difference"] < 0
        return {
            "win": win,
            "loss": table["plain"]["ci_low"] > 0,
            "added": win and table["marginal"]["ci_high"] < 0,
        }

    @staticmethod
    def selected(flags: dict[str, dict[str, bool]], kind: str) -> list[str]:
        """List groups carrying one decision flag.

        Args:
            flags (dict[str, dict[str, bool]]): Decision flags per group.
            kind (str): Flag to select.

        Returns:
            list[str]: Matching group names.
        """
        return [family for family, entry in flags.items() if entry[kind]]

    @classmethod
    def summarize(cls, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply the declared screening rule using paired instance comparisons.

        Args:
            rows (list[dict[str, Any]]): Normalized observations in matched instance order.

        Returns:
            dict[str, Any]: Paired comparisons and the decision to advance to fresh confirmation.
        """
        tables = {family: cls.comparisons(rows, family) for family in FAMILIES}
        flags = {family: cls.flags(table) for family, table in tables.items()}
        wins, losses, added = (cls.selected(flags, kind) for kind in ("win", "loss", "added"))
        return {
            "comparisons": tables,
            "wins": wins,
            "losses": losses,
            "neural_over_marginal": added,
            "advance": len(wins) >= 3 and not losses and bool(added),
        }


def main() -> int:
    """Parse workload settings, run the screen and write a reproducible result file.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--budget", type=float, default=2.0)
    parser.add_argument("--instances", type=int, default=4)
    parser.add_argument("--first-seed", type=int, default=3500)
    parser.add_argument("--generator", choices=("bottleneck", "coupling"), default="bottleneck")
    parser.add_argument("--deployment", choices=("decoder", "seeds"), default="decoder")
    parser.add_argument("--preparation", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-cap", type=float, default=0.05)
    parser.add_argument("--archive-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("results/neural-adapter-screen.json"))
    options = parser.parse_args()
    design = AdapterDesign(
        options.size,
        options.budget,
        options.instances,
        options.first_seed,
        generator=options.generator,
        deployment=options.deployment,
        preparation_seconds=options.preparation,
        epochs=options.epochs,
        train_cap=options.train_cap,
        archive_size=options.archive_size,
    )
    study = AdapterStudy(design)
    rows = study.run()
    report = {"design": asdict(design), "summary": study.summarize(rows), "trials": rows}
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
