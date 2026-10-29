"""Automatic engine selection by short simulation on the user's own problem.

Objective:
- Choose the engine that finds the best solutions for one specific problem,
  instead of relying on a fixed rule that may not hold for that objective function.

Technical approach:
- Enumerate the candidates that can run the problem (Python processes, Rust, Rust with all
  threads, C++ reference) and run each for a short budget, twice, with the same two seeds.
- Score each candidate by the mean cost of its probes and estimate its noise from the
  spread of the two probes; the cheapest candidate replaces the evidence-based preferred one
  only if it beats it by more than that noise and by more than 5 percent of its cost, because
  a short probe can favor an engine that loses over the full budget (it did on a pure-Python
  objective, where processes led a 0.1 s probe by 2 percent and lost the 10 s run).
- The best solution of any probe is kept, so the simulation time is not wasted.

Flow (pseudocode):
1. Build the candidate list for the problem kind and the installed extension.
2. Run `REPEATS` probes per candidate with shared seeds and a per-probe budget.
3. Drop candidates whose probes fail, pick the winner with `decide` and report everything.

Important criteria:
- The selection budget is separate from the search `time_limit` and is reported.
- The preferred order encodes the calibration evidence: the Rust engine first (its
  per-candidate cost is far below the Python engine's even when callbacks hold the
  interpreter lock), then all Rust threads, processes and the C++ reference.
- A failing candidate (for example unpicklable callbacks for processes) is excluded and
  its error is kept; if every candidate fails the first error is raised.

Inputs/Outputs:
- Input: a configured solver object, the search time limit and optional selection budget.
- Output: an `AutoSelection` record plus the best probe solution.
"""

import math
import numpy as np
import os
import statistics
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class ProbeResult(Protocol):
    """Describe the part of a solve result the selection reads.

    Attributes:
        cost (float): Best cost found.
        keys (tuple[float, ...]): Best random-key vector.
    """

    @property
    def cost(self) -> float:
        """Return the best cost found."""
        ...

    @property
    def keys(self) -> tuple[float, ...]:
        """Return the best random-key vector."""
        ...


class Probeable(Protocol):
    """Describe a solver copy that can run one probe."""

    def solve(self, time_limit: float) -> ProbeResult:
        """Run the probe and return its best solution.

        Args:
            time_limit (float): Search seconds of the probe.

        Returns:
            ProbeResult: Best cost and keys.
        """
        ...


class SelectableSolver(Protocol):
    """Describe the solver surface the selection needs.

    Attributes:
        random_state (int | None): Seed the probe seeds derive from.
    """

    random_state: int | None

    def _kind(self) -> str:
        """Return `callbacks`, `batch` or `native`."""
        ...

    def _variant(self, candidate: AutoCandidate, seed: int | None, probe: bool = True) -> Probeable:
        """Return a copy running the candidate engine, quiet when `probe` is true."""
        ...


@dataclass(frozen=True)
class AutoCandidate:
    """Describe one engine configuration the selection may choose.

    Attributes:
        name (str): Public candidate name.
        engine (str): Engine passed to the solver.
        execution (str | None): Python-engine execution passed to the solver, or None.
        n_jobs (int | None): Worker limit passed to the solver.
    """

    name: str
    engine: str
    execution: str | None = None
    n_jobs: int | None = None


@dataclass(frozen=True)
class AutoProbe:
    """Record one short probe run.

    Attributes:
        candidate (str): Candidate name.
        seed (int): Seed used for the probe.
        cost (float | None): Best cost found, or None when the probe failed.
        seconds (float): Wall seconds the probe took, including startup.
        error (str | None): Failure message, or None.
    """

    candidate: str
    seed: int
    cost: float | None
    seconds: float
    error: str | None = None


@dataclass(frozen=True)
class AutoSelection:
    """Describe how the automatic engine was chosen.

    Attributes:
        chosen (str): Name of the selected candidate.
        candidates (tuple[str, ...]): Candidates that were tried, in preference order.
        probes (tuple[AutoProbe, ...]): Every probe, including failures.
        scores (tuple[tuple[str, float], ...]): Mean probe cost of each usable candidate.
        margin (float): Noise margin used when comparing with the preferred candidate.
        reason (str): Short explanation of the decision.
        probe_seconds (float): Requested search seconds of one probe.
        selection_seconds (float): Wall seconds spent on the whole selection.
        improved_by_probe (bool): Whether the returned solution came from a probe.
    """

    chosen: str
    candidates: tuple[str, ...]
    probes: tuple[AutoProbe, ...]
    scores: tuple[tuple[str, float], ...]
    margin: float
    reason: str
    probe_seconds: float
    selection_seconds: float
    improved_by_probe: bool = False


class AutoEngine:
    """Enumerate candidates, run the probes and decide.

    Responsibilities:
    - Build the candidate list for a problem kind and the installed extension.
    - Probe each candidate under identical conditions and tolerate failures.
    - Decide with a noise- and margin-aware rule that keeps the evidence-based preference.

    Attributes:
        NAMES (tuple[str, ...]): Accepted candidate names.
        REPEATS (int): Probes per candidate.
        FRACTION (float): Default selection budget as a fraction of the time limit.
        MINIMUM_PROBE (float): Smallest per-probe budget in seconds.
        SWITCH_MARGIN (float): Relative cost advantage, on top of the probe noise, a candidate
            needs before the selection leaves the preferred candidate.
    """

    NAMES = ("process", "rust", "rust-threads", "cpp")
    REPEATS = 2
    FRACTION = 0.1
    MINIMUM_PROBE = 0.05
    SWITCH_MARGIN = 0.05

    @staticmethod
    def build(name: str, cpus: int) -> AutoCandidate:
        """Create the candidate object for a public name.

        Args:
            name (str): One of `NAMES`.
            cpus (int): Usable logical CPUs, used for the all-threads candidate.

        Returns:
            AutoCandidate: Engine, execution and worker limit to run.
        """
        return {
            "process": AutoCandidate("process", "python", "process"),
            "rust": AutoCandidate("rust", "rust"),
            "rust-threads": AutoCandidate("rust-threads", "rust", None, max(2, cpus)),
            "cpp": AutoCandidate("cpp", "cpp"),
        }[name]

    @classmethod
    def candidates(
        cls,
        kind: str,
        cpus: int,
        cpp_available: bool,
        requested: Sequence[str] | None = None,
    ) -> list[AutoCandidate]:
        """List the candidates that can run this problem, most preferred first.

        Args:
            kind (str): `callbacks`, `batch` or `native` (built-in or compiled problem).
            cpus (int): Usable logical CPUs, used for the all-threads candidate.
            cpp_available (bool): Whether the extension includes the C++ reference.
            requested (Sequence[str] | None): Optional subset of `NAMES` to restrict to.

        Returns:
            list[AutoCandidate]: Ordered candidates.

        Raises:
            ValueError: If a requested name is unknown or none can run this problem.
        """
        if requested is not None:
            unknown = [name for name in requested if name not in cls.NAMES]
            if unknown or not requested:
                raise ValueError(f"'auto_candidates' must be a nonempty subset of {cls.NAMES}.")
        order = {
            "callbacks": ("rust", "rust-threads", "process", "cpp"),
            "batch": ("rust", "rust-threads", "cpp"),
            "native": ("rust", "cpp"),
        }[kind]
        chosen = [
            cls.build(name, cpus)
            for name in order
            if (requested is None or name in requested) and (name != "cpp" or cpp_available)
        ]
        if not chosen:
            raise ValueError("No requested engine can run this problem in this installation.")
        return chosen

    @staticmethod
    def decide(
        scores: dict[str, list[float]], prior: Sequence[str]
    ) -> tuple[str, float, str, dict[str, float]]:
        """Pick the winner, preferring the evidence-based candidate inside the noise.

        Args:
            scores (dict[str, list[float]]): Probe costs of each usable candidate.
            prior (Sequence[str]): Candidates from most to least preferred.

        Returns:
            tuple[str, float, str, dict[str, float]]: Winner, noise margin, reason and the
            mean cost of every usable candidate.

        Raises:
            ValueError: If no candidate has a usable score.
        """
        usable = {name: costs for name, costs in scores.items() if costs}
        if not usable:
            raise ValueError("No candidate produced a usable probe.")
        rank = {name: index for index, name in enumerate(prior)}
        means = {name: statistics.fmean(costs) for name, costs in usable.items()}
        noise = {name: (max(costs) - min(costs)) / 2 for name, costs in usable.items()}
        best = min(means, key=lambda name: (means[name], rank.get(name, len(rank))))
        preferred = min(usable, key=lambda name: rank.get(name, len(rank)))
        margin = max(
            noise[best], noise[preferred], AutoEngine.SWITCH_MARGIN * abs(means[preferred])
        )
        if preferred != best and means[preferred] - means[best] <= margin + 1e-12:
            return preferred, margin, "no clear advantage over the preferred engine", means
        return best, margin, "lowest mean probe cost", means

    @classmethod
    def select(
        cls,
        solver: SelectableSolver,
        time_limit: float,
        auto_time: float | None,
        requested: Sequence[str] | None,
        cpp_available: bool,
        clock: Callable[[], float],
    ) -> tuple[AutoSelection, tuple[float, ...] | None, float]:
        """Run every probe and return the decision and the best probe solution.

        Args:
            solver (SelectableSolver): Configured solver with `_variant`, `_kind`, `random_state`.
            time_limit (float): Search seconds of the final run, used for the default budget.
            auto_time (float | None): Total selection budget, or 10 percent of `time_limit`.
            requested (Sequence[str] | None): Optional candidate subset.
            cpp_available (bool): Whether the C++ reference is installed.
            clock (Callable[[], float]): Monotonic clock function.

        Returns:
            tuple[AutoSelection, tuple[float, ...] | None, float]: Decision, best probe keys
            and best probe cost (infinity when every probe failed).

        Raises:
            RuntimeError: If every candidate fails; the first error is chained in the text.
        """
        started = clock()
        cpus = max(1, os.process_cpu_count() or 1)
        candidates = cls.candidates(solver._kind(), cpus, cpp_available, requested)
        budget = cls.FRACTION * time_limit if auto_time is None else auto_time
        per_probe = max(budget / (len(candidates) * cls.REPEATS), cls.MINIMUM_PROBE)
        sequence = np.random.SeedSequence(solver.random_state)
        seeds = [int(item.generate_state(1)[0]) for item in sequence.spawn(cls.REPEATS)]
        probes: list[AutoProbe] = []
        costs: dict[str, list[float]] = {}
        best_keys: tuple[float, ...] | None = None
        best_cost = math.inf
        for candidate in candidates:
            for seed in seeds:
                began = clock()
                try:
                    result = solver._variant(candidate, seed).solve(per_probe)
                except Exception as failure:
                    probes.append(
                        AutoProbe(
                            candidate.name,
                            seed,
                            None,
                            clock() - began,
                            f"{type(failure).__name__}: {failure}",
                        )
                    )
                    break
                probes.append(AutoProbe(candidate.name, seed, result.cost, clock() - began))
                costs.setdefault(candidate.name, []).append(result.cost)
                if result.cost < best_cost:
                    best_cost, best_keys = result.cost, result.keys
        failed = {probe.candidate for probe in probes if probe.error is not None}
        usable = {name: values for name, values in costs.items() if name not in failed}
        if not usable:
            first = next((probe.error for probe in probes if probe.error), "no candidates")
            raise RuntimeError(f"Automatic engine selection failed for every candidate: {first}")
        prior = [candidate.name for candidate in candidates]
        chosen, margin, reason, means = cls.decide(usable, prior)
        selection = AutoSelection(
            chosen=chosen,
            candidates=tuple(prior),
            probes=tuple(probes),
            scores=tuple(sorted(means.items(), key=lambda item: item[1])),
            margin=margin,
            reason=reason,
            probe_seconds=per_probe,
            selection_seconds=clock() - started,
        )
        return selection, best_keys, best_cost
