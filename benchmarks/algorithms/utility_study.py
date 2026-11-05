"""Manifest, sequential runner and row analysis of the expanded utility study.

Objective:
- Freeze a fresh, fingerprinted instance plan, run it one trial at a time in a seeded
  order that cannot depend on quality, and turn the raw rows into class decisions.

Technical approach:
- `UtilityManifest` draws pilot or confirmation inputs and records path, digest and the
  exact reference where one exists.
- `UtilityRunner` pairs baseline and candidate trials, shuffles pairs by an order seed,
  appends rows to a new file and aborts on the first error or when the time ceiling ends.
- `UtilityStudy` applies the registered win rules to rows and classifies the claims.

Flow (pseudocode):
1. Build the manifest and freeze its digest.
2. Plan every (instance, profile, seed, arm) trial and run it sequentially.
3. Group rows per instance and profile, compute wins and decide each claim.

Important criteria:
- No planned instance is dropped; a wrong seed count is an error and an absent instance
  leaves its claim incomplete.
- Output files are created exclusively and never overwritten.

Inputs/Outputs:
- Input: a `UtilityInputs` factory, an executor callable and raw trial rows.
- Output: manifest entries, JSONL rows and per-class win lists.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from benchmarks.algorithms.utility_analysis import SEEDS
from benchmarks.algorithms.utility_analysis import UtilityAnalysis
from benchmarks.algorithms.utility_inputs import UtilityInputs
from benchmarks.problems.knapsack_utils import exact_profit
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from examples.run_kp_cpp_instances import read_kp_instance
from pathlib import Path
from typing import Any

CLASSES = ("item", "route", "continuous")
SIZES = {"item": (500, 1000), "route": (100, 300), "continuous": (100, 300)}
ARMS = ("baseline", "candidate")
SOLVER_SEED = 20262001


class UtilityManifest:
    """Draw and record the pilot or confirmation instances.

    Responsibilities:
    - Build 60 confirmation or 12 pilot entries with path, fingerprint and reference.
    - Digest the canonical entry list to identify the frozen plan.

    Attributes:
        inputs (UtilityInputs): Factory for fresh inputs.
        role (str): `pilot` or `confirmation`.
    """

    def __init__(self, inputs: UtilityInputs, role: str) -> None:
        """Bind the manifest to a factory and role.

        Args:
            inputs (UtilityInputs): Factory for fresh inputs.
            role (str): `pilot` or `confirmation`.
        """
        self.inputs = inputs
        self.role = role

    def build(self) -> list[dict[str, Any]]:
        """Fetch, validate and fingerprint every planned instance.

        Returns:
            list[dict[str, Any]]: One entry per instance.

        Raises:
            ValueError: If an input is invalid or a pool is exhausted.
            OSError: If an input cannot be fetched or written.
        """
        confirm = self.role == "confirmation"
        entries: list[dict[str, Any]] = []
        for size in SIZES["item"]:
            entries += self._items(size, 6 if confirm else 1)
        for cities in SIZES["route"]:
            entries += self._routes(cities, 15 if confirm else 1)
        for dimension in SIZES["continuous"]:
            entries += self._continuous(dimension, 30 if confirm else 2)
        return entries

    def _entry(
        self, kind: str, size: int, labels: tuple[str, str], path: Path, reference: int | None
    ) -> dict[str, Any]:
        """Record one instance with its content fingerprint.

        Args:
            kind (str): Problem class.
            size (int): Size stratum value.
            labels (tuple[str, str]): Stable instance name and its subtype (KPLIB type,
                layout or `shift`).
            path (Path): Native-readable file.
            reference (int | None): Exact optimum for item selection, otherwise None.

        Returns:
            dict[str, Any]: Manifest entry.
        """
        return {
            "class": kind,
            "stratum": f"{kind}-{size}",
            "instance": labels[0],
            "subtype": labels[1],
            "path": str(path),
            "sha256": self.inputs.fingerprint(path),
            "reference": reference,
        }

    def _items(self, size: int, per_type: int) -> list[dict[str, Any]]:
        """Draw item-selection inputs; a pilot keeps two of one per type.

        Args:
            size (int): KPLIB item count.
            per_type (int): Confirmation inputs per type, or one for the pilot draw.

        Returns:
            list[dict[str, Any]]: Entries with exact references.

        Raises:
            ValueError: If a pool is exhausted or a file is invalid.
            OSError: If a file cannot be fetched.
        """
        pairs = self.inputs.draw(size, per_type, self.role)
        if self.role == "pilot":
            pairs = sorted(random.Random(f"pilot:{size}").sample(pairs, 2))
        entries = []
        for kind, item_id in pairs:
            path = self.inputs.kplib_path(kind, size, item_id)
            reference = exact_profit(read_kp_instance(path))
            name = f"{kind}/n{size:05d}/s{item_id:03d}"
            entries.append(self._entry("item", size, (name, kind), path, reference))
        return entries

    def _routes(self, cities: int, per_layout: int) -> list[dict[str, Any]]:
        """Write routing inputs for both layouts.

        Args:
            cities (int): City count.
            per_layout (int): Instances per layout.

        Returns:
            list[dict[str, Any]]: Entries without a reference.

        Raises:
            OSError: If a file cannot be written.
        """
        entries = []
        for layout in ("uniform", "clustered"):
            for index in range(per_layout):
                path = self.inputs.tsp_path(layout, cities, index, self.role)
                name = f"route/{layout}-{cities}-{self.role}-{index:02d}"
                entries.append(self._entry("route", cities, (name, layout), path, None))
        return entries

    def _continuous(self, dimension: int, count: int) -> list[dict[str, Any]]:
        """Write shifted Rastrigin inputs.

        Args:
            dimension (int): Coordinate count.
            count (int): Number of shifts.

        Returns:
            list[dict[str, Any]]: Entries without a reference.

        Raises:
            OSError: If a file cannot be written.
        """
        entries = []
        for index in range(count):
            path = self.inputs.rastrigin_path(dimension, index, self.role)
            name = f"continuous/shift-{dimension}-{self.role}-{index:02d}"
            entries.append(self._entry("continuous", dimension, (name, "shift"), path, None))
        return entries

    @staticmethod
    def digest(entries: Sequence[Mapping[str, Any]]) -> str:
        """Hash the canonical JSON of the entries.

        Args:
            entries (Sequence[Mapping[str, Any]]): Manifest entries.

        Returns:
            str: SHA-256 hex digest; paths are excluded so the digest names content only.
        """
        canonical = [{k: v for k, v in entry.items() if k != "path"} for entry in entries]
        encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()


class UtilityRunner:
    """Plan and run trials sequentially in a seeded, quality-independent order.

    Responsibilities:
    - Pair baseline and candidate trials and shuffle the pairs by the order seed.
    - Append rows to a new file, abort on the first error and stop at the time ceiling.

    Attributes:
        entries (Sequence[Mapping[str, Any]]): Manifest entries.
        profiles (Sequence[str]): Budget profile names.
        order_seed (int): Seed of the execution order.
    """

    def __init__(
        self, entries: Sequence[Mapping[str, Any]], profiles: Sequence[str], order_seed: int
    ) -> None:
        """Store the plan inputs.

        Args:
            entries (Sequence[Mapping[str, Any]]): Manifest entries.
            profiles (Sequence[str]): Budget profile names.
            order_seed (int): Seed of the execution order.
        """
        self.entries = entries
        self.profiles = profiles
        self.order_seed = order_seed

    def plan(self) -> list[dict[str, Any]]:
        """List every trial once, pairs adjacent, in the seeded order.

        Returns:
            list[dict[str, Any]]: Trials with instance, profile, replicate and arm.
        """
        stream = random.Random(f"order:{self.order_seed}")
        where = {entry["instance"]: entry for entry in self.entries}
        pairs = [
            (entry["instance"], profile, replicate)
            for entry in self.entries
            for profile in self.profiles
            for replicate in range(SEEDS)
        ]
        stream.shuffle(pairs)
        trials = []
        for instance, profile, replicate in pairs:
            arms = list(ARMS)
            stream.shuffle(arms)
            for arm in arms:
                trials.append(
                    {
                        "class": where[instance]["class"],
                        "stratum": where[instance]["stratum"],
                        "instance": instance,
                        "profile": profile,
                        "replicate": replicate,
                        "seed": SOLVER_SEED + replicate,
                        "arm": arm,
                    }
                )
        return trials

    def run(
        self,
        execute: Callable[[dict[str, Any]], dict[str, Any]],
        output: Path,
        max_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> dict[str, int]:
        """Run the plan once and append each row before the next trial starts.

        Args:
            execute (Callable[[dict[str, Any]], dict[str, Any]]): Runs one trial.
            output (Path): New JSONL path.
            max_seconds (float): Cumulative time ceiling.
            clock (Callable[[], float]): Monotonic clock, injectable for tests.

        Returns:
            dict[str, int]: Planned and completed trial counts.

        Raises:
            FileExistsError: If the output exists.
            TimeoutError: If the ceiling is reached before the plan completes.
        """
        trials = self.plan()
        started = clock()
        completed = 0
        with output.open("x") as handle:
            for trial in trials:
                if clock() - started > max_seconds:
                    raise TimeoutError(
                        f"time ceiling reached after {completed} of {len(trials)} trials"
                    )
                handle.write(json.dumps(execute(trial), allow_nan=False) + "\n")
                handle.flush()
                completed += 1
        return {"planned": len(trials), "completed": completed}


class UtilityStudy:
    """Turn raw trial rows into per-class win lists.

    Responsibilities:
    - Group seeds per instance, profile and arm and apply the registered win rule.
    """

    @staticmethod
    def class_wins(
        rows: Sequence[Mapping[str, Any]],
        entries: Sequence[Mapping[str, Any]],
        profiles: Sequence[str],
    ) -> dict[str, list[bool]]:
        """Compute one win flag per instance that has rows, grouped by class.

        Args:
            rows (Sequence[Mapping[str, Any]]): Raw trial rows.
            entries (Sequence[Mapping[str, Any]]): Planned manifest entries.
            profiles (Sequence[str]): Required budget profiles.

        Returns:
            dict[str, list[bool]]: Win flags per class; an instance without any row is omitted.

        Raises:
            ValueError: If an instance has a wrong number of seeds in any arm or profile.
        """
        grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[(row["instance"], row["profile"], row["arm"])].append(row)
        wins: dict[str, list[bool]] = {}
        for entry in entries:
            if not any(key[0] == entry["instance"] for key in grouped):
                continue
            flags = {
                profile: UtilityStudy._profile(grouped, entry, profile) for profile in profiles
            }
            wins.setdefault(entry["class"], []).append(UtilityAnalysis.instance_win(flags))
        return wins

    @staticmethod
    def _profile(
        grouped: Mapping[tuple[str, str, str], list[Mapping[str, Any]]],
        entry: Mapping[str, Any],
        profile: str,
    ) -> bool:
        """Evaluate one instance in one profile.

        Args:
            grouped (Mapping[tuple[str, str, str], list[Mapping[str, Any]]]): Rows by key.
            entry (Mapping[str, Any]): Manifest entry.
            profile (str): Budget profile.

        Returns:
            bool: Whether the candidate wins in this profile.

        Raises:
            ValueError: If either arm lacks exactly the planned number of seeds.
        """
        arms = {}
        for arm in ARMS:
            found = sorted(
                grouped.get((entry["instance"], profile, arm), []), key=lambda r: r["replicate"]
            )
            if len(found) != SEEDS:
                raise ValueError(f"{entry['instance']} {profile} {arm}: expected {SEEDS} seeds")
            arms[arm] = found
        return UtilityAnalysis.profile_win(
            entry["class"],
            [row["score"] for row in arms["baseline"]],
            [row["score"] for row in arms["candidate"]],
            [bool(row["feasible"]) for row in arms["candidate"]],
        )
