"""Size-aware, outcome-independent inputs for the expanded utility study.

Objective:
- Provide fresh KPLIB, routing and continuous inputs that never reuse an earlier cohort.

Technical approach:
- Treat each KPLIB (size, type, id) as a distinct input and exclude every consumed pool.
- Generate routing and Rastrigin files from namespaced seeds, never from solver outcomes.
- Score routes with the existing rounded, index-stable independent objective.

Flow (pseudocode):
1. Rebuild the consumed pools of the calibration, test, confirmation and screen cohorts.
2. Draw pilot inputs first, then draw confirmation inputs from the remainder per type.
3. Fetch and validate each file by its declared item count, or write a generated file.

Important criteria:
- A 1000-item file is never cached or accepted under a 500-item path.
- An exhausted pool is an error, not a reason to recycle identifiers.
- Draws depend only on role, size, group and seed.

Inputs/Outputs:
- Input: cache directory, role (`pilot` or `confirmation`), size and group counts.
- Output: identifiers, validated file paths and independent scorers.
"""

from __future__ import annotations

import hashlib
import random
import urllib.request
from benchmarks.algorithms.sa_temperature_screen import ScreenInputs
from benchmarks.engines.cpp_python_knapsack import FAMILIES
from benchmarks.engines.cpp_python_knapsack import KPLIB_RAW
from benchmarks.engines.cpp_python_knapsack import KPLIB_REVISION
from benchmarks.engines.gate_instances import GateInstances
from benchmarks.problems.numeric import tsp_objectives
from collections.abc import Callable
from pathlib import Path

UTILITY_SEED = 20261201
KPLIB_SIZES = (500, 1000)
KPLIB_IDS = 100
MAX_KPLIB_BYTES = 1 << 20
TSP_KINDS = ("uniform", "clustered")
TSP_EXTENT = 10_000
TSP_CLUSTERS = 6
TSP_SPREAD = 600
ROLES = ("pilot", "confirmation")
PILOT_PER_TYPE = 1


class UtilityInputs:
    """Select, fetch and generate the study inputs without consulting any solver result.

    Responsibilities:
    - Rebuild consumed KPLIB pools and draw disjoint pilot and confirmation inputs.
    - Cache KPLIB files under their item count and refuse a mismatched header.
    - Write routing and Rastrigin inputs and expose the independent route scorer.

    Attributes:
        cache_dir (Path): Cache root for downloads and generated inputs.
    """

    def __init__(self, cache_dir: Path) -> None:
        """Bind the input factory to a cache directory without touching the network.

        Args:
            cache_dir (Path): Cache root for downloads and generated inputs.
        """
        self.cache_dir = cache_dir

    def consumed(self, size: int) -> set[tuple[str, int]]:
        """Return the KPLIB (type, id) pairs already used at one item count.

        Args:
            size (int): Item count of the file.

        Returns:
            set[tuple[str, int]]: Consumed pairs; empty for sizes no cohort used.

        Raises:
            ValueError: If the size is not a supported item count.
        """
        if size not in KPLIB_SIZES:
            raise ValueError(f"unsupported KPLIB size {size}")
        if size != 500:
            return set()
        cohorts = (
            GateInstances(self.cache_dir, "calibration", 100),
            GateInstances(self.cache_dir, "test", 100),
            GateInstances(self.cache_dir, "confirmation", 100),
            ScreenInputs(self.cache_dir, 100),
        )
        return {pair for cohort in cohorts for pair in cohort.knapsack_ids()}

    def eligible(self, size: int) -> dict[str, list[int]]:
        """List the unused identifiers per KPLIB type at one item count.

        Args:
            size (int): Item count of the file.

        Returns:
            dict[str, list[int]]: Sorted unused identifiers per type.

        Raises:
            ValueError: If the size is not a supported item count.
        """
        used = self.consumed(size)
        return {
            kind: [item for item in range(KPLIB_IDS) if (kind, item) not in used]
            for kind in FAMILIES
        }

    def draw(self, size: int, per_type: int, role: str) -> list[tuple[str, int]]:
        """Draw KPLIB inputs per type without replacement, pilot before confirmation.

        Args:
            size (int): Item count of the file.
            per_type (int): Positive inputs per type.
            role (str): `pilot` or `confirmation`.

        Returns:
            list[tuple[str, int]]: Type and identifier pairs, grouped by type.

        Raises:
            ValueError: If the role or size is unknown or an eligible pool is exhausted.
        """
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        pools = self.eligible(size)
        return [
            pair
            for kind, ids in pools.items()
            for pair in self._draw_group(kind, ids, size, per_type, role)
        ]

    @staticmethod
    def _draw_group(
        kind: str, ids: list[int], size: int, per_type: int, role: str
    ) -> list[tuple[str, int]]:
        """Draw one type's identifiers; confirmation excludes the pilot draw first.

        Args:
            kind (str): KPLIB type.
            ids (list[int]): Eligible identifiers of that type.
            size (int): Item count.
            per_type (int): Positive number to draw.
            role (str): `pilot` or `confirmation`.

        Returns:
            list[tuple[str, int]]: Sorted pairs of this type.

        Raises:
            ValueError: If the request is outside the role's reserved share of the pool.
        """
        order = random.Random(f"{UTILITY_SEED}:{size}:{kind}").sample(ids, len(ids))
        limit = PILOT_PER_TYPE if role == "pilot" else len(order) - PILOT_PER_TYPE
        if not 1 <= per_type <= limit:
            raise ValueError(f"eligible pool exhausted for {kind} at {size} items")
        start = 0 if role == "pilot" else PILOT_PER_TYPE
        return [(kind, item) for item in sorted(order[start : start + per_type])]

    def kplib_path(self, kind: str, size: int, item_id: int) -> Path:
        """Return a validated, size-namespaced KPLIB file, downloading it when absent.

        Args:
            kind (str): KPLIB type directory.
            size (int): Declared item count.
            item_id (int): Instance number.

        Returns:
            Path: Cached file whose header declares exactly `size` items.

        Raises:
            ValueError: If the file is oversized, malformed or declares another item count.
            OSError: If the file cannot be downloaded or written.
        """
        target = self.cache_dir / "kplib-sized" / f"n{size:05d}" / kind / f"s{item_id:03d}.kp"
        if not target.exists():
            url = f"{KPLIB_RAW}/{KPLIB_REVISION}/{kind}/n{size:05d}/R01000/s{item_id:03d}.kp"
            data = self._download(url)
            if len(data) > MAX_KPLIB_BYTES:
                raise ValueError(f"unexpectedly large KPLIB input: {kind}/{item_id}")
            self._check_header(data, size, kind, item_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        self._check_header(target.read_bytes(), size, kind, item_id)
        return target

    @staticmethod
    def _check_header(data: bytes, size: int, kind: str, item_id: int) -> None:
        """Require the declared item count and a complete body.

        Args:
            data (bytes): File contents.
            size (int): Expected item count.
            kind (str): KPLIB type, for the message.
            item_id (int): Instance number, for the message.

        Raises:
            ValueError: If the count differs or the file is truncated or malformed.
        """
        tokens = data.split()
        if len(tokens) != 2 + 2 * size or int(tokens[0]) != size:
            raise ValueError(f"{kind}/s{item_id:03d} does not declare exactly {size} items")

    def _download(self, url: str) -> bytes:
        """Fetch at most the size cap of one URL.

        Args:
            url (str): Source URL.

        Returns:
            bytes: Response body up to one byte over the cap.

        Raises:
            OSError: If the request fails.
        """
        with urllib.request.urlopen(url, timeout=30) as response:
            return response.read(MAX_KPLIB_BYTES + 1)

    def tsp_path(self, kind: str, cities: int, index: int, role: str) -> Path:
        """Write one deterministic routing instance with integer coordinates.

        Args:
            kind (str): `uniform` or `clustered`.
            cities (int): City count, at least four.
            index (int): Instance number within its role.
            role (str): `pilot` or `confirmation`.

        Returns:
            Path: Native-readable file: city count, then one `x y` pair per line.

        Raises:
            ValueError: If the kind, role or city count is invalid.
            OSError: If the file cannot be written.
        """
        if kind not in TSP_KINDS or cities < 4 or index < 0:
            raise ValueError("invalid routing instance request")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        generator = random.Random(f"{UTILITY_SEED}:tsp:{role}:{kind}:{cities}:{index}")
        lines = [str(cities), *self._tsp_points(generator, kind, cities)]
        path = self.cache_dir / "utility-routes" / f"{role}-{kind}-{cities}-{index:02d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    def rastrigin_path(self, dimension: int, index: int, role: str) -> Path:
        """Write one deterministic shifted Rastrigin instance.

        Args:
            dimension (int): Positive coordinate count.
            index (int): Instance number within its role.
            role (str): `pilot` or `confirmation`.

        Returns:
            Path: File with the dimension, then one shift per line.

        Raises:
            ValueError: If the dimension, index or role is invalid.
            OSError: If the file cannot be written.
        """
        if dimension < 1 or index < 0:
            raise ValueError("invalid continuous instance request")
        if role not in ROLES:
            raise ValueError(f"unknown role {role!r}")
        generator = random.Random(f"{UTILITY_SEED}:rastrigin:{role}:{dimension}:{index}")
        shifts = [round(generator.uniform(-2.0, 2.0), 3) for _ in range(dimension)]
        path = self.cache_dir / "utility-continuous" / f"{role}-{dimension}-{index:02d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join([str(dimension), *(repr(shift) for shift in shifts)]) + "\n")
        return path

    @staticmethod
    def _tsp_points(generator: random.Random, kind: str, cities: int) -> list[str]:
        """Draw integer city coordinates as `x y` lines.

        Args:
            generator (random.Random): Namespaced stream.
            kind (str): `uniform` or `clustered`.
            cities (int): City count.

        Returns:
            list[str]: One `x y` line per city.
        """
        centers = [
            (generator.randrange(TSP_EXTENT), generator.randrange(TSP_EXTENT))
            for _ in range(TSP_CLUSTERS)
        ]
        top = TSP_EXTENT - 1
        lines = []
        for _ in range(cities):
            if kind == "uniform":
                x, y = generator.randrange(TSP_EXTENT), generator.randrange(TSP_EXTENT)
            else:
                cx, cy = generator.choice(centers)
                x = min(top, max(0, round(generator.gauss(cx, TSP_SPREAD))))
                y = min(top, max(0, round(generator.gauss(cy, TSP_SPREAD))))
            lines.append(f"{x} {y}")
        return lines

    def route_scorer(self, path: Path) -> Callable[[list[float]], float]:
        """Build the independent rounded closed-tour scorer for one routing file.

        Args:
            path (Path): File written by `tsp_path`.

        Returns:
            Callable[[list[float]], float]: Tour length of a key vector, ties by index.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the file is malformed.
        """
        _, _, blocked = tsp_objectives(path)
        return lambda keys: float(blocked([keys])[0])

    @staticmethod
    def fingerprint(path: Path) -> str:
        """Hash the exact bytes of one input file.

        Args:
            path (Path): Input file.

        Returns:
            str: SHA-256 hex digest of the platform-native bytes.

        Raises:
            OSError: If the file cannot be read.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()
