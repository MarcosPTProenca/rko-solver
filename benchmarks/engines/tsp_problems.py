"""Euclidean traveling-salesman instances, priors and a residual-priority sort decoder.

Objective:
- Provide a second problem class, a permutation sort decoder, for testing whether a prior tour
  inside the decoder beats the same prior seeded through `initial_keys`.

Technical approach:
- Two generators: uniform cities in the unit square and cities drawn around a few cluster
  centers. Distances are Euclidean and unrounded.
- Two priors are tours: nearest neighbour from city zero, and that tour improved by
  best-improvement 2-opt over a full delta matrix.
- A Held-Karp 1-tree lower bound by projected subgradient gives an absolute quality scale.
- `TspDecoder` sorts cities by the rank-normalized position in the prior tour plus a key-driven
  residual, or by key alone when no prior is given.

Flow (pseudocode):
1. Generate an instance from a family name and a seed.
2. Build a prior tour, then encode it as one key vector or a set of perturbed ones.
3. Decode a key vector by sorting prior position plus spread times the centered key.
4. Evaluate the closed tour length.

Important criteria:
- Every decoded order is a permutation of the cities.
- The lower bound never exceeds the length of any tour.
- Instances are deterministic functions of family, size and seed.

Inputs/Outputs:
- Input: family name, number of cities, seeds and tours.
- Output: instances, tours, key vectors, decoded orders, lengths and a lower bound.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from dataclasses import field

KINDS = ("uniform", "clustered")
CLUSTERS = 6
CLUSTER_SPREAD = 0.06
BOUND_STEPS = 300


@dataclass(slots=True, eq=False)
class TspInstance:
    """A Euclidean traveling-salesman instance.

    Attributes:
        name (str): `<kind>-<cities>-<seed>`.
        coordinates (np.ndarray): City coordinates, one row per city.
    """

    name: str
    coordinates: np.ndarray

    @property
    def n_cities(self) -> int:
        """Return the number of cities.

        Returns:
            int: Row count of the coordinates.
        """
        return int(self.coordinates.shape[0])


class TspFamily:
    """Generate instances and build tours, keys and bounds.

    Responsibilities:
    - Generate uniform and clustered instances deterministically.
    - Build nearest-neighbour and 2-opt tours.
    - Encode tours as key vectors and compute a Held-Karp lower bound.
    """

    @staticmethod
    def make(kind: str, n_cities: int, seed: int) -> TspInstance:
        """Generate one instance.

        Args:
            kind (str): `uniform` or `clustered`.
            n_cities (int): Number of cities, at least four.
            seed (int): Instance seed.

        Returns:
            TspInstance: Generated instance.

        Raises:
            ValueError: If the kind is unknown or there are fewer than four cities.
        """
        if kind not in KINDS:
            raise ValueError(f"Unknown kind {kind!r}; expected one of {KINDS}.")
        if n_cities < 4:
            raise ValueError("An instance needs at least four cities.")
        rng = np.random.default_rng([seed, n_cities, KINDS.index(kind)])
        if kind == "uniform":
            points = rng.random((n_cities, 2))
        else:
            centers = rng.random((CLUSTERS, 2))
            members = rng.integers(0, CLUSTERS, n_cities)
            points = np.clip(
                centers[members] + CLUSTER_SPREAD * rng.standard_normal((n_cities, 2)), 0.0, 1.0
            )
        return TspInstance(f"{kind}-{n_cities}-{seed}", points)

    @staticmethod
    def length(instance: TspInstance, order: np.ndarray) -> float:
        """Compute the closed tour length of a visiting order.

        Args:
            instance (TspInstance): Instance.
            order (np.ndarray): Permutation of the cities.

        Returns:
            float: Closed tour length.

        Raises:
            ValueError: If the order does not have one entry per city.
        """
        order = np.asarray(order)
        if order.shape != (instance.n_cities,):
            raise ValueError("The order must have one entry per city.")
        path = instance.coordinates[order]
        return float(np.linalg.norm(path - np.roll(path, -1, axis=0), axis=1).sum())

    @staticmethod
    def nearest_neighbour(instance: TspInstance) -> np.ndarray:
        """Build the nearest-neighbour tour from city zero.

        Args:
            instance (TspInstance): Instance.

        Returns:
            np.ndarray: Permutation of the cities starting at city zero.
        """
        points = instance.coordinates
        distance = np.linalg.norm(points[:, None] - points[None], axis=2)
        tour = np.zeros(instance.n_cities, dtype=np.int64)
        left = np.ones(instance.n_cities, dtype=bool)
        left[0] = False
        for step in range(1, instance.n_cities):
            candidates = np.where(left, distance[tour[step - 1]], np.inf)
            tour[step] = int(np.argmin(candidates))
            left[tour[step]] = False
        return tour

    @staticmethod
    def two_opt(instance: TspInstance, tour: np.ndarray) -> np.ndarray:
        """Improve a tour with best-improvement 2-opt until no move shortens it.

        Args:
            instance (TspInstance): Instance.
            tour (np.ndarray): Permutation of the cities.

        Returns:
            np.ndarray: A 2-opt local optimum no longer than the input.
        """
        points = instance.coordinates
        distance = np.linalg.norm(points[:, None] - points[None], axis=2)
        count = instance.n_cities
        allowed = np.triu(np.ones((count, count), dtype=bool), 1)
        allowed[0, count - 1] = False
        tour = np.asarray(tour).copy()
        while True:
            local = distance[np.ix_(tour, tour)]
            edge = np.diagonal(np.roll(local, -1, axis=1))
            delta = np.where(
                allowed,
                local + np.roll(local, (-1, -1), axis=(0, 1)) - edge[:, None] - edge[None],
                0.0,
            )
            first, second = np.unravel_index(int(np.argmin(delta)), delta.shape)
            if delta[first, second] > -1e-12:
                break
            tour[first + 1 : second + 1] = tour[first + 1 : second + 1][::-1].copy()
        return tour

    @staticmethod
    def lower_bound(instance: TspInstance, steps: int = BOUND_STEPS) -> float:
        """Compute a Held-Karp 1-tree lower bound by projected subgradient ascent.

        Args:
            instance (TspInstance): Instance.
            steps (int): Subgradient iterations.

        Returns:
            float: A value no larger than any tour length.
        """
        points = instance.coordinates
        distance = np.linalg.norm(points[:, None] - points[None], axis=2)
        upper = TspFamily.length(instance, TspFamily.nearest_neighbour(instance))
        penalty = np.zeros(instance.n_cities)
        best, scale, stale = -np.inf, 2.0, 0
        for _ in range(steps):
            value, degree = TspFamily.one_tree(distance + penalty[:, None] + penalty[None])
            value -= 2.0 * penalty.sum()
            if value > best + 1e-9:
                best, stale = value, 0
            else:
                stale += 1
                if stale == 10:
                    scale, stale = scale / 2.0, 0
            slope = degree - 2.0
            norm = float(slope @ slope)
            if norm == 0.0:
                break
            penalty = penalty + scale * max(upper - value, 1e-12) / norm * slope
        return float(best)

    @staticmethod
    def one_tree(weights: np.ndarray) -> tuple[float, np.ndarray]:
        """Build a minimum 1-tree over the weights.

        The tree spans cities one and up and adds city zero with its two cheapest edges.

        Args:
            weights (np.ndarray): Symmetric edge weights.

        Returns:
            tuple[float, np.ndarray]: Total weight and the degree of every city in the 1-tree.
        """
        count = weights.shape[0]
        degree = np.zeros(count)
        reach = weights[1].copy()
        parent = np.ones(count, dtype=np.int64)
        left = np.ones(count, dtype=bool)
        left[:2] = False
        total = 0.0
        for _ in range(count - 2):
            node = int(np.argmin(np.where(left, reach, np.inf)))
            total += float(reach[node])
            degree[node] += 1.0
            degree[parent[node]] += 1.0
            left[node] = False
            closer = left & (weights[node] < reach)
            reach[closer] = weights[node][closer]
            parent[closer] = node
        nearest = np.argsort(weights[0, 1:])[:2] + 1
        total += float(weights[0, nearest].sum())
        degree[nearest] += 1.0
        degree[0] = 2.0
        return total, degree

    @staticmethod
    def position_keys(tour: np.ndarray) -> list[float]:
        """Encode a tour as one key vector whose ascending order is the tour.

        Args:
            tour (np.ndarray): Permutation of the cities.

        Returns:
            list[float]: One key in (0, 1) per city.
        """
        tour = np.asarray(tour)
        keys = np.empty(len(tour))
        keys[tour] = (np.arange(len(tour)) + 0.5) / len(tour)
        return keys.tolist()

    @staticmethod
    def perturbed_keys(tour: np.ndarray, count: int, noise: float, seed: int) -> list[list[float]]:
        """Encode a tour and noisy variants of it as key vectors.

        Args:
            tour (np.ndarray): Permutation of the cities.
            count (int): Number of key vectors, at least one.
            noise (float): Standard deviation, as a fraction of the tour, added to positions.
            seed (int): Perturbation seed.

        Returns:
            list[list[float]]: The exact tour first, then `count - 1` perturbed orders.

        Raises:
            ValueError: If the count is not positive or the noise is negative.
        """
        if count < 1:
            raise ValueError("'count' must be at least one.")
        if noise < 0.0:
            raise ValueError("'noise' must not be negative.")
        tour = np.asarray(tour)
        rng = np.random.default_rng(seed)
        rows = [TspFamily.position_keys(tour)]
        for _ in range(count - 1):
            shifted = np.arange(len(tour)) / len(tour) + noise * rng.standard_normal(len(tour))
            rows.append(TspFamily.position_keys(tour[np.argsort(shifted, kind="stable")]))
        return rows


@dataclass(slots=True, eq=False)
class TspDecoder:
    """Decode keys into a visiting order through an optional prior tour.

    Responsibilities:
    - Sort cities by rank-normalized prior position plus a key-driven residual, or by key alone.

    Attributes:
        instance (TspInstance): Instance.
        tour (np.ndarray | None): Prior tour, None for the plain sort decoder.
        spread (float): Residual scale; large values approach the plain sort decoder.
    """

    instance: TspInstance
    tour: np.ndarray | None = None
    spread: float = 1.0
    base: np.ndarray | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Precompute the normalized position of every city in the prior tour."""
        if self.tour is not None:
            count = self.instance.n_cities
            self.base = np.empty(count)
            self.base[np.asarray(self.tour)] = np.arange(count) / max(count - 1, 1)

    def __call__(self, keys: np.ndarray) -> np.ndarray:
        """Decode one key vector.

        Args:
            keys (np.ndarray): One key per city.

        Returns:
            np.ndarray: Permutation of the cities.
        """
        keys = np.asarray(keys)
        priority = keys if self.base is None else self.base + self.spread * (keys - 0.5)
        return np.argsort(priority, kind="stable")


@dataclass(slots=True)
class TspCost:
    """Closed tour length of a decoded order.

    Attributes:
        instance (TspInstance): Instance.
    """

    instance: TspInstance

    def __call__(self, order: np.ndarray) -> float:
        """Compute the minimization objective.

        Args:
            order (np.ndarray): Permutation of the cities.

        Returns:
            float: Closed tour length.
        """
        return TspFamily.length(self.instance, order)
