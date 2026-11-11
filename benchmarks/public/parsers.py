"""Parsers for TSPLIB and OR-Library multi-dimensional knapsack files.

Objective:
- Turn public text files into plain numeric records, with no dependency on the solver.

Technical approach:
- TSPLIB: read `DIMENSION`, `EDGE_WEIGHT_TYPE` and `NODE_COORD_SECTION`; accept only EUC_2D and
  round distances with TSPLIB's nearest-integer rule, floor(x + 0.5).
- TSPLIB solutions: lines of the form `name : value`.
- OR-Library MDKP: a leading instance count, then for each instance `n m optimum`, n profits, m rows
  of n weights and m capacities. The optimum is None when the file stores 0.

Flow (pseudocode):
1. Read whitespace-separated tokens, or lines for the TSPLIB header.
2. Validate counts and lengths against the declared dimensions.
3. Return immutable records.

Important criteria:
- Any count or length mismatch raises `ValueError`; nothing is padded or guessed.
- Only EUC_2D TSPLIB instances are accepted.

Inputs/Outputs:
- Input: file text.
- Output: `TspInstance`, `Solutions` or a tuple of `MdkpInstance`.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TspInstance:
    """A symmetric Euclidean TSP with rounded integer distances.

    Attributes:
        name (str): TSPLIB name.
        distances (np.ndarray): Square matrix of rounded distances.
    """

    name: str
    distances: np.ndarray


@dataclass(frozen=True, slots=True)
class MdkpInstance:
    """A 0/1 multi-dimensional knapsack.

    Attributes:
        name (str): Source file and instance index.
        profits (np.ndarray): Length-n profits.
        weights (np.ndarray): m-by-n weight matrix.
        capacities (np.ndarray): Length-m capacities.
        optimum (float | None): Declared optimum, or None when the file gives none.
    """

    name: str
    profits: np.ndarray
    weights: np.ndarray
    capacities: np.ndarray
    optimum: float | None


class PublicParsers:
    """Parse TSPLIB and OR-Library text into immutable records.

    Responsibilities:
    - Validate declared sizes and reject unsupported TSPLIB variants.
    """

    @staticmethod
    def tsplib(name: str, text: str) -> TspInstance:
        """Parse one EUC_2D TSPLIB instance and build its rounded distance matrix.

        Args:
            name (str): Instance name used in records.
            text (str): File content.

        Returns:
            TspInstance: Symmetric matrix of nearest-integer Euclidean distances.

        Raises:
            ValueError: If the header is missing, the type is not EUC_2D, or the coordinate count
                differs from DIMENSION.
        """
        header, coordinates = PublicParsers._sections(text)
        if header.get("EDGE_WEIGHT_TYPE") != "EUC_2D":
            raise ValueError(f"{name}: only EUC_2D TSPLIB instances are supported.")
        dimension = int(header["DIMENSION"])
        if len(coordinates) != dimension:
            raise ValueError(f"{name}: {len(coordinates)} coordinates, DIMENSION says {dimension}.")
        xy = np.array(coordinates, dtype=float)
        diff = xy[:, None, :] - xy[None, :, :]
        distances = np.floor(np.sqrt((diff**2).sum(axis=2)) + 0.5)
        return TspInstance(name, distances)

    @staticmethod
    def _sections(text: str) -> tuple[dict[str, str], list[tuple[float, float]]]:
        """Split TSPLIB text into its `KEY : VALUE` header and its node coordinates.

        Args:
            text (str): File content.

        Returns:
            tuple[dict[str, str], list[tuple[float, float]]]: Header fields and (x, y) pairs.
        """
        header: dict[str, str] = {}
        coordinates: list[tuple[float, float]] = []
        in_coordinates = False
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            if line == "NODE_COORD_SECTION":
                in_coordinates = True
                continue
            if line == "EOF":
                break
            if in_coordinates:
                parts = line.split()
                coordinates.append((float(parts[1]), float(parts[2])))
            elif ":" in line:
                key, _, value = line.partition(":")
                header[key.strip()] = value.strip()
        return header, coordinates

    @staticmethod
    def solutions(text: str) -> dict[str, float]:
        """Parse a TSPLIB `solutions` file into optimal values by instance name.

        Args:
            text (str): File content with lines `name : value`.

        Returns:
            dict[str, float]: Optimal tour length for each named instance.

        Raises:
            ValueError: If a line has a non-numeric value.
        """
        values: dict[str, float] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            name, _, value = line.partition(":")
            values[name.strip()] = float(value.split()[0])
        return values

    @staticmethod
    def mdkp(prefix: str, text: str) -> tuple[MdkpInstance, ...]:
        """Parse an OR-Library multi-dimensional knapsack file.

        Args:
            prefix (str): File name used in instance names.
            text (str): File content.

        Returns:
            tuple[MdkpInstance, ...]: Every instance in file order.

        Raises:
            ValueError: If the token count does not match the declared sizes.
        """
        tokens = [float(token) for token in text.split()]
        count = int(tokens[0])
        cursor = 1
        instances = []
        for index in range(1, count + 1):
            n, m, optimum = int(tokens[cursor]), int(tokens[cursor + 1]), tokens[cursor + 2]
            cursor += 3
            profits = np.array(tokens[cursor : cursor + n])
            cursor += n
            weights = np.array(tokens[cursor : cursor + n * m]).reshape(m, n)
            cursor += n * m
            capacities = np.array(tokens[cursor : cursor + m])
            cursor += m
            instances.append(
                MdkpInstance(
                    f"{prefix}#{index}",
                    profits,
                    weights,
                    capacities,
                    optimum if optimum > 0 else None,
                )
            )
        if cursor != len(tokens):
            raise ValueError(f"{prefix}: {len(tokens) - cursor} unread tokens after {count} rows.")
        return tuple(instances)
