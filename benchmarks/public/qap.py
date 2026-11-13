"""Quadratic assignment (QAP) instances in QAPLIB format, with a decoder and an objective.

Objective:
- Parse QAPLIB `.dat` and enriched `.qap` text exactly, and score an assignment of facilities to
  locations by the sum of flow times distance over all facility pairs.

Technical approach:
- The first line holds the size, optionally followed by the optimum and the best-known value. A
  negative optimum means the optimum is unknown and its magnitude is a lower bound.
- The flow and distance matrices are read in order as n x n blocks, exactly 2 n^2 numbers in all.
- The random-key decoder sorts keys ascending; the resulting order places facility `order[k]` at
  location `k`.

Flow (pseudocode):
1. Read the header tokens; accept one token (size) or three (size, optimum, best known).
2. Read 2 n^2 numeric tokens; build the flow and distance matrices.
3. Decode keys to `argsort(keys)`; cost is the sum of `flow[order[k], order[l]] * distance[k, l]`.

Important criteria:
- Malformed or non-square input raises ValueError before any search starts.
- Ties in the keys are broken stably, so decoding is deterministic.

Inputs/Outputs:
- Input: QAPLIB-format text, or two square matrices and a key vector.
- Output: a QapInstance, a permutation of facilities, or a float cost.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class QapInstance:
    """Parsed QAP instance with its optional reference values.

    Responsibilities:
    - Hold the flow and distance matrices exactly as parsed.
    - Carry the optimum and best-known value when the file records them.

    Attributes:
        size (int): Number of facilities and locations.
        flow (np.ndarray): Facility flow matrix.
        distance (np.ndarray): Location distance matrix.
        optimum (int | None): Proven optimum, or None when unknown.
        best_known (int | None): Best-known objective value, or None when absent.
    """

    size: int
    flow: np.ndarray
    distance: np.ndarray
    optimum: int | None
    best_known: int | None


def parse_qap(text: str) -> QapInstance:
    """Parse QAPLIB `.dat` or enriched `.qap` text.

    Args:
        text (str): Full file content.

    Returns:
        QapInstance: The parsed instance.

    Raises:
        ValueError: If the header is malformed, a value is not numeric, or the number of matrix
            values is not 2 n^2.
    """
    lines = text.strip().split("\n", 1)
    header = lines[0].split() if lines and lines[0].strip() else []
    if len(header) not in (1, 3):
        raise ValueError("QAP header must hold the size, or the size, optimum and best known")
    size = int(header[0])
    if size < 1:
        raise ValueError("QAP size must be positive")
    optimum, best_known = _header_values(header)
    flow, distance = _matrices(lines[1] if len(lines) > 1 else "", size)
    return QapInstance(size, flow, distance, optimum, best_known)


def _matrices(body: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Read the flow and distance matrices from the body of a QAPLIB file.

    Args:
        body (str): Text after the header line.
        size (int): Number of facilities and locations.

    Returns:
        tuple[np.ndarray, np.ndarray]: Flow matrix and distance matrix, both size by size.

    Raises:
        ValueError: If a value is not numeric or the count is not 2 n^2.
    """
    values = [float(token) for token in body.split()]
    if len(values) != 2 * size * size:
        raise ValueError(f"QAP needs {2 * size * size} matrix values, found {len(values)}")
    block = size * size
    flow = np.array(values[:block], dtype=float).reshape(size, size)
    distance = np.array(values[block:], dtype=float).reshape(size, size)
    return flow, distance


def _header_values(header: list[str]) -> tuple[int | None, int | None]:
    """Read the optional optimum and best-known values from the header tokens.

    Args:
        header (list[str]): Header tokens of the first line.

    Returns:
        tuple[int | None, int | None]: Optimum (None when unknown or a bound) and best-known value.
    """
    if len(header) == 1:
        return None, None
    raw_optimum = int(header[1])
    optimum = raw_optimum if raw_optimum >= 0 else None
    return optimum, int(header[2])


class QapObjective:
    """Assignment cost over a flow and distance pair, decoded from random keys.

    Responsibilities:
    - Decode a key vector into a facility order.
    - Score a facility order by flow times distance over all pairs.

    Attributes:
        minimize (bool): Always True; lower assignment cost is better.
        flow (np.ndarray): Facility flow matrix.
        distance (np.ndarray): Location distance matrix.
        size (int): Number of facilities.
    """

    minimize = True

    def __init__(self, flow: np.ndarray, distance: np.ndarray) -> None:
        """Store the two matrices after checking they are square and of equal size.

        Args:
            flow (np.ndarray): Square facility flow matrix.
            distance (np.ndarray): Square location distance matrix of the same size.

        Raises:
            ValueError: If either matrix is not square or the sizes differ.
        """
        if flow.ndim != 2 or flow.shape[0] != flow.shape[1]:
            raise ValueError("flow must be a square matrix")
        if distance.shape != flow.shape:
            raise ValueError("distance must have the same square shape as flow")
        self.flow = flow
        self.distance = distance
        self.size = flow.shape[0]

    def decode(self, keys: np.ndarray) -> np.ndarray:
        """Return the facility order that sorts the keys ascending.

        Args:
            keys (np.ndarray): One key per facility.

        Returns:
            np.ndarray: Facility indices in location order; a permutation of 0 to n-1.
        """
        return np.argsort(keys, kind="stable")

    def cost(self, order: np.ndarray) -> float:
        """Score an assignment given as a facility order.

        Args:
            order (np.ndarray): Facility placed at each location, a permutation of 0 to n-1.

        Returns:
            float: Sum over all location pairs of flow between their facilities times distance.
        """
        return float(np.sum(self.flow[np.ix_(order, order)] * self.distance))
