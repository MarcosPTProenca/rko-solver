"""Immutable sparse plan preparation for continuous restoration.

Objective:
- Normalize continuous row supports without losing original coefficients.

Technical approach:
- Compile each row independently into linear-size owned arrays.

Flow (pseudocode):
1. Locate movable continuous coordinates.
2. Fold fixed/discrete effects and normalize row supports safely.
3. Freeze plan buffers as bytes on the numerical snapshot.

Important criteria:
- Never allocate a dense constraint matrix or silently discard nonzero coefficients.

Inputs/Outputs:
- Input: validated sparse numerical model.
- Output: immutable continuous restoration plan.
"""

from __future__ import annotations

import math
import numpy as np
from .records import LinearRow
from .storage import LinearStorage
from typing import Any


class RepairPlan:
    """Compile a portable row projection plan for one immutable model.

    Responsibilities:
    - Validate normalized sparse arithmetic.
    - Keep temporary compilation state out of the published snapshot.

    Attributes:
        model (LinearStorage): Validated target storage.
        positions (dict[int, int]): Original to continuous-coordinate mapping.
    """

    def __init__(self, model: LinearStorage) -> None:
        """Locate eligible continuous coordinates and initialize linear-size buffers.

        Args:
            model (LinearStorage): Validated original model.
        """
        self.model = model
        free = np.frombuffer(model._free, dtype=np.intp)
        self.lower = np.frombuffer(model._lower)
        self.upper = np.frombuffer(model._upper)
        self.selected = (
            [i for i, column in enumerate(free) if model.variables[column].domain == "continuous"]
            if model.repair == "continuous"
            else []
        )
        self.original = free[self.selected]
        self.positions = {int(column): i for i, column in enumerate(self.original)}
        self.widths = self.upper[self.selected] - self.lower[self.selected]
        self.buffers: dict[str, list[Any]] = {
            name: [] for name in ("columns", "coefficients", "offsets", "scales", "norms", "frozen")
        }
        self.buffers["indptr"] = [0]
        self.entry = 0

    def apply(self) -> None:
        """Compile rows in original order and freeze every plan buffer.

        Raises:
            ValueError: If normalization underflows or exceeds safe arithmetic.
        """
        for index, row in enumerate(self.model.rows if self.selected else ()):
            self._row(index, row)
        fixed = {
            "free": (self.original, np.intp),
            "lower": (self.lower[self.selected], float),
            "upper": (self.upper[self.selected], float),
            "width": (self.widths, float),
        }
        for name, (values, dtype) in fixed.items():
            object.__setattr__(
                self.model, "_repair_" + name, np.asarray(values, dtype=dtype).tobytes()
            )
        for name, entries in self.buffers.items():
            dtype = np.intp if name in ("indptr", "columns", "frozen") else float
            target = "_repair_frozen_entries" if name == "frozen" else "_repair_" + name
            object.__setattr__(self.model, target, np.asarray(entries, dtype=dtype).tobytes())

    def _row(self, index: int, row: LinearRow) -> None:
        """Compile one continuous support and original affine offset.

        Args:
            index (int): Original row position.
            row (LinearRow): Original sparse affine row.

        Raises:
            ValueError: If normalized coefficients or bounds are unsafe.
        """
        self.terms = [float(np.frombuffer(self.model._row_constants)[index])]
        self.slopes: list[float] = []
        self.row_columns: list[int] = []
        self.frozen_magnitudes: list[float] = []
        for column, coefficient in row.coefficients:
            if not self.model.domains[column].fixed and coefficient != 0:
                self._term(column, coefficient, row.identifier)
        scale = max(map(abs, self.slopes), default=1.0)
        normalized = self._normalized(scale, row.identifier)
        offset = math.fsum(self.terms)
        bounds = [abs(v) for v in (row.lower, row.upper) if math.isfinite(v)]
        self.model._safe(
            math.fsum([max(bounds, default=0.0), abs(offset), *self.frozen_magnitudes]) / scale,
            f"repair row {row.identifier!r} normalized arithmetic",
        )
        self.buffers["columns"].extend(self.row_columns)
        self.buffers["coefficients"].extend(normalized)
        self.buffers["indptr"].append(len(self.buffers["columns"]))
        self.buffers["offsets"].append(offset)
        self.buffers["scales"].append(scale)
        self.buffers["norms"].append(math.fsum(v * v for v in normalized))

    def _term(self, column: int, coefficient: float, identifier: str) -> None:
        """Compile one nonfixed nonzero term without changing its domain.

        Args:
            column (int): Original variable index.
            coefficient (float): Original finite coefficient.
            identifier (str): Original row diagnostic identity.

        Raises:
            ValueError: If continuous scaling underflows.
        """
        if column in self.positions:
            position = self.positions[column]
            slope = coefficient * self.widths[position]
            if slope == 0:
                raise ValueError(f"Repair row {identifier!r}: coefficient underflow.")
            self.slopes.append(slope)
            self.row_columns.append(position)
            self.terms.append(coefficient * self.lower[self.selected[position]])
        else:
            self.buffers["frozen"].append(self.entry)
            domain = self.model.domains[column]
            self.frozen_magnitudes.append(
                abs(coefficient) * max(abs(domain.lower), abs(domain.upper))
            )
        self.entry += 1

    def _normalized(self, scale: float, identifier: str) -> list[float]:
        """Normalize a row without silently dropping tiny coefficients.

        Args:
            scale (float): Maximum nonzero continuous slope magnitude.
            identifier (str): Original row identity.

        Returns:
            list[float]: Finite normalized support coefficients.

        Raises:
            ValueError: If normalization or squaring underflows.
        """
        normalized = [value / scale for value in self.slopes]
        if any(value == 0 or value * value == 0 for value in normalized):
            raise ValueError(f"Repair row {identifier!r}: normalization underflow.")
        return normalized
