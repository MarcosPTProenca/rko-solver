"""Strict PuLP source extraction.

Objective:
- Preserve ID-based PuLP linear semantics.

Technical approach:
- Validate original frontend semantics before owning portable data.

Flow (pseudocode):
1. Audit supported source features.
2. Preserve stable identities and affine records.
3. Return immutable portable extraction.

Important criteria:
- Never silently ignore unsupported source features or alter objective semantics.

Inputs/Outputs:
- Input: trusted constructed native source model.
- Output: strictly validated owned original linear records.
"""

from __future__ import annotations

import importlib
import math
from .records import LinearRow
from .records import LinearVariable
from .source import ExtractedLinearModel
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

if TYPE_CHECKING:
    from pulp import LpConstraint
    from pulp import LpVariable


class PulpExtractor:
    """Audit and extract one qualified frontend without invoking its optimizer.

    Responsibilities:
    - Reject unsupported source constructs before numerical compilation.
    - Preserve stable variable identity and complete affine semantics.
    """

    @staticmethod
    def extract(model: object, installed: str) -> tuple[ExtractedLinearModel, tuple[Any, ...]]:
        """Extract PuLP 4 public ID-based data without name-keyed lossy JSON conversion.

        Args:
            model (object): Trusted PuLP 4 problem.
            installed (str): Qualified installed PuLP version.

        Returns:
            tuple: Immutable extraction and stable native variable handles.

        Raises:
            ValueError: If type, SOS, objective or variable/row semantics are unsupported.
            ImportError: If optional PuLP native runtime cannot be loaded.
        """
        p = importlib.import_module("pulp")
        if not isinstance(model, p.LpProblem):
            raise ValueError("PuLP conversion requires an LpProblem.")
        if model.has_sos():
            raise ValueError("PuLP SOS constraints are unsupported.")
        if model.objective is None:
            raise ValueError(
                "PuLP requires one explicit linear objective, including constant models."
            )
        if model.sense not in (p.LpMinimize, p.LpMaximize):
            raise ValueError("Unsupported PuLP objective direction.")
        native = tuple(sorted(model.variables(), key=lambda variable: variable.id))
        positions = {variable.id: index for index, variable in enumerate(native)}
        variables = PulpExtractor._variables(native, p)
        objective = [0.0] * len(native)
        for variable, coefficient in model.objective.items():
            objective[positions[variable.id]] = float(coefficient)
        rows = [
            PulpExtractor._row(constraint, positions)
            for constraint in sorted(model.constraints(), key=lambda row: row.id)
        ]
        return ExtractedLinearModel(
            "pulp",
            installed,
            model.name,
            tuple(variables),
            tuple(objective),
            float(model.objective.constant),
            model.sense == p.LpMaximize,
            tuple(rows),
        ), native

    @staticmethod
    def _row(constraint: LpConstraint, positions: dict[int, int]) -> LinearRow:
        """Preserve one original PuLP row's sense, coefficients and affine offset.

        Args:
            constraint (Any): Original native constraint.
            positions (dict[int, int]): Stable original variable positions.

        Returns:
            LinearRow: Complete original sparse row.

        Raises:
            ValueError: If the original row sense is unsupported.
        """
        if constraint.sense not in (-1, 0, 1):
            raise ValueError(f"PuLP row {constraint.id}: unsupported constraint sense.")
        return LinearRow(
            f"pulp:{constraint.id}",
            tuple(
                (positions[variable.id], float(coefficient))
                for variable, coefficient in constraint.items()
            ),
            -float(constraint.constant) if constraint.sense in (0, 1) else -math.inf,
            -float(constraint.constant) if constraint.sense in (0, -1) else math.inf,
        )

    @staticmethod
    def _variables(native: tuple[LpVariable, ...], p: ModuleType) -> list[LinearVariable]:
        """Validate and preserve one original frontend structure.

        Args:
            native (tuple): Stable native handles.
            p (Any): Qualified frontend.

        Returns:
            list: Exact owned variable domains.

        Raises:
            ValueError: If original semantics are unsupported.
        """
        variables = []
        for variable in native:
            if variable.cat not in (p.LpContinuous, p.LpInteger, p.LpBinary):
                raise ValueError(f"PuLP variable {variable.id}: unsupported domain {variable.cat}.")
            kind: Literal["continuous", "integer", "binary"] = (
                "binary"
                if variable.isBinary()
                else ("integer" if variable.cat == p.LpInteger else "continuous")
            )
            variables.append(
                LinearVariable(
                    f"pulp:{variable.id}",
                    variable.name,
                    -math.inf if variable.lowBound is None else float(variable.lowBound),
                    math.inf if variable.upBound is None else float(variable.upBound),
                    kind,
                )
            )
        return variables
