"""Strict MathOpt source extraction.

Objective:
- Reject non-primary and nonlinear populated model features.

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
from .records import LinearRow
from .records import LinearVariable
from .source import ExtractedLinearModel
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from ortools.math_opt.model_pb2 import ModelProto


class MathOptExtractor:
    """Audit and extract one qualified frontend without invoking its optimizer.

    Responsibilities:
    - Reject unsupported source constructs before numerical compilation.
    - Preserve stable variable identity and complete affine semantics.
    """

    @staticmethod
    def extract(model: object, installed: str) -> tuple[ExtractedLinearModel, tuple[Any, ...]]:
        """Audit every populated MathOpt proto field before extracting ID-based linear data.

        Args:
            model (object): Trusted OR-Tools MathOpt model, not CP-SAT or MPSolver.
            installed (str): Qualified installed OR-Tools version.

        Returns:
            tuple: Immutable extraction and matching stable native variable handles.

        Raises:
            ValueError: If type or any populated non-linear/non-primary feature is unsupported.
            ImportError: If the optional MathOpt native runtime cannot be loaded.
        """
        module = importlib.import_module("ortools.math_opt.python.model")
        if not isinstance(model, module.Model):
            raise ValueError(
                "OR-Tools conversion supports MathOpt Model only, not CP-SAT/MPSolver."
            )
        proto = model.export_model()
        MathOptExtractor._audit(proto)
        data = proto.variables
        positions = {identity: index for index, identity in enumerate(data.ids)}
        native = tuple(model.get_variable(identity) for identity in data.ids)
        variables = tuple(
            LinearVariable(
                f"mathopt:{identity}",
                variable.name,
                float(lb),
                float(ub),
                "integer" if integer else "continuous",
            )
            for identity, variable, lb, ub, integer in zip(
                data.ids,
                native,
                data.lower_bounds,
                data.upper_bounds,
                data.integers,
                strict=True,
            )
        )
        objective = [0.0] * len(variables)
        for identity, coefficient in zip(
            proto.objective.linear_coefficients.ids,
            proto.objective.linear_coefficients.values,
            strict=True,
        ):
            objective[positions[identity]] = float(coefficient)
        rows = MathOptExtractor._rows(proto, positions)
        return ExtractedLinearModel(
            "mathopt",
            installed,
            proto.name,
            variables,
            tuple(objective),
            float(proto.objective.offset),
            bool(proto.objective.maximize),
            rows,
        ), native

    @staticmethod
    def _audit(proto: ModelProto) -> None:
        """Validate and preserve one original frontend structure.

        Args:
            proto (Any): Complete exported original model.

        Returns:
            None: Validates every populated feature.

        Raises:
            ValueError: If original semantics are unsupported.
        """
        supported = {
            "name",
            "variables",
            "objective",
            "linear_constraints",
            "linear_constraint_matrix",
        }
        for field, _ in proto.ListFields():
            if field.name not in supported:
                raise ValueError(f"MathOpt unsupported populated model feature: {field.name}.")
        for field, _ in proto.objective.ListFields():
            if field.name not in {"maximize", "offset", "linear_coefficients", "name"}:
                raise ValueError(f"MathOpt unsupported objective feature: {field.name}.")

    @staticmethod
    def _rows(proto: ModelProto, positions: dict[int, int]) -> tuple[LinearRow, ...]:
        """Validate and preserve one original frontend structure.

        Args:
            proto (Any): Audited original model proto.
            positions (dict): Stable original variable positions.

        Returns:
            tuple: Original sparse affine row records.

        Raises:
            ValueError: If original semantics are unsupported.
        """
        row_data = proto.linear_constraints
        row_positions = {identity: index for index, identity in enumerate(row_data.ids)}
        coefficients: list[list[tuple[int, float]]] = [[] for _ in row_data.ids]
        matrix = proto.linear_constraint_matrix
        for row, column, coefficient in zip(
            matrix.row_ids, matrix.column_ids, matrix.coefficients, strict=True
        ):
            coefficients[row_positions[row]].append((positions[column], float(coefficient)))
        rows = tuple(
            LinearRow(f"mathopt:{identity}", tuple(entries), float(lb), float(ub))
            for identity, entries, lb, ub in zip(
                row_data.ids,
                coefficients,
                row_data.lower_bounds,
                row_data.upper_bounds,
                strict=True,
            )
        )
        return rows
