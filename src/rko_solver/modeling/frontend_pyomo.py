"""Strict Pyomo source extraction.

Objective:
- Preserve active bounded linear Pyomo semantics.

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
from .linear import CompiledLinearModel
from .records import LinearRow
from .records import LinearVariable
from .source import ExtractedLinearModel
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

if TYPE_CHECKING:
    from pyomo.core.base.block import BlockData
    from pyomo.core.base.constraint import ConstraintData
    from pyomo.core.base.objective import ObjectiveData
    from pyomo.core.base.var import VarData


class PyomoExtractor:
    """Audit and extract one qualified frontend without invoking its optimizer.

    Responsibilities:
    - Reject unsupported source constructs before numerical compilation.
    - Preserve stable variable identity and complete affine semantics.
    """

    @staticmethod
    def extract(model: object, installed: str) -> tuple[ExtractedLinearModel, tuple[Any, ...]]:
        """Audit active Pyomo components and extract only closed linear domains/expressions.

        Args:
            model (object): Trusted Pyomo concrete block/model instance.
            installed (str): Qualified installed Pyomo version.

        Returns:
            tuple: Immutable extraction and stable native variable handles.

        Raises:
            ValueError: If active components, domains, objectives or expressions are unsupported.
            ImportError: If the optional Pyomo runtime cannot be loaded.
        """
        p = importlib.import_module("pyomo.environ")
        block = importlib.import_module("pyomo.core.base.block")
        if not isinstance(model, block.BlockData) or not model.is_constructed():
            raise ValueError("Pyomo conversion requires a constructed concrete model/block.")
        PyomoExtractor._audit(model, p)
        native = tuple(
            sorted(
                model.component_data_objects(p.Var, active=True, descend_into=True),
                key=lambda variable: str(p.ComponentUID(variable, context=model)),
            )
        )
        positions = {id(variable): index for index, variable in enumerate(native)}
        variables = []
        structure = []
        for variable in native:
            record, structure_entry = PyomoExtractor._variable(variable, model, p)
            variables.append(record)
            structure.append(structure_entry)
        objective, constant, rows, expression_structure, maximize = PyomoExtractor._expressions(
            model, p, positions
        )
        structure.extend(expression_structure)
        return ExtractedLinearModel(
            "pyomo",
            installed,
            model.name,
            tuple(variables),
            tuple(objective),
            constant,
            maximize,
            tuple(rows),
            tuple(structure),
        ), native

    @staticmethod
    def _audit(model: BlockData, p: ModuleType) -> None:
        """Validate and preserve one original frontend structure.

        Args:
            model (Any): Trusted native model.
            p (Any): Loaded qualified frontend.

        Returns:
            None: Validates supported components.

        Raises:
            ValueError: If original semantics are unsupported.
        """
        allowed = {
            p.Block,
            p.Var,
            p.Constraint,
            p.Objective,
            p.Param,
            p.Set,
            p.RangeSet,
            p.Expression,
            p.Suffix,
            p.BuildAction,
            p.BuildCheck,
        }
        for component in model.component_objects(active=True, descend_into=True):
            if component.ctype is p.Suffix and component.export_enabled() and len(component):
                raise ValueError(f"Pyomo export suffix {component.name!r} is unsupported.")
            if component.ctype not in allowed:
                raise ValueError(
                    f"Pyomo active component {component.name!r}: "
                    f"unsupported {component.ctype.__name__}."
                )

    @staticmethod
    def _variable(variable: VarData, model: BlockData, p: ModuleType) -> tuple[LinearVariable, str]:
        """Validate and preserve one original frontend structure.

        Args:
            variable (Any): Original variable.
            model (Any): Original root.
            p (Any): Qualified frontend.

        Returns:
            tuple: Owned variable and original domain structure.

        Raises:
            ValueError: If original semantics are unsupported.
        """
        number = CompiledLinearModel._number
        identity = str(p.ComponentUID(variable, context=model))
        kind = PyomoExtractor._domain(variable, identity)
        record = LinearVariable(
            f"pyomo:{identity}",
            variable.name,
            -math.inf if variable.lb is None else number(p.value(variable.lb), identity),
            math.inf if variable.ub is None else number(p.value(variable.ub), identity),
            kind,
            number(p.value(variable), identity) if variable.fixed else None,
        )
        return record, str(variable.domain)

    @staticmethod
    def _domain(variable: VarData, identity: str) -> Literal["continuous", "integer", "binary"]:
        """Validate and preserve one original frontend structure.

        Args:
            variable (Any): Original variable domain.
            identity (str): Stable diagnostic identity.

        Returns:
            str: Supported exact domain kind.

        Raises:
            ValueError: If original semantics are unsupported.
        """
        domain = variable.domain
        lb, ub, step = domain.get_interval()
        ranges = tuple(domain.ranges()) if step == 0 else ()
        if step == 0 and len(ranges) == 1 and ranges[0].closed == (True, True):
            kind: Literal["continuous", "integer", "binary"] = "continuous"
        elif step == 1 and all(bound is None or float(bound).is_integer() for bound in (lb, ub)):
            kind = "binary" if variable.is_binary() else "integer"
        else:
            raise ValueError(f"Pyomo variable {identity!r}: unsupported domain {domain}.")
        return kind

    @staticmethod
    def _expression(
        expression: ConstraintData | ObjectiveData,
        body: object,
        positions: dict[int, int],
        p: ModuleType,
    ) -> tuple[list[tuple[int, float]], float]:
        """Validate and preserve one original frontend structure.

        Args:
            expression (Any): Original row/objective.
            body (Any): Original expression body.
            positions (dict): Stable original variable positions.
            p (Any): Qualified frontend.

        Returns:
            tuple: Owned coefficients and original affine constant.

        Raises:
            ValueError: If original semantics are unsupported.
        """
        number = CompiledLinearModel._number
        linear = importlib.import_module("pyomo.repn.standard_repn").generate_standard_repn(
            body, compute_values=True
        )
        if not linear.is_linear():
            raise ValueError(f"Pyomo {expression.name!r}: nonlinear expressions are unsupported.")
        coefficients = []
        for variable, coefficient in zip(linear.linear_vars, linear.linear_coefs, strict=True):
            if id(variable) not in positions:
                raise ValueError(
                    f"Pyomo {expression.name!r}: references an inactive/foreign variable."
                )
            coefficients.append(
                (positions[id(variable)], number(p.value(coefficient), expression.name))
            )
        offset = number(p.value(linear.constant), expression.name)
        return coefficients, offset

    @staticmethod
    def _expressions(
        model: BlockData, p: ModuleType, positions: dict[int, int]
    ) -> tuple[list[float], float, list[LinearRow], list[str], bool]:
        """Validate and preserve one original frontend structure.

        Args:
            model (Any): Constructed original model.
            p (Any): Qualified frontend.
            positions (dict): Original variable positions.

        Returns:
            tuple: Objective, constant, rows, original structure and direction.

        Raises:
            ValueError: If original semantics are unsupported.
        """
        structure = []
        number = CompiledLinearModel._number
        objectives = tuple(
            model.component_data_objects(p.Objective, active=True, descend_into=True)
        )
        if len(objectives) != 1:
            raise ValueError("Pyomo requires exactly one active linear objective.")
        expressions = [objectives[0]] + list(
            model.component_data_objects(p.Constraint, active=True, descend_into=True)
        )
        rows = []
        objective = [0.0] * len(positions)
        constant = 0.0
        for index, expression in enumerate(expressions):
            body = expression.expr if index == 0 else expression.body
            coefficients, offset = PyomoExtractor._expression(expression, body, positions, p)
            structure.append(str(body))
            if index == 0:
                for column, coefficient in coefficients:
                    objective[column] = coefficient
                constant = offset
            else:
                rows.append(
                    LinearRow(
                        f"pyomo:{p.ComponentUID(expression, context=model)}",
                        tuple(coefficients),
                        -math.inf
                        if expression.lower is None
                        else number(p.value(expression.lower), expression.name),
                        math.inf
                        if expression.upper is None
                        else number(p.value(expression.upper), expression.name),
                        offset,
                    )
                )
        return objective, constant, rows, structure, objectives[0].sense == p.maximize
