"""Portable original frontend model semantics.

Objective:
- Keep original semantics independent of search overrides.

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

import hashlib
from .linear import CompiledLinearModel
from .options import SearchOptions
from .records import LinearRow
from .records import LinearVariable
from dataclasses import dataclass
from typing import Literal
from typing import Unpack


@dataclass(frozen=True, slots=True)
class ExtractedLinearModel:
    """Own original frontend semantics without model references or compilation overrides.

    Responsibilities:
    - Preserve source identity and qualified frontend version separately from search settings.
    - Supply the shared numerical compiler and source-compatibility fingerprint.

    Attributes:
        frontend (str): Pyomo, pulp or mathopt frontend identifier.
        version (str): Qualified installed package version.
        name (str): Source display name.
        variables (tuple[LinearVariable, ...]): Stable original variable domains and fixed values.
        objective (tuple[float, ...]): Original objective coefficients in stable variable order.
        objective_constant (float): Original objective offset.
        maximize (bool): Original objective direction.
        rows (tuple[LinearRow, ...]): Original active affine rows with both sides.
        structure (tuple[str, ...]): Additional original Pyomo expression/domain structure.
    """

    frontend: Literal["pyomo", "pulp", "mathopt"]
    version: str
    name: str
    variables: tuple[LinearVariable, ...]
    objective: tuple[float, ...]
    objective_constant: float
    maximize: bool
    rows: tuple[LinearRow, ...]
    structure: tuple[str, ...] = ()

    @property
    def fingerprint(self) -> str:
        """Hash canonical extracted source semantics, excluding mutable solution guesses.

        Returns:
            str: SHA256 source fingerprint, independent of restrictive compilation bounds.
        """
        return hashlib.sha256(repr(self).encode("utf-8")).hexdigest()

    def compile(self, **options: Unpack[SearchOptions]) -> CompiledLinearModel:
        """Compile independent numerical settings without changing original semantics.

        Args:
            options (SearchOptions): Restrictive bounds, scoring, tolerances and repair.

        Returns:
            CompiledLinearModel: Owned pickleable numerical snapshot.

        Raises:
            ValueError: If the numerical model or policy is unsafe or unsupported.
            TypeError: If an unknown keyword is supplied.
        """
        return CompiledLinearModel(
            self.variables,
            self.objective,
            self.rows,
            objective_constant=self.objective_constant,
            maximize=self.maximize,
            **options,
        )
