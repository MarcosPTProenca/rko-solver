"""Strict optional frontend dispatch and explicit native assignment boundary.

Objective:
- Isolate native handles from portable bounded linear source records.

Technical approach:
- Audit only qualified frontend versions and dispatch cohesive source extractors.
- Revalidate original fingerprints before mapping or transactional write-back.

Flow (pseudocode):
1. Identify and version-check a trusted native model.
2. Extract complete supported original semantics without invoking an optimizer.
3. Revalidate source identity before explicitly requested native export.

Important criteria:
- No silent feature omission, name-only identities or implicit model modification.

Inputs/Outputs:
- Input: trusted native model and explicit completed export evidence.
- Output: immutable owned source data or transactional native value updates.
"""

from __future__ import annotations

from .frontend_mathopt import MathOptExtractor
from .frontend_pulp import PulpExtractor
from .frontend_pyomo import PyomoExtractor
from .linear import CompiledLinearModel
from .records import ModelEvaluation
from .source import ExtractedLinearModel
from .write_back import AssignmentTransaction
from importlib.metadata import version
from typing import Any

__all__ = ["ExtractedLinearModel", "ModelAdapter"]


class ModelAdapter:
    """Dispatch strict qualified source extraction and explicit validated export.

    Responsibilities:
    - Keep native frontend concerns outside numerical callbacks and process IPC.
    - Verify original source compatibility before native assignment transactions.

    Attributes:
        _VERSIONS (dict): Exact released frontend versions qualified by tests.
    """

    _VERSIONS = {"pyomo": "6.10.1", "pulp": "4.2.0", "mathopt": "9.15.6755"}

    @classmethod
    def extract(cls, model: object) -> ExtractedLinearModel:
        """Extract supported original semantics without invoking an optimizer.

        Args:
            model (object): Constructed trusted Pyomo, PuLP or MathOpt model.

        Returns:
            ExtractedLinearModel: Owned original-source representation.

        Raises:
            ValueError: If model, installed version or source features are unsupported.
            ImportError: If the selected optional frontend cannot load.
        """
        return cls._inspect(model)[0]

    @classmethod
    def _inspect(cls, model: object) -> tuple[ExtractedLinearModel, tuple[Any, ...]]:
        """Identify one qualified frontend and obtain stable local native handles.

        Args:
            model (object): Trusted native model.

        Returns:
            tuple: Original owned source and matching stable native handles.

        Raises:
            ValueError: If type, version or supported source semantics disagree.
            ImportError: If the selected optional frontend is unavailable.
        """
        roots = {base.__module__.split(".")[0] for base in type(model).__mro__}
        frontend = next(
            (
                name
                for root, name in (("pyomo", "pyomo"), ("pulp", "pulp"), ("ortools", "mathopt"))
                if root in roots
            ),
            None,
        )
        if frontend is None:
            raise ValueError("Expected a Pyomo, PuLP 4 or OR-Tools MathOpt model object.")
        package = "ortools" if frontend == "mathopt" else frontend
        installed = version(package)
        if installed != cls._VERSIONS[frontend]:
            raise ValueError(
                f"Unqualified {frontend} version {installed}; install rko-solver[{frontend}] "
                f"with {package}=={cls._VERSIONS[frontend]}."
            )
        try:
            extractors = {
                "pyomo": PyomoExtractor.extract,
                "pulp": PulpExtractor.extract,
                "mathopt": MathOptExtractor.extract,
            }
            return extractors[frontend](model, installed)
        except ImportError as exc:
            raise ImportError(
                f"Cannot load {frontend}; install the rko-solver[{frontend}] extra."
            ) from exc

    @classmethod
    def variable_mapping(cls, model: object, source: ExtractedLinearModel) -> dict[str, Any]:
        """Map native variables only after checking original source compatibility.

        Args:
            model (object): Explicit original or compatible reconstructed native model.
            source (ExtractedLinearModel): Original owned source semantics.

        Returns:
            dict[str, Any]: Stable source IDs mapped to local native handles.

        Raises:
            ValueError: If the original source semantics or version changed.
            ImportError: If the selected optional runtime cannot load.
        """
        current, native = cls._inspect(model)
        if current.fingerprint != source.fingerprint:
            raise ValueError(
                "Native model changed or is incompatible; reconvert before assignment export."
            )
        return {
            variable.identifier: target
            for variable, target in zip(source.variables, native, strict=True)
        }

    @classmethod
    def write_back(
        cls,
        model: object,
        source: ExtractedLinearModel,
        snapshot: CompiledLinearModel,
        evaluation: ModelEvaluation,
    ) -> None:
        """Export only explicitly validated feasible evidence with transactional rollback.

        Args:
            model (object): Compatible Pyomo/PuLP native model receiving values.
            source (ExtractedLinearModel): Original owned source semantics.
            snapshot (CompiledLinearModel): Completed witness's numerical policy.
            evaluation (ModelEvaluation): Exact completed feasible evidence.

        Raises:
            NotImplementedError: If MathOpt native assignment slots are requested.
            ValueError: If provenance, evidence or restorable prior values are invalid.
            RuntimeError: If application fails, including explicit rollback failures.
            ImportError: If the selected optional frontend cannot load.
        """
        if source.frontend == "mathopt":
            raise NotImplementedError(
                "MathOpt has no native assignment slot; export assignments instead."
            )
        mapping = cls.variable_mapping(model, source)
        AssignmentTransaction(source, snapshot, evaluation, mapping).apply()
