"""Bounded owned linear-model transport for isolated assignment construction.

Objective:
- Carry original model semantics and effective policy without native objects or pickle.
Technical approach:
- Use strict versioned JSON, explicit absent-side bounds and original source fingerprints.
Flow (pseudocode):
1. Compile/validate original owned records and explicit numerical policy.
2. Encode bounded data with no NaN/Infinity or executable values.
3. Restore public records and verify source identity before isolated execution.
Important criteria:
- Reference declarations remain parent-only and never enter construction data.
- Transport validity is not a claim of native-package or optimizer certification.
Inputs/Outputs:
- Input: owned extracted model, effective policy and optional trusted reference declaration.
- Output: bounded data and immutable protected numerical evaluation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numpy as np
import os
import typing
from pathlib import Path
from rko_solver.experimental.self_evolve.artifacts import DecoderArtifact
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.options import SearchOptions
from rko_solver.modeling.records import LinearRow
from rko_solver.modeling.records import LinearVariable
from rko_solver.modeling.records import ModelEvaluation
from rko_solver.modeling.source import ExtractedLinearModel

MODEL_FINALIZATION_ALLOWANCE = 0.05


@dataclasses.dataclass(frozen=True, slots=True)
class ModelReference:
    """Own an explicit parent-side reference, not a claimed solver certificate.

    Responsibilities:
    - Keep original objective and declared provenance out of candidate input.

    Attributes:
        value (float): Predeclared trustworthy original reference objective.
        provenance (str): Owner-supplied source of the reference, not proof by itself.
    """

    value: float
    provenance: str

    def __post_init__(self) -> None:
        """Reject unknown/nonfinite values and absent reference provenance.

        Raises:
            ValueError: If reference scalar or provenance is malformed.
        """
        if (
            type(self.value) not in (int, float)
            or abs(self.value) > 1e100
            or not math.isfinite(self.value)
        ):
            raise ValueError("Model reference must be a finite original objective.")
        if abs(self.value) > 1e100 or float(self.value) != self.value:
            raise ValueError("Model reference exceeds exact supported scalar precision.")
        if not isinstance(self.provenance, str) or not 0 < len(self.provenance) <= 1000:
            raise ValueError("Model reference requires bounded explicit provenance.")


@dataclasses.dataclass(frozen=True, slots=True)
class ModelProblem:
    """Own bounded original mathematical data and independent model evaluation.

    Responsibilities:
    - Keep original extraction separate from numerical policy and references.
    - Reject malformed/drifting data before runtime setup or candidate execution.

    Attributes:
        data (dict): Detached versioned model/policy JSON without references.
        reference (ModelReference | None): Optional parent-only campaign reference.
        source (ExtractedLinearModel): Restored owned original semantics.
        snapshot (CompiledLinearModel): Protected immutable numerical evaluator.
    """

    data: dict[str, typing.Any]
    reference: ModelReference | None = None
    source: ExtractedLinearModel = dataclasses.field(init=False, repr=False)
    snapshot: CompiledLinearModel = dataclasses.field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and detach a strict bounded owned-model envelope.

        Raises:
            ValueError: If schema, types, numerical policy or source identity disagree.
        """
        try:
            payload = json.dumps(self.data, allow_nan=False).encode("utf-8")
            if len(payload) > 24576:
                raise ValueError("Model data exceed the 24 KiB worker envelope.")
            data = json.loads(payload)
            self._fields(data, {"schema", "source", "policy", "implementation"})
            if data["implementation"] != self.implementation():
                raise ValueError(
                    "Model runtime implementation changed; use the pinned source/runtime."
                )
            if type(data["schema"]) is not int or data["schema"] != 1:
                raise ValueError("Unsupported model data schema.")
            source = self._restore(data["source"])
            if not isinstance(data["policy"], dict):
                raise ValueError("Model policy must be explicit JSON settings.")
            snapshot = source.compile(**data["policy"])
            if source.fingerprint != data["source"]["fingerprint"]:
                raise ValueError("Original model source fingerprint mismatch.")
        except (TypeError, KeyError, OverflowError, RecursionError) as error:
            raise ValueError("Malformed bounded model source/policy data.") from error
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "snapshot", snapshot)
        if self.reference is not None and not isinstance(self.reference, ModelReference):
            raise ValueError("Expected explicit model reference provenance.")

    @staticmethod
    def _fields(value: object, expected: set[str]) -> None:
        """Require an exact object shape before interpreting mathematical fields.

        Args:
            value (object): Untrusted JSON object.
            expected (set[str]): Required exact keys.

        Raises:
            ValueError: If object type or fields are incompatible.
        """
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError("Model data fields do not match their declared schema.")

    @staticmethod
    def _restore(raw: dict[str, typing.Any]) -> ExtractedLinearModel:
        """Restore public source records without importing a native modeling package.

        Args:
            raw (dict): Strict owned source encoding.

        Returns:
            ExtractedLinearModel: Original ordered source representation.

        Raises:
            ValueError: If metadata, records or absent-side declarations are malformed.
            TypeError: If a nested source container has an incompatible shape.
        """
        ModelProblem._fields(
            raw,
            {
                "frontend",
                "version",
                "name",
                "variables",
                "objective",
                "objective_constant",
                "maximize",
                "rows",
                "structure",
                "fingerprint",
            },
        )
        ModelProblem._metadata(raw)
        for key in ("variables", "objective", "rows", "structure"):
            if not isinstance(raw[key], list):
                raise ValueError("Original model sequences must be bounded JSON lists.")
        if any(not isinstance(value, str) for value in raw["structure"]):
            raise ValueError("Original model structure must contain strings.")
        variables = tuple(ModelProblem._variable(item) for item in raw["variables"])
        rows = tuple(ModelProblem._row(item) for item in raw["rows"])
        return ExtractedLinearModel(
            raw["frontend"],
            raw["version"],
            raw["name"],
            variables,
            tuple(raw["objective"]),
            raw["objective_constant"],
            raw["maximize"],
            rows,
            tuple(raw["structure"]),
        )

    @staticmethod
    def _metadata(raw: dict[str, typing.Any]) -> None:
        """Validate bounded original provenance without claiming native runtime qualification.

        Args:
            raw (dict): Schema-checked original source metadata.

        Raises:
            ValueError: If source namespace, metadata types or lengths are invalid.
        """
        if raw["frontend"] not in ("pyomo", "pulp", "mathopt"):
            raise ValueError("Unsupported original source frontend.")
        for key in ("version", "name", "fingerprint"):
            if not isinstance(raw[key], str) or len(raw[key]) > 1024:
                raise ValueError("Invalid original model metadata.")

    @staticmethod
    def _variable(raw: dict[str, typing.Any]) -> LinearVariable:
        """Restore original variable bounds without conflating absent sides/fixed values.

        Args:
            raw (dict): One original variable record.

        Returns:
            LinearVariable: Public original-domain record.

        Raises:
            ValueError: If its fields differ from the source schema.
        """
        ModelProblem._fields(raw, {"identifier", "label", "lower", "upper", "domain", "fixed"})
        return LinearVariable(
            **{
                **raw,
                "lower": -math.inf if raw["lower"] is None else raw["lower"],
                "upper": math.inf if raw["upper"] is None else raw["upper"],
            }
        )

    @staticmethod
    def _row(raw: dict[str, typing.Any]) -> LinearRow:
        """Restore sparse original-coordinate row arithmetic and both original sides.

        Args:
            raw (dict): One original affine row record.

        Returns:
            LinearRow: Public original mathematical row.

        Raises:
            ValueError: If its fields or coefficient container differ from the schema.
            TypeError: If a sparse entry is not a coordinate/coefficient sequence.
        """
        ModelProblem._fields(raw, {"identifier", "coefficients", "lower", "upper", "constant"})
        if not isinstance(raw["coefficients"], list):
            raise ValueError("Expected original sparse row coefficients.")
        return LinearRow(
            raw["identifier"],
            tuple(tuple(pair) for pair in raw["coefficients"]),
            -math.inf if raw["lower"] is None else raw["lower"],
            math.inf if raw["upper"] is None else raw["upper"],
            raw["constant"],
        )

    @classmethod
    def from_source(
        cls,
        source: ExtractedLinearModel,
        *,
        policy: SearchOptions | None = None,
        reference: ModelReference | None = None,
    ) -> ModelProblem:
        """Encode validated original source without native references or reference leakage.

        Args:
            source (ExtractedLinearModel): Trusted owned extraction.
            policy (SearchOptions | None): Explicit original numerical compilation settings.
            reference (ModelReference | None): Optional trusted parent-only declaration.

        Returns:
            ModelProblem: Detached identity-checked numerical model problem.

        Raises:
            ValueError: If source/policy are unsafe or exceed the transport envelope.
            TypeError: If unsupported numerical policy keywords are supplied.
        """
        options: SearchOptions = {"repair": "continuous", **(policy or {})}
        source.compile(**options)
        raw = dataclasses.asdict(source)
        for record in (*raw["variables"], *raw["rows"]):
            record["lower"] = None if record["lower"] == -math.inf else record["lower"]
            record["upper"] = None if record["upper"] == math.inf else record["upper"]
        raw["fingerprint"] = source.fingerprint
        return cls(
            {"schema": 1, "source": raw, "policy": options, "implementation": cls.implementation()},
            reference,
        )

    @staticmethod
    def implementation() -> dict[str, str]:
        """Bind actual Python implementation bytes independently of native source mathematics.

        Returns:
            dict[str, str]: Package-relative source hashes excluding optional test files.

        Raises:
            OSError: If installed source cannot be read for frozen identity.
        """
        root = Path(__file__).resolve().parents[2]
        identity = {}
        for directory, children, names in os.walk(root):
            children[:] = sorted(child for child in children if child != "tests")
            for name in sorted(names):
                if name.endswith(".py"):
                    path = os.path.join(directory, name)
                    relative = os.path.relpath(path, root).replace(os.sep, "/")
                    with open(path, "rb") as source:
                        identity[relative] = hashlib.file_digest(source, "sha256").hexdigest()
        return identity

    def verify_witness(self, raw: object) -> ModelEvaluation:
        """Independently check reported coordinates without invoking construction or repair.

        Args:
            raw (dict): Reported keys/values and original numerical claims.

        Returns:
            ModelEvaluation: Fresh verification-only original-unit evidence.

        Raises:
            ValueError: If provenance-free claims disagree with protected numerical evaluation.
        """
        if not isinstance(raw, dict) or not isinstance(raw.get("keys"), list):
            raise ValueError("Missing original model witness keys.")
        if any(type(value) not in (int, float) or not 0 <= value <= 1 for value in raw["keys"]):
            raise ValueError("Malformed original model witness keys.")
        result = self.snapshot.evaluate_assignment(np.asarray(raw["keys"]), raw.get("values"))
        if type(raw.get("feasible")) is not bool or raw["feasible"] != result.feasible:
            raise ValueError("Model witness feasibility mismatch.")
        for key, expected in (("objective", result.objective), ("score", result.score)):
            self._claim(raw.get(key), expected)
        return result

    @staticmethod
    def _claim(actual: object, expected: float) -> None:
        """Check bounded finite numerical claims without weakening original row feasibility.

        Args:
            actual (object): Untrusted objective or score scalar.
            expected (float): Independently recomputed protected value.

        Raises:
            ValueError: If a scalar is malformed, nonfinite, out of range or inconsistent.
        """
        if type(actual) not in (int, float):
            raise ValueError("Model witness objective/search-score types are invalid.")
        number = typing.cast(int | float, actual)
        if abs(number) > 1e100 or not math.isclose(number, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Model witness objective/search-score mismatch.")

    def verify_record(self, record: dict[str, typing.Any]) -> dict[str, typing.Any]:
        """Verify both retained witnesses and preserve observed versus checked score identity.

        Args:
            record (dict): Bounded untrusted model worker evidence.

        Returns:
            dict: Authoritative primary objective/feasibility and separate verification values.

        Raises:
            ValueError: If a witness, primary selection or feasible archive is inconsistent.
        """
        self._identity(record)
        incumbent = self.verify_witness(record.get("search_incumbent"))
        raw_best = record.get("best_feasible")
        best = self.verify_witness(raw_best) if raw_best is not None else None
        primary = self._select_witness(incumbent, best)
        if (best is None) != (record["infeasible_evaluations"] == record["evaluations"]):
            raise ValueError("Model feasible archive and completed counts disagree.")
        selected = raw_best if raw_best is not None else record["search_incumbent"]
        if record.get("score") != selected.get("score"):
            raise ValueError("Observed model score identity mismatch.")
        checked = self.verify_witness({**record, "values": record.get("solution")})
        if checked.keys != primary.keys or checked.values != primary.values:
            raise ValueError("Primary model witness identity mismatch.")
        return {
            "objective": primary.objective,
            "feasible": primary.feasible,
            "score": record["score"],
            "verified_score": primary.score,
            "verified_rows": [dataclasses.asdict(row) for row in primary.rows],
            "observed_objective": record["objective"],
            "model_status": "feasible" if best is not None else "no_feasible_found",
        }

    def _identity(self, record: dict[str, typing.Any]) -> None:
        """Bind model evidence to actual original source, effective policy and construction use.

        Args:
            record (dict): Untrusted numerical worker record.

        Raises:
            ValueError: If reported model/policy or zero-key provenance is inconsistent.
        """
        expected = {
            "model_source_fingerprint": self.source.fingerprint,
            "model_policy_fingerprint": self.policy_fingerprint,
            "no_search": self.dimension == 0,
            "construction_used": self.dimension != 0,
        }
        if any(
            type(record.get(key)) is not type(value) or record.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("Model evidence source/policy/construction identity mismatch.")

    def _select_witness(
        self, incumbent: ModelEvaluation, best: ModelEvaluation | None
    ) -> ModelEvaluation:
        """Require an actually retained feasible archive without changing original direction.

        Args:
            incumbent (ModelEvaluation): Verified actual score-selected witness.
            best (ModelEvaluation | None): Verified original-objective feasible witness.

        Returns:
            ModelEvaluation: Primary feasible witness or the diagnostic infeasible incumbent.

        Raises:
            ValueError: If feasibility retention or original-objective ordering is inconsistent.
        """
        if best is not None and not best.feasible:
            raise ValueError("Primary model witness is not feasible.")
        if incumbent.feasible and best is None:
            raise ValueError("A feasible model incumbent was lost.")
        if incumbent.feasible and best is not None:
            worse = (
                best.objective < incumbent.objective
                if self.source.maximize
                else best.objective > incumbent.objective
            )
            if worse:
                raise ValueError("Best feasible model witness is worse than a retained incumbent.")
        return best if best is not None else incumbent

    def proposer_data(self) -> dict[str, typing.Any]:
        """Expose mathematical construction data without case identity or reference leakage.

        Returns:
            dict: Immutable-context field descriptions and original finite JSON mathematics.
        """
        raw = self.data["source"]
        return {
            key: raw[key]
            for key in ("variables", "rows", "objective", "objective_constant", "maximize")
        } | {
            "domains": [dataclasses.asdict(domain) for domain in self.snapshot.domains],
            "n_keys": self.dimension,
            "protected_policy": self.data["policy"],
        }

    @staticmethod
    def seed_artifact() -> DecoderArtifact:
        """Return an explicitly requested existing-mapper seed, never a hidden initialization.

        Returns:
            DecoderArtifact: Generic original-domain mapper with no model-specific identifiers.

        Raises:
            ValueError: If built-in source fails the current artifact contract.
        """
        source = """import math

def decode(keys, instance):
    values = []
    free = 0
    for variable, domain in zip(instance.variables, instance.domains):
        value = domain.lower
        if not domain.fixed:
            key = keys[free]
            free += 1
            width = domain.upper - domain.lower
            if variable.domain == "continuous":
                value = domain.lower + key * width
            else:
                value = domain.lower + math.floor(key * (width + 1))
            value = min(domain.upper, max(domain.lower, value))
        values.append(value)
    return values
"""
        return DecoderArtifact.create("linear_model", source)

    @property
    def family(self) -> str:
        """Identify assignment construction without inferring a domain-specific formulation.

        Returns:
            str: Distinct generic linear-model artifact family.
        """
        return "linear_model"

    @property
    def dimension(self) -> int:
        """Derive keys from nonfixed effective coordinates, never an assumed training size.

        Returns:
            int: Actual model key dimension.
        """
        return self.snapshot.n_keys

    @property
    def policy_fingerprint(self) -> str:
        """Identify the actual compiled policy independently of original source identity.

        Returns:
            str: SHA256 of immutable numerical snapshot representation.
        """
        return hashlib.sha256(repr(self.snapshot).encode("utf-8")).hexdigest()

    def optimum(self) -> float:
        """Require an explicit parent-side reference for automatic campaign comparison.

        Returns:
            float: Trusted predeclared original reference objective.

        Raises:
            ValueError: If no reference provenance was supplied.
        """
        if self.reference is None:
            raise ValueError("Automatic model comparison requires a trustworthy reference.")
        return self.reference.value
