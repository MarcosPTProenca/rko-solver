"""Bounded data-only artifacts for a small frozen contextual action-scoring network.

Objective:
- Evaluate independently trained weights using NumPy without a neural runtime dependency.

Technical approach:
- Use the existing four-array hidden-tanh scoring shape with validated immutable arrays.

Flow (pseudocode):
1. Validate and freeze family-specific numerical weights.
2. Evaluate finite four-feature action rows.
3. Export or reload a bounded data-only artifact with a content fingerprint.

Important criteria:
- Model loading never executes pickled objects or imports a training framework.

Inputs/Outputs:
- Input: finite network weights, feature rows or a bounded artifact.
- Output: frozen contextual scores and stable artifact identity.
"""

from __future__ import annotations

import hashlib
import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar


@dataclass(frozen=True, slots=True, eq=False)
class FrozenScorer:
    """Evaluate a small hidden-tanh network from frozen numerical weights.

    Responsibilities:
    - Detach numerical parameters from their caller-owned arrays.
    - Preserve a stable content identity through data-only artifact exchange.

    Attributes:
        family (str): Task family associated with the feature schema.
        w1 (np.ndarray): Four-feature input-to-hidden weights.
        b1 (np.ndarray): Hidden biases.
        w2 (np.ndarray): Hidden-to-score weights.
        b2 (np.ndarray): Scalar score bias.
    """

    family: str
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    NAMES: ClassVar[tuple[str, ...]] = ("w1", "b1", "w2", "b2")

    def __post_init__(self) -> None:
        """Validate and detach arrays onto immutable byte-backed storage.

        Raises:
            ValueError: If the feature family, array shapes or numerical weights are invalid.
        """
        arrays = tuple(np.asarray(getattr(self, name), dtype=np.float64) for name in self.NAMES)
        self.validate(self.family, arrays)
        for name, values in zip(self.NAMES, arrays, strict=True):
            frozen = np.frombuffer(values.tobytes(), dtype=np.float64).reshape(values.shape)
            object.__setattr__(self, name, frozen)

    @staticmethod
    def validate(family: str, arrays: tuple[np.ndarray, ...]) -> None:
        """Validate a bounded four-feature, at-most-64-hidden-unit model.

        Args:
            family (str): Declared feature-schema family.
            arrays (tuple[np.ndarray, ...]): Four numerical parameter arrays.

        Raises:
            ValueError: If family, dimensions or values violate the frozen model schema.
        """
        if family not in ("tour", "cover"):
            raise ValueError("The model family must be 'tour' or 'cover'.")
        FrozenScorer._validate_shapes(arrays)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("Model weights must be finite.")

    @staticmethod
    def _validate_shapes(arrays: tuple[np.ndarray, ...]) -> None:
        """Check the four parameter arrays form one consistent hidden-tanh network.

        Args:
            arrays (tuple[np.ndarray, ...]): Four numerical parameter arrays.

        Raises:
            ValueError: If dimensions, hidden width or parameter shapes are inconsistent.
        """
        first = arrays[0]
        if first.ndim != 2 or first.shape[0] != 4 or not 1 <= first.shape[1] <= 64:
            raise ValueError("Model input weights require four features and 1..64 hidden units.")
        width = first.shape[1]
        shapes = ((4, width), (width,), (width,), ())
        if any(values.shape != shape for values, shape in zip(arrays, shapes, strict=True)):
            raise ValueError("Model parameter shapes do not match the scoring network.")

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        """Reconstruct through validation when transferring owned weights to a solver worker.

        Returns:
            tuple[object, tuple[object, ...]]: Constructor and immutable-model arguments.
        """
        return type(self), (self.family, self.w1, self.b1, self.w2, self.b2)

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Evaluate hidden-tanh scores for legal-action feature rows.

        Args:
            features (np.ndarray): Matrix containing one four-feature row per legal action.

        Returns:
            np.ndarray: One finite score per legal action, clipped to [-10, 10].

        Raises:
            ValueError: If features or resulting scores are nonfinite or malformed.
        """
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
            raise ValueError("Action features must be a finite matrix with four columns.")
        with np.errstate(over="ignore", invalid="ignore"):
            scores = np.asarray(np.tanh(rows @ self.w1 + self.b1) @ self.w2 + self.b2)
        if not np.isfinite(scores).all():
            raise ValueError("Model evaluation produced nonfinite scores.")
        return np.clip(scores, -10.0, 10.0)

    def record(self) -> dict[str, object]:
        """Represent fixed weights with ordinary data-only JSON values.

        Returns:
            dict[str, object]: Versioned family and weight data.
        """
        return {
            "format": "quantile-scorer-v1",
            "family": self.family,
            "weights": {name: getattr(self, name).tolist() for name in self.NAMES},
        }

    @property
    def fingerprint(self) -> str:
        """Hash the canonical family, format and numerical parameters.

        Returns:
            str: Hexadecimal content digest.
        """
        return hashlib.sha256(json.dumps(self.record(), sort_keys=True).encode()).hexdigest()

    def save(self, path: Path) -> None:
        """Atomically export numerical weights without executable serialization.

        Args:
            path (Path): Output file in an existing directory.

        Raises:
            OSError: If the artifact cannot be written or replaced.
        """
        record = {**self.record(), "fingerprint": self.fingerprint}
        with TemporaryDirectory(dir=path.parent) as temporary:
            draft = Path(temporary) / "scorer.json"
            draft.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
            draft.replace(path)

    @classmethod
    def load(cls, path: Path) -> FrozenScorer:
        """Read numerical parameters from a data-only JSON artifact.

        Args:
            path (Path): Previously exported model artifact.

        Returns:
            FrozenScorer: Frozen scoring model.

        Raises:
            OSError: If the artifact cannot be read.
            ValueError: If the JSON or numerical arrays are malformed.
        """
        record = cls._read_record(path)
        weights = record["weights"]
        if not isinstance(weights, dict) or set(weights) != set(cls.NAMES):
            raise ValueError("The model artifact weight schema is invalid.")
        arrays = [np.asarray(weights[name]) for name in cls.NAMES]
        model = cls(str(record["family"]), arrays[0], arrays[1], arrays[2], arrays[3])
        if record["fingerprint"] != model.fingerprint:
            raise ValueError("The model artifact fingerprint does not match its weights.")
        return model

    @staticmethod
    def _read_record(path: Path) -> dict[str, object]:
        """Read and schema-check the bounded JSON artifact before any array conversion.

        Args:
            path (Path): Previously exported model artifact.

        Returns:
            dict[str, object]: Schema-checked artifact fields.

        Raises:
            OSError: If the artifact cannot be read.
            ValueError: If the artifact size, JSON shape or format is invalid.
        """
        with path.open("rb") as source:
            raw = source.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError("Model artifact exceeds the supported size.")
        record = json.loads(raw)
        fields = {"format", "family", "weights", "fingerprint"}
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("The model artifact schema is invalid.")
        if record["format"] != "quantile-scorer-v1":
            raise ValueError("The model artifact schema version is unsupported.")
        return record


@dataclass(frozen=True, slots=True)
class LinearRule:
    """Provide a declared cheap linear comparator on the identical contextual features.

    Responsibilities:
    - Separate nonlinear learning from the value of feature engineering and score scale.

    Attributes:
        family (str): Feature-schema family.
        coefficients (tuple[float, ...]): Exactly four fixed finite feature coefficients.
    """

    family: str
    coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        """Freeze and validate a bounded linear scoring rule.

        Raises:
            ValueError: If family or feature coefficients are invalid.
        """
        values = tuple(float(value) for value in self.coefficients)
        if self.family not in ("tour", "cover"):
            raise ValueError("The linear-rule family is unsupported.")
        if len(values) != 4 or not np.isfinite(values).all():
            raise ValueError("The linear rule needs four finite coefficients.")
        object.__setattr__(self, "coefficients", values)

    @property
    def fingerprint(self) -> str:
        """Identify the exact family and linear coefficients.

        Returns:
            str: Hexadecimal content digest for this control.
        """
        data = json.dumps(("linear-v1", self.family, self.coefficients))
        return hashlib.sha256(data.encode()).hexdigest()

    def scores(self, features: np.ndarray) -> np.ndarray:
        """Apply the fixed linear rule with the same logit bound as the neural comparator.

        Args:
            features (np.ndarray): Finite four-feature legal-action rows.

        Returns:
            np.ndarray: One bounded score per action.

        Raises:
            ValueError: If features or resulting scores are malformed or nonfinite.
        """
        rows = np.asarray(features, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != 4 or not np.isfinite(rows).all():
            raise ValueError("Action features must be a finite matrix with four columns.")
        with np.errstate(over="ignore", invalid="ignore"):
            values = rows @ np.asarray(self.coefficients)
        if not np.isfinite(values).all():
            raise ValueError("The linear rule produced nonfinite scores.")
        return np.clip(values, -10.0, 10.0)
