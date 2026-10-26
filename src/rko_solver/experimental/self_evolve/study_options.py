"""Typed frozen experiment declaration.

Objective:
- Retain explicit caller keyword names, required inputs and original defaults.
Technical approach:
- Use TypedDict/Unpack at the public boundary and a typed settings dataclass.
- Validate sources/runtime, randomization, numeric limits, external power and prior identities.
Flow (pseudocode):
1. Normalize the typed keyword declaration without altering input values.
2. Short-circuit dimension checks in the original evaluation order.
3. Require valid external power declarations and canonical prior-instance identities.
Important criteria:
- Preserve specification schema, operation order, reservations and inference gates.
Inputs/Outputs:
- Input: explicit artifact, instance, runtime, resource and analysis keyword settings.
- Output: a typed experiment declaration or the original validation exception.
"""

from __future__ import annotations

import math
from .artifacts import FAMILIES
from .artifacts import DecoderArtifact
from .problems import Problem
from .sandbox import JOB_GRACE_SECONDS
from dataclasses import dataclass
from typing import Any
from typing import NotRequired
from typing import TypedDict


class StudyOptions(TypedDict):
    """Type the cohesive experiment declaration with original caller keywords.

    Responsibilities:
    - Type the cohesive experiment declaration with original caller keywords.

    Attributes:
        artifacts (dict): Frozen family sources.
        instances (dict): Fresh test instances.
        seeds (list[int]): Fixed paired solver panel.
        image (str): Immutable runtime identity.
        seconds (float): Common job allowance.
        cpu_budget (float): Conservative CPU envelope.
        elapsed_budget (float): Wall-time envelope.
        minimum_effect (float): Predeclared inference threshold.
        prior_instance_hashes (list[str]): Previously exposed identities.
        execution_seed (NotRequired[int]): Paired-block and within-block shuffle seed.
        claim (NotRequired[str]): Exploratory or confirmatory evidence label.
        power_plan (NotRequired[dict | None]): External power justification.
        allow_runtime_rebind (NotRequired[bool]): Explicit unchanged-source rebinding approval.
        bootstrap_samples (NotRequired[int]): Frozen instance-bootstrap count.
        permutation_samples (NotRequired[int]): Frozen Monte Carlo sign-flip count.
        strata (NotRequired[dict | None]): Fixed sampling labels.
        transport (NotRequired[str]): Per-job or persistent fresh-process transport.
    """

    artifacts: dict[str, dict[str, DecoderArtifact]]
    instances: dict[str, list[Problem]]
    prior_instance_hashes: list[str]
    image: str
    seeds: list[int]
    seconds: float
    cpu_budget: float
    elapsed_budget: float
    minimum_effect: float
    execution_seed: NotRequired[int]
    claim: NotRequired[str]
    power_plan: NotRequired[dict[str, Any] | None]
    allow_runtime_rebind: NotRequired[bool]
    bootstrap_samples: NotRequired[int]
    permutation_samples: NotRequired[int]
    strata: NotRequired[dict[str, list[str]] | None]
    transport: NotRequired[str]


@dataclass
class StudySettings:
    """Normalize and validate one cohesive experiment declaration.

    Responsibilities:
    - Normalize and validate one cohesive experiment declaration.

    Attributes:
        artifacts (dict): Frozen seed, first-shot and candidate sources per family.
        instances (dict): Fresh independent test instances per family.
        prior_instance_hashes (list[str]): Previously exposed instance identities.
        image (str): Immutable local runtime identity.
        seeds (list[int]): Distinct solver seeds nested within instances.
        seconds (float): Common bounded job allowance.
        cpu_budget (float): Conservative resource envelope.
        elapsed_budget (float): Campaign wall-time envelope.
        minimum_effect (float): Required absolute bounded-loss reduction.
        execution_seed (int): Paired-block and within-block shuffle seed.
        claim (str): Exploratory or confirmatory evidence label.
        power_plan (dict | None): External power justification.
        allow_runtime_rebind (bool): Explicit unchanged-source rebinding approval.
        bootstrap_samples (int): Frozen instance-bootstrap count.
        permutation_samples (int): Frozen Monte Carlo sign-flip count.
        strata (dict | None): Fixed sampling labels.
        transport (str): Per-job or persistent fresh-process transport.
    """

    artifacts: dict[str, dict[str, DecoderArtifact]]
    instances: dict[str, list[Problem]]
    prior_instance_hashes: list[str]
    image: str
    seeds: list[int]
    seconds: float
    cpu_budget: float
    elapsed_budget: float
    minimum_effect: float
    execution_seed: int = 42
    claim: str = "exploratory"
    power_plan: dict[str, Any] | None = None
    allow_runtime_rebind: bool = False
    bootstrap_samples: int = 10000
    permutation_samples: int = 9999
    strata: dict[str, list[str]] | None = None
    transport: str = "per_job"

    def _invalid_sources_runtime(self) -> bool:
        """Check exact family coverage and immutable local image syntax.

        Returns:
            bool: Whether any original constraint rejects the declaration.
        """
        return (
            set(self.artifacts) != set(FAMILIES)
            or set(self.instances) != set(FAMILIES)
            or (not isinstance(self.image, str))
            or (len(self.image) != 71)
            or (not self.image.startswith("sha256:"))
            or any(c not in "0123456789abcdef" for c in self.image[7:])
        )

    def _invalid_randomization(self) -> bool:
        """Check distinct bounded solver seeds and the execution shuffle seed.

        Returns:
            bool: Whether any original constraint rejects the declaration.
        """
        return (
            not 2 <= len(self.seeds) <= 100
            or any(type(s) is not int or not 0 <= s < 2**32 for s in self.seeds)
            or len(set(self.seeds)) != len(self.seeds)
            or (type(self.execution_seed) is not int)
            or (not 0 <= self.execution_seed < 2**32)
        )

    def _invalid_numeric_settings(self) -> bool:
        """Check numeric types before validating finite job allowances and effect sizes.

        Returns:
            bool: Whether any original constraint rejects the declaration.
        """
        return (
            any(
                type(value) not in (int, float)
                for value in (
                    self.seconds,
                    self.cpu_budget,
                    self.elapsed_budget,
                    self.minimum_effect,
                )
            )
            or not math.isfinite(self.seconds)
            or (not 0 < self.seconds <= 30)
            or (not math.isfinite(self.minimum_effect))
            or (not 0 < self.minimum_effect <= 1)
        )

    def _invalid_envelope_claim(self) -> bool:
        """Check budget admission, evidence scope and declared transport.

        Returns:
            bool: Whether any original constraint rejects the declaration.
        """
        return (
            any(
                not math.isfinite(b) or not self.seconds + JOB_GRACE_SECONDS < b <= 604800
                for b in (self.cpu_budget, self.elapsed_budget)
            )
            or self.claim not in {"exploratory", "confirmatory"}
            or self.transport not in {"per_job", "persistent_exec"}
        )

    def validate(self) -> None:
        """Reject invalid dimensions, external power declarations and prior identities.

        Raises:
            ValueError: If any frozen input fails its original validation.
        """
        if (
            self._invalid_sources_runtime()
            or self._invalid_randomization()
            or self._invalid_numeric_settings()
            or self._invalid_envelope_claim()
        ):
            raise ValueError(
                "Invalid frozen study dimensions, runtime, budgets or effect threshold."
            )
        self._validate_power()
        self._validate_prior_identities()

    def _validate_power(self) -> None:
        """Require an external power justification for confirmatory declarations.

        Raises:
            ValueError: If a confirmatory declaration lacks a valid power plan.
        """
        if self.claim == "confirmatory" and (
            not isinstance(self.power_plan, dict)
            or not isinstance(self.power_plan.get("method"), str)
            or (not self.power_plan["method"])
            or (len(self.power_plan["method"]) > 2000)
            or (type(self.power_plan.get("planned_power")) not in (int, float))
            or (not 0.8 <= self.power_plan["planned_power"] <= 1)
            or (type(self.power_plan.get("calibration_instances")) is not int)
            or (self.power_plan["calibration_instances"] < 10)
        ):
            raise ValueError("Confirmation requires an explicit externally justified power plan.")

    def _validate_prior_identities(self) -> None:
        """Require canonical SHA-256 identities for all prior exposures.

        Raises:
            ValueError: If a prior identity is not a lowercase SHA-256 digest.
        """
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
            for value in self.prior_instance_hashes
        ):
            raise ValueError("Prior instance identities must be lowercase SHA-256 hashes.")
