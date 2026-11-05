"""Validated TOML and programmatic metaheuristic construction.

Objective:
- Build fresh algorithms with partial offline overrides and consistent validation.

Technical approach:
- Normalize selected algorithm aliases and snapshot named scalar mappings.
- Merge overrides into freshly loaded defaults before constructor validation.

Flow (pseudocode):
1. Validate override names, scalar types and offline control mode.
2. Read TOML defaults and merge owned override values.
3. Construct an algorithm using the existing scalar and joint-domain checks.

Important criteria:
- Overrides never write TOMLs or silently select additional algorithms.
- Adaptive grids remain TOML-configured rather than replacing requested scalars.

Inputs/Outputs:
- Input: configured portfolio, partial numeric mappings and problem callbacks.
- Output: owned normalized overrides or a fully validated algorithm.
"""

from __future__ import annotations

import math
from ..components.solution_pool import ElitePool
from ..metaheuristics.base import MetaheuristicBase
from ..parameters.loader import load_offline
from ..parameters.loader import load_online_grid
from .configuration import MH_SPECS
from .configuration import canonical_mh_name
from .types import CostFn
from .types import DecoderFn
from .types import Hyperparameters
from .types import SearchMetadata
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def _as_int(params: dict[str, Any], key: str, mh_name: str) -> int:
    """Read an integral numeric parameter without truncating fractional values.

    Args:
        params (dict[str, Any]): Parsed offline table.
        key (str): Required parameter name.
        mh_name (str): Algorithm identifier for diagnostics.

    Returns:
        int: Exact integral numeric value.

    Raises:
        ValueError: If the value is missing, nonfinite, fractional or nonnumeric.
    """
    if key in params and type(params[key]) is int:
        return params[key]
    value = _as_float(params, key, mh_name)
    if not value.is_integer():
        raise ValueError(f"{mh_name}: '{key}' must be integral; fractional values are invalid.")
    return int(value)


def _as_float(params: dict[str, Any], key: str, mh_name: str) -> float:
    """Read a finite TOML number without coercing strings or booleans.

    Args:
        params (dict[str, Any]): Parsed offline table.
        key (str): Required parameter name.
        mh_name (str): Algorithm identifier for diagnostics.

    Returns:
        float: Finite numeric parameter.

    Raises:
        ValueError: If the value is missing, nonfinite or nonnumeric.
    """
    try:
        value = params[key]
    except KeyError as exc:
        raise ValueError(f"Missing offline parameter '{key}' for '{mh_name}'.") from exc
    message = f"{mh_name}: '{key}' must be a finite number, not a string or boolean."
    if type(value) not in (int, float):
        raise ValueError(message)
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(message) from exc
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def normalize_hyperparameters(
    hyperparameters: Hyperparameters | None,
    algorithms: tuple[str, ...],
    control: int,
) -> dict[str, dict[str, int | float]]:
    """Snapshot partial offline scalars using canonical selected algorithm names.

    Args:
        hyperparameters (Hyperparameters | None): Algorithm-to-scalar mappings.
        algorithms (tuple[str, ...]): Selected canonical algorithm names.
        control (int): Configured offline or adaptive mode.

    Returns:
        dict[str, dict[str, int | float]]: Detached, plain, pickleable dictionaries.

    Raises:
        ValueError: If mappings, names, mode or numeric scalar values are invalid.
    """
    if hyperparameters is None:
        return {}
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("'hyperparameters' must map algorithm names to parameter mappings.")
    if hyperparameters and control != 0:
        raise ValueError("Hyperparameter overrides require control=0; use TOML for online grids.")
    result = {}
    for name, values in hyperparameters.items():
        if not isinstance(name, str):
            raise ValueError("Hyperparameter algorithm names must be strings.")
        canonical = canonical_mh_name(name)
        if canonical not in algorithms:
            raise ValueError(f"Hyperparameters target an unselected algorithm: '{name}'.")
        if canonical in result:
            raise ValueError(f"Duplicate hyperparameter mappings for '{canonical}'.")
        if not isinstance(values, Mapping):
            raise ValueError(f"{canonical}: hyperparameters must be a named scalar mapping.")
        spec = MH_SPECS[canonical]
        if any(key not in spec.int_params + spec.float_params for key in values):
            raise ValueError(
                f"{canonical}: hyperparameter keys must come from "
                f"{spec.int_params + spec.float_params}."
            )
        snapshot = dict(values)
        result[canonical] = {
            key: _as_int(snapshot, key, canonical)
            if key in spec.int_params
            else _as_float(snapshot, key, canonical)
            for key in snapshot
        }
    return result


def build_metaheuristic(
    mh_name: str,
    pool: ElitePool,
    decoder: DecoderFn,
    cost: CostFn,
    tam_solution: int,
    control: int,
    params_dir: Path | None,
    hyperparameters: Mapping[str, int | float] | None = None,
) -> MetaheuristicBase:
    """Construct a configured algorithm after merging optional offline scalars.

    Args:
        mh_name (str): Registered algorithm name or alias.
        pool (ElitePool): Shared elite context.
        decoder (DecoderFn): Problem decoder.
        cost (CostFn): Objective callback.
        tam_solution (int): Random-key dimension.
        control (int): Offline or adaptive control mode.
        params_dir (Path | None): Optional TOML directory.
        hyperparameters (Mapping[str, int | float] | None): Partial offline overrides.

    Returns:
        MetaheuristicBase: Fresh configured algorithm.

    Raises:
        ValueError: If names, overrides or constructor domains are invalid.
        OSError: If required parameter files cannot be read.
    """
    canonical_name = canonical_mh_name(mh_name)
    spec = MH_SPECS.get(canonical_name)
    if spec is None:
        raise ValueError(f"Unknown metaheuristic name: '{mh_name}'")

    overrides = normalize_hyperparameters(
        {canonical_name: hyperparameters} if hyperparameters is not None else None,
        (canonical_name,),
        control,
    )
    offline = load_offline(canonical_name, params_dir)
    offline.update(overrides.get(canonical_name, {}))
    online_grid = load_online_grid(canonical_name, params_dir) if control == 1 else []

    kwargs: dict[str, Any] = {
        "solution_pool": pool,
        "decoder": decoder,
        "cost": cost,
        "tam_solution": tam_solution,
    }

    for key in spec.int_params:
        kwargs[key] = _as_int(offline, key, canonical_name)
    for key in spec.float_params:
        kwargs[key] = _as_float(offline, key, canonical_name)

    if spec.supports_control:
        kwargs["control"] = control
        if online_grid:
            kwargs["parameters"] = online_grid

    algorithm = spec.constructor(**kwargs)
    algorithm._search_metadata = SearchMetadata(
        algorithm=canonical_name,
        control=control if spec.supports_control else 0,
        parameters=tuple((key, kwargs[key]) for key in spec.int_params + spec.float_params),
        parameter_grid=tuple(tuple(axis) for axis in algorithm.parameters)
        if spec.supports_control and control == 1
        else (),
    )
    return algorithm
