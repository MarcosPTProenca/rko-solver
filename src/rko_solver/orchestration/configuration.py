"""Configuration parsing and the metaheuristic registry.

Objective:
- Turn packaged or user-supplied TOML into validated, typed run settings.
- Keep one registry that maps canonical algorithm names to their constructors.

Technical approach:
- Resolve display aliases, reject unsupported keys and validate scalar domains.
- Accept either the readable `adaptation` name or the legacy integer `control`.

Flow (pseudocode):
1. Read the required `run` and `active_metaheuristics` tables.
2. Reject unsupported keys and normalize the adaptation mode.
3. Validate pool capacity, algorithm names and duplicates.
4. Return an immutable RunConfig for the orchestration layer.

Important criteria:
- Missing or contradictory settings raise ValueError instead of defaulting silently.
- Algorithm names are canonicalized before duplicate and membership checks.

Inputs/Outputs:
- Input: the parsed TOML mapping.
- Output: RunConfig with mode, pool capacity and active algorithm names.
"""

from __future__ import annotations

from ..metaheuristics.brkga import BRKGA
from ..metaheuristics.brkga_cs import BRKGA_CS
from ..metaheuristics.ga import GeneticAlgorithm
from ..metaheuristics.grasp import GRASP
from ..metaheuristics.ils import ILS
from ..metaheuristics.ipr import IPR
from ..metaheuristics.lns import LNS
from ..metaheuristics.multi_start import MultiStart
from ..metaheuristics.pso import PSO
from ..metaheuristics.sa import SA
from ..metaheuristics.vns import VNS
from .adaptation import normalize_adaptation
from .types import MhSpec
from .types import RunConfig

MH_ALIASES: dict[str, str] = {
    "GeneticAlgorithm": "GA",
}

MH_SPECS: dict[str, MhSpec] = {
    "SA": MhSpec(
        constructor=SA,
        int_params=("sa_max",),
        float_params=("t0", "alpha", "beta_min", "beta_max"),
        supports_control=True,
    ),
    "MultiStart": MhSpec(constructor=MultiStart),
    "ILS": MhSpec(
        constructor=ILS,
        float_params=("beta_min", "beta_max"),
        supports_control=True,
    ),
    "VNS": MhSpec(
        constructor=VNS,
        int_params=("k_max",),
        float_params=("beta_min",),
        supports_control=True,
    ),
    "LNS": MhSpec(
        constructor=LNS,
        float_params=("beta_min", "beta_max", "t0", "alpha_lns"),
        supports_control=True,
    ),
    "GRASP": MhSpec(
        constructor=GRASP,
        float_params=("hs", "he"),
        supports_control=True,
    ),
    "BRKGA": MhSpec(
        constructor=BRKGA,
        int_params=("p",),
        float_params=("pe", "pm", "rhoe"),
        supports_control=True,
    ),
    "GA": MhSpec(
        constructor=GeneticAlgorithm,
        int_params=("size_pop",),
        float_params=("prob_cros", "prob_mut"),
        supports_control=True,
    ),
    "PSO": MhSpec(
        constructor=PSO,
        int_params=("p_size",),
        float_params=("c1", "c2", "w"),
        supports_control=True,
    ),
    "BRKGA_CS": MhSpec(
        constructor=BRKGA_CS,
        int_params=("p",),
        float_params=("pe", "pm", "rhoe"),
        supports_control=True,
    ),
    "IPR": MhSpec(constructor=IPR),
}


def canonical_mh_name(name: str) -> str:
    """Return the canonical metaheuristic name (resolving aliases)."""
    normalized = name.strip()
    return MH_ALIASES.get(normalized, normalized)


def parse_run_config(cfg: dict[str, object]) -> RunConfig:
    """Validate and normalize run configuration loaded from ``config.toml``."""
    try:
        run_section = cfg["run"]
        active_section = cfg["active_metaheuristics"]
    except KeyError as exc:
        raise ValueError("Invalid config.toml: missing required sections.") from exc

    if not isinstance(run_section, dict) or not isinstance(active_section, dict):
        raise ValueError(
            "Invalid config.toml: sections 'run' and 'active_metaheuristics' must be tables."
        )

    unsupported = set(run_section) - {"control", "adaptation", "pool_size", "prefill_pool"}
    if unsupported:
        raise ValueError(f"Unsupported run settings: {', '.join(sorted(unsupported))}")

    try:
        control = normalize_adaptation(run_section.get("adaptation"), run_section.get("control"))
        pool_size = run_section["pool_size"]
        active_raw = active_section["list"]
    except KeyError as exc:
        raise ValueError("Invalid config.toml: required keys are missing.") from exc

    if control is None:
        raise ValueError(
            "Invalid config.toml: 'run.adaptation' must be 'fixed' or 'adaptive' "
            "(or the legacy 'run.control' 0/1)."
        )
    if type(pool_size) is not int or pool_size <= 0:
        raise ValueError("Invalid config.toml: 'run.pool_size' must be > 0.")
    if not isinstance(active_raw, list) or not active_raw:
        raise ValueError(
            "Invalid config.toml: 'active_metaheuristics.list' must be a non-empty list."
        )

    if any(not isinstance(name, str) for name in active_raw):
        raise ValueError("Metaheuristic names must be strings.")
    prefill_pool = run_section.get("prefill_pool", False)
    if type(prefill_pool) is not bool:
        raise ValueError("Invalid config.toml: 'run.prefill_pool' must be a boolean.")
    active = tuple(canonical_mh_name(name) for name in active_raw)
    if len(set(active)) != len(active):
        raise ValueError("Duplicate metaheuristic names are not supported.")
    unknown = sorted(name for name in active if name not in MH_SPECS)
    if unknown:
        raise ValueError(f"Unknown metaheuristic names in config.toml: {', '.join(unknown)}")

    return RunConfig(
        control=control,
        pool_size=pool_size,
        active_metaheuristics=active,
        prefill_pool=prefill_pool,
    )
