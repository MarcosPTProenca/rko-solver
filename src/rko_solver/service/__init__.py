"""Lazy public bindings for the independently optional RKO HTTP service.

Objective:
- Keep FastAPI/Pydantic/Uvicorn out of ordinary core imports.

Technical approach:
- Resolve explicit service exports only when requested by application code.

Flow (pseudocode):
1. Import core RKO without HTTP dependencies.
2. Resolve a requested service binding lazily.
3. Explain the optional extra if an HTTP dependency is missing.

Important criteria:
- Service adapter wiring does not change the ordinary three-function problem contract.

Inputs/Outputs:
- Input: explicit service export name.
- Output: trusted service API/type, or an actionable optional-install error.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .api import create_app
    from .models import ProblemDefinition
    from .models import ServiceAdapter
    from .models import ServiceLimits

__all__ = ["ProblemDefinition", "ServiceAdapter", "ServiceLimits", "create_app"]


def __getattr__(name: str) -> object:
    """Load explicit optional service bindings without changing core dependencies.

    Args:
        name (str): Requested public service export.

    Returns:
        object: Requested factory or typed service binding.

    Raises:
        AttributeError: If the export is unknown.
        ImportError: If the service extra is missing.
    """
    if name not in __all__:
        raise AttributeError(name)
    try:
        module = importlib.import_module(".api" if name == "create_app" else ".models", __name__)
    except ModuleNotFoundError as exc:
        if exc.name in ("fastapi", "pydantic", "starlette", "threadpoolctl"):
            raise ImportError(
                "Install rko-solver[service] to use the optional HTTP service."
            ) from exc
        raise
    return getattr(module, name)
