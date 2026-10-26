"""Reference ASGI entry point for the bounded private RKO scheduling service.

Objective:
- Bind trusted adapter code and deployment-injected authentication without client execution.

Technical approach:
- Build the optional app factory with one active job and independent report support.
- Leave numerical callbacks unchanged in the application adapter.

Flow (pseudocode):
1. Import the installed developer-owned adapter/schema.
2. Require the deployment secret from the environment, never a committed default.
3. Let the maintained factory own process supervision and ephemeral retention.

Important criteria:
- Launch exactly one HTTP worker inside the supplied resource-limited Linux container.
- Report support is independent of service/core dependencies and explicitly enabled here.

Inputs/Outputs:
- Input: trusted adapter module and required RKO_SERVICE_API_KEY deployment environment.
- Output: ASGI application; no optimization occurs during module import.
"""

import os
from adapter import InstanceInput
from adapter import build_problem
from rko_solver.service import ServiceAdapter
from rko_solver.service import ServiceLimits
from rko_solver.service import create_app

app = create_app(
    ServiceAdapter(InstanceInput, build_problem),
    api_key=os.environ["RKO_SERVICE_API_KEY"],
    limits=ServiceLimits(allow_reports=True),
)
