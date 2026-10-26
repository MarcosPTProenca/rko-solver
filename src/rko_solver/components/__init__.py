"""RKO framework components.

Objective:
- Group reusable primitives for local search, shaking, initialization, and adaptive control.

Technical approach:
- Organize functions and classes by responsibility in dedicated submodules.
- Provide a common base for integration with metaheuristics.

Flow (pseudocode):
1. Select a specialized component.
2. Apply it to a random-key solution.
3. Return the updated structure to the caller algorithm.

Important criteria:
- Reuse across different metaheuristics.
- Typed interfaces for composition.

Inputs/Outputs:
- Input: random-key vectors, decoder/cost callbacks, and control parameters.
- Output: transformed solutions, costs, and auxiliary states.
"""

from .adaptive_control.q_learning import QLearningController
from .adaptive_control.q_learning import QState
from .blending.blending import blending
from .initialization.initialization import create_initial_solution
from .initialization.initialization import create_pool_solutions
from .initialization.initialization import update_pool_solutions
from .local_search.farey_ls import FAREY_INTERVALS
from .local_search.farey_ls import farey_ls
from .local_search.mirror_ls import mirror_ls
from .local_search.nelder_mead_ls import nelder_mead_search
from .local_search.rvnd import rvnd
from .local_search.swap_ls import swap_ls
from .shaking.mirror import mirror_shake
from .shaking.neighbor_swap import neighbor_swap_shake
from .shaking.random import shake
from .shaking.swap import swap_shake
from .solution_pool import ElitePool
from .solution_pool import SolutionPool
from .solution_pool import SolutionPoolWriter

__all__ = [
    "QLearningController",
    "QState",
    "blending",
    "create_initial_solution",
    "create_pool_solutions",
    "update_pool_solutions",
    "farey_ls",
    "FAREY_INTERVALS",
    "mirror_ls",
    "nelder_mead_search",
    "rvnd",
    "swap_ls",
    "mirror_shake",
    "neighbor_swap_shake",
    "shake",
    "swap_shake",
    "ElitePool",
    "SolutionPool",
    "SolutionPoolWriter",
]
