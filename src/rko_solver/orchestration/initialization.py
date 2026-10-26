"""Bounded reference-style shared-pool initialization before search release.

Objective:
- Prefill only when explicitly requested, accounting for startup work separately.

Technical approach:
- Use an owned RNG branch and the ordinary isolated finite-objective boundary.
- Keep private slots for reference selection sorting and one cost-tie mutation pass.
- Publish final/partial slots and the best completed observation through the pool.

Flow (pseudocode):
1. Evaluate P random vectors using a separate cooperative startup allowance.
2. Selection-sort slots; perturb adjacent cost ties once, then re-sort if needed.
3. On timeout retain completed work; publish without another objective call.
4. Report actual seed, completed evaluations, elapsed time and admitted occupancy.

Important criteria:
- At most 2P-1 evaluations; no retry-until-unique loop or arbitrary callback reset.
- Selection-sort swap ordering follows C++ even for ties; RNG engines differ.
- Search allowances are fresh after startup; active callbacks cannot be interrupted.

Inputs/Outputs:
- Input: validated pool/callbacks, dimension, positive capacity/duration and seed source.
- Output: shared elites and an immutable startup report, or the original callback error.
"""

import math
import numpy as np
import time
from ..components.random_streams import RandomStreams
from ..components.solution_pool import ElitePool
from ..tracing import TraceConfig
from .search import SearchSession
from .search import _BudgetExpired
from .types import CostFn
from .types import DecoderFn
from .types import PoolInitializationMetadata
from typing import Literal


def initialize_pool(
    pool: ElitePool,
    decoder: DecoderFn,
    cost: CostFn,
    dimension: int,
    capacity: int,
    duration: float,
    random_state: int | None,
    trace: TraceConfig | None = None,
) -> PoolInitializationMetadata:
    """Fill reference slots safely without altering algorithm stream derivation.

    Args:
        pool (ElitePool): Empty shared pool, with no active search producers.
        decoder (DecoderFn): Original callback; payload mutations remain isolated.
        cost (CostFn): Original objective, required to return a finite value.
        dimension (int): Validated positive chromosome length.
        capacity (int): Validated positive slot count and pool capacity.
        duration (float): Independent positive cooperative startup allowance.
        random_state (int | None): Seed source; a reserved grandchild branch is used.
        trace (TraceConfig | None): Optional bounded original-decision recording.

    Returns:
        PoolInitializationMetadata: Actual completed startup work and admitted size.

    Raises:
        Exception: Original callback or pool-publication failure.
        ValueError: If an objective observation is nonfinite.
    """
    started = time.perf_counter()
    seed = int(np.random.SeedSequence(random_state, spawn_key=(0, 0)).generate_state(1)[0])
    streams = RandomStreams(seed)
    session = (
        SearchSession(decoder, cost)
        if trace is None
        else SearchSession(decoder, cost, trace=trace, trace_source="initial_pool")
    )
    session.deadline = started + duration
    slots: list[tuple[float, np.ndarray]] = []
    reason: Literal["completed", "time_limit"] = "completed"

    def sort_slots() -> None:
        """Apply the reference selection-sort swaps with cooperative outer checks.

        Raises:
            _BudgetExpired: When startup expires after a finite observation.
        """
        for index in range(len(slots) - 1):
            session._check()
            minimum = index
            for other in range(index + 1, len(slots)):
                if slots[other][0] < slots[minimum][0]:
                    minimum = other
            if minimum != index:
                slots[index], slots[minimum] = slots[minimum], slots[index]

    try:
        with streams.activate():
            for _ in range(capacity):
                keys = streams.numpy.random(dimension)
                value = session.evaluate(session.decode(keys))
                slots.append((value, keys))
            sort_slots()
            changed = False
            for index in range(len(slots) - 1, 0, -1):
                session._check()
                if slots[index][0] == slots[index - 1][0]:
                    keys = slots[index][1].copy()
                    for _ in range(math.ceil(0.2 * dimension)):
                        position = streams.python.randint(0, dimension - 1)
                        keys[position] = streams.python.random()
                    value = session.evaluate(session.decode(keys))
                    slots[index] = (value, keys)
                    changed = True
            if changed:
                sort_slots()
    except _BudgetExpired:
        reason = "time_limit"
    for value, keys in slots:
        pool.insert((value, keys.tolist()), "initial_pool")
    assert session.best is not None
    keys, value = session.best
    pool.insert((value, keys.tolist()), "initial_pool")
    finished = time.perf_counter()
    if finished >= session.deadline:
        reason = "time_limit"
    return PoolInitializationMetadata(
        random_state=seed,
        n_evaluations=session.n_evaluations,
        elapsed_seconds=finished - started,
        pool_size=pool.size,
        stop_reason=reason,
        time_limit=duration,
    )
