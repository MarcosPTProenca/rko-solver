"""Portable Exact Knapsack Oracle.

Objective:
- Compute exact 0/1 Knapsack profits for benchmark validation.

Technical approach:
- Maintain a NumPy dynamic-programming table indexed by capacity.
- Process each item with a reverse-capacity recurrence encoded as vector slices.

Flow (pseudocode):
1. Allocate a zero-filled profit table through the capacity limit.
2. For each item, update every feasible capacity from the prior item state.
3. Return the table value at the full capacity.

Important criteria:
- Each item is selected at most once.
- Runtime is O(item_count * capacity); memory is O(capacity).

Inputs/Outputs:
- Input: KpInstance containing integer capacity, benefits, and weights.
- Output: integer optimal profit under the capacity constraint.
"""

import numpy as np
from rko_solver.examples.run_kp_cpp_instances import KpInstance


def exact_profit(instance: KpInstance) -> int:
    """Compute the exact 0/1 optimum with a capacity-indexed DP table.

    Args:
        instance (KpInstance): Integer-weight Knapsack data.

    Returns:
        int: Optimal feasible profit.
    """
    values = np.zeros(instance.capacity + 1, dtype=np.int64)
    for profit, weight in zip(instance.benefits, instance.weights, strict=True):
        if weight <= instance.capacity:
            values[weight:] = np.maximum(values[weight:], values[:-weight] + profit)
    return int(values[-1])
