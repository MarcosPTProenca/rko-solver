"""Original-unit row-activity finite checks at the generated callback boundary.

Objective:
- Reject nonfinite row activities even when objective and adjusted violations are finite.

Technical approach:
- Inject a damaged numerical row state beneath the public immutable decoder boundary.

Flow (pseudocode):
1. Compile a valid finite bounded linear fixture.
2. Replace only its row-state arithmetic with a nonfinite activity.
3. Verify that public decoding rejects the incomplete numerical evidence.

Important criteria:
- Finite score and zero side violations cannot legitimize a nonfinite activity.

Inputs/Outputs:
- Input: valid numerical snapshot and isolated damaged row-state arithmetic.
- Output: explicit public evaluation failure without a completed payload.
"""

import numpy as np
import pytest
from rko_solver.modeling.linear import CompiledLinearModel
from rko_solver.modeling.linear import LinearRow
from rko_solver.modeling.linear import LinearVariable


def test_nonfinite_activity_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a nonfinite original row activity despite finite score and side arrays.

    Args:
        monkeypatch (pytest.MonkeyPatch): Isolated numerical-boundary substitution.
    """
    snapshot = CompiledLinearModel(
        (LinearVariable("x", "x", 0.0, 1.0),),
        (1.0,),
        (LinearRow("row", ((0, 1.0),), upper=1.0),),
    )

    def damaged_rows(model: CompiledLinearModel, values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Return a damaged activity without contaminating adjusted violation arrays.

        Args:
            model (CompiledLinearModel): Bound immutable snapshot, deliberately unused.
            values (np.ndarray): Original assignment, deliberately unused.

        Returns:
            tuple[np.ndarray, ...]: Nonfinite activity and otherwise finite zero row sides.
        """
        return np.full(1, np.nan), *(np.zeros(1) for _ in range(4))

    monkeypatch.setattr(CompiledLinearModel, "_row_state", damaged_rows)
    with pytest.raises(ValueError, match="nonfinite arithmetic"):
        snapshot.decode(np.array([0.5]))
