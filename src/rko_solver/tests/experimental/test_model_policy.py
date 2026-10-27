"""Exact model promotion panel regressions independent of optimizer throughput.

Objective:
- Require strict complete primary-loss improvement and per-case matched-job feasibility.
Technical approach:
- Supply explicit synthetic completed RKO panels to the model-only policy.
Flow (pseudocode):
1. Declare exactly three cases and two frozen solver seeds.
2. Compare complete feasible wins and deliberate missing/tied/worse-case panels.
3. Reject any false evolved-win label without executing source or a study.
Important criteria:
- Feasible callback counts are not feasible matched-job rates.
Inputs/Outputs:
- Input: clearly synthetic durable evaluation records.
- Output: exact fail-closed policy decisions.
"""

import copy
import pytest
from rko_solver.experimental.self_evolve.model_campaign import ModelCampaignPolicy


def _panel() -> tuple[dict, list[dict]]:
    """Make a complete synthetic strict-improvement comparison.

    Returns:
        tuple[dict, list[dict]]: Minimal frozen coordinates and completed records.
    """
    spec = {
        "family": "linear_model",
        "seed": "seed",
        "instances": [0, 1, 2],
        "solver_seeds": [0, 1],
    }
    records = [
        {
            "kind": "evaluation",
            "mode": "rko",
            "stage": "development",
            "candidate": candidate,
            "partition": partition,
            "instance": index,
            "seed": seed,
            "status": "ok",
            "feasible": True,
            "loss": loss,
        }
        for candidate, partition, loss in (("seed", "baseline", 0.3), ("child", "candidate", 0.1))
        for index in range(3)
        for seed in (0, 1)
    ]
    return spec, records


def test_complete_strict_model_improvement_is_eligible() -> None:
    """A real complete mean-loss win with nonworse per-case success meets the policy."""
    spec, records = _panel()
    ModelCampaignPolicy.promote(spec, records, "child")


@pytest.mark.parametrize("kind", ["tie", "missing", "duplicate", "no_feasible", "worse_case"])
def test_model_panel_rejects_false_win_labels(kind: str) -> None:
    """Panel shape, strictness and per-case success cannot be replaced by a global headline.

    Args:
        kind (str): Deliberate complete-panel promotion violation.
    """
    spec, records = _panel()
    if kind == "tie":
        for row in records:
            row["loss"] = 0.3
    elif kind == "missing":
        records.pop()
    elif kind == "duplicate":
        records.append(copy.deepcopy(records[-1]))
    elif kind == "no_feasible":
        for row in records[6:]:
            row.update(feasible=False, loss=1.0)
    else:
        for row in records:
            row["feasible"] = (row["instance"] == 0) == (row["partition"] == "baseline")
            row["loss"] = 0.0 if row["feasible"] else 1.0
    with pytest.raises(ValueError):
        ModelCampaignPolicy.promote(spec, records, "child")
