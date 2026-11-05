"""Confirmation-cohort selection contracts.

Objective:
- Keep confirmation inputs separate without changing historical gate inputs.

Technical approach:
- Exhaust all supported counts and verify historical selection fingerprints.
- Exercise generated Rastrigin inputs without executing a search.

Flow (pseudocode):
1. Reserve calibration and historical test cohorts.
2. Compare confirmation counts with the reserved inputs and each other.
3. Check input fingerprints and command-line validation.

Important criteria:
- Confirmation never reuses reserved KPLIB IDs; old defaults stay unchanged.

Inputs/Outputs:
- Input: temporary cache directories and fixed selection counts.
- Output: deterministic assertions without network access.
"""

import hashlib
import json
import pytest
from benchmarks.engines import gate
from benchmarks.engines.gate_instances import GateInstances
from pathlib import Path


def test_confirmation_is_disjoint_stable_and_bounded(tmp_path: Path) -> None:
    """Exhaust confirmation counts against all reserved historical cohorts.

    Args:
        tmp_path (Path): Isolated input cache root.
    """
    calibration = set(GateInstances(tmp_path, "calibration", 100).knapsack_ids())
    historical = {
        pair
        for count in range(1, 101)
        for pair in GateInstances(tmp_path, "test", count).knapsack_ids()
    }
    full = GateInstances(tmp_path, "confirmation", 100).knapsack_ids()
    assert len(full) == len(set(full)) == 100
    assert not set(full) & (calibration | historical)
    for count in range(1, 101):
        selected = GateInstances(tmp_path, "confirmation", count).knapsack_ids()
        assert len(selected) == len(set(selected)) == count
        assert selected == full[:count]
        assert selected == GateInstances(tmp_path, "confirmation", count).knapsack_ids()
    with pytest.raises(ValueError, match="100"):
        GateInstances(tmp_path, "confirmation", 101)


@pytest.mark.parametrize(
    ("split", "count", "digest"),
    [
        ("calibration", 15, "03c6f5101d7878c375ddf894520c018b65a1cba40d6cefcb08de6aa06b012fd3"),
        ("calibration", 100, "6cc74cd6f143d7b11644e6d21443f4c5e1a8b5497cf014bfb7da49b6d4d2c325"),
        ("test", 15, "eb243433ab96ef7f525a86c6c6bdb26753757188a9666d8fb9d171a9dc8e0d05"),
        ("test", 100, "fd05a7a99f53612f695638d97fc3bc92a12e3f4d4f58310ab7afadc49d171fbe"),
    ],
)
def test_historical_knapsack_fingerprints(split: str, count: int, digest: str) -> None:
    """Preserve selections measured on the merged gate implementation.

    Args:
        split (str): Historical cohort name.
        count (int): Number of selected inputs.
        digest (str): Original serialized-selection SHA-256 digest.
    """
    selected = GateInstances(Path("."), split, count).knapsack_ids()
    assert hashlib.sha256(json.dumps(selected).encode()).hexdigest() == digest


def test_confirmation_rastrigin_inputs_are_distinct(tmp_path: Path) -> None:
    """Keep generated shifts repeatable and distinct from both old splits.

    Args:
        tmp_path (Path): Isolated generated-input cache root.
    """
    historical_hashes = {
        100: (
            "dfe21b70bf70c0409f07b463b01b3670aa975490303cd85ed82b2b6e21398fb4",
            "02a075ee3846a47b18a40d003373e1f4b13e7e390cdc851a5adcb3a732bdb51f",
        ),
        300: (
            "c30c50bcad98de9f9a1cb877e46b82eeaa27df59e35ee1340d11f7e10709fba3",
            "8dbe73773657c8fade05faa67ec0216c50751d933570822c97cbb7723ccac358",
        ),
    }
    for dimension in (100, 300):
        instances = [
            next(GateInstances(tmp_path, split, 1).build(f"rastrigin-{dimension}"))
            for split in ("calibration", "test", "confirmation")
        ]
        assert tuple(instance.sha256 for instance in instances[:2]) == historical_hashes[dimension]
        assert len({instance.name for instance in instances}) == 3
        assert len({instance.sha256 for instance in instances}) == 3
        repeated = next(GateInstances(tmp_path, "confirmation", 1).build(f"rastrigin-{dimension}"))
        assert repeated.sha256 == instances[-1].sha256
        assert repeated.name.startswith("confirmation-")


def test_confirmation_cli_accepts_new_split(tmp_path: Path) -> None:
    """Accept the additive split through the real argument parser.

    Args:
        tmp_path (Path): Destination for the unused trial output.
    """
    args = gate._build_parser().parse_args(
        ["--split", "confirmation", "--output", str(tmp_path / "rows.jsonl")]
    )
    assert args.split == "confirmation"
    defaults = gate._build_parser().parse_args(["--output", str(tmp_path / "default.jsonl")])
    assert defaults.split == "test"
