"""Trust-boundary and budget regressions for decoder artifacts and independent instances.

Objective:
- Verify campaign resource limits and artifact or instance trust-boundary validation.
Technical approach:
- Inject costly runner evidence and mutate exported metadata or supply malformed instances.
Flow (pseudocode):
1. Calibrate a campaign with conservatively expensive job records.
2. Alter exported evidence or runtime requirements before loading artifacts.
3. Assert budget stops and rejection of incompatible or malformed external data.
Important criteria:
- Calibration cannot overspend; validation fails without importing generated source.
Inputs/Outputs:
- Input: temporary campaign and export paths, runner records and malformed instance mappings.
- Output: assertions on resource charges, integrity failures and schema validation.
"""

import json
import pytest
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve.campaign import BudgetExpired
from rko_solver.tests.experimental.test_self_evolve_campaign import FixtureRunner
from rko_solver.tests.experimental.test_self_evolve_campaign import make_campaign
from typing import Any


class ExpensiveRunner(FixtureRunner):
    """Report conservative high job usage to exercise the public hard-budget path.

    Responsibilities:
    - Return valid evidence with deliberately expensive declared resource consumption.

    Attributes:
        image (str): Inherited immutable fixture runtime identity.
    """

    def run(
        self, artifact: DecoderArtifact, problem: Problem, *, mode: str, seed: int, seconds: float
    ) -> dict[str, Any]:
        """Return a valid witness while reporting expensive externally bounded work.

        Args:
            artifact (DecoderArtifact): Unexecuted artifact identity.
            problem (Problem): Frozen instance.
            mode (str): Evaluation mode.
            seed (int): Solver seed.
            seconds (float): Nominal inner allowance.

        Returns:
            dict[str, Any]: Valid evidence with a high charge.
        """
        result = super().run(artifact, problem, mode=mode, seed=seed, seconds=seconds)
        return {**result, "cpu_seconds": 50, "outer_wall_seconds": 50}


def test_baseline_budget_stops_before_overspending(tmp_path: Path) -> None:
    """A costly calibration cannot exceed its frozen allocation or request proposals.

    Args:
        tmp_path (Path): Temporary parent for the campaign with expensive calibration jobs.
    """
    campaign = make_campaign(tmp_path / "campaign")
    with pytest.raises(BudgetExpired, match="baseline"):
        campaign.calibrate(ExpensiveRunner())
    assert sum(e.get("charge", 0) for e in campaign.events()) <= 600
    assert not any(e["kind"] == "calibrated" for e in campaign.events())
    with pytest.raises(ValueError, match="Calibrate"):
        campaign.request()


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r", "mixed"])
@pytest.mark.parametrize("large", [False, True])
def test_artifact_preserves_exact_source_bytes(tmp_path: Path, newline: str, large: bool) -> None:
    """Source identities survive export/load without platform newline translation.

    Args:
        tmp_path (Path): New artifact destination outside the repository.
        newline (str): Original newline convention, including mixed line endings.
        large (bool): Whether source bytes approach the maximum artifact size.
    """
    padding = "x" * (32000 if large else 1)
    source = f'def decode(keys, instance):\n    """{padding}"""\n    return []\n'
    source = (
        source.replace("\n", "\r\n", 1) if newline == "mixed" else source.replace("\n", newline)
    )
    artifact = DecoderArtifact.create("knapsack", source)
    output = tmp_path / "artifact"
    artifact.export(output)
    assert (output / "decoder.py").read_bytes() == source.encode("utf-8")
    loaded = DecoderArtifact.load(output)
    assert loaded.source == source
    assert loaded.digest == artifact.digest


def test_exported_evidence_is_hash_bound(tmp_path: Path) -> None:
    """Modified evidence is rejected even when source bytes remain unchanged.

    Args:
        tmp_path (Path): Temporary parent for the campaign and export with altered evidence.
    """
    campaign = make_campaign(tmp_path / "campaign")
    campaign.calibrate(FixtureRunner())
    output = tmp_path / "export"
    campaign.export(campaign.spec["seed"], output)
    (output / "evidence.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence hash"):
        DecoderArtifact.load(output)


def test_incompatible_runtime_and_schema_fail_without_import(tmp_path: Path) -> None:
    """Metadata cannot request an unsupported runtime or silently change the decoder contract.

    Args:
        tmp_path (Path): Temporary parent for the export with incompatible runtime metadata.
    """
    output = tmp_path / "export"
    DecoderArtifact.create("knapsack", Problem.seed_source("knapsack")).export(output)
    manifest = output / "manifest.json"
    data = json.loads(manifest.read_text())
    data["runtime"]["python"] = ">=3.99"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="runtime"):
        DecoderArtifact.load(output)


@pytest.mark.parametrize(
    "data",
    [
        {"weights": 3, "profits": [3], "capacity": 1},
        {"weights": [True], "profits": [3], "capacity": 1},
        {"weights": [2], "profits": [3], "capacity": []},
    ],
)
def test_malformed_instance_fields_are_rejected(data: dict[str, object]) -> None:
    """Malformed external data must fail before any generated source executes.

    Args:
        data (dict[str, object]): Parametrized knapsack instance with an invalid field value.
    """
    with pytest.raises(ValueError):
        Problem("knapsack", data)
