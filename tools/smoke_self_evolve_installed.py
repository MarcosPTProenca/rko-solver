"""Installed-package smoke for non-executing experimental decoder artifact contracts.

Objective:
- Verify wheel/sdist include the opt-in namespace and offline artifact format.
Technical approach:
- Construct a known trusted seed, export/load it and check a hand-worked objective.
Flow (pseudocode):
1. Import only from the installed package.
2. Round-trip a temporary artifact without executing source.
3. Verify independent problem semantics.
Important criteria:
- This smoke needs neither Podman nor an LLM and does not certify live isolation.
Inputs/Outputs:
- Input: the package installed in an isolated environment.
- Output: assertion failure on packaging or artifact regressions.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from rko_solver.experimental.self_evolve import DecoderArtifact
from rko_solver.experimental.self_evolve import Problem
from rko_solver.experimental.self_evolve.analysis import PairedAnalysis
from rko_solver.experimental.self_evolve.campaign import Campaign
from rko_solver.experimental.self_evolve.pi_config import PiConfig
from rko_solver.experimental.self_evolve.proposal_journal import ProposalJournal
from rko_solver.experimental.self_evolve.proposals import PiCampaign
from rko_solver.experimental.self_evolve.study import ConfirmationStudy


def main() -> None:
    """Round-trip an explicit seed without importing generated code or provider libraries.

    Raises:
        AssertionError: If artifact identity or independent objective semantics differ.
    """
    original = DecoderArtifact.create("knapsack", Problem.seed_source("knapsack"))
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / "artifact"
        original.export(destination)
        restored = DecoderArtifact.load(destination)
        assert original == restored
        config = PiConfig(Path(sys.executable), Path(directory) / "unused-profile", "gpt-fixture")
        assert config.descriptor()["access_mode"] == "oauth"
        assert not config.profile.exists()
    assert ProposalJournal([]).entries == {}
    assert issubclass(PiCampaign, Campaign)
    help_result = subprocess.run(
        [sys.executable, "-m", "rko_solver.experimental.self_evolve", "--help"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert "approve-proposal" in help_result.stdout
    assert "pi-profile" in help_result.stdout
    problem = Problem("knapsack", {"weights": [2, 3, 4], "profits": [3, 5, 6], "capacity": 5})
    assert problem.evaluate([0, 1])["objective"] == 8
    assert problem.optimum() == 8
    effect = PairedAnalysis(bootstrap_samples=100, permutation_samples=100).compare(
        [[1.0]] * 4, [[0.0]] * 4
    )
    assert effect["p_value"] == 1 / 16
    assert ConfirmationStudy.instance_hash(problem)
    print("Installed experimental contracts passed; no generated-source execution or study run.")


if __name__ == "__main__":
    main()
