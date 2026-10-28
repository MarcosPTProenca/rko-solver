"""Session startup charges and explicit transport binding at the collector boundary.

Objective:
- Verify persistent-session startup accounting and frozen transport identity.
Technical approach:
- Use deterministic runner evidence with a persistent transport label and reopen studies.
Flow (pseudocode):
1. Freeze a study with an explicit transport.
2. Collect a job with a startup charge and resume collection.
3. Check retained charges and reject transport mismatches.
Important criteria:
- Startup consumption is not free; collection cannot relabel a frozen transport.
Inputs/Outputs:
- Input: temporary study paths, deterministic runner records and a startup charge.
- Output: assertions on cumulative accounting and approval failures.
"""

import pytest
from pathlib import Path
from tests.experimental.test_confirmation_study import create_study
from tests.experimental.test_self_evolve_campaign import FixtureRunner


class SessionFixture(FixtureRunner):
    """Represent qualified fresh-process transport without executing source in unit tests.

    Responsibilities:
    - Identify the frozen transport while reusing trusted fixture results.

    Attributes:
        transport (str): Explicit fresh-process measurement transport.
    """

    transport = "persistent_exec"


def test_session_startup_is_charged_in_addition_to_jobs(tmp_path: Path) -> None:
    """Amortizing startup must not turn setup consumption into unrecorded free evaluation.

    Args:
        tmp_path (Path): Temporary parent for the persistent-transport study and saved progress.
    """
    study = create_study(tmp_path / "study", transport="persistent_exec")
    progress = study.run(SessionFixture(), approved_spec=study.digest, max_jobs=1, startup_charge=7)
    assert progress["session_startup_upper_bound"] == 7
    assert progress["charged_cpu_upper_bound"] == pytest.approx(7.3)
    reopened = type(study)(study.path)
    result = reopened.run(SessionFixture(), approved_spec=study.digest)
    assert result["session_startup_upper_bound"] == 7


def test_transport_cannot_change_after_freezing_the_panel(tmp_path: Path) -> None:
    """Transport changes require a new manifest, never relabelled measurements.

    Args:
        tmp_path (Path): Temporary parent for the study with a conflicting frozen transport.
    """
    study = create_study(tmp_path / "study")
    with pytest.raises(ValueError, match="approval"):
        study.run(SessionFixture(), approved_spec=study.digest)
