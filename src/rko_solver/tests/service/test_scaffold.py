"""Verify that installed canonical service files are copied without overwriting users.

Objective:
- Check the small scaffold independently of optional HTTP/runtime dependencies.

Technical approach:
- Compare copied resource bytes and validate bounded same-version local wheel metadata.

Flow (pseudocode):
1. Create a new target from canonical package resources.
2. Reject existing destinations and invalid SDK wheels before mutation.
3. Confirm valid local wheels are copied without extraction or publication.

Important criteria:
- No Docker daemon, network, actual secret or FastAPI installation is needed.

Inputs/Outputs:
- Input: temporary targets and bounded synthetic wheel metadata.
- Output: assertions about preserved files and deterministic reference layout.
"""

import pytest
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from rko_solver.service.scaffold import ServiceScaffold
from zipfile import ZipFile


def test_scaffold_copies_canonical_layout_only(tmp_path: Path) -> None:
    """Copy exactly the maintained layout into a new target without hidden downloads.

    Args:
        tmp_path (Path): Fixture-owned output root.
    """
    destination = ServiceScaffold.create(tmp_path / "new")
    canonical = files("rko_solver.service").joinpath("templates")
    for name in ServiceScaffold.filenames:
        assert (destination / name).read_bytes() == canonical.joinpath(name).read_bytes()
    assert list((destination / "wheels").iterdir()) == []
    assert "127.0.0.1:8000:8000" in (destination / "compose.yaml").read_text()
    assert "latest" not in (destination / "Dockerfile").read_text()


def test_scaffold_never_overwrites_existing_directory(tmp_path: Path) -> None:
    """Preserve both empty targets and preexisting user code without mutation.

    Args:
        tmp_path (Path): Fixture-owned preexisting target.
    """
    sentinel = tmp_path / "app.py"
    sentinel.write_text("application-owned")
    with pytest.raises(FileExistsError):
        ServiceScaffold.create(tmp_path)
    assert sentinel.read_text() == "application-owned"
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        ServiceScaffold.create(empty)
    assert not list(empty.iterdir())


@pytest.mark.parametrize("matching", [True, False])
def test_scaffold_checks_sdk_metadata_before_creation(tmp_path: Path, matching: bool) -> None:
    """Require the installed SDK version and preserve original reviewed wheel bytes.

    Args:
        tmp_path (Path): Fixture-owned wheel/output root.
        matching (bool): Whether synthetic wheel metadata matches this installed SDK.
    """
    current = version("rko-solver") if matching else "0.0.0"
    wheel = tmp_path / f"rko_solver-{current}-py3-none-any.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(
            f"rko_solver-{current}.dist-info/METADATA", f"Name: rko-solver\nVersion: {current}\n"
        )
    destination = tmp_path / "generated"
    if matching:
        ServiceScaffold.main(["scaffold", str(destination), "--wheel", str(wheel)])
        assert (destination / "wheels" / wheel.name).read_bytes() == wheel.read_bytes()
    else:
        with pytest.raises(ValueError, match="matching"):
            ServiceScaffold.create(destination, wheel)
        assert not destination.exists()


def test_scaffold_rejects_bad_archives_and_symlinks(tmp_path: Path) -> None:
    """Reject invalid local artifacts without extracting files or creating a target.

    Args:
        tmp_path (Path): Fixture-owned invalid wheel/output root.
    """
    wheel = tmp_path / "rko_solver-0.3.0-py3-none-any.whl"
    wheel.write_bytes(b"not a wheel")
    with pytest.raises(ValueError, match="archive"):
        ServiceScaffold.create(tmp_path / "bad", wheel)
    assert not (tmp_path / "bad").exists()
    target = tmp_path / "link"
    try:
        target.symlink_to(tmp_path / "absent", target_is_directory=True)
    except OSError:
        pytest.skip("host does not allow test symlinks")
    with pytest.raises(FileExistsError):
        ServiceScaffold.create(target)
