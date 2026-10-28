"""Named online-grid and configuration persistence contracts.

Objective:
- Keep parameter meanings independent of TOML order and preserve valid files.

Technical approach:
- Load reordered grids and round-trip escaped scalar values.
- Simulate atomic replacement failure without damaging the original file.

Flow (pseudocode):
1. Write representative user parameter files.
2. Load or replace their online grids.
3. Verify canonical axes, exact offline values and failure cleanup.

Important criteria:
- Invalid input must fail before the destination is overwritten.

Inputs/Outputs:
- Input: temporary TOML files and injected filesystem failures.
- Output: parameter semantics and data-integrity assertions.
"""

import pytest
import tomllib
from pathlib import Path
from rko_solver.parameters import loader


@pytest.fixture
def parameters(tmp_path: Path) -> Path:
    """Write a valid GA file with reordered keys and escaped metadata.

    Args:
        tmp_path (Path): Temporary parameter root.

    Returns:
        Path: Parameter directory.
    """
    (tmp_path / "ga.toml").write_text(
        "[offline]\nsize_pop=30\nprob_cros=0.8\nprob_mut=0.02\n"
        'label="stable \\"quoted\\" \\n Unicode 😀"\n'
        "[online]\nprob_mut=[0.02]\nprob_cros=[0.8]\nsize_pop=[30.0]\n",
        encoding="utf-8",
    )
    return tmp_path


def test_grid_order_is_defined_by_parameter_names(parameters: Path) -> None:
    """Ignore textual order when constructing positional algorithm grids.

    Args:
        parameters (Path): Valid reordered file.
    """
    assert loader.load_online_grid("GA", parameters) == [[30.0], [0.8], [0.02]]


def test_save_preserves_scalars_and_canonical_axes(parameters: Path) -> None:
    """Round-trip strings without corrupting their TOML representation.

    Args:
        parameters (Path): Valid reordered file.
    """
    original = loader.load_offline("GA", parameters)
    loader.save_online_grid("GA", [[40.0], [0.9], [0.05]], params_dir=parameters)
    assert loader.load_offline("GA", parameters) == original
    assert loader.load_online_grid("GA", parameters) == [[40.0], [0.9], [0.05]]


@pytest.mark.parametrize(
    "grid", [[[0.02], [0.8]], [[float("nan")], [0.8], [0.02]], [[], [0.8], [0.02]]]
)
def test_invalid_grid_does_not_touch_original(parameters: Path, grid: list[list[float]]) -> None:
    """Reject malformed grids before overwriting the original file.

    Args:
        parameters (Path): Valid parameter directory.
        grid (list[list[float]]): Invalid replacement grid.
    """
    path = parameters / "ga.toml"
    original = path.read_bytes()
    with pytest.raises(ValueError):
        loader.save_online_grid("GA", grid, params_dir=parameters)
    assert path.read_bytes() == original


def test_replace_failure_preserves_original_and_cleans_temp(
    parameters: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave the existing TOML usable when committing a replacement fails.

    Args:
        parameters (Path): Valid parameter directory.
        monkeypatch (pytest.MonkeyPatch): Inject replacement failure.
    """
    path = parameters / "ga.toml"
    original = path.read_bytes()

    def fail_replace(source: object, target: object) -> None:
        """Simulate a filesystem failure before atomic replacement.

        Args:
            source (object): Temporary file path.
            target (object): Existing destination.

        Raises:
            OSError: Always, to simulate replacement failure.
        """
        raise OSError("replacement failed")

    monkeypatch.setattr(loader.os, "replace", fail_replace)
    with pytest.raises(OSError, match="replacement failed"):
        loader.save_online_grid("GA", [[40.0], [0.9], [0.05]], params_dir=parameters)
    assert path.read_bytes() == original
    assert list(parameters.iterdir()) == [path]
    assert tomllib.loads(path.read_text(encoding="utf-8"))["offline"]["size_pop"] == 30
