"""Fresh-input, size-awareness and routing-score contracts of the utility study.

Objective:
- Prove inputs are disjoint from every earlier cohort and cannot be mis-sized or mis-scored.

Technical approach:
- Rebuild the consumed pools with the production selectors and compare draws.
- Use synthetic KPLIB bytes instead of the network.
- Check the rounded route scorer against hand-computed tours and the native parser format.

Flow (pseudocode):
1. Draw pilot and confirmation inputs and compare them with consumed and each other.
2. Fetch synthetic files of correct and wrong sizes.
3. Score hand-built tours and tied keys.

Important criteria:
- No network access, no optimizer run and no outcome-dependent choice.

Inputs/Outputs:
- Input: temporary caches, synthetic file bytes and tiny coordinates.
- Output: deterministic assertions.
"""

import pytest
import random
from benchmarks.algorithms.sa_temperature_screen import ScreenInputs
from benchmarks.algorithms.utility_inputs import FAMILIES
from benchmarks.algorithms.utility_inputs import UtilityInputs
from benchmarks.engines.gate_instances import GateInstances
from pathlib import Path


def kp_bytes(items: int, capacity: int = 100) -> bytes:
    """Build a syntactically valid KPLIB-style file.

    Args:
        items (int): Declared and actual item count.
        capacity (int): Knapsack capacity.

    Returns:
        bytes: Header plus one `weight profit` pair per line.
    """
    body = "".join(f"{3 + index % 7} {5 + index % 11}\n" for index in range(items))
    return f"{items} {capacity}\n{body}".encode()


def test_consumed_pools_cover_every_earlier_cohort_at_500_only(tmp_path: Path) -> None:
    """Rebuild all four earlier pools at 500 items and none at 1000.

    Args:
        tmp_path (Path): Input cache root.
    """
    inputs = UtilityInputs(tmp_path)
    expected = (
        set(GateInstances(tmp_path, "calibration", 100).knapsack_ids())
        | set(GateInstances(tmp_path, "test", 100).knapsack_ids())
        | set(GateInstances(tmp_path, "confirmation", 100).knapsack_ids())
        | set(ScreenInputs(tmp_path, 100).knapsack_ids())
    )
    assert len(expected) == 4 * 20 * len(FAMILIES)
    assert inputs.consumed(500) == expected
    assert inputs.consumed(1000) == set()
    with pytest.raises(ValueError, match="size"):
        inputs.consumed(250)


def test_eligible_pools_are_the_complement(tmp_path: Path) -> None:
    """Return only identifiers no earlier cohort consumed.

    Args:
        tmp_path (Path): Input cache root.
    """
    inputs = UtilityInputs(tmp_path)
    used = inputs.consumed(500)
    pools = inputs.eligible(500)
    assert set(pools) == set(FAMILIES)
    assert all(len(ids) == 20 and ids == sorted(ids) for ids in pools.values())
    assert not {(kind, item) for kind, ids in pools.items() for item in ids} & used
    assert all(len(ids) == 100 for ids in inputs.eligible(1000).values())


@pytest.mark.parametrize("size", [500, 1000])
def test_pilot_and_confirmation_are_disjoint_repeatable_and_fresh(
    tmp_path: Path, size: int
) -> None:
    """Draw per type without replacement, pilot before confirmation, outside consumed pools.

    Args:
        tmp_path (Path): Input cache root.
        size (int): Item count under test.
    """
    inputs = UtilityInputs(tmp_path)
    pilot = inputs.draw(size, 1, "pilot")
    confirm = inputs.draw(size, 6, "confirmation")
    assert len(pilot) == len(set(pilot)) == len(FAMILIES)
    assert len(confirm) == len(set(confirm)) == 6 * len(FAMILIES)
    assert not set(pilot) & set(confirm)
    assert not (set(pilot) | set(confirm)) & inputs.consumed(size)
    assert confirm == inputs.draw(size, 6, "confirmation")
    assert pilot == inputs.draw(size, 1, "pilot")
    assert {kind for kind, _ in confirm} == set(FAMILIES)


def test_draw_does_not_depend_on_other_sizes_and_refuses_exhaustion(tmp_path: Path) -> None:
    """Keep sizes independent and refuse to recycle identifiers.

    Args:
        tmp_path (Path): Input cache root.
    """
    inputs = UtilityInputs(tmp_path)
    assert inputs.draw(500, 6, "confirmation") != inputs.draw(1000, 6, "confirmation")
    with pytest.raises(ValueError, match="exhausted"):
        inputs.draw(500, 20, "confirmation")
    with pytest.raises(ValueError, match="role"):
        inputs.draw(500, 1, "test")


def test_kplib_path_is_size_namespaced_and_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cache a correct file under its size and refuse a header with another count.

    Args:
        tmp_path (Path): Input cache root.
        monkeypatch (pytest.MonkeyPatch): Replaces only the network boundary.
    """
    inputs = UtilityInputs(tmp_path)
    urls: list[str] = []

    def fake(self: UtilityInputs, url: str) -> bytes:
        """Serve a 1000-item file for every URL and record the URL.

        Args:
            self (UtilityInputs): Calling factory.
            url (str): Requested URL.

        Returns:
            bytes: Synthetic file contents.
        """
        urls.append(url)
        return kp_bytes(1000)

    monkeypatch.setattr(UtilityInputs, "_download", fake)
    path = inputs.kplib_path("00Uncorrelated", 1000, 7)
    assert "n01000" in urls[0] and urls[0].endswith("00Uncorrelated/n01000/R01000/s007.kp")
    assert path == tmp_path / "kplib-sized" / "n01000" / "00Uncorrelated" / "s007.kp"
    assert inputs.kplib_path("00Uncorrelated", 1000, 7) == path and len(urls) == 1
    with pytest.raises(ValueError, match="500"):
        inputs.kplib_path("00Uncorrelated", 500, 7)
    assert not (tmp_path / "kplib-sized" / "n00500" / "00Uncorrelated" / "s007.kp").exists()


def test_cached_file_with_a_wrong_count_is_refused_not_trusted(tmp_path: Path) -> None:
    """Validate a pre-existing cached file instead of trusting its location.

    Args:
        tmp_path (Path): Input cache root.
    """
    target = tmp_path / "kplib-sized" / "n01000" / "00Uncorrelated" / "s003.kp"
    target.parent.mkdir(parents=True)
    target.write_bytes(kp_bytes(500))
    with pytest.raises(ValueError, match="1000"):
        UtilityInputs(tmp_path).kplib_path("00Uncorrelated", 1000, 3)


def test_oversized_download_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject a response beyond the byte cap before writing it.

    Args:
        tmp_path (Path): Input cache root.
        monkeypatch (pytest.MonkeyPatch): Replaces only the network boundary.
    """
    monkeypatch.setattr(UtilityInputs, "_download", lambda self, url: b"1 " * (1 << 20))
    with pytest.raises(ValueError, match="large"):
        UtilityInputs(tmp_path).kplib_path("00Uncorrelated", 500, 1)
    assert not list((tmp_path / "kplib-sized").rglob("*.kp"))


def test_routing_files_are_deterministic_native_format_and_role_separated(tmp_path: Path) -> None:
    """Write integer-coordinate files the native parser reads, different per role and kind.

    Args:
        tmp_path (Path): Input cache root.
    """
    inputs = UtilityInputs(tmp_path)
    first = inputs.tsp_path("uniform", 100, 0, "pilot")
    again = inputs.tsp_path("uniform", 100, 0, "pilot")
    assert first.read_bytes() == again.read_bytes()
    lines = first.read_text().splitlines()
    assert lines[0] == "100" and len(lines) == 101
    assert all(len(line.split()) == 2 for line in lines[1:])
    assert all(0 <= int(value) < 10_000 for line in lines[1:] for value in line.split())
    contents = {
        inputs.fingerprint(inputs.tsp_path(kind, 100, index, role))
        for kind in ("uniform", "clustered")
        for index in (0, 1)
        for role in ("pilot", "confirmation")
    }
    assert len(contents) == 8
    with pytest.raises(ValueError):
        inputs.tsp_path("ring", 100, 0, "pilot")
    with pytest.raises(ValueError):
        inputs.tsp_path("uniform", 3, 0, "pilot")
    with pytest.raises(ValueError, match="role"):
        inputs.tsp_path("uniform", 100, 0, "test")


def test_route_scorer_rounds_distances_and_breaks_ties_by_index(tmp_path: Path) -> None:
    """Match the native rule: round each edge to nearest, sort keys with ties by index.

    Args:
        tmp_path (Path): Scratch directory.
    """
    path = tmp_path / "square.txt"
    path.write_text("4\n0 0\n3 0\n3 4\n0 4\n")
    score = UtilityInputs(tmp_path).route_scorer(path)
    assert score([0.1, 0.2, 0.3, 0.4]) == 3 + 4 + 3 + 4
    assert score([0.1, 0.3, 0.2, 0.4]) == 5 + 4 + 5 + 4
    assert score([0.5, 0.5, 0.5, 0.5]) == score([0.0, 0.1, 0.2, 0.3])
    half = tmp_path / "half.txt"
    half.write_text("2\n0 0\n1 1\n")
    assert UtilityInputs(tmp_path).route_scorer(half)([0.0, 1.0]) == 2.0


def test_route_scorer_is_invariant_to_key_scale(tmp_path: Path) -> None:
    """Hold for random instances: only the key order decides the tour.

    Args:
        tmp_path (Path): Input cache root.
    """
    inputs = UtilityInputs(tmp_path)
    score = inputs.route_scorer(inputs.tsp_path("clustered", 100, 0, "pilot"))
    generator = random.Random(5)
    for _ in range(25):
        keys = [generator.random() for _ in range(100)]
        assert score(keys) == score([0.5 * key + 0.25 for key in keys])


def test_continuous_files_are_deterministic_bounded_and_role_separated(tmp_path: Path) -> None:
    """Write shifts in [-2, 2] that differ by role and index and never reuse earlier seeds.

    Args:
        tmp_path (Path): Input cache root.
    """
    inputs = UtilityInputs(tmp_path)
    path = inputs.rastrigin_path(100, 0, "pilot")
    assert path.read_bytes() == inputs.rastrigin_path(100, 0, "pilot").read_bytes()
    values = path.read_text().splitlines()
    assert values[0] == "100" and len(values) == 101
    assert all(-2.0 <= float(value) <= 2.0 for value in values[1:])
    digests = {
        inputs.fingerprint(inputs.rastrigin_path(100, index, role))
        for index in (0, 1)
        for role in ("pilot", "confirmation")
    }
    assert len(digests) == 4
    gate_seeds = (20261120, 20261111, 20261121, 20261122)
    for seed in gate_seeds:
        generator = random.Random(f"{seed}:100:0")
        old = [round(generator.uniform(-2.0, 2.0), 3) for _ in range(100)]
        assert [float(value) for value in values[1:]] != old
    with pytest.raises(ValueError):
        inputs.rastrigin_path(0, 0, "pilot")
    with pytest.raises(ValueError, match="role"):
        inputs.rastrigin_path(100, 0, "test")
