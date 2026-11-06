"""Manifest, trial-order, runner and row-analysis contracts of the utility study.

Objective:
- Prove the registered plan is complete, outcome-independent and aborts on the first error.

Technical approach:
- Build manifests from synthetic KPLIB bytes and generated files, never the network.
- Drive the runner with an injected executor and the analysis with synthetic rows.

Flow (pseudocode):
1. Build pilot and confirmation manifests and compare counts, balance and disjointness.
2. Plan and run trials with a fake executor, including a failure and the time ceiling.
3. Analyze synthetic rows into class claims.

Important criteria:
- No optimizer or network call; every planned instance enters the analysis.

Inputs/Outputs:
- Input: temporary caches, fake executors and synthetic rows.
- Output: deterministic assertions.
"""

import hashlib
import json
import pytest
from benchmarks.algorithms.utility_inputs import UtilityInputs
from benchmarks.algorithms.utility_study import CLASSES
from benchmarks.algorithms.utility_study import UtilityManifest
from benchmarks.algorithms.utility_study import UtilityRunner
from benchmarks.algorithms.utility_study import UtilityStudy
from pathlib import Path
from typing import Any


@pytest.fixture
def inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UtilityInputs:
    """Serve synthetic KPLIB files of the requested size for every URL.

    Args:
        tmp_path (Path): Input cache root.
        monkeypatch (pytest.MonkeyPatch): Replaces only the network boundary.

    Returns:
        UtilityInputs: Factory bound to the temporary cache.
    """

    def fake(self: UtilityInputs, url: str) -> bytes:
        """Return a valid file whose item count follows the URL's size directory.

        Args:
            self (UtilityInputs): Calling factory.
            url (str): Requested URL.

        Returns:
            bytes: Synthetic file.
        """
        size = int(url.split("/n")[1][:5])
        body = "".join(f"{3 + i % 7} {5 + i % 11}\n" for i in range(size))
        capacity = 100 + int(hashlib.sha256(url.encode()).hexdigest(), 16) % 5000
        return f"{size} {capacity}\n{body}".encode()

    monkeypatch.setattr(UtilityInputs, "_download", fake)
    return UtilityInputs(tmp_path)


def test_confirmation_manifest_is_complete_balanced_and_fingerprinted(
    inputs: UtilityInputs,
) -> None:
    """Hold 60 instances per class, balanced inside each size, with real fingerprints.

    Args:
        inputs (UtilityInputs): Factory with a faked network.
    """
    entries = UtilityManifest(inputs, "confirmation").build()
    for kind in CLASSES:
        rows = [row for row in entries if row["class"] == kind]
        assert len(rows) == 60 and len({row["instance"] for row in rows}) == 60
        assert {row["stratum"] for row in rows} == {f"{kind}-{n}" for n in _sizes(kind)}
        assert all(len(row["sha256"]) == 64 and Path(row["path"]).exists() for row in rows)
    items = [row for row in entries if row["class"] == "item"]
    assert all(isinstance(row["reference"], int) and row["reference"] > 0 for row in items)
    kinds = {row["subtype"] for row in items if row["stratum"] == "item-500"}
    assert len(kinds) == 5
    routes = [row for row in entries if row["stratum"] == "route-300"]
    assert sorted({row["subtype"] for row in routes}) == ["clustered", "uniform"]
    assert sum(row["subtype"] == "uniform" for row in routes) == 15
    assert all(row["reference"] is None for row in entries if row["class"] != "item")


def _sizes(kind: str) -> tuple[int, int]:
    """Return the two size strata of a class.

    Args:
        kind (str): Problem class.

    Returns:
        tuple[int, int]: Small and large size.
    """
    return (500, 1000) if kind == "item" else (100, 300)


def test_pilot_is_two_per_stratum_and_disjoint_from_confirmation(inputs: UtilityInputs) -> None:
    """Keep the baseline-only pilot small and separate from the confirmation set.

    Args:
        inputs (UtilityInputs): Factory with a faked network.
    """
    pilot = UtilityManifest(inputs, "pilot").build()
    confirm = UtilityManifest(inputs, "confirmation").build()
    assert len(pilot) == 12
    assert all(sum(r["stratum"] == s for r in pilot) == 2 for s in {r["stratum"] for r in pilot})
    assert not {r["instance"] for r in pilot} & {r["instance"] for r in confirm}
    assert not {r["sha256"] for r in pilot} & {r["sha256"] for r in confirm}


def test_manifest_digest_is_stable_and_changes_with_content(inputs: UtilityInputs) -> None:
    """Identify the frozen plan by a digest of its canonical content.

    Args:
        inputs (UtilityInputs): Factory with a faked network.
    """
    first = UtilityManifest(inputs, "pilot")
    second = UtilityManifest(inputs, "pilot")
    assert first.digest(first.build()) == second.digest(second.build())
    other = UtilityManifest(inputs, "confirmation")
    assert other.digest(other.build()) != first.digest(first.build())


def plan_entries() -> list[dict[str, Any]]:
    """Return three tiny manifest entries.

    Returns:
        list[dict[str, Any]]: Entries with the fields the runner reads.
    """
    return [{"class": "item", "stratum": "item-500", "instance": f"i{n}"} for n in range(3)]


def test_trial_order_is_seeded_complete_paired_and_outcome_independent() -> None:
    """Plan every pair once, adjacent, in an order set only by the order seed."""
    runner = UtilityRunner(plan_entries(), ("evaluations", "time"), order_seed=7)
    trials = runner.plan()
    assert len(trials) == 3 * 2 * 5 * 2
    keys = {(t["instance"], t["profile"], t["replicate"], t["arm"]) for t in trials}
    assert len(keys) == len(trials)
    for first, second in zip(trials[::2], trials[1::2], strict=True):
        assert (first["instance"], first["profile"], first["replicate"]) == (
            second["instance"],
            second["profile"],
            second["replicate"],
        )
        assert {first["arm"], second["arm"]} == {"baseline", "candidate"}
    assert trials == UtilityRunner(plan_entries(), ("evaluations", "time"), order_seed=7).plan()
    assert trials != UtilityRunner(plan_entries(), ("evaluations", "time"), order_seed=8).plan()
    arms = {trial["arm"] for trial in trials[::2]}
    assert arms == {"baseline", "candidate"}


def test_runner_writes_rows_sequentially_and_refuses_an_existing_file(tmp_path: Path) -> None:
    """Run each trial once in plan order and never overwrite earlier evidence.

    Args:
        tmp_path (Path): Output directory.
    """
    runner = UtilityRunner(plan_entries()[:1], ("evaluations",), order_seed=1)
    seen: list[dict[str, Any]] = []

    def execute(trial: dict[str, Any]) -> dict[str, Any]:
        """Record the trial and return a score row.

        Args:
            trial (dict[str, Any]): Planned trial.

        Returns:
            dict[str, Any]: Row with a score.
        """
        seen.append(trial)
        return {**trial, "score": 1}

    output = tmp_path / "rows.jsonl"
    summary = runner.run(execute, output, max_seconds=60.0)
    assert seen == runner.plan() and summary["completed"] == summary["planned"] == 10
    assert [json.loads(line)["score"] for line in output.read_text().splitlines()] == [1] * 10
    with pytest.raises(FileExistsError):
        runner.run(execute, output, max_seconds=60.0)


def test_first_error_aborts_and_keeps_the_partial_evidence(tmp_path: Path) -> None:
    """Stop at the first failing trial instead of continuing or dropping it.

    Args:
        tmp_path (Path): Output directory.
    """
    runner = UtilityRunner(plan_entries(), ("evaluations",), order_seed=2)
    calls = {"count": 0}

    def execute(trial: dict[str, Any]) -> dict[str, Any]:
        """Fail on the fourth trial.

        Args:
            trial (dict[str, Any]): Planned trial.

        Returns:
            dict[str, Any]: Score row.

        Raises:
            RuntimeError: On the fourth call.
        """
        calls["count"] += 1
        if calls["count"] == 4:
            raise RuntimeError("native failure")
        return {**trial, "score": 1}

    output = tmp_path / "rows.jsonl"
    with pytest.raises(RuntimeError, match="native failure"):
        runner.run(execute, output, max_seconds=60.0)
    assert calls["count"] == 4 and len(output.read_text().splitlines()) == 3


def test_ceiling_stops_the_campaign_as_incomplete(tmp_path: Path) -> None:
    """Stop when the cumulative time budget is exhausted and say so.

    Args:
        tmp_path (Path): Output directory.
    """
    runner = UtilityRunner(plan_entries(), ("evaluations",), order_seed=3)
    ticks = iter(range(1000))
    with pytest.raises(TimeoutError, match="ceiling"):
        runner.run(
            lambda trial: {**trial, "score": 1},
            tmp_path / "rows.jsonl",
            max_seconds=2.5,
            clock=lambda: float(next(ticks)),
        )
    assert 0 < len((tmp_path / "rows.jsonl").read_text().splitlines()) < 30


def row(
    instance: str, arm: str, profile: str, replicate: int, score: float, **extra: bool
) -> dict[str, Any]:
    """Build one analysis row.

    Args:
        instance (str): Instance identifier.
        arm (str): `baseline` or `candidate`.
        profile (str): Budget profile.
        replicate (int): Seed index.
        score (float): Final score.
        **extra (Any): Additional fields.

    Returns:
        dict[str, Any]: Row with a feasible score.
    """
    return {
        "class": "item",
        "instance": instance,
        "arm": arm,
        "profile": profile,
        "replicate": replicate,
        "score": score,
        "feasible": True,
        **extra,
    }


def test_claims_require_every_profile_and_keep_every_planned_instance() -> None:
    """Turn rows into wins: both profiles must win; an instance with no rows is left out."""
    rows = []
    for name, ev_win, time_win in (("a", True, True), ("b", True, False), ("c", False, False)):
        for rep in range(5):
            rows += [row(name, "baseline", "evaluations", rep, 100)]
            rows += [row(name, "candidate", "evaluations", rep, 90 if ev_win else 100)]
            rows += [row(name, "baseline", "time", rep, 100)]
            rows += [row(name, "candidate", "time", rep, 90 if time_win else 100)]
    entries = [{"class": "item", "instance": n} for n in ("a", "b", "c", "d")]
    wins = UtilityStudy.class_wins(rows, entries, ("evaluations", "time"))
    assert wins["item"] == [True, False, False]


def test_an_infeasible_candidate_row_cannot_win() -> None:
    """Carry feasibility from the rows into the win rule."""
    rows = []
    for rep in range(5):
        rows += [row("a", "baseline", "evaluations", rep, 100)]
        rows += [row("a", "candidate", "evaluations", rep, 10, feasible=rep != 2)]
    entries = [{"class": "item", "instance": "a"}]
    assert UtilityStudy.class_wins(rows, entries, ("evaluations",))["item"] == [False]


def test_incomplete_seed_rows_are_an_error_not_a_non_win() -> None:
    """Abort when an instance has the wrong number of seeds instead of silently dropping it."""
    rows = [row("a", "baseline", "evaluations", r, 100) for r in range(5)]
    rows += [row("a", "candidate", "evaluations", r, 90) for r in range(4)]
    with pytest.raises(ValueError, match="seeds"):
        UtilityStudy.class_wins(rows, [{"class": "item", "instance": "a"}], ("evaluations",))
