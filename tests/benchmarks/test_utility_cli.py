"""Command-line contracts of the expanded utility study.

Objective:
- Prove the manifest, run and analyze commands freeze, record and refuse correctly.

Technical approach:
- Fake the KPLIB download, replace only the budget with a tiny one and run the real
  executors on tiny generated inputs.

Flow (pseudocode):
1. Build a pilot manifest and check its digest.
2. Run a mechanism on it and read the rows and the provenance sidecar.
3. Analyze synthetic rows of both mechanisms into six decisions.

Important criteria:
- Outputs are created exclusively; incomplete claims are reported, never hidden.

Inputs/Outputs:
- Input: argument vectors and temporary files.
- Output: JSON files and exit codes.
"""

import json
import pytest
from benchmarks.algorithms import utility_cli
from benchmarks.algorithms import utility_study
from benchmarks.algorithms.utility_cli import UtilityCli
from benchmarks.algorithms.utility_inputs import UtilityInputs
from pathlib import Path


@pytest.fixture
def pilot_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a pilot manifest through the real command with a faked download.

    Args:
        tmp_path (Path): Scratch directory.
        monkeypatch (pytest.MonkeyPatch): Replaces only the network boundary.

    Returns:
        Path: Manifest file.
    """

    def fake(self: UtilityInputs, url: str) -> bytes:
        """Serve a small valid file of the size named in the URL.

        Args:
            self (UtilityInputs): Calling factory.
            url (str): Requested URL.

        Returns:
            bytes: Synthetic file.
        """
        size = int(url.split("/n")[1][:5])
        body = "".join(f"{3 + i % 7} {5 + i % 11}\n" for i in range(size))
        return f"{size} 300\n{body}".encode()

    monkeypatch.setattr(UtilityInputs, "_download", fake)
    manifest = tmp_path / "pilot.json"
    code = UtilityCli.main(
        [
            "manifest",
            "--role",
            "pilot",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(manifest),
        ]
    )
    assert code == 0
    return manifest


def test_manifest_command_writes_entries_and_a_matching_digest(pilot_manifest: Path) -> None:
    """Record 12 pilot entries and a digest that recomputes from them.

    Args:
        pilot_manifest (Path): Manifest file.
    """
    data = json.loads(pilot_manifest.read_text())
    assert data["role"] == "pilot" and len(data["entries"]) == 12
    assert data["digest"] == utility_study.UtilityManifest.digest(data["entries"])
    assert UtilityCli.main(["manifest", "--role", "pilot", "--output", str(pilot_manifest)]) != 0


def test_run_refuses_an_unknown_mechanism_and_an_existing_output(
    tmp_path: Path, pilot_manifest: Path
) -> None:
    """Fail fast on bad arguments before any trial.

    Args:
        tmp_path (Path): Scratch directory.
        pilot_manifest (Path): Manifest file.
    """
    rows = tmp_path / "rows.jsonl"
    rows.write_text("evidence\n")
    base = ["run", "--manifest", str(pilot_manifest), "--output", str(rows), "--order-seed", "1"]
    with pytest.raises(SystemExit):
        UtilityCli.main([*base, "--mechanism", "other"])
    assert UtilityCli.main([*base, "--mechanism", "adaptive", "--max-hours", "1"]) != 0
    assert rows.read_text() == "evidence\n"


def test_run_executes_the_plan_and_records_provenance(
    tmp_path: Path, pilot_manifest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run two real adaptive trials per planned pair and write rows plus a sidecar.

    Args:
        tmp_path (Path): Scratch directory.
        pilot_manifest (Path): Manifest file.
        monkeypatch (pytest.MonkeyPatch): Shrinks only the budget.
    """
    monkeypatch.setattr(utility_cli, "BUDGET", (0.1, 200))
    data = json.loads(pilot_manifest.read_text())
    keep = [entry for entry in data["entries"] if entry["stratum"] == "route-100"]
    small = tmp_path / "small.json"
    digest = utility_study.UtilityManifest.digest(keep)
    small.write_text(json.dumps({**data, "digest": digest, "entries": keep}))
    rows = tmp_path / "rows.jsonl"
    code = UtilityCli.main(
        [
            "run",
            "--manifest",
            str(small),
            "--output",
            str(rows),
            "--order-seed",
            "5",
            "--mechanism",
            "adaptive",
            "--profiles",
            "evaluations",
            "--max-hours",
            "1",
        ]
    )
    assert code == 0
    lines = [json.loads(line) for line in rows.read_text().splitlines()]
    assert len(lines) == 2 * 5 * 2 and {row["arm"] for row in lines} == {"baseline", "candidate"}
    meta = json.loads(Path(f"{rows}.meta.json").read_text())
    assert meta["manifest_digest"] == digest and meta["status"] == "complete"
    assert meta["mechanism"] == "adaptive" and meta["order_seed"] == 5
    assert meta["planned"] == meta["completed"] == 20 and "git_commit" in meta


def test_analyze_reports_six_decisions_and_marks_missing_mechanisms_incomplete(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print one status per registered claim, incomplete when rows are absent.

    Args:
        tmp_path (Path): Scratch directory.
        capsys (pytest.CaptureFixture[str]): Output capture.
    """
    entries = [{"class": "item", "instance": f"i{n}", "stratum": "item-500"} for n in range(60)]
    manifest = tmp_path / "m.json"
    digest = utility_study.UtilityManifest.digest(entries)
    manifest.write_text(json.dumps({"role": "confirmation", "digest": digest, "entries": entries}))
    rows = tmp_path / "rows.jsonl"
    with rows.open("w") as handle:
        for entry in entries:
            for profile in ("evaluations", "time"):
                for replicate in range(5):
                    for arm, score in (("baseline", 100), ("candidate", 50)):
                        handle.write(
                            json.dumps(
                                {
                                    "class": "item",
                                    "instance": entry["instance"],
                                    "profile": profile,
                                    "arm": arm,
                                    "replicate": replicate,
                                    "score": score,
                                    "feasible": True,
                                }
                            )
                            + "\n"
                        )
    code = UtilityCli.main(["analyze", "--manifest", str(manifest), "--rows", f"adaptive={rows}"])
    output = capsys.readouterr().out
    assert code == 0
    assert "adaptive:item" in output and "evidence" in output
    assert output.split("JSON ", 1)[0].count("incomplete") == 5
    assert set(json.loads(output.split("JSON ", 1)[1])) == set(utility_cli.CLAIMS)


def test_analyze_and_run_refuse_a_manifest_whose_digest_does_not_match(tmp_path: Path) -> None:
    """Protect the frozen plan from silent edits.

    Args:
        tmp_path (Path): Scratch directory.
    """
    entries = [{"class": "item", "instance": "i0", "stratum": "item-500"}]
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"role": "pilot", "digest": "bad", "entries": entries}))
    with pytest.raises(ValueError, match="digest"):
        UtilityCli.main(["analyze", "--manifest", str(manifest)])
    with pytest.raises(ValueError, match="digest"):
        UtilityCli.main(
            [
                "run",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "r.jsonl"),
                "--order-seed",
                "1",
                "--mechanism",
                "adaptive",
            ]
        )
