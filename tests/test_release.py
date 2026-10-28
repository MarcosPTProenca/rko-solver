"""Contract checks for non-publishing prerelease preparation and immutable promotion.

Objective:
- Reject ambiguous tags, changed payloads and unsafe publication provenance before upload.

Technical approach:
- Construct tiny archive fixtures and use temporary Git repositories/service metadata.
- Inspect workflow structure without requesting credentials or publishing anything.

Flow (pseudocode):
1. Validate canonical prerelease identifiers and tagged main ancestry.
2. Build and mutate synthetic distribution payloads to test integrity and exclusions.
3. Reject unprotected environments, existing versions and untrusted promotion runs.

Important criteria:
- These tests never create real repository tags/environments or upload to an index.
- Real artifact and installed-extra checks remain separate mandatory CI jobs.

Inputs/Outputs:
- Input: temporary archives, Git histories and mocked service responses.
- Output: assertions for release identity, integrity and least-privilege controls.
"""

import io
import json
import pytest
import shutil
import subprocess
import tarfile
from pathlib import Path
from tools.release import ReleaseManager
from unittest.mock import Mock
from zipfile import ZipFile


@pytest.fixture
def payload(tmp_path: Path) -> tuple[ReleaseManager, Path]:
    """Create minimal typed artifacts with real approved license bytes and source identity.

    Args:
        tmp_path (Path): Isolated payload directory.

    Returns:
        tuple[ReleaseManager, Path]: Trusted source validator and synthetic artifact directory.
    """
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "source"
    source.mkdir()
    for name in ReleaseManager.licenses:
        destination = source / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(root / name, destination)
    (source / "pyproject.toml").write_text('[project]\nname="rko-solver"\nversion="0.3.0a1"\n')
    (source / "CHANGELOG.md").write_text("## 0.3.0a1\n")
    subprocess.run(["git", "init", "-b", "main", str(source)], check=True, capture_output=True)
    for key, value in (
        ("user.name", "Release Test"),
        ("user.email", "release@example.invalid"),
        ("commit.gpgsign", "false"),
    ):
        subprocess.run(["git", "config", key, value], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-m", "fixture"], cwd=source, check=True, capture_output=True)
    release = tmp_path / "release"
    dist = release / "dist"
    dist.mkdir(parents=True)
    metadata = (
        b"Metadata-Version: 2.4\nName: rko-solver\nVersion: 0.3.0a1\n"
        b"Requires-Python: >=3.14\nLicense-Expression: Apache-2.0 AND MIT\n"
        b"Requires-Dist: numpy>=2.0.0\n\nDescription\n"
    )
    resources = {
        "examples/instances/scheduling.json": b"{}",
        "examples/instances/kp10.txt": b"1",
        "parameters/SA.toml": b"x=1",
    }
    licenses = {name: (root / name).read_bytes() for name in ReleaseManager.licenses}
    runtime = ["rko_solver.libs/libgomp-1a2b3c.so.1.0.0"]
    wheels = {
        "manylinux_2_28_x86_64": ("rko_solver/_native.abi3.so", runtime),
        "manylinux_2_28_aarch64": ("rko_solver/_native.abi3.so", runtime),
        "win_amd64": ("rko_solver/_native.pyd", []),
        "win_arm64": ("rko_solver/_native.pyd", []),
        "macosx_11_0_arm64": ("rko_solver/_native.abi3.so", []),
        "macosx_10_12_x86_64": ("rko_solver/_native.abi3.so", []),
    }
    for tag, (module, libraries) in wheels.items():
        with ZipFile(dist / f"rko_solver-0.3.0a1-cp314-abi3-{tag}.whl", "w") as archive:
            prefix = "rko_solver-0.3.0a1.dist-info/"
            archive.writestr(prefix + "METADATA", metadata)
            archive.writestr(
                prefix + "WHEEL",
                f"Root-Is-Purelib: false\nTag: cp314-abi3-{tag}\n",
            )
            archive.writestr(module, b"binary")
            for library in libraries:
                archive.writestr(library, b"runtime")
            for name, content in resources.items():
                archive.writestr("rko_solver/" + name, content)
            for name, content in licenses.items():
                archive.writestr(prefix + "licenses/" + name, content)
    entries = {
        "PKG-INFO": metadata,
        "Cargo.lock": b"",
        "engines/rust/rko-py/Cargo.toml": b"",
        "engines/cpp/build.rs": b"",
        **licenses,
        **{"src/rko_solver/" + name: content for name, content in resources.items()},
    }
    with tarfile.open(dist / "rko_solver-0.3.0a1.tar.gz", "w:gz") as archive:
        for name, content in entries.items():
            member = tarfile.TarInfo("rko_solver-0.3.0a1/" + name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    return ReleaseManager(source), release


@pytest.mark.parametrize(
    "tag",
    [
        "0.3.0a1",
        "v0.3.0",
        "v0.3.0.dev0",
        "v0.3.0a01",
        "v0.3.0a1+local",
        "--help",
        "v0.3.0a1\nsecret",
    ],
)
def test_noncanonical_or_stable_tags_are_rejected(tag: str) -> None:
    """Keep the first release workflow limited to explicit canonical prereleases.

    Args:
        tag (str): Unsupported version/ref input.
    """
    with pytest.raises(ValueError):
        ReleaseManager.version(tag)
    assert ReleaseManager.version("v0.3.0a1") == "0.3.0a1"


@pytest.mark.parametrize("content", [b"*", b"*\n", b"unreviewed hidden content"])
def test_only_exact_uv_output_marker_is_removed(
    payload: tuple[ReleaseManager, Path],
    content: bytes,
) -> None:
    """Admit uv's generated marker without accepting arbitrary hidden publication files.

    Args:
        payload (tuple[ReleaseManager, Path]): Dedicated temporary artifact output.
        content (bytes): Generated marker or unrecognized hidden content.
    """
    manager, directory = payload
    marker = directory / "dist" / ".gitignore"
    marker.write_bytes(content)
    if content in (b"*", b"*\n"):
        manager.create(directory)
        assert not marker.exists()
        manager.verify(directory)
    else:
        with pytest.raises(ValueError, match="marker"):
            manager.create(directory)
        assert marker.read_bytes() == content


def test_exact_payload_survives_transfer_and_changed_bytes_fail(
    payload: tuple[ReleaseManager, Path],
) -> None:
    """Retain immutable artifact hashes and fail rather than silently regenerating them.

    Args:
        payload (tuple[ReleaseManager, Path]): Valid synthetic artifact set.
    """
    manager, directory = payload
    manifest = manager.create(directory)
    assert manager.verify(directory, "v0.3.0a1", manifest["source_sha"]) == manifest
    assert sorted(item["platform"] for item in manifest["files"]) == [
        "linux-aarch64",
        "linux-x86_64",
        "macos-arm64",
        "macos-x86_64",
        "sdist",
        "windows-arm64",
        "windows-x86_64",
    ]
    (manager.root / "pyproject.toml").write_text('[project]\nversion="0.3.0a2"\n')
    assert manager.verify(directory, "v0.3.0a1", manifest["source_sha"]) == manifest
    with pytest.raises(ValueError, match="identity"):
        manager.verify(directory, source_sha="0" * 40)
    wheel = next((directory / "dist").glob("*manylinux*x86_64.whl"))
    with wheel.open("ab") as output:
        output.write(b"changed after testing")
    with pytest.raises(ValueError, match="immutable"):
        manager.verify(directory)


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        "rko_solver/_native.so",
        "rko_solver/other.dll",
        "rko_solver/.env",
        "rko_solver/results/private.json",
        "rko_solver/target/binary",
        "rko_solver/examples/instances/secret.csv",
    ],
)
def test_private_unsafe_or_native_archive_entries_fail(
    payload: tuple[ReleaseManager, Path], name: str
) -> None:
    """Prevent source leaks/native binaries from entering the initial pure payload.

    Args:
        payload (tuple[ReleaseManager, Path]): Synthetic artifact set to mutate.
        name (str): Forbidden archive entry.
    """
    manager, directory = payload
    wheel = next((directory / "dist").glob("*win_amd64.whl"))
    with ZipFile(wheel, "a") as archive:
        archive.writestr(name, b"private")
    with pytest.raises(ValueError, match="excluded|unreviewed"):
        manager.create(directory)


def test_dirty_source_and_unsupported_wheel_tags_cannot_be_promoted(
    payload: tuple[ReleaseManager, Path],
) -> None:
    """Refuse false source identities and wheels outside the supported platform set.

    Args:
        payload (tuple[ReleaseManager, Path]): Valid synthetic candidate source/artifacts.
    """
    manager, directory = payload
    wheel = next((directory / "dist").glob("*win_amd64.whl"))
    wheel.rename(wheel.with_name("rko_solver-0.3.0a1-cp314-cp314-win_amd64.whl"))
    with pytest.raises(ValueError, match="abi3"):
        manager.create(directory)
    wheel = next((directory / "dist").glob("*cp314-cp314-win_amd64.whl"))
    wheel.rename(wheel.with_name("rko_solver-0.3.0a1-cp314-abi3-win32.whl"))
    with pytest.raises(ValueError, match="platform tag"):
        manager.create(directory)
    wheel = next((directory / "dist").glob("*win32.whl"))
    wheel.rename(wheel.with_name("rko_solver-0.3.0a1-py3-none-any.whl"))
    with pytest.raises(ValueError, match="abi3"):
        manager.create(directory)
    (manager.root / "pyproject.toml").write_text("changed source")
    with pytest.raises(ValueError, match="Commit"):
        manager.create(directory)


def test_every_platform_wheel_is_required_and_structurally_checked(
    payload: tuple[ReleaseManager, Path],
) -> None:
    """Reject missing platforms, missing OpenMP runtimes and misplaced shared libraries.

    Args:
        payload (tuple[ReleaseManager, Path]): Valid synthetic artifact set.
    """
    manager, directory = payload
    dist = directory / "dist"
    windows = next(dist.glob("*win_amd64.whl"))
    saved = windows.read_bytes()
    windows.unlink()
    with pytest.raises(ValueError, match="one wheel per platform"):
        manager.create(directory)
    windows.write_bytes(saved)
    with ZipFile(windows, "a") as archive:
        archive.writestr("rko_solver.libs/libgomp-1.so", b"x")
    with pytest.raises(ValueError, match="Only the Linux wheel"):
        manager.create(directory)
    windows.write_bytes(saved)
    linux = next(dist.glob("*manylinux*x86_64.whl"))
    with ZipFile(linux) as source, ZipFile(directory / "stripped.whl", "w") as target:
        for entry in source.infolist():
            if not entry.filename.startswith("rko_solver.libs/"):
                target.writestr(entry, source.read(entry.filename))
    linux.unlink()
    (directory / "stripped.whl").rename(linux)
    with pytest.raises(ValueError, match="OpenMP"):
        manager.create(directory)


def test_wheel_without_the_engine_module_fails(payload: tuple[ReleaseManager, Path]) -> None:
    """Refuse a wheel that was built without the compiled extension.

    Args:
        payload (tuple[ReleaseManager, Path]): Valid synthetic artifact set.
    """
    manager, directory = payload
    macos = next((directory / "dist").glob("*macosx*.whl"))
    with ZipFile(macos) as source, ZipFile(directory / "hollow.whl", "w") as target:
        for entry in source.infolist():
            if "_native" not in entry.filename:
                target.writestr(entry, source.read(entry.filename))
    macos.unlink()
    (directory / "hollow.whl").rename(macos)
    with pytest.raises(ValueError, match="compiled engine"):
        manager.create(directory)


def test_smoke_installs_only_this_runners_wheel_and_the_sdist(
    payload: tuple[ReleaseManager, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pick the matching platform wheel and run the C++ check on Linux only.

    Args:
        payload (tuple[ReleaseManager, Path]): Valid synthetic artifact set.
        monkeypatch (pytest.MonkeyPatch): Captures commands instead of installing.
    """
    manager, directory = payload
    manager.create(directory)
    run = Mock()
    monkeypatch.setattr("tools.release.subprocess.run", run)
    runners = (
        ("Linux", "x86_64", 5),
        ("Linux", "aarch64", 5),
        ("Windows", "AMD64", 4),
        ("Windows", "ARM64", 4),
        ("Darwin", "arm64", 4),
        ("Darwin", "x86_64", 4),
    )
    for system, machine, expected in runners:
        run.reset_mock()
        monkeypatch.setattr("tools.release.platform.system", lambda system=system: system)
        monkeypatch.setattr("tools.release.platform.machine", lambda machine=machine: machine)
        manager.smoke(directory)
        assert run.call_count == expected
        requirements = [
            call.args[0][call.args[0].index("--with") + 1] for call in run.call_args_list
        ]
        assert sum("tar.gz" in item for item in requirements) == 2
    monkeypatch.setattr("tools.release.platform.system", lambda: "Linux")
    monkeypatch.setattr("tools.release.platform.machine", lambda: "riscv64")
    with pytest.raises(ValueError, match="matches this runner"):
        manager.smoke(directory)


def test_manifest_identity_and_extra_artifacts_fail(payload: tuple[ReleaseManager, Path]) -> None:
    """Reject renamed releases and unlisted output files.

    Args:
        payload (tuple[ReleaseManager, Path]): Valid synthetic artifact set.
    """
    manager, directory = payload
    manager.create(directory)
    path = directory / "manifest.json"
    data = json.loads(path.read_text())
    data["distribution"] = "another-package"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        manager.verify(directory)
    (directory / "dist" / "private.json").write_text("{}")
    with pytest.raises(ValueError, match="fresh"):
        manager.create(directory)


def test_publication_requires_reviewers_and_an_unused_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Fail closed on missing environment approvals and existing immutable index releases.

    Args:
        monkeypatch (pytest.MonkeyPatch): Substitutes public service metadata, not account rights.
        tmp_path (Path): Trusted empty validation root.
    """
    manager = ReleaseManager(tmp_path)
    protected = {"protection_rules": [{"type": "required_reviewers", "reviewers": ["owner"]}]}
    query = Mock(side_effect=[{"protection_rules": []}, protected, {"urls": []}, protected, None])
    monkeypatch.setattr(manager, "_json", query)
    with pytest.raises(ValueError, match="reviewers"):
        manager.publication_gate("pypi", "v0.3.0a1")
    with pytest.raises(ValueError, match="already exists"):
        manager.publication_gate("pypi", "v0.3.0a1")
    manager.publication_gate("testpypi", "v0.3.0a1")


def test_promotion_rejects_pr_runs_and_changed_index_hashes(
    payload: tuple[ReleaseManager, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require a successful main manual release origin and matching TestPyPI artifacts.

    Args:
        payload (tuple[ReleaseManager, Path]): Original artifact set.
        monkeypatch (pytest.MonkeyPatch): Provides controlled run/index responses.
    """
    manager, directory = payload
    manifest = manager.create(directory)
    run = {
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_branch": "main",
        "path": ".github/workflows/release.yml",
        "head_repository": {"full_name": manager.repository},
    }
    index = {
        "urls": [
            {"filename": item["filename"], "digests": {"sha256": item["sha256"]}}
            for item in manifest["files"]
        ]
    }
    query = Mock(side_effect=[{**run, "event": "pull_request"}, run, {"urls": []}, run, index])
    monkeypatch.setattr(manager, "_json", query)
    with pytest.raises(ValueError, match="origin"):
        manager.promotion("123", manifest)
    with pytest.raises(ValueError, match="exact"):
        manager.promotion("123", manifest)
    manager.promotion("123", manifest)


def test_tag_resolution_checks_version_and_main_ancestry(tmp_path: Path) -> None:
    """Resolve temporary Git tags without creating tags in the actual project.

    Args:
        tmp_path (Path): Disposable Git repository.
    """
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], check=True, capture_output=True)
    for key, value in (
        ("user.name", "Release Test"),
        ("user.email", "release@example.invalid"),
        ("commit.gpgsign", "false"),
    ):
        subprocess.run(["git", "config", key, value], cwd=tmp_path, check=True)
    (tmp_path / "pyproject.toml").write_text('[project]\nname="rko-solver"\nversion="0.3.0a1"\n')
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "fixture"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=tmp_path, check=True
    )
    for tag in ("v0.3.0a1", "v0.3.0a2"):
        subprocess.run(["git", "tag", tag], cwd=tmp_path, check=True)
    manager = ReleaseManager(tmp_path)
    assert manager.resolve("v0.3.0a1")["version"] == "0.3.0a1"
    with pytest.raises(ValueError, match="match"):
        manager.resolve("v0.3.0a2")
    subprocess.run(
        ["git", "checkout", "--orphan", "outside-main"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "pyproject.toml").write_text('[project]\nname="rko-solver"\nversion="0.3.0b1"\n')
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "independent fixture"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "tag", "v0.3.0b1"], cwd=tmp_path, check=True)
    with pytest.raises(subprocess.CalledProcessError):
        manager.resolve("v0.3.0b1")


def test_index_install_separates_dependency_and_target_indexes(
    payload: tuple[ReleaseManager, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install the target package without resolving dependencies from TestPyPI.

    Args:
        payload (tuple[ReleaseManager, Path]): Verified synthetic release artifacts.
        monkeypatch (pytest.MonkeyPatch): Captures commands without installing or publishing.
    """
    manager, directory = payload
    manager.create(directory)
    monkeypatch.setattr(manager, "index_matches", Mock())
    run = Mock()
    monkeypatch.setattr("tools.release.subprocess.run", run)
    manager.installed_index("testpypi", directory)
    commands = [call.args[0] for call in run.call_args_list]
    assert commands[1][commands[1].index("--index-url") + 1] == "https://pypi.org/simple"
    assert commands[2][commands[2].index("--index-url") + 1] == "https://test.pypi.org/simple"
    assert "--no-deps" in commands[2] and "rko-solver==0.3.0a1" in commands[2]
    assert all("--extra-index-url" not in command for command in commands)
    assert "-I" in commands[3] and commands[3][-1].endswith("smoke_installed.py")
    assert commands[4][-1].endswith("smoke_cpp_installed.py")


def test_workflows_publish_only_manual_main_and_promote_without_rebuilding() -> None:
    """Check the credential boundary and build-once promotion structure without running it."""
    yaml = pytest.importorskip("yaml")
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.load(
        (root / ".github/workflows/release.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    jobs = workflow["jobs"]
    assert "refs/heads/main" in jobs["resolve"]["if"]
    assert jobs["candidate"]["if"] == "inputs.index == 'testpypi'"
    assert jobs["publish"]["environment"]["name"] == "${{ inputs.index }}"
    assert jobs["publish"]["permissions"]["id-token"] == "write"
    assert "id-token" not in jobs["verify-index"]["permissions"]
    for name, job in jobs.items():
        if name != "publish":
            assert "id-token" not in job.get("permissions", {})
        for step in job.get("steps", []):
            assert "password" not in step.get("with", {})
            assert "uv build" not in step.get("run", "")
            if "uses" in step:
                reference = step["uses"].split("@")[1]
                assert len(reference) == 40 and all(
                    char in "0123456789abcdef" for char in reference
                )
    publisher = next(
        step for step in jobs["publish"]["steps"] if "pypi-publish@" in step.get("uses", "")
    )
    assert publisher["with"]["skip-existing"] == "false"
    assert publisher["with"]["attestations"] == "true"


@pytest.mark.parametrize(
    ("requirement", "accepted"),
    [
        ('fastapi>=0.1; extra == "service"', True),
        ("fastapi>=0.1", False),
        ('twine>=6; extra == "service"', False),
    ],
)
def test_service_extra_may_use_fastapi_but_runtime_may_not(
    payload: tuple[ReleaseManager, Path], requirement: str, accepted: bool
) -> None:
    """Allow optional extras their own dependencies while keeping tooling out of the core.

    Args:
        payload (tuple[ReleaseManager, Path]): Valid synthetic artifact set.
        requirement (str): Requires-Dist line added to the sdist metadata.
        accepted (bool): Whether the validator should admit it.
    """
    manager, directory = payload
    sdist = directory / "dist" / "rko_solver-0.3.0a1.tar.gz"
    with tarfile.open(sdist) as source:
        members = {member.name: source.extractfile(member).read() for member in source.getmembers()}
    key = "rko_solver-0.3.0a1/PKG-INFO"
    members[key] = members[key].replace(
        b"Requires-Dist: numpy>=2.0.0\n",
        b"Requires-Dist: numpy>=2.0.0\nRequires-Dist: " + requirement.encode() + b"\n",
    )
    with tarfile.open(sdist, "w:gz") as target:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            target.addfile(member, io.BytesIO(content))
    if accepted:
        manager.create(directory)
    else:
        with pytest.raises(ValueError, match="runtime dependency"):
            manager.create(directory)
