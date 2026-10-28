"""Validate immutable platform-wheel prerelease payloads and controlled promotion.

Objective:
- Keep publication tied to a canonical prerelease tag and the exact tested artifacts.

Technical approach:
- Resolve tags on main, inspect archives without extracting them, and hash the payload.
- Admit exactly one abi3 wheel per supported platform plus the source distribution.
- Require protected publisher environments and a successful main TestPyPI run for promotion.

Flow (pseudocode):
1. Resolve a canonical prerelease tag and check source-version/main ancestry.
2. Validate wheel/sdist metadata, native module, licenses and publication exclusions.
3. Snapshot hashes and verify them again after artifact transfer.
4. Check owner-configured protection/index state before publishing elsewhere.
5. Verify index hashes and a clean install without mixing package indexes.

Important criteria:
- This tool never uploads, creates tags, changes environments or requests stored credentials.
- Owner licensing/account confirmations and per-release approval are human responsibilities.
- PyPI promotion reuses a successful TestPyPI run's bytes; it does not rebuild them.

Inputs/Outputs:
- Input: repository/tag, payload directory and an allowlisted index/run identifier.
- Output: validated metadata, immutable manifest and failures on ambiguous publication state.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tarfile
import tempfile
import time
import tomllib
from email import policy
from email.parser import BytesParser
from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
from packaging.version import Version
from pathlib import Path
from pathlib import PurePosixPath
from urllib.error import HTTPError
from urllib.request import Request
from urllib.request import urlopen
from zipfile import ZipFile


class ReleaseManager:
    """Validate the protected platform-wheel prerelease lifecycle.

    Responsibilities:
    - Bind tested artifacts to their tag/source commit without executing archive contents.
    - Reject unsafe publication state, unreviewed promotion origins and changed payloads.

    Attributes:
        root (Path): Trusted workflow checkout containing validation and smoke tools.
        repository (str): Canonical owner/repository allowed to publish this distribution.
        license_expression (str): Approved licenses for current Python code and fixtures.
    """

    repository = "MarcosPTProenca/rko-solver"
    license_expression = "Apache-2.0 AND MIT"
    licenses = (
        "LICENSE",
        "NOTICE",
        "engines/cpp/vendor/rko_cpp_v2/LICENSE",
        "engines/THIRD_PARTY_LICENSES.txt",
    )
    platforms = {
        "linux-x86_64": re.compile(r"manylinux_\d+_\d+_x86_64"),
        "linux-aarch64": re.compile(r"manylinux_\d+_\d+_aarch64"),
        "windows-x86_64": re.compile(r"win_amd64"),
        "windows-arm64": re.compile(r"win_arm64"),
        "macos-arm64": re.compile(r"macosx_\d+_\d+_arm64"),
        "macos-x86_64": re.compile(r"macosx_\d+_\d+_x86_64"),
    }
    native_modules = {
        name: "rko_solver/_native.pyd"
        if name.startswith("windows")
        else "rko_solver/_native.abi3.so"
        for name in platforms
    }
    machines = {
        ("Linux", "x86_64"): "linux-x86_64",
        ("Linux", "aarch64"): "linux-aarch64",
        ("Linux", "arm64"): "linux-aarch64",
        ("Windows", "AMD64"): "windows-x86_64",
        ("Windows", "ARM64"): "windows-arm64",
        ("Darwin", "arm64"): "macos-arm64",
        ("Darwin", "x86_64"): "macos-x86_64",
    }
    binary_suffix = re.compile(r"\.(so|pyd|dll|dylib)(\.|$)")

    def __init__(self, root: Path) -> None:
        """Store the trusted checkout without importing optimizer modules.

        Args:
            root (Path): Repository containing release metadata and test scripts.
        """
        self.root = root.resolve()

    @staticmethod
    def version(tag: str) -> str:
        """Accept only canonical non-development alpha/beta/RC tags for this pipeline.

        Args:
            tag (str): Explicit v-prefixed prerelease identifier.

        Returns:
            str: Canonical package version without the v prefix.

        Raises:
            ValueError: If the tag is malformed, stable, local, epoch, post or development.
        """
        if not isinstance(tag, str) or not re.fullmatch(r"v[0-9A-Za-z.]{1,63}", tag):
            raise ValueError("Use a canonical prerelease tag such as v0.3.0a1.")
        parsed = Version(tag[1:])
        if (
            str(parsed) != tag[1:]
            or not parsed.is_prerelease
            or parsed.is_devrelease
            or parsed.is_postrelease
            or parsed.local is not None
            or parsed.epoch
        ):
            raise ValueError("This workflow permits canonical alpha/beta/RC prereleases only.")
        return str(parsed)

    def resolve(self, tag: str) -> dict[str, str]:
        """Resolve one immutable commit from a prerelease tag already reachable from main.

        Args:
            tag (str): Existing canonical release tag.

        Returns:
            dict[str, str]: Source commit, version and tag for downstream jobs.

        Raises:
            ValueError: If source metadata does not match the tag.
            subprocess.CalledProcessError: If the tag is absent or not reachable from main.
        """
        version = self.version(tag)
        sha = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}"],
            cwd=self.root,
            text=True,
        ).strip()
        if not re.fullmatch(r"[0-9a-f]{40}", sha):
            raise ValueError("Release source must resolve to a commit SHA.")
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", sha, "origin/main"], cwd=self.root, check=True
        )
        source = subprocess.check_output(
            ["git", "show", f"{sha}:pyproject.toml"], cwd=self.root, text=True
        )
        project = tomllib.loads(source)["project"]
        if project["name"] != "rko-solver" or project["version"] != version:
            raise ValueError("Tag and tagged project name/version must match exactly.")
        return {"source_sha": sha, "version": version, "tag": tag}

    @staticmethod
    def _json(url: str, missing: bool = False) -> dict[str, object] | None:
        """Read bounded metadata from a fixed caller-selected GitHub/PyPI service URL.

        Args:
            url (str): Service URL constructed internally, never an arbitrary CLI endpoint.
            missing (bool): Whether HTTP404 denotes an absent version.

        Returns:
            dict[str, object] | None: JSON object, or permitted HTTP404 absence.

        Raises:
            ValueError: If metadata is oversized or not a JSON object.
            HTTPError: If service access fails outside an allowed absence.
            OSError: If the network request fails.
        """
        headers = {"Accept": "application/json", "User-Agent": "RKO-release-validation"}
        if url.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
        try:
            with urlopen(Request(url, headers=headers), timeout=30) as response:
                data = response.read(2_000_001)
        except HTTPError as exc:
            if missing and exc.code == 404:
                return None
            raise
        if len(data) > 2_000_000:
            raise ValueError("Release service metadata exceeds its bound.")
        decoded = json.loads(data)
        if not isinstance(decoded, dict):
            raise ValueError("Release service must return a JSON object.")
        return decoded

    @staticmethod
    def _index(index: str, version: str) -> str:
        """Choose one index without accepting arbitrary upload/download URLs.

        Args:
            index (str): testpypi or pypi.
            version (str): Validated canonical prerelease version.

        Returns:
            str: Exact version-metadata URL.

        Raises:
            ValueError: If the index is not allowlisted.
        """
        hosts = {"testpypi": "test.pypi.org", "pypi": "pypi.org"}
        if index not in hosts:
            raise ValueError("Index must be testpypi or pypi.")
        return f"https://{hosts[index]}/pypi/rko-solver/{version}/json"

    def publication_gate(self, index: str, tag: str) -> None:
        """Require reviewer protection and refuse an existing immutable index version.

        Args:
            index (str): Explicit target environment/index.
            tag (str): Canonical prerelease tag.

        Raises:
            ValueError: If required reviewers are absent or the version already exists.
            OSError: If GitHub/index protection metadata cannot be checked.
        """
        version = self.version(tag)
        endpoint = self._index(index, version)
        environment = self._json(
            f"https://api.github.com/repos/{self.repository}/environments/{index}"
        )
        rules = environment.get("protection_rules", [])
        if not any(
            rule.get("type") == "required_reviewers" and rule.get("reviewers") for rule in rules
        ):
            raise ValueError("Configure required reviewers on the publishing environment first.")
        if self._json(endpoint, missing=True) is not None:
            raise ValueError(
                "Index version already exists; do not overwrite or skip existing files."
            )

    @classmethod
    def _platform(cls, tags: object) -> str:
        """Classify the compatibility tags of one wheel into a supported platform.

        Args:
            tags (object): Tags parsed from the wheel filename.

        Returns:
            str: One of the supported platform names.

        Raises:
            ValueError: If the wheel is not a CPython 3.14 abi3 build for exactly one platform.
        """
        found = set()
        for tag in tags:
            if tag.interpreter != "cp314" or tag.abi != "abi3":
                raise ValueError("Release wheels must be CPython 3.14 abi3 builds.")
            matches = [name for name, rule in cls.platforms.items() if rule.fullmatch(tag.platform)]
            if len(matches) != 1:
                raise ValueError(f"Unsupported wheel platform tag: {tag.platform}.")
            found.add(matches[0])
        if len(found) != 1:
            raise ValueError("A wheel must target exactly one supported platform.")
        return found.pop()

    @classmethod
    def _inventory(cls, items: list[dict[str, object]]) -> None:
        """Require one wheel per supported platform and one source distribution.

        Args:
            items (list[dict[str, object]]): Validated artifact descriptions.

        Raises:
            ValueError: If a platform is missing, duplicated or unexpected.
        """
        if sorted(str(item["platform"]) for item in items) != sorted([*cls.platforms, "sdist"]):
            raise ValueError("Payload must contain one wheel per platform and one sdist.")

    def _artifact(self, path: Path, version: str, compare_source: bool) -> dict[str, object]:
        """Inspect one release archive without extracting or importing its contents.

        Args:
            path (Path): Wheel or source distribution.
            version (str): Expected canonical version.
            compare_source (bool): Compare bundled license bytes to this build checkout.

        Returns:
            dict[str, object]: Filename, platform, size and SHA256 of the validated artifact.

        Raises:
            ValueError: If archive paths, metadata, resources or artifact kind are invalid.
            OSError: If artifact/source files cannot be read.
        """
        if path.stat().st_size > 100_000_000:
            raise ValueError("Release artifact exceeds the 100MB first-pipeline bound.")
        licenses = self.licenses
        if path.suffix == ".whl":
            name, wheel_version, _, tags = parse_wheel_filename(path.name)
            if name != "rko-solver" or str(wheel_version) != version:
                raise ValueError("Wheel name or version differs from the release.")
            target = self._platform(tags)
            prefix = f"rko_solver-{version}.dist-info/"
            with ZipFile(path) as archive:
                entries = archive.infolist()
                if len(entries) > 5000 or sum(entry.file_size for entry in entries) > 100_000_000:
                    raise ValueError("Wheel inventory exceeds its expanded-size/count bound.")
                if any(entry.file_size > 5_000_000 for entry in entries):
                    raise ValueError("Wheel member exceeds its individual size bound.")
                names = [entry.filename for entry in entries]
                metadata = archive.read(prefix + "METADATA")
                wheel = archive.read(prefix + "WHEEL").decode()
                if "Root-Is-Purelib: false" not in wheel:
                    raise ValueError("A platform wheel must declare a platform installation.")
                for name in licenses:
                    content = archive.read(prefix + "licenses/" + name)
                    if compare_source and content != (self.root / name).read_bytes():
                        raise ValueError("Wheel license/notice differs from approved source.")
            resource_prefix = "rko_solver/"
            if self.native_modules[target] not in names:
                raise ValueError("Wheel lacks the compiled engine module.")
            grafted = [name for name in names if name.startswith("rko_solver.libs/")]
            if target.startswith("linux") and not any(
                name.startswith("rko_solver.libs/libgomp") for name in grafted
            ):
                raise ValueError(
                    "The Linux wheels must bundle the OpenMP runtime of the C++ engine."
                )
            if not target.startswith("linux") and grafted:
                raise ValueError("Only the Linux wheels may bundle shared libraries.")
            allowed = {self.native_modules[target], *grafted}
        elif path.name == f"rko_solver-{version}.tar.gz":
            target = "sdist"
            allowed = set()
            prefix = f"rko_solver-{version}/"
            with tarfile.open(path) as archive:
                members = []
                expanded = 0
                for member in archive:
                    expanded += member.size
                    if len(members) >= 5000 or member.size > 5_000_000 or expanded > 100_000_000:
                        raise ValueError("Source archive exceeds its expanded-size/count bound.")
                    members.append(member)
                names = [member.name for member in members]
                if any(member.issym() or member.islnk() for member in members):
                    raise ValueError("Release source archives cannot contain links.")
                metadata = archive.extractfile(prefix + "PKG-INFO").read()
                for name in licenses:
                    content = archive.extractfile(prefix + name).read()
                    if compare_source and content != (self.root / name).read_bytes():
                        raise ValueError("Source license/notice differs from approved source.")
            resource_prefix = prefix + "src/rko_solver/"
            for required in (
                "Cargo.lock",
                "engines/rust/rko-py/Cargo.toml",
                "engines/cpp/build.rs",
            ):
                if prefix + required not in names:
                    raise ValueError("Source distribution lacks the sources to build the engine.")
        else:
            raise ValueError("Unexpected distribution filename.")
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names are ambiguous.")
        forbidden = {
            ".git",
            ".venv",
            "__pycache__",
            "audits",
            "plans",
            "migrations",
            "superpowers",
            "results",
            "site",
            "target",
            "release",
        }
        for name in names:
            parts = PurePosixPath(name).parts
            if (
                PurePosixPath(name).is_absolute()
                or ".." in parts
                or forbidden.intersection(parts)
                or any(part == ".env" or part.startswith(".env.") for part in parts)
                or name.endswith(".pyc")
                or (self.binary_suffix.search(name) and name not in allowed)
            ):
                raise ValueError("Archive contains excluded/private/native build material.")
        if any(name.startswith(resource_prefix + "examples/") for name in names):
            raise ValueError("Archive contains example material that is excluded from the release.")
        message = BytesParser(policy=policy.default).parsebytes(metadata)
        if (
            message["Name"] != "rko-solver"
            or message["Version"] != version
            or message["Requires-Python"] != ">=3.14"
            or message["License-Expression"] != self.license_expression
        ):
            raise ValueError("Distribution metadata differs from approved release policy.")
        for requirement in message.get_all("Requires-Dist", []):
            parsed = Requirement(requirement)
            optional = parsed.marker is not None and "extra" in str(parsed.marker)
            banned = (
                ("mkdocs", "twine", "pytest")
                if optional
                else (
                    "mkdocs",
                    "twine",
                    "pytest",
                    "fastapi",
                )
            )
            if any(tool in parsed.name.lower() for tool in banned):
                raise ValueError("Release tooling must not become a runtime dependency.")
        if not any(
            name.startswith(resource_prefix + "parameters/") and name.endswith(".toml")
            for name in names
        ):
            raise ValueError("Packaged optimizer parameters are missing.")
        return {
            "filename": path.name,
            "platform": target,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    def create(self, payload: Path, tag: str | None = None) -> dict[str, object]:
        """Create a payload manifest for one build, before cross-platform tests/transfer.

        Args:
            payload (Path): Directory containing only the platform wheels and sdist in dist/.
            tag (str | None): Expected tag, or infer the current candidate for PR checks.

        Returns:
            dict[str, object]: Artifact hashes, tag/version and exact source commit.

        Raises:
            ValueError: If source, changelog or distribution inventory is inconsistent.
            OSError: If files cannot be read/written.
            subprocess.CalledProcessError: If source commit cannot be identified.
        """
        if (payload / "manifest.json").exists():
            raise ValueError("Use a fresh payload; never replace an existing manifest.")
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=self.root, text=True)
        if dirty.strip():
            raise ValueError("Commit the reviewed source before binding artifacts to a commit.")
        version = tomllib.loads((self.root / "pyproject.toml").read_text())["project"]["version"]
        tag = tag or f"v{version}"
        if (
            self.version(tag) != version
            or f"## {version}" not in (self.root / "CHANGELOG.md").read_text()
        ):
            raise ValueError("Candidate version, tag and changelog must agree.")
        marker = payload / "dist" / ".gitignore"
        if marker.exists():
            if (
                marker.is_symlink()
                or not marker.is_file()
                or marker.read_bytes() not in (b"*", b"*\n")
            ):
                raise ValueError("Unexpected uv output marker; use a fresh dedicated payload.")
            marker.unlink()
        paths = sorted((payload / "dist").iterdir())
        files = [self._artifact(path, version, True) for path in paths]
        self._inventory(files)
        manifest = {
            "schema_version": 1,
            "distribution": "rko-solver",
            "version": version,
            "tag": tag,
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
            ).strip(),
            "files": files,
        }
        (payload / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (payload / "SHA256SUMS").write_text(
            "".join(f"{item['sha256']}  dist/{item['filename']}\n" for item in manifest["files"]),
            encoding="utf-8",
        )
        return manifest

    def verify(
        self, payload: Path, tag: str | None = None, source_sha: str | None = None
    ) -> dict[str, object]:
        """Verify every transferred byte before testing or publishing, without rebuilding.

        Args:
            payload (Path): Downloaded immutable artifact payload.
            tag (str | None): Expected promotion tag when publishing.
            source_sha (str | None): Expected immutable tag commit when publishing.

        Returns:
            dict[str, object]: Validated manifest.

        Raises:
            ValueError: If manifest fields, hashes or artifact inventory changed.
            OSError: If payload files cannot be read.
        """
        manifest_path = payload / "manifest.json"
        if manifest_path.stat().st_size > 64_000:
            raise ValueError("Release manifest exceeds its bound.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        version = self.version(manifest["tag"])
        if (
            manifest["schema_version"] != 1
            or manifest["distribution"] != "rko-solver"
            or manifest["version"] != version
            or not re.fullmatch(r"[0-9a-f]{40}", manifest["source_sha"])
            or (tag is not None and manifest["tag"] != tag)
            or (source_sha is not None and manifest["source_sha"] != source_sha)
        ):
            raise ValueError("Manifest identity does not match the release source/tag.")
        paths = sorted((payload / "dist").iterdir())
        actual = [self._artifact(path, version, False) for path in paths]
        self._inventory(actual)
        if actual != manifest["files"]:
            raise ValueError("Transferred artifacts differ from their immutable manifest.")
        checksums = "".join(f"{item['sha256']}  dist/{item['filename']}\n" for item in actual)
        if (payload / "SHA256SUMS").read_text(encoding="utf-8") != checksums:
            raise ValueError("Checksum inventory differs from the artifact manifest.")
        return manifest

    def smoke(self, payload: Path) -> None:
        """Install the sdist and this runner's wheel outside the checkout, with core and extras.

        Args:
            payload (Path): Previously built payload to test without replacing files.

        Raises:
            ValueError: If payload verification fails or the runner has no matching wheel.
            subprocess.CalledProcessError: If any isolated installed check fails.
        """
        manifest = self.verify(payload)
        local = self.machines.get((platform.system(), platform.machine()), "")
        chosen = [item for item in manifest["files"] if item["platform"] in (local, "sdist")]
        if len(chosen) != 2:
            raise ValueError("No wheel in the payload matches this runner's platform.")
        with tempfile.TemporaryDirectory(prefix="rko-release-smoke-") as directory:
            for item in chosen:
                artifact = payload / "dist" / str(item["filename"])
                checks = [("", "smoke_installed.py"), ("[reports]", "smoke_reports_installed.py")]
                if str(item["platform"]).startswith("linux"):
                    checks.append(("", "smoke_cpp_installed.py"))
                for extra, script in checks:
                    requirement = f"rko-solver{extra} @ {artifact.resolve().as_uri()}"
                    subprocess.run(
                        [
                            "uv",
                            "run",
                            "--isolated",
                            "--no-project",
                            "--with",
                            requirement,
                            "python",
                            "-I",
                            str(self.root / "tools" / script),
                        ],
                        cwd=directory,
                        check=True,
                    )

    def index_matches(self, index: str, manifest: dict[str, object]) -> None:
        """Require published index hashes to equal every tested artifact, without uploading.

        Args:
            index (str): Exact testpypi/pypi service.
            manifest (dict[str, object]): Verified artifact/source manifest.

        Raises:
            ValueError: If published files are absent or differ from the tested payload.
            OSError: If the index cannot be queried.
        """
        data = self._json(self._index(index, manifest["version"]), missing=True)
        if data is None:
            raise ValueError("Index has not published the expected version yet.")
        published = {item["filename"]: item["digests"]["sha256"] for item in data["urls"]}
        expected = {item["filename"]: item["sha256"] for item in manifest["files"]}
        if published != expected:
            raise ValueError("Index distributions differ from the exact tested payload.")

    def promotion(self, run_id: str, manifest: dict[str, object]) -> None:
        """Admit promotion only from a successful main manual release run and TestPyPI bytes.

        Args:
            run_id (str): Previous successful TestPyPI workflow run ID.
            manifest (dict[str, object]): Payload downloaded from that run.

        Raises:
            ValueError: If run provenance or TestPyPI hashes are not qualified.
            OSError: If service metadata is unavailable.
        """
        self.promotion_origin(run_id)
        self.index_matches("testpypi", manifest)

    def promotion_origin(self, run_id: str) -> None:
        """Check run provenance before downloading any cross-run publication payload.

        Args:
            run_id (str): Successful same-repository main/manual release run ID.

        Raises:
            ValueError: If run identity, event or completion is untrusted.
            OSError: If GitHub metadata cannot be retrieved.
        """
        if not re.fullmatch(r"[0-9]{1,20}", run_id):
            raise ValueError("Promotion requires a numeric successful TestPyPI run ID.")
        run = self._json(f"https://api.github.com/repos/{self.repository}/actions/runs/{run_id}")
        if (
            run.get("conclusion") != "success"
            or run.get("event") != "workflow_dispatch"
            or run.get("head_branch") != "main"
            or run.get("path") != ".github/workflows/release.yml"
            or run.get("head_repository", {}).get("full_name") != self.repository
        ):
            raise ValueError(
                "Promotion origin must be this repository's successful main release workflow."
            )

    def installed_index(self, index: str, payload: Path) -> None:
        """Verify uploaded hashes and install the index wheel with dependencies from PyPI only.

        Args:
            index (str): Target index after an explicitly approved publication.
            payload (Path): Original tested artifacts/manifest, not rebuilt distributions.

        Raises:
            ValueError: If index metadata does not converge to the expected artifacts.
            OSError: If service or filesystem operations fail.
            subprocess.CalledProcessError: If the clean index installation or smoke fails.
        """
        manifest = self.verify(payload)
        for attempt in range(12):
            try:
                self.index_matches(index, manifest)
                break
            except ValueError:
                if attempt == 11:
                    raise
                time.sleep(10)
        host = "test.pypi.org" if index == "testpypi" else "pypi.org"
        with tempfile.TemporaryDirectory(prefix="rko-index-check-") as directory:
            environment = Path(directory) / "venv"
            subprocess.run(["uv", "venv", "--python", "3.14", str(environment)], check=True)
            python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--index-url",
                    "https://pypi.org/simple",
                    "numpy>=2.0.0",
                ],
                check=True,
            )
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    "--index-url",
                    f"https://{host}/simple",
                    "--no-deps",
                    "--only-binary",
                    ":all:",
                    f"rko-solver=={manifest['version']}",
                ],
                check=True,
            )
            for script in ("smoke_installed.py", "smoke_cpp_installed.py"):
                subprocess.run(
                    [str(python), "-I", str(self.root / "tools" / script)],
                    cwd=directory,
                    check=True,
                )


def main() -> None:
    """Run an explicit validation command; none performs publication or tag creation.

    Raises:
        ValueError: If a release policy or identity check fails.
        OSError: If required metadata/services are unavailable.
        subprocess.CalledProcessError: If Git/build-install smoke operations fail.
    """
    parser = argparse.ArgumentParser(description="Validate controlled RKO prerelease payloads")
    parser.add_argument(
        "command",
        choices=("resolve", "create", "verify", "smoke", "gate", "promotion", "origin", "index"),
    )
    parser.add_argument("--tag")
    parser.add_argument("--source-sha")
    parser.add_argument("--payload", type=Path, default=Path("release"))
    parser.add_argument("--index", choices=("testpypi", "pypi"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    manager = ReleaseManager(Path(__file__).resolve().parents[1])
    result = None
    if args.command == "resolve":
        if not args.tag:
            parser.error("resolve requires --tag")
        result = manager.resolve(args.tag)
        if os.environ.get("GITHUB_OUTPUT"):
            with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as output:
                output.write("".join(f"{key}={value}\n" for key, value in result.items()))
    elif args.command == "create":
        result = manager.create(args.payload, args.tag)
    elif args.command == "verify":
        result = manager.verify(args.payload, args.tag, args.source_sha)
    elif args.command == "smoke":
        manager.smoke(args.payload)
    elif args.command == "gate":
        if not args.tag or not args.index:
            parser.error("gate requires --tag and --index")
        manager.publication_gate(args.index, args.tag)
    elif args.command == "origin":
        if not args.run_id:
            parser.error("origin requires --run-id")
        manager.promotion_origin(args.run_id)
    elif args.command == "promotion":
        if not args.run_id:
            parser.error("promotion requires --run-id")
        manager.promotion(args.run_id, manager.verify(args.payload, args.tag, args.source_sha))
    elif args.command == "index":
        if not args.index:
            parser.error("index requires --index")
        manager.installed_index(args.index, args.payload)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
