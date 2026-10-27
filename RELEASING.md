# Controlled prereleases

This procedure prepares and publishes **explicitly approved prereleases**, not
stable releases or automatic uploads from main. A source version/changelog is not
proof that a package is available on an index.

## Owner decisions and scope

The distribution is **rko-solver** (import `rko_solver`). On 2026-10-24 the repository owner
approved the current combined Python code's rights, licenses and attribution, recorded in
NOTICE as an owner attestation, not an independent legal certification. Source uses Apache-2.0; the bundled
reference fixtures retain MIT, hence the combined distribution expression
`Apache-2.0 AND MIT`. Review new third-party additions before every release.

The current candidate is **0.3.0a3**. (0.3.0a1 reached TestPyPI only, where its post-upload
check failed on a test-harness bug, and 0.3.0a2 is already on PyPI; an index version can never
be reused, so each candidate gets a new number.) The payload is **one CPython 3.14 abi3
wheel per platform plus the sdist**: Linux x86_64 and aarch64 (`manylinux_2_28`, with
the optional C++ reference engine and its OpenMP runtime bundled), Windows x86_64 and
ARM64, and macOS arm64 and x86_64. Core is NumPy-only; reports are optional. The
compiled engine is part of every wheel, but `engine="rust"` and `engine="cpp"`
stay opt-in and the default engine and execution do not change. Platforms without a
wheel (musl Linux, 32-bit systems, other architectures) install from the sdist, which
needs a Rust toolchain; free-threaded CPython is checked by installing the sdist, not
by a wheel. Other interpreters are not certified, and metadata's >=3.14 is not evidence
that every future Python release was tested. No older-Python support is added.

The validator admits exactly that set: the wheel tags must be `cp314-abi3` for one
of the six platforms, each wheel must contain the compiled module, only the Linux
wheels may bundle shared libraries (the OpenMP runtime), and the sdist must carry the
Cargo workspace needed to build. Third-party Rust crate notices
(`engines/THIRD_PARTY_LICENSES.txt`, checked against `Cargo.lock` in CI), the vendored
C++ MIT license, `LICENSE` and `NOTICE` must be byte-identical to the reviewed source.
Speed and quality claims are limited to the evidence in
[benchmarks](benchmarks/README.md); no C++ parity beyond it, optimality or universal
callback speed is implied.

## One-time owner setup

1. Protect main and version tags `v*` against unauthorized changes/deletion.
2. Create GitHub environments named **testpypi** and **pypi**, each with at least
   one required reviewer and main-only deployment rules. Prefer separate reviewers
   and prevent self-review when the maintainer team permits it. The validator
   fails if required-reviewer protection is missing; naming an environment alone
   is not adequate protection.
3. Create the PyPI account (with two-factor authentication) and, in Account settings →
   Publishing, add a **pending** GitHub Trusted Publisher for the new project name
   **rko-solver**: owner `MarcosPTProenca`, repository **rko-solver**, workflow filename
   **release.yml**, environment **pypi**. Repository casing/rename matters. The first
   upload creates the project.
4. Independently configure the **TestPyPI** account/project or pending publisher
   for distribution rko-solver and the same repository/workflow, environment **testpypi**.
   PyPI rights do not grant TestPyPI rights. If that name is unavailable there,
   stop and coordinate ownership; do not publish under a different test name and
   pretend it proves promotion of the same distribution.
5. Use no API token/password secret. Publishing is short-lived OIDC through the
   pinned official PyPA action; its PEP740 attestations remain enabled.

The agent has not created environments/publishers/tags or uploaded packages.
Owner confirmation authorizes preparation, not an implicit upload. Ordinary PR
jobs remain read-only and receive no publisher OIDC permission.

## Prepare and review a candidate

- Review CHANGELOG, NOTICE, supported targets,
  version and source/package URLs. Keep exact tag/version correspondence.
- Run the mandatory library checks plus:

```bash
uv run --locked --group release pytest src/rko_solver/tests/test_release.py -q
uv run --locked python tools/third_party_licenses.py --check
```

The six platform wheels can only be produced by the **Wheels** workflow (maturin,
manylinux container, Windows and macOS runners); a payload is created only from all of
them plus the sdist, so `create` runs in **Distribution checks**. To rehearse by hand,
download the `wheel-*` and `sdist` artifacts of a successful run into `release/dist`
(an empty directory) and run:

```bash
uv run --locked --only-group release twine check --strict release/dist/*
uv run --locked --only-group release python tools/release.py create --payload release
uv run --locked --only-group release python tools/release.py smoke --payload release
```

Commit the reviewed source before binding artifacts to a commit. The validator
rejects dirty source and replacement manifests. Use an empty/new payload for each
candidate. Do not overwrite an old manifest to disguise modified artifacts. Output is ignored by Git.
The exact `*` ignore marker generated by uv inside its output directory is removed
before manifest creation; unrecognized hidden files are not silently discarded. The first
pipeline caps archives, rejects private/unsafe paths and any compiled material other
than the expected engine module and the Linux OpenMP runtime, and checks platform tags,
metadata, resources and license bytes. These technical checks do not replace
human secret/provenance review.

The **Distribution checks** workflow builds the six wheels and the sdist once (the
**Wheels** workflow), assembles them on Ubuntu and retains one immutable
`release-candidate` artifact with the seven files, manifest and SHA256SUMS. A runner of
each platform (including the hosted ARM runners) installs the sdist and its own wheel
outside the checkout, with core and report extras, without editable imports or
PYTHONPATH; the Linux wheels also run a C++ engine check. Installing an sdist naturally compiles a
temporary installation wheel; it does not replace the publication payload.
Existing independent quality builds are not publication artifacts.

After the reviewed candidate is merged to protected main, the owner creates the
matching immutable tag (currently `v0.3.0a3`). No tag is created by these tools.
The resolver accepts canonical alpha/beta/RC tags reachable from main and checks
the tagged pyproject version; stable/dev/local/post tags are intentionally rejected.

## TestPyPI first

Run **Controlled prerelease** manually, selecting branch **main**, the exact tag,
and index **testpypi**. Do not dispatch it from a feature branch. Review/approve
the testpypi environment only after checking the source commit and intended version.

The workflow resolves an immutable tag SHA, calls the same build-once/matrix-test
workflow, downloads the exact tested payload, verifies its manifest and environment
protection, refuses an existing index version, then invokes the PyPA OIDC uploader.
No distribution is built in the privileged publish job. It inspects archives but
does not install candidate package code there.

An unprivileged follow-up job verifies index filenames/hashes and a clean core
installation/real solve from that exact index. It first installs NumPy from approved
PyPI, then installs the exact rko-solver version from TestPyPI with **--no-deps**. There
is no extra-index-url mixing or silent dependency lookup through TestPyPI.

Save the **successful TestPyPI workflow run ID**. A partial upload, failed run,
missing approval or failed post-upload verification is not qualified promotion.
No successful public upload has been exercised during implementation of this PR.

## Promote the same bytes to PyPI

Dispatch **Controlled prerelease** again on main, same tag, index **pypi**, and the
successful TestPyPI run ID. This path **does not run a build**. It admits only this
repository's successful main/manual release workflow origin, downloads that run's
immutable payload, matches its tag/commit/hashes, and checks TestPyPI filenames/
digests against the original manifest. Approve the separate pypi environment.

The original artifact is retained for 30 days; promote within that window. If it
expires, do not rebuild and call the result the same candidate. Prepare a new
version with a new test run. Reusing an arbitrary PR/artifact run is forbidden.
The target version must be absent; skip-existing is disabled. Protected tags and
SHA checks prevent tag retargeting from silently selecting different source.

After PyPI upload, the unprivileged verification repeats index hashes and a fresh
exact-version core installation outside the checkout. Only after that succeeds
should maintainers announce availability and update website/README install commands
to the actual verified version/extras. Keep prerelease selection explicit; an
unqualified pip install can still select the old stable environment API.

## Failure, withdrawal and audit trail

- Before upload, fix the candidate and rerun normal checks; never alter a tag that
  already identifies published/tested artifacts. Prefer a fresh prerelease number.
- PyPI releases/files are immutable. A partial upload or publication error is not
  fixed by overwriting files or enabling skip-existing. Publish a corrected **new**
  version after review; use the index's documented yank/withdrawal mechanisms when
  appropriate, without claiming that installed environments automatically roll back.
- Preserve the source tag, manifest, checksums, successful test run, publisher
  approvals and PEP740 attestations. No credentials belong in archives or logs.
- A README/version bump is not a release. Source code, index metadata and actual
  clean installation must agree before claiming publication.
- Stable publication, wheels for other platforms, unsupported Python targets and service
  deployment are intentionally not authorized by this prerelease preparation.
