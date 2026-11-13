"""Contract tests for the digest-checked store of moved benchmark reference files.

Objective:
- Verify that a reference file is returned only when its SHA-256 matches the manifest, that a
  download is verified before it is kept, and that a missing file never passes silently.

Technical approach:
- Use a temporary cache directory and an injected downloader, so no test touches the network.

Flow (pseudocode):
1. Return a cached file whose digest matches, without downloading.
2. Download an absent file, verify it, and store it.
3. Refuse a download whose digest differs, and store nothing.
4. Report an unreachable source as unavailable, not as a pass.
5. Check that the manifest lists every file that the code or tests read.

Important criteria:
- A digest mismatch is never repaired silently.
- Unknown names are rejected before any download.

Inputs/Outputs:
- Input: a reference name, a cache directory and a downloader function.
- Output: a verified local path, or one of the store's errors.
"""

import hashlib
import pytest
from benchmarks.references import store
from pathlib import Path

PAYLOAD = b"id,value\n1,2\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()


@pytest.fixture
def manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the manifest with one known entry for the duration of a test.

    Args:
        monkeypatch (pytest.MonkeyPatch): Used to swap the loaded manifest.
    """
    monkeypatch.setattr(
        store,
        "load_manifest",
        lambda: {
            "base_url": "https://example.invalid/refs/",
            "files": [{"name": "sample.csv", "sha256": DIGEST, "bytes": len(PAYLOAD)}],
        },
    )


def test_cached_file_with_matching_digest_is_returned_without_download(
    tmp_path: Path, manifest: None
) -> None:
    """A correct cached copy is used and the downloader is never called."""
    (tmp_path / "sample.csv").write_bytes(PAYLOAD)

    def fail(url: str) -> bytes:
        raise AssertionError(f"unexpected download of {url}")

    assert store.ensure("sample.csv", cache_dir=tmp_path, fetch=fail) == tmp_path / "sample.csv"


def test_absent_file_is_downloaded_verified_and_stored(tmp_path: Path, manifest: None) -> None:
    """A missing file is fetched from the manifest URL and kept only after its digest matches."""
    urls: list[str] = []

    def fetch(url: str) -> bytes:
        urls.append(url)
        return PAYLOAD

    path = store.ensure("sample.csv", cache_dir=tmp_path, fetch=fetch)
    assert path.read_bytes() == PAYLOAD
    assert urls == ["https://example.invalid/refs/sample.csv"]


def test_digest_mismatch_is_refused_and_nothing_is_stored(tmp_path: Path, manifest: None) -> None:
    """A download with the wrong content raises and leaves no file behind."""
    with pytest.raises(store.ReferenceDigestError, match="sample.csv"):
        store.ensure("sample.csv", cache_dir=tmp_path, fetch=lambda url: b"tampered")
    assert not (tmp_path / "sample.csv").exists()


def test_unreachable_source_is_unavailable_not_a_pass(tmp_path: Path, manifest: None) -> None:
    """A failed download raises ReferenceUnavailable, so callers cannot treat it as success."""

    def offline(url: str) -> bytes:
        raise OSError("no route to host")

    with pytest.raises(store.ReferenceUnavailable, match="sample.csv"):
        store.ensure("sample.csv", cache_dir=tmp_path, fetch=offline)


def test_unknown_name_is_rejected_before_any_download(tmp_path: Path, manifest: None) -> None:
    """A name that the manifest does not list fails without touching the network."""

    def fail(url: str) -> bytes:
        raise AssertionError("no download expected")

    with pytest.raises(KeyError, match="not-listed.csv"):
        store.ensure("not-listed.csv", cache_dir=tmp_path, fetch=fail)


def test_manifest_lists_every_reference_the_code_and_tests_read() -> None:
    """Every file named by a reader is in the manifest, so no read can point at nothing."""
    listed = {entry["name"] for entry in store.load_manifest()["files"]}
    for name in store.READ_BY_CODE_AND_TESTS:
        assert name in listed, name


def test_concurrent_writers_do_not_share_a_partial_file(tmp_path: Path, manifest: None) -> None:
    """A stale partial file from another writer does not break or replace a verified store."""
    (tmp_path / "sample.csv.stale.part").write_bytes(b"left by another writer")
    path = store.ensure("sample.csv", cache_dir=tmp_path, fetch=lambda url: PAYLOAD)
    assert path.read_bytes() == PAYLOAD
    assert (tmp_path / "sample.csv.stale.part").read_bytes() == b"left by another writer"


def test_refused_replace_keeps_an_already_verified_copy(
    tmp_path: Path, manifest: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep a verified copy that another worker wrote when this platform refuses the replace."""
    (tmp_path / "sample.csv").write_bytes(PAYLOAD)

    def refuse(source: str, target: str) -> None:
        raise PermissionError("target is open in another process")

    monkeypatch.setattr(store.os, "replace", refuse)
    path = store.ensure("sample.csv", cache_dir=tmp_path, fetch=lambda url: PAYLOAD)
    assert path.read_bytes() == PAYLOAD
    assert not list(tmp_path.glob("sample.csv.*.part"))
