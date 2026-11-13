"""Digest-checked access to benchmark reference files that live outside Git.

Objective:
- Give code and tests one way to obtain a moved reference file, verified by SHA-256, from the
  release assets listed in MANIFEST.json.

Technical approach:
- The manifest records each file's name, byte size and SHA-256. A cached copy is used only when
  its digest matches. A download is written to a temporary file, verified, then moved into place.
- The downloader is injectable, so tests never touch the network.

Flow (pseudocode):
1. Look up the name in the manifest; an unknown name raises KeyError before any I/O.
2. If the cached file exists and its digest matches, return its path.
3. Otherwise fetch `base_url + name`; any transport error raises ReferenceUnavailable.
4. Verify the digest of the downloaded bytes; a mismatch raises ReferenceDigestError,
   and nothing is stored.
5. Move the verified file into the cache and return its path.

Important criteria:
- A missing or altered file never passes silently: callers get an error, not partial data.
- The cache directory defaults to a per-user location and can be set by RKO_REFERENCE_CACHE.

Inputs/Outputs:
- Input: a reference file name, an optional cache directory and an optional fetch function.
- Output: a Path to a verified local file, or a ReferenceUnavailable, ReferenceDigestError
  or KeyError.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

MANIFEST_PATH = Path(__file__).with_name("MANIFEST.json")

READ_BY_CODE_AND_TESTS = (
    "general-sa-preset-engines-reference.json",
    "engine-ratio-reference.json",
    "general-sa-preset-reference.json",
    "modeling-validation-box-reference-13b6934.jsonl",
    "modeling-validation-reference-13b6934.jsonl",
    "native-evidence.csv.gz",
    "prefill-expanded-pairs.csv",
    "prefill-expanded-summary.json",
    "prefill-work-reference.jsonl",
    "timing-cv-adaptive-pinned-cpu1-wsl2.jsonl",
    "timing-cv-temperature-free-wsl2.jsonl",
    "timing-cv-temperature-pinned-cpu1-wsl2.jsonl",
    "tuning-wide-expanded-confirm-reference.jsonl",
    "tuning-wide-expanded-reference.jsonl",
    "tuning-wide-original-confirm-reference.jsonl",
    "tuning-wide-original-reference.jsonl",
)


class ReferenceUnavailable(RuntimeError):
    """Raised when a reference file cannot be obtained from its source."""


class ReferenceDigestError(RuntimeError):
    """Raised when a downloaded or cached reference does not match its manifest digest."""


def load_manifest() -> dict[str, Any]:
    """Read the committed manifest of reference files.

    Returns:
        dict[str, Any]: The manifest, with `base_url` and the list of `files`.
    """
    try:
        data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"reference manifest is unreadable: {MANIFEST_PATH}") from error
    data["base_url"] = data.get(
        "base_url",
        "https://github.com/MarcosPTProenca/rko-solver/releases/download/references-v1/",
    )
    return data


def cache_directory() -> Path:
    """Return the directory where verified reference copies are kept.

    Returns:
        Path: `RKO_REFERENCE_CACHE` when set, otherwise a per-user cache directory.
    """
    configured = os.environ.get("RKO_REFERENCE_CACHE")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "rko-references"


def resolve(path: Path | None, name: str) -> Path:
    """Return an explicit path when given, otherwise the verified copy of a reference file.

    Args:
        path (Path | None): Path supplied by the caller, or None.
        name (str): Reference file name used when no path is supplied.

    Returns:
        Path: The explicit path, or the digest-verified local copy of `name`.
    """
    return path if path is not None else ensure(name)


def _entry(name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Find one file's manifest entry.

    Args:
        name (str): File name as listed in the manifest.
        manifest (dict[str, Any]): The loaded manifest.

    Returns:
        dict[str, Any]: The entry with `name`, `sha256` and `bytes`.

    Raises:
        KeyError: If the manifest does not list the name.
    """
    for entry in manifest["files"]:
        if entry["name"] == name:
            return entry
    raise KeyError(f"{name!r} is not listed in the reference manifest")


def _download(url: str) -> bytes:
    """Fetch one URL with the standard library.

    Args:
        url (str): Absolute URL of the reference file.

    Returns:
        bytes: The response body.
    """
    request = urllib.request.Request(url, headers={"User-Agent": "rko-solver-references"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def ensure(
    name: str,
    cache_dir: Path | None = None,
    fetch: Callable[[str], bytes] | None = None,
) -> Path:
    """Return a local, digest-verified copy of a reference file.

    Args:
        name (str): File name listed in the manifest.
        cache_dir (Path | None): Directory for verified copies; None uses `cache_directory()`.
        fetch (Callable[[str], bytes] | None): Downloader taking a URL; None uses HTTPS.

    Returns:
        Path: Path of the verified local file.

    Raises:
        KeyError: If the manifest does not list the name.
        ReferenceDigestError: If the file's content does not match its digest.
        ReferenceUnavailable: If the file is absent and cannot be downloaded.
    """
    manifest = load_manifest()
    entry = _entry(name, manifest)
    directory = cache_dir if cache_dir is not None else cache_directory()
    target = directory / name
    if _is_verified(target, entry):
        return target
    content = _fetch_verified(manifest, entry, fetch)
    return _store(directory, target, content, entry)


def _is_verified(path: Path, entry: dict[str, Any]) -> bool:
    """Tell whether a local file exists and matches its manifest digest.

    Args:
        path (Path): Local file to check.
        entry (dict[str, Any]): Manifest entry carrying the expected digest.

    Returns:
        bool: True when the file exists and its SHA-256 matches.
    """
    return path.exists() and _digest(path.read_bytes()) == entry["sha256"]


def _fetch_verified(
    manifest: dict[str, Any], entry: dict[str, Any], fetch: Callable[[str], bytes] | None
) -> bytes:
    """Download one reference file and check its digest before any use.

    Args:
        manifest (dict[str, Any]): Loaded manifest, for the base URL.
        entry (dict[str, Any]): Manifest entry of the file.
        fetch (Callable[[str], bytes] | None): Downloader; None uses HTTPS.

    Returns:
        bytes: The verified content.

    Raises:
        ReferenceUnavailable: If the download fails.
        ReferenceDigestError: If the content does not match the digest.
    """
    download = fetch if fetch is not None else _download
    url = manifest["base_url"] + entry["name"]
    try:
        content = download(url)
    except OSError as error:
        raise ReferenceUnavailable(
            f"could not download {entry['name']} from {url}: {error}"
        ) from error
    if _digest(content) != entry["sha256"]:
        raise ReferenceDigestError(f"downloaded {entry['name']} does not match its manifest digest")
    return content


def _store(directory: Path, target: Path, content: bytes, entry: dict[str, Any]) -> Path:
    """Write verified content atomically, tolerating a concurrent writer on refused replaces.

    Args:
        directory (Path): Cache directory, created when missing.
        target (Path): Final path of the reference file.
        content (bytes): Verified content to store.
        entry (dict[str, Any]): Manifest entry, to recognise a copy another writer finished.

    Returns:
        Path: The stored file.

    Raises:
        PermissionError: If the replace is refused and no verified copy exists.
    """
    directory.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=target.name + ".", suffix=".part")
    with os.fdopen(handle, "wb") as sink:
        sink.write(content)
    try:
        os.replace(temporary, target)
    except PermissionError:
        Path(temporary).unlink(missing_ok=True)
        if _is_verified(target, entry):
            return target
        raise
    return target


def main(argv: list[str] | None = None) -> int:
    """Print the verified local path of a reference file, fetching it when needed.

    Args:
        argv (list[str] | None): Command-line arguments; None reads `sys.argv`.

    Returns:
        int: Process exit status, zero on success.
    """
    parser = argparse.ArgumentParser(description="Print the verified path of a reference file.")
    parser.add_argument("name")
    print(ensure(parser.parse_args(argv).name))
    return 0


def _digest(content: bytes) -> str:
    """Return the SHA-256 hex digest of some bytes.

    Args:
        content (bytes): Bytes to hash.

    Returns:
        str: Lowercase hexadecimal SHA-256.
    """
    return hashlib.sha256(content).hexdigest()


if __name__ == "__main__":
    sys.exit(main())
