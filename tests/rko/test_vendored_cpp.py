"""Integrity test for the vendored C++ reference sources.

Objective:
- Guarantee that the files copied from RKO C++ v2.0 still match the recorded digests, so the
  attribution in NOTICE keeps describing byte-for-byte upstream code.

Technical approach:
- Hash every file listed in `SHA256SUMS` and compare; skip outside a source checkout.

Flow (pseudocode):
1. Locate the vendor directory relative to the repository root.
2. Hash each listed file and assert equality, and that no unlisted file exists.

Important criteria:
- Behavior changes belong in `cpp/rko_cpp.cpp`, never in the vendored headers.

Inputs/Outputs:
- Input: the vendored files and their manifest.
- Output: pass/fail assertions.
"""

import hashlib
import pytest
from pathlib import Path

VENDOR = Path(__file__).resolve().parents[2] / "engines" / "cpp" / "vendor" / "rko_cpp_v2"


@pytest.mark.skipif(not VENDOR.is_dir(), reason="not a source checkout")
def test_vendored_files_match_the_recorded_digests() -> None:
    """Every vendored file hashes to its manifest entry and none is unlisted."""
    listed: dict[str, str] = {}
    for line in (VENDOR / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        listed[name] = digest
    actual = {
        path.relative_to(VENDOR).as_posix()
        for path in VENDOR.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert actual == set(listed)
    for name, digest in listed.items():
        assert hashlib.sha256((VENDOR / name).read_bytes()).hexdigest() == digest, name
