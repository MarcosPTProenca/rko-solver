"""Contract tests for the pinned QAPLIB source catalog.

Objective:
- Verify that every QAP instance the study may use has a pinned mirror URL and a well-formed SHA-256
  digest, and that the catalog cannot drift from the digest table.

Technical approach:
- Read the catalog and the digest table only; no download happens in these tests.

Flow (pseudocode):
1. Build the QAP sources from the catalog.
2. Check the names, the mirror URLs and the digests against the table.

Important criteria:
- Every source has a 64-character lowercase hexadecimal digest.
- Sources come from the pinned mirror directory, not from a moving listing.

Inputs/Outputs:
- Input: the catalog and the QAP digest table.
- Output: assertions on names, URLs and digests.
"""

import re
from benchmarks.public.sources import QAP_DIGESTS
from benchmarks.public.sources import QAP_MIRROR
from benchmarks.public.sources import PublicSources


def test_qap_catalog_lists_every_digest_in_the_table() -> None:
    """Each table entry becomes one source with the same digest and a `.qap` path."""
    sources = PublicSources.qap()
    assert {source.relative for source in sources} == {f"qap/{name}.qap" for name in QAP_DIGESTS}
    for source in sources:
        name = source.relative.removeprefix("qap/").removesuffix(".qap")
        assert source.sha256 == QAP_DIGESTS[name]
        assert source.url == f"{QAP_MIRROR}/{name}.qap"


def test_qap_digests_are_lowercase_sha256_hex() -> None:
    """Every pinned digest has the 64-character lowercase hexadecimal form."""
    assert len(QAP_DIGESTS) == 6
    for name, digest in QAP_DIGESTS.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest), name


def test_qap_mirror_is_the_qaplib_directory_of_the_named_repository() -> None:
    """The mirror path points at the QAPLIB directory of the named mirror repository."""
    assert QAP_MIRROR == "https://raw.githubusercontent.com/didoudiaz/QAP-instances/main/QAPLIB"
