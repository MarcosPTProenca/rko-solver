"""Verified download catalog for public benchmark instances.

Objective:
- Fetch every public input from a pinned URL, verify its SHA-256 and keep it in a local cache,
  so runs never depend on whatever a remote host serves later.

Technical approach:
- A frozen `Source` record holds the relative cache path, the URL and the expected digest.
- The TSPLIB files come from a public mirror of the TSPLIB95 archive; the optimal lengths come
  from the mirror's `solutions` file, not from memory.
- KPLIB inputs reuse the pinned revision and digests of `benchmarks.engines.cpp_python_knapsack`.
- OR-Library files are fetched from the Brunel University page that hosts them.

Flow (pseudocode):
1. Build the catalog of sources for the requested families.
2. For each source, reuse the cached file when its digest matches.
3. Otherwise download it, verify the digest, and write it atomically; raise on any mismatch.

Important criteria:
- A digest mismatch deletes the downloaded file and raises; it is never used.
- The catalog is the only place where URLs and digests are written down.

Inputs/Outputs:
- Input: cache directory and the family names to fetch.
- Output: local paths keyed by the source name.
"""

from __future__ import annotations

import hashlib
import urllib.request
from benchmarks.engines.cpp_python_knapsack import INSTANCE_SHA256
from benchmarks.engines.cpp_python_knapsack import KPLIB_RAW
from benchmarks.engines.cpp_python_knapsack import KPLIB_REVISION
from dataclasses import dataclass
from pathlib import Path

TSPLIB_MIRROR = "https://raw.githubusercontent.com/mastqe/tsplib/master"
ORLIB_BASE = "https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files"
TSPLIB_DIGESTS = {
    "eil51": "66c05ad077ffd64d5778e358bbd16be049f8c8ae880cbd24f034487331d1ccce",
    "berlin52": "8496a5838133e3c3eb25a9fa8893b1ac302593fc8dc325350643a0cabd8abadc",
    "st70": "60c09e8886609bf22d6dc699fe7d7aa1924207d084578aad1a377ec65cf15b1b",
    "kroA100": "e103100c1cf31dfc06be95a9d04011b5a8753bb65a3339594ca34404e574bdf5",
    "kroB100": "283d8c912e3334deea76cc9fe95e915d09111979e7753d0affaf14d9aa21cdbe",
    "ch150": "05abd424f773269b13477d367e534797aac080fb536a28074a72a9c0ba8064f5",
    "tsp225": "3d356ab7c1a724cd1e05e0809cf2db11932c3bc8f54f6124e02c2b6c7f5fd901",
    "a280": "3fb489af4be6d04f2e84e73ec8302c32c65e6d344187b2347685fb311416e7bf",
    "pcb442": "0f58bc2f1aa0023764ebc14b3b0dcb6e765ffe173963fdc1cb8443d969842b51",
    "rat783": "b9535dc5e44549f613bf6e34ee8667eed6f7a13e6b74c0b0451de95a480cabde",
}
TSPLIB_SOLUTIONS_DIGEST = "d6fa81c754f9b559bfbf19121c577d06aa3da195142d16d5fc2a14569f9be47c"
ORLIB_DIGESTS = {
    "mknap1.txt": "727c5f90b6acafa0896ce4b5b5559e2995303b735ee083a07e9b724738fac283",
    "mknapcb1.txt": "53011406c49617c8fe0bd783cf4bd3f0552657a70d56529c926304f750d995c3",
}


@dataclass(frozen=True, slots=True)
class Source:
    """One downloadable file with its pinned location and digest.

    Responsibilities:
    - Describe where a file lives and what content it must have.

    Attributes:
        relative (str): Path under the cache directory.
        url (str): Pinned download location.
        sha256 (str): Expected lowercase hexadecimal digest of the file content.
    """

    relative: str
    url: str
    sha256: str


class PublicSources:
    """Build the verified source catalog and fetch files into a cache.

    Responsibilities:
    - Enumerate every pinned file and resolve it to a verified local path.
    """

    @staticmethod
    def tsplib() -> tuple[Source, ...]:
        """Return the TSPLIB instance sources followed by the optimal-values file.

        Returns:
            tuple[Source, ...]: One source per instance and one for `solutions`.
        """
        instances = tuple(
            Source(f"tsplib/{name}.tsp", f"{TSPLIB_MIRROR}/{name}.tsp", digest)
            for name, digest in TSPLIB_DIGESTS.items()
        )
        solutions = Source(
            "tsplib/solutions", f"{TSPLIB_MIRROR}/solutions", TSPLIB_SOLUTIONS_DIGEST
        )
        return (*instances, solutions)

    @staticmethod
    def orlib() -> tuple[Source, ...]:
        """Return the OR-Library multi-dimensional knapsack files.

        Returns:
            tuple[Source, ...]: Sources for `mknap1.txt` and `mknapcb1.txt`.
        """
        return tuple(
            Source(f"orlib/{name}", f"{ORLIB_BASE}/{name}", digest)
            for name, digest in ORLIB_DIGESTS.items()
        )

    @staticmethod
    def kplib() -> tuple[Source, ...]:
        """Return the pinned KPLIB 500-item knapsack sources.

        Returns:
            tuple[Source, ...]: One source per instance in the pinned digest table.
        """
        sources = []
        for path, digest in INSTANCE_SHA256.items():
            kind, filename = path.split("/")
            url = f"{KPLIB_RAW}/{KPLIB_REVISION}/{kind}/n00500/R01000/{filename}"
            sources.append(Source(f"kplib/{path}", url, digest))
        return tuple(sources)

    @staticmethod
    def fetch(source: Source, cache_dir: Path) -> Path:
        """Return the verified local copy of one source, downloading it when needed.

        Args:
            source (Source): Pinned location and digest.
            cache_dir (Path): Root cache directory.

        Returns:
            Path: Local file whose digest equals `source.sha256`.

        Raises:
            ValueError: If the downloaded content has a different digest.
            OSError: If the download fails.
        """
        path = cache_dir / source.relative
        if path.exists() and PublicSources._digest(path) == source.sha256:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        with urllib.request.urlopen(source.url, timeout=60) as response:
            partial.write_bytes(response.read())
        if PublicSources._digest(partial) != source.sha256:
            partial.unlink()
            raise ValueError(f"Digest mismatch for {source.url}; file discarded.")
        partial.replace(path)
        return path

    @staticmethod
    def _digest(path: Path) -> str:
        """Hash the exact downloaded bytes.

        Args:
            path (Path): Local file to verify.

        Returns:
            str: Lowercase SHA-256 hexadecimal digest.

        Raises:
            OSError: If the file cannot be read.
        """
        return hashlib.sha256(path.read_bytes()).hexdigest()
