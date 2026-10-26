"""Immutable experimental decoder artifacts and non-executing preflight checks.

Objective:
- Preserve seed/candidate identity without importing untrusted source on the host.
Technical approach:
- Validate a restricted Python module shape and hash UTF-8 source bytes.
- Export source plus bounded JSON metadata, never pickle or credentials.
Flow (pseudocode):
1. Validate source and construct a content-addressed descriptor.
2. Export into a new directory; load by checking metadata and source hash.
Important criteria:
- Syntax checks are defense in depth, not a Python security sandbox.
Inputs/Outputs:
- Input: source, family and optional parent hash.
- Output: experimental artifact; execution is exclusively an isolated-worker operation.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

FAMILIES = ("knapsack", "scheduling", "set_cover")
MAX_SOURCE = 32768
_FORBIDDEN = {
    "open",
    "exec",
    "eval",
    "compile",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "input",
    "breakpoint",
    "help",
    "exit",
    "quit",
    "memoryview",
    "load",
    "save",
    "savez",
    "savez_compressed",
    "loadtxt",
    "savetxt",
    "fromfile",
    "tofile",
    "memmap",
    "ctypes",
    "lib",
    "f2py",
    "random",
    "recfromcsv",
    "recfromtxt",
}


def validate_source(source: str) -> None:
    """Reject malformed or obviously unauthorized decoder modules without executing them.

    Args:
        source (str): Python module defining decode(keys, instance).

    Raises:
        ValueError: If module size, interface or syntactic policy is invalid.
    """
    if not isinstance(source, str) or not 0 < len(source.encode("utf-8")) <= MAX_SOURCE:
        raise ValueError("Decoder source must contain at most 32768 UTF-8 bytes.")
    try:
        tree = ast.parse(source)
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Decoder source is not valid Python.") from error
    decoders = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.Import):
            if any(alias.name not in {"numpy", "math"} for alias in node.names):
                raise ValueError("Only numpy and math imports are permitted.")
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list or node.args.defaults or node.args.kw_defaults:
                raise ValueError("Decorators and argument defaults are not permitted.")
            if node.name == "decode":
                decoders.append(node)
        else:
            raise ValueError("Module scope permits only imports, docstrings and functions.")
    if len(decoders) != 1:
        raise ValueError("Exactly one decode(keys, instance) function is required.")
    args = decoders[0].args
    if (
        [a.arg for a in args.posonlyargs + args.args] != ["keys", "instance"]
        or args.vararg
        or args.kwarg
        or args.kwonlyargs
    ):
        raise ValueError("Decoder signature must be decode(keys, instance).")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            a.name not in {"numpy", "math"} for a in node.names
        ):
            raise ValueError(
                "Only numpy and math imports are permitted, including inside functions."
            )
        if isinstance(
            node, (ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef)
        ):
            raise ValueError("Unsupported decoder construct.")
        name = (
            node.id
            if isinstance(node, ast.Name)
            else (node.attr if isinstance(node, ast.Attribute) else "")
        )
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            raise ValueError("Attribute mutation/persistent function state is not permitted.")
        if "__" in name or name in _FORBIDDEN:
            raise ValueError(f"Forbidden decoder name: {name}")


@dataclass(frozen=True)
class DecoderArtifact:
    """Own a family decoder and its immutable experimental identity.

    Responsibilities:
    - Separate portable source identity from runtime execution.

    Attributes:
        family (str): Supported problem family.
        source (str): Validated module text, not yet executed.
        digest (str): SHA-256 source hash.
        parent (str | None): Prior source hash, absent for a seed.
        status (str): Always experimental in this initial implementation.
        qualified_image (str | None): Pinned runtime required for evidence-compatible replay.
    """

    family: str
    source: str
    digest: str
    parent: str | None = None
    status: str = "experimental"
    qualified_image: str | None = None

    @classmethod
    def create(cls, family: str, source: str, parent: str | None = None) -> DecoderArtifact:
        """Create a source-validated artifact without executing it.

        Args:
            family (str): Supported family.
            source (str): Decoder module.
            parent (str | None): Optional lowercase SHA-256 lineage parent.

        Returns:
            DecoderArtifact: Experimental descriptor.

        Raises:
            ValueError: If family, parent or source violates the contract.
        """
        if family not in FAMILIES:
            raise ValueError("Unsupported decoder family.")
        if parent is not None and (
            not isinstance(parent, str)
            or len(parent) != 64
            or any(c not in "0123456789abcdef" for c in parent)
        ):
            raise ValueError("Parent must be a lowercase SHA-256 hash.")
        validate_source(source)
        return cls(family, source, hashlib.sha256(source.encode("utf-8")).hexdigest(), parent)

    def metadata(self) -> dict[str, Any]:
        """Return portable contract metadata with no instance data or credentials.

        Returns:
            dict[str, Any]: Versioned experimental descriptor.
        """
        return {
            "schema": 1,
            "family": self.family,
            "sha256": self.digest,
            "parent": self.parent,
            "status": self.status,
            "qualified_image": self.qualified_image,
            "entrypoint": "decode(keys, instance)",
            "solution": "unique zero-based selected indices; scheduling requires a permutation",
            "dimension": "number of items/jobs/subsets in the supplied instance",
            "runtime": {
                "python": ">=3.14",
                "numpy": ">=2",
                "created_python": platform.python_version(),
                "created_rko": importlib.metadata.version("rko-solver"),
            },
        }

    def export(self, path: Path) -> None:
        """Write a new non-overwriting bundle; never execute its source.

        Args:
            path (Path): Nonexistent destination directory.

        Raises:
            FileExistsError: If destination already exists.
            OSError: If source or metadata cannot be written.
        """
        verified = self.create(self.family, self.source, self.parent)
        if verified.digest != self.digest or self.status != "experimental":
            raise ValueError("Artifact hash/status mismatch.")
        path.mkdir(parents=True, exist_ok=False, mode=0o700)
        (path / "decoder.py").write_text(self.source, encoding="utf-8")
        (path / "manifest.json").write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DecoderArtifact:
        """Read and verify a bundle without importing generated code.

        Args:
            path (Path): Artifact directory.

        Returns:
            DecoderArtifact: Hash-verified experimental artifact.

        Raises:
            ValueError: If schema, size, status or source hash is invalid.
            OSError: If bundle files cannot be read.
        """
        manifest_path, source_path = path / "manifest.json", path / "decoder.py"
        if manifest_path.stat().st_size > 16384 or source_path.stat().st_size > MAX_SOURCE:
            raise ValueError("Artifact file size limit exceeded.")
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != 1
            or metadata.get("status") != "experimental"
        ):
            raise ValueError("Unsupported artifact schema/status.")
        artifact = cls.create(
            metadata.get("family"), source_path.read_text(encoding="utf-8"), metadata.get("parent")
        )
        if artifact.digest != metadata.get("sha256"):
            raise ValueError("Artifact source hash mismatch.")
        expected = artifact.metadata()
        for key in ("entrypoint", "solution", "dimension"):
            if metadata.get(key) != expected[key]:
                raise ValueError("Incompatible artifact input/output contract.")
        runtime = metadata.get("runtime")
        if (
            not isinstance(runtime, dict)
            or runtime.get("python") != ">=3.14"
            or runtime.get("numpy") != ">=2"
        ):
            raise ValueError("Incompatible artifact runtime requirements.")
        image = metadata.get("qualified_image")
        if image is not None and (
            not isinstance(image, str)
            or len(image) != 71
            or not image.startswith("sha256:")
            or any(c not in "0123456789abcdef" for c in image[7:])
        ):
            raise ValueError("Invalid qualified runtime image identity.")
        if "evidence_sha256" in metadata:
            evidence = path / "evidence.json"
            if evidence.stat().st_size > 1048576:
                raise ValueError("Artifact evidence exceeds 1 MiB.")
            if hashlib.sha256(evidence.read_bytes()).hexdigest() != metadata["evidence_sha256"]:
                raise ValueError("Artifact evidence hash mismatch.")
        return replace(artifact, qualified_image=image)
