#!/usr/bin/env python3
"""Detect temporary and unapproved Markdown files in repository changes.

Objective:
- Prevent session notes and scratch documents from entering version control.

Technical approach:
- Reject tracked files in temp_docs except its README.
- Compare new or renamed Markdown files with an allowlist.
- Support both staged local changes and base-to-HEAD pull request diffs.

Flow (pseudocode):
1. Find tracked files under temp_docs and reject unauthorized entries.
2. Collect staged paths or diff paths against the required base revision.
3. Filter Markdown paths through the allowlist and report violations.

Important criteria:
- Existing documents do not fail the base-to-HEAD check.
- CI pull request checks require an available base revision.
- Local checks without a base revision skip the diff check unless required.

Inputs/Outputs:
- Input: repository root, optional base revision, and staged-check selection.
- Output: zero on success; nonzero when document noise or comparison errors exist.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path

TEMP_DOCS_DIR = "temp_docs"
TEMP_DOCS_ALLOWED = frozenset({"temp_docs/README.md"})
DOC_ALLOW = frozenset(
    {
        "README.md",
        "AGENTS.md",
        "AGENTS.override.md",
        "CLAUDE.md",
        "CONTRIBUTING.md",
        "CHANGELOG.md",
        "SECURITY.md",
        "LICENSE",
        "LICENSE.md",
    }
)


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one Git command from the selected repository root.

    Args:
        root (Path): Repository directory used as the command working directory.
        *args (str): Git subcommand and arguments.

    Returns:
        subprocess.CompletedProcess[str]: Captured command result.

    Raises:
        OSError: If the Git executable cannot be started.
    """
    return subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)


def tracked_temp_docs(root: Path) -> list[str]:
    """Return tracked scratch documents outside the permitted README.

    Args:
        root (Path): Repository directory to inspect.

    Returns:
        list[str]: Sorted tracked paths that violate the scratch-directory rule.

    Raises:
        ValueError: If Git cannot list tracked scratch files.
    """
    result = _git(root, "ls-files", "-z", "--", TEMP_DOCS_DIR)
    if result.returncode != 0:
        raise ValueError(f"git ls-files failed: {result.stderr.strip()}")
    return sorted(
        path for path in result.stdout.split("\0") if path and path not in TEMP_DOCS_ALLOWED
    )


def base_ref(environ: Mapping[str, str] | None = None) -> str:
    """Read the comparison base revision from the environment.

    Args:
        environ (Mapping[str, str] | None): Optional environment mapping; defaults to os.environ.

    Returns:
        str: Base revision from BASE_REF, or an empty string when unset.
    """
    env = dict(os.environ if environ is None else environ)
    return env.get("BASE_REF", "")


def _is_allowed(path: str) -> bool:
    """Check a path against exact-path and basename allowlist entries.

    Args:
        path (str): Repository-relative Markdown path.

    Returns:
        bool: Whether the path is permitted.
    """
    return path in DOC_ALLOW or Path(path).name in DOC_ALLOW


def _noise_paths(listing: str) -> list[str]:
    """Filter a Git path listing to unapproved Markdown files.

    Args:
        listing (str): Newline-delimited repository-relative paths.

    Returns:
        list[str]: Sorted Markdown paths outside temp_docs and the allowlist.
    """
    return sorted(
        path
        for path in listing.splitlines()
        if path.endswith(".md")
        and not path.startswith(f"{TEMP_DOCS_DIR}/")
        and not _is_allowed(path)
    )


def new_doc_noise(root: Path, base: str, *, require_base: bool = False) -> list[str]:
    """List unapproved Markdown documents added or renamed since a base revision.

    Args:
        root (Path): Repository directory to inspect.
        base (str): Git revision used as the left side of the three-dot diff.
        require_base (bool): Fail rather than skip when the base cannot be used.

    Returns:
        list[str]: Sorted unapproved Markdown paths.

    Raises:
        ValueError: If a required base is missing or the Git comparison fails.
    """
    if not base:
        if require_base:
            raise ValueError("base reference is required; provide BASE_REF")
        return []
    result = _git(
        root, "diff", "--no-ext-diff", "--diff-filter=AR", "--name-only", f"{base}...HEAD"
    )
    if result.returncode != 0:
        if require_base:
            raise ValueError(f"cannot compare {base}...HEAD: {result.stderr.strip()}")
        return []
    return _noise_paths(result.stdout)


def staged_doc_noise(root: Path) -> list[str]:
    """List unapproved Markdown documents added or renamed in the Git index.

    Args:
        root (Path): Repository directory to inspect.

    Returns:
        list[str]: Sorted unapproved staged Markdown paths.

    Raises:
        ValueError: If Git cannot read the staged diff.
    """
    result = _git(root, "diff", "--cached", "--no-ext-diff", "--diff-filter=AR", "--name-only")
    if result.returncode != 0:
        raise ValueError(f"cannot read staged changes: {result.stderr.strip()}")
    return _noise_paths(result.stdout)


def main() -> int:
    """Run document-noise checks and return the command exit status.

    Returns:
        int: Zero when all checks pass; one when a check fails.

    Raises:
        SystemExit: If command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base-ref", default=base_ref())
    parser.add_argument("--require-base", action="store_true")
    parser.add_argument("--staged", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()

    try:
        stray = tracked_temp_docs(root)
        if stray:
            print(
                "Document noise gate failed: temp_docs/ must stay untracked "
                "(git-ignored scratch space); promote with 'git mv' instead:\n- "
                + "\n- ".join(stray)
            )
            return 1
        if args.staged:
            noise = staged_doc_noise(root)
        else:
            noise = new_doc_noise(root, args.base_ref, require_base=args.require_base)
    except ValueError as error:
        print(f"Document noise gate failed: {error}")
        return 1

    if noise:
        print(
            "Document noise gate failed: new docs outside the allowlist look "
            "like session noise. Keep session scratch in temp_docs/, or add the "
            "path to DOC_ALLOW in tools/document_noise_gate.py if it is real project doc:\n- "
            + "\n- ".join(noise)
        )
        return 1

    print("Document noise gate: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
