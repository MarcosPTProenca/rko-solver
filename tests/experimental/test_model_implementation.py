"""Complete protected implementation identity without per-file pathlib rebuilding.

Objective:
- Preserve the exact eligible Python source set and content hashes while reducing setup work.
Technical approach:
- Place a small package tree with nested test exclusions and known source bytes.
Flow (pseudocode):
1. Point the public identity operation at an isolated package layout.
2. Include ordinary/nested Python and exclude nested tests/non-Python data.
3. Assert portable relative keys and known-byte hashes.
Important criteria:
- No sampled or cached-away file checks; later byte changes must be detected.
Inputs/Outputs:
- Input: temporary owned package files.
- Output: complete fresh source identity.
"""

import hashlib
from pathlib import Path
from pytest import MonkeyPatch
from rko_solver.experimental.self_evolve import model_data
from rko_solver.experimental.self_evolve.model_data import ModelProblem


def test_model_implementation_checks_all_original_source_bytes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Source identity preserves coverage/exclusions and notices later on-disk changes.

    Args:
        tmp_path (Path): Temporary package source root.
        monkeypatch (MonkeyPatch): Point only the module location at the owned fixture.
    """
    root = tmp_path / "package"
    contents = {
        "plain.py": b"plain",
        "nested/code.py": b"nested",
        "experimental/self_evolve/model_data.py": b"owned",
        "tests/ignored.py": b"ignored",
        "nested/tests/ignored.py": b"ignored",
        "nested/data.txt": b"not source",
    }
    for relative, content in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(
        model_data, "__file__", str(root / "experimental/self_evolve/model_data.py")
    )
    expected = {
        "plain.py": hashlib.sha256(b"plain").hexdigest(),
        "nested/code.py": hashlib.sha256(b"nested").hexdigest(),
        "experimental/self_evolve/model_data.py": hashlib.sha256(b"owned").hexdigest(),
    }
    assert ModelProblem.implementation() == expected
    (root / "nested/code.py").write_bytes(b"changed")
    expected["nested/code.py"] = hashlib.sha256(b"changed").hexdigest()
    assert ModelProblem.implementation() == expected
