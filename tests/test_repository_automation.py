"""Repository automation and quality-policy contracts.

Objective:
- Keep local checks, hosted checks, dependency automation and contributor policy aligned.

Technical approach:
- Parse the maintained YAML, TOML and Markdown policy files and compare their declared contracts.
- Derive hosted-check expectations from the workflow itself rather than a second hand-kept list.

Flow (pseudocode):
1. Read the hosted quality job and the local hook configuration.
2. Require every locally runnable hosted check to have a matching local hook.
3. Check dependency automation, contributor templates, the security policy and the coverage floor.

Important criteria:
- Network-bound audits are the only hosted checks exempt from a local hook.
- Dependency automation must not rewrite deliberately pinned manifests.
- The documented coverage floor equals the enforced one.

Inputs/Outputs:
- Input: files tracked at the repository root.
- Output: assertions that fail when a policy file drifts from its counterpart.
"""

import re
import tomllib
import yaml
from pathlib import Path
from tools import document_noise_gate

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CHECKS = (
    "document_noise_gate.py",
    "quality_gate.py",
    "lint-imports",
    "debt_counter.py",
    "vulture",
)
NETWORK_AUDITS = ("pip-audit", "cargo-deny")
ECOSYSTEM_MANIFESTS = {
    "github-actions": (".github/workflows",),
    "cargo": ("Cargo.toml", "Cargo.lock"),
    "uv": ("pyproject.toml", "uv.lock"),
}


def load_yaml(relative: str) -> dict:
    """Parse a tracked YAML file.

    Args:
        relative (str): Repository-relative path.

    Returns:
        dict: Parsed mapping.
    """
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def local_hooks() -> list[dict]:
    """List the hooks that define their own command in the pre-commit configuration.

    Returns:
        list[dict]: Hook mappings with an `entry` command.
    """
    config = load_yaml(".pre-commit-config.yaml")
    return [hook for repo in config["repos"] for hook in repo["hooks"] if "entry" in hook]


def test_every_hosted_quality_check_runnable_locally_has_a_hook() -> None:
    """A check that fails only in hosted CI costs a round trip; mirror it in a local hook."""
    workflow = load_yaml(".github/workflows/ci.yml")
    steps = workflow["jobs"]["quality-gate"]["steps"]
    hosted = " ".join(step.get("run", "") + " " + step.get("uses", "") for step in steps)
    assert all(check in hosted for check in (*LOCAL_CHECKS, *NETWORK_AUDITS)), hosted
    entries = [hook["entry"] for hook in local_hooks()]
    missing = [c for c in LOCAL_CHECKS if not any(c in entry for entry in entries)]
    assert not missing, f"Hosted checks without a local hook: {missing}"


def test_slow_checks_run_before_push_and_both_hook_types_install() -> None:
    """Keep the commit hook fast while the type ratchet and debt counter still run before a push."""
    config = load_yaml(".pre-commit-config.yaml")
    assert set(config["default_install_hook_types"]) == {"pre-commit", "pre-push"}
    by_entry = {hook["entry"]: hook for hook in local_hooks()}
    pushed = {hook["id"] for hook in by_entry.values() if hook.get("stages") == ["pre-push"]}
    assert {"debt-counter", "quality-gate-types"} <= pushed
    commit = [hook for hook in by_entry.values() if "stages" not in hook]
    assert any("document_noise_gate.py" in hook["entry"] for hook in commit)
    assert any("vulture" in hook["entry"] for hook in commit)


def test_ruff_hook_revision_matches_the_locked_development_pin() -> None:
    """Two ruff versions would make local formatting disagree with the hosted check."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins = [d for d in project["dependency-groups"]["dev"] if d.startswith("ruff==")]
    config = load_yaml(".pre-commit-config.yaml")
    revisions = [r["rev"] for r in config["repos"] if r["repo"].endswith("ruff-pre-commit")]
    assert revisions == ["v" + pins[0].split("==")[1]]


def test_dependency_automation_covers_every_ecosystem_without_rewriting_pins() -> None:
    """Refresh locks and actions on a quiet schedule; exact pins and tested minimums stay manual."""
    config = load_yaml(".github/dependabot.yml")
    assert config["version"] == 2
    updates = {item["package-ecosystem"]: item for item in config["updates"]}
    assert set(updates) == set(ECOSYSTEM_MANIFESTS)
    for ecosystem, item in updates.items():
        assert item["directory"] == "/"
        assert item["schedule"]["interval"] == "weekly"
        assert item["cooldown"]["default-days"] >= 7
        assert item["open-pull-requests-limit"] <= 3
        assert item["groups"], ecosystem
        assert re.fullmatch(r"\[L[123]\] chore\(deps-[a-z]+\)", item["commit-message"]["prefix"])
        assert all((ROOT / name).exists() for name in ECOSYSTEM_MANIFESTS[ecosystem])
    for ecosystem in ("cargo", "uv"):
        assert updates[ecosystem]["versioning-strategy"] == "lockfile-only"


def test_pull_request_template_follows_the_maintained_convention() -> None:
    """The template keeps the sections reviewers rely on and the complexity-level prefix."""
    text = (ROOT / ".github" / "pull_request_template.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", text, re.M)
    assert headings == [
        "Why",
        "Summary",
        "Evidence",
        "Tests executed",
        "How to validate",
        "Merge risk",
        "Checklist",
    ]
    for required in ("Reversal", "Blast radius", "NOT RUN", "[L1]", "[L2]", "[L3]"):
        assert required in text


def test_security_policy_states_scope_route_and_limits() -> None:
    """Reporters learn what is supported, how to report privately and what is out of scope."""
    text = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    for heading in ("## Supported versions", "## Reporting a vulnerability", "## Scope"):
        assert heading in text
    assert "docs/status.md" in text
    assert "private" in text.lower()
    assert "do not" in text.lower()


def test_document_noise_gate_admits_the_policy_documents() -> None:
    """Policy files are maintained project documents, not session scratch."""
    listing = ".github/pull_request_template.md\nSECURITY.md\ndocs/scratch-notes.md"
    assert document_noise_gate._noise_paths(listing) == ["docs/scratch-notes.md"]


def test_documented_coverage_floor_equals_the_enforced_floor() -> None:
    """The contributor guide must state the floor that the coverage job enforces."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    enforced = project["tool"]["coverage"]["report"]["fail_under"]
    guide = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    documented = re.search(r"at least\s+(\d+)\s+percent\s+combined line and branch", guide)
    assert documented is not None
    assert int(documented.group(1)) == enforced
    assert enforced >= 89
