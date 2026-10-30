"""Scientific reference index contract tests.

Objective:
- Keep the reference index complete, truthful about provenance and linked from the code it cites.

Technical approach:
- Parse `docs/references.md` and compare it with the repository tree: algorithm modules, local
  search operators, experiment modules, local paper copies and module docstrings.

Flow (pseudocode):
1. Split the page into sections and collect headings, links and repository paths.
2. Compare each implemented algorithm and experiment with the page.
3. Check local copies against recorded hashes and licenses, and docstrings against anchors.

Important criteria:
- Every algorithm, local search operator and experiment appears on the page.
- Every section cites a source or says that none exists; no plain-HTTP links.
- A local PDF is committed only with its SHA-256 and a license that permits redistribution.
- Unpublished and machine-generated proposals are labelled as such.

Inputs/Outputs:
- Input: the documentation page, navigation, source tree and local PDFs.
- Output: assertions on coverage, links, licenses and cross-references.
"""

import hashlib
import pytest
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "references.md"
PAPERS = ROOT / "docs" / "references" / "papers"
ALGORITHM_DIR = ROOT / "src" / "rko_solver" / "metaheuristics"
SEARCH_DIR = ROOT / "src" / "rko_solver" / "components" / "local_search"
NOT_ALGORITHMS = {"__init__", "_solvers", "base"}
EXPERIMENTS = (
    "examples/priority_decoders.py",
    "benchmarks/engines/priority_decoder_study.py",
    "benchmarks/engines/surrogate_seed_study.py",
    "benchmarks/engines/mlp_surrogate.py",
    "benchmarks/engines/neural_adapter_study.py",
    "benchmarks/engines/neural_key_adapter.py",
    "benchmarks/engines/coupled_key_distribution.py",
    "benchmarks/engines/packing_problems.py",
    "benchmarks/engines/priority_scorer.py",
    "benchmarks/engines/learned_priority_study.py",
)
PROPOSALS = (
    "Neural Decoders for Automatic Random-Key Optimization",
    "RKO-SR",
    "Decoder-aware convergence bounds for random-key metaheuristics",
)
SOURCE_GROUPS = ("Algorithms", "Shared components", "Learning components", "Convergence")


def text() -> str:
    """Read the reference page.

    Returns:
        str: Page Markdown.
    """
    return PAGE.read_text(encoding="utf-8")


def slug(heading: str) -> str:
    """Compute the anchor that both GitHub and MkDocs give a heading without punctuation clashes.

    Args:
        heading (str): Heading text without the leading hashes.

    Returns:
        str: Lowercase anchor with spaces turned into hyphens.
    """
    cleaned = re.sub(r"[^\w\s-]", "", heading).strip().lower()
    return re.sub(r"\s", "-", cleaned)


def sections() -> dict[str, dict[str, str]]:
    """Split the page into level-two groups and their level-three entries.

    Returns:
        dict[str, dict[str, str]]: Group heading mapped to entry heading mapped to its body.
    """
    groups: dict[str, dict[str, str]] = {}
    group, entry = "", ""
    for line in text().splitlines():
        if line.startswith("## "):
            group, entry = line[3:].strip(), ""
            groups[group] = {}
        elif line.startswith("### "):
            entry = line[4:].strip()
            groups[group][entry] = ""
        elif group and entry:
            groups[group][entry] += line + "\n"
    return groups


def test_page_exists_and_is_linked_from_navigation_and_index() -> None:
    """The page is published through the navigation and reachable from the documentation index."""
    assert PAGE.is_file()
    assert "references.md" in (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "references.md" in (ROOT / "docs" / "index.md").read_text(encoding="utf-8")


def test_every_algorithm_and_local_search_module_is_listed() -> None:
    """Each implemented algorithm and local search operator names its module on the page."""
    page = text()
    modules = [
        path for path in sorted(ALGORITHM_DIR.glob("*.py")) if path.stem not in NOT_ALGORITHMS
    ]
    modules += [path for path in sorted(SEARCH_DIR.glob("*.py")) if path.stem != "__init__"]
    assert len(modules) >= 11
    for path in modules:
        assert f"`{path.relative_to(ROOT).as_posix()}`" in page, path.name


def test_every_tested_experiment_module_is_listed() -> None:
    """Each experiment this repository ran against a paper or without one is on the page."""
    page = text()
    for path in EXPERIMENTS:
        assert f"`{path}`" in page, path


def test_every_backticked_repository_path_exists() -> None:
    """A path written in code font must point at a file or directory that exists."""
    paths = re.findall(r"`((?:src|benchmarks|examples|engines|docs|tests|tools)/[^`\s]+)`", text())
    assert paths
    missing = [path for path in paths if not (ROOT / path.split("#")[0]).exists()]
    assert not missing


def test_every_entry_cites_a_source_or_says_none_exists() -> None:
    """An entry without a link must say explicitly that no published reference exists."""
    for group in SOURCE_GROUPS:
        entries = sections().get(group, {})
        assert entries, group
        for heading, body in entries.items():
            assert "https://" in body or "No published reference" in body, f"{group}/{heading}"


def test_links_use_https_and_doi_links_are_well_formed() -> None:
    """No plain-HTTP link, and every DOI link carries a syntactically valid DOI."""
    page = text()
    assert "http://" not in page
    for doi in re.findall(r"https://doi\.org/(\S+?)\)", page):
        assert re.fullmatch(r"10\.\d{4,9}/\S+", doi), doi


def test_proposals_are_labelled_unpublished_and_machine_generated() -> None:
    """The three Novix proposals are never presented as literature."""
    body = "".join(sections().get("Unpublished proposals", {}).values())
    for title in PROPOSALS:
        assert title in body, title
    assert "machine-generated" in body and "not peer reviewed" in body
    assert "https://" not in body


def test_local_paper_copies_carry_hash_and_a_redistribution_license() -> None:
    """A committed PDF is listed with its SHA-256 and the license that allows the copy."""
    page = text()
    pdfs = sorted(PAPERS.glob("*.pdf"))
    assert pdfs
    for pdf in pdfs:
        assert pdf.read_bytes()[:5] == b"%PDF-", pdf.name
        assert hashlib.sha256(pdf.read_bytes()).hexdigest() in page, pdf.name
        line = next(line for line in page.splitlines() if pdf.name in line)
        assert "CC BY 4.0" in line, pdf.name


def test_implementation_docstrings_point_at_existing_sections() -> None:
    """Every algorithm, local search and Q-learning module header names its page section.

    Headers write `docs/references.md, section <anchor>` instead of a `#` link: a `#` inside a
    docstring changes which clones the duplicate detector of `tools/debt_counter.py` reports.
    """
    anchors = {slug(line[4:]) for line in text().splitlines() if line.startswith("### ")}
    modules = [
        path for path in sorted(ALGORITHM_DIR.glob("*.py")) if path.stem not in NOT_ALGORITHMS
    ]
    modules += [path for path in sorted(SEARCH_DIR.glob("*.py")) if path.stem != "__init__"]
    modules.append(
        ROOT / "src" / "rko_solver" / "components" / "adaptive_control" / "q_learning.py"
    )
    for path in modules:
        found = re.findall(
            r"docs/references\.md, section ([\w-]+)", path.read_text(encoding="utf-8")
        )
        assert found, path.name
        assert all(anchor in anchors for anchor in found), (path.name, found)


def test_every_anchor_link_into_the_page_resolves() -> None:
    """Internal links and links from other documents must name a real heading."""
    headings = {slug(line[3:]) for line in text().splitlines() if line.startswith("## ")}
    headings |= {slug(line[4:]) for line in text().splitlines() if line.startswith("### ")}
    internal = re.findall(r"\]\(#([\w-]+)\)", text())
    assert internal
    assert all(anchor in headings for anchor in internal), sorted(set(internal) - headings)
    outside = [ROOT / "benchmarks" / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for path in outside:
        for anchor in re.findall(r"references\.md#([\w-]+)", path.read_text(encoding="utf-8")):
            assert anchor in headings, (path.name, anchor)


@pytest.mark.parametrize("heading", ["Algorithms", "Shared components", "Convergence"])
def test_required_groups_are_present(heading: str) -> None:
    """The page keeps its documented top-level groups.

    Args:
        heading (str): Required group heading.
    """
    assert heading in sections()
