"""Publication-boundary and project-subpath checks for the optional MkDocs site.

Objective:
- Prevent private working files and broken project-relative navigation from publication.

Technical approach:
- Resolve rendered local links against a synthetic project URL without network requests.
- Build a temporary docs project containing adversarial non-public input files.

Flow (pseudocode):
1. Check negative fixtures for root-path escapes and missing assets.
2. When MkDocs is installed, build the real maintained pages in strict mode.
3. Inspect output and search data for allowlist, provenance and local-link integrity.

Important criteria:
- Docs dependencies remain optional; core installations skip only the real site build.
- Tests never execute Markdown examples, profiling experiments or user callbacks.

Inputs/Outputs:
- Input: HTML fixtures and a temporary copy of maintained documentation/configuration.
- Output: assertions for public-file selection, version notices and subpath-safe links.
"""

import importlib.util
import json
import pytest
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import URLError
from urllib.parse import unquote
from urllib.parse import urljoin
from urllib.parse import urlsplit
from urllib.request import urlopen


class SiteLinks(HTMLParser):
    """Collect local rendered links and enforce the GitHub project URL boundary.

    Responsibilities:
    - Audit existing local pages/assets without making external network requests.
    - Reject domain-root paths that would fail when deployed as a project site.

    Attributes:
        root (Path): Resolved static output directory.
        page (Path): Current rendered HTML file.
        links (list[str]): Collected href/src targets.
    """

    def __init__(self, root: Path, page: Path) -> None:
        """Initialize an isolated parser for one built page.

        Args:
            root (Path): Static output directory.
            page (Path): Current rendered page inside root.
        """
        super().__init__()
        self.root = root.resolve()
        self.page = page
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Collect literal href/src attributes from ordinary generated elements.

        Args:
            tag (str): Parsed HTML tag name.
            attrs (list[tuple[str, str | None]]): Parsed attribute names/values.
        """
        self.links.extend(value for key, value in attrs if key in ("href", "src") and value)

    def verify(self) -> None:
        """Assert that same-origin links remain inside the project and resolve to files.

        Raises:
            AssertionError: If a link escapes the project prefix or references a missing asset.
        """
        prefix = "/rko-solver/"
        origin = "https://marcosptproenca.github.io"
        current = origin + prefix + self.page.relative_to(self.root).as_posix()
        for link in self.links:
            parsed = urlsplit(urljoin(current, link))
            if parsed.scheme not in ("http", "https") or parsed.netloc != urlsplit(origin).netloc:
                continue
            assert parsed.path.startswith(prefix), f"Project-prefix escape: {link} in {self.page}"
            target = (self.root / unquote(parsed.path[len(prefix) :])).resolve()
            assert target.is_relative_to(self.root), f"Output-directory escape: {link}"
            if target.is_dir():
                target /= "index.html"
            assert target.is_file(), f"Missing local link: {link} in {self.page}"


@pytest.mark.parametrize("link", ["/getting-started/", "../outside/", "missing.css"])
def test_site_links_reject_missing_assets_and_root_escapes(tmp_path: Path, link: str) -> None:
    """Fail realistic website mistakes without requiring MkDocs or network access.

    Args:
        tmp_path (Path): Temporary static output root.
        link (str): Broken local/project link.
    """
    page = tmp_path / "index.html"
    page.write_text("test", encoding="utf-8")
    parser = SiteLinks(tmp_path, page)
    parser.feed(f'<a href="{link}">Link</a>')
    with pytest.raises(AssertionError):
        parser.verify()


def test_docs_dependencies_and_deployment_permissions_are_isolated() -> None:
    """Keep docs tools out of runtime requirements and privileged deploys off PR jobs."""
    root = Path(__file__).resolve().parents[3]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = metadata["project"]["dependencies"]
    runtime += [
        item for group in metadata["project"]["optional-dependencies"].values() for item in group
    ]
    assert not any("mkdocs" in item.lower() for item in runtime)
    assert any("mkdocs" in item.lower() for item in metadata["dependency-groups"]["docs"])
    yaml = pytest.importorskip("yaml")
    workflow = yaml.load(
        (root / ".github/workflows/docs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert workflow["permissions"] == {"contents": "read"}
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert "pull_request" in workflow["on"]
    assert "pull_request_target" not in workflow["on"]
    configuration = yaml.load(
        (root / "mkdocs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert configuration["site_name"] == "RKO"
    assert configuration["repo_url"] == metadata["project"]["urls"]["Repository"]
    assert configuration["repo_name"] == "MarcosPTProenca/rko-solver"
    assert configuration["site_url"] == "https://marcosptproenca.github.io/rko-solver/"
    assert configuration["site_url"] == metadata["project"]["urls"]["Documentation"]
    pages: set[str] = set()
    pending: list[object] = [configuration["nav"]]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif isinstance(item, str) and item.endswith(".md"):
            pages.add(item)
    if (root / ".git").exists():
        tracked = subprocess.run(
            ["git", "ls-files", "docs/*.md"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        maintained = {
            Path(name).relative_to("docs").as_posix() for name in tracked if (root / name).is_file()
        }
        assert maintained <= pages
    assert all((root / "docs" / name).is_file() for name in pages)
    build = workflow["jobs"]["documentation"]
    assert "permissions" not in build
    assert build["steps"][0]["with"]["persist-credentials"] == "false"
    deploy = workflow["jobs"]["deploy"]
    assert deploy["permissions"] == {"pages": "write", "id-token": "write"}
    assert "refs/heads/main" in deploy["if"] and "pull_request" in deploy["if"]
    assert deploy["environment"]["name"] == "github-pages"
    assert deploy["needs"] == "documentation"
    assert all("enablement" not in step.get("with", {}) for step in deploy["steps"])
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "uses" in step:
                reference = step["uses"].split("@")[1]
                assert len(reference) == 40 and all(
                    char in "0123456789abcdef" for char in reference
                )


def test_site_links_accept_local_assets_and_deliberate_repository_urls(tmp_path: Path) -> None:
    """Keep valid project assets and explicit external repository links usable.

    Args:
        tmp_path (Path): Temporary static output root.
    """
    page = tmp_path / "index.html"
    page.write_text("test", encoding="utf-8")
    (tmp_path / "base.css").write_text("body{}", encoding="utf-8")
    parser = SiteLinks(tmp_path, page)
    parser.feed('<link href="/rko-solver/base.css"><a href="https://github.com/example">Repo</a>')
    parser.verify()


@pytest.mark.skipif(importlib.util.find_spec("mkdocs") is None, reason="Install the docs group")
def test_preview_serves_pages_assets_and_search_at_project_prefix() -> None:
    """Exercise the real preview server without launching a browser or leaving it running."""
    root = Path(__file__).resolve().parents[3]
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "-m", "mkdocs", "serve", "--dev-addr", f"127.0.0.1:{port}"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}/rko-solver/"
    try:
        for _ in range(100):
            if server.poll() is not None:
                pytest.fail("Preview server exited before becoming ready.")
            try:
                with urlopen(base, timeout=1.0) as response:
                    home = response.read()
                break
            except URLError, TimeoutError:
                time.sleep(0.2)
        else:
            pytest.fail("Preview server did not become ready within the startup allowance.")
        assert b"Development documentation" in home
        for path in ("getting-started/", "css/base.css", "search/search_index.json"):
            with urlopen(base + path, timeout=5.0) as response:
                assert response.status == 200
                data = response.read()
                if path.endswith(".json"):
                    assert len(json.loads(data)["docs"]) >= 28
    finally:
        server.terminate()
        try:
            server.communicate(timeout=10.0)
        except subprocess.TimeoutExpired:
            server.kill()
            server.communicate(timeout=5.0)


@pytest.mark.skipif(importlib.util.find_spec("mkdocs") is None, reason="Install the docs group")
def test_strict_site_excludes_private_inputs_and_supports_project_subpath(tmp_path: Path) -> None:
    """Build real pages with hostile local files and verify public output/navigation.

    Args:
        tmp_path (Path): Isolated project and generated-site workspace.
    """
    root = Path(__file__).resolve().parents[3]
    project = tmp_path / "project"
    project.mkdir()
    shutil.copytree(root / "docs", project / "docs")
    for name in ("mkdocs.yml", "pyproject.toml"):
        shutil.copy(root / name, project / name)
    (project / "tools").mkdir()
    shutil.copy(root / "tools" / "docs_hooks.py", project / "tools" / "docs_hooks.py")
    sentinel = "PRIVATE_WORKING_MATERIAL_MUST_NEVER_BE_PUBLISHED_70412"
    for name in (
        "ROOT-LOCAL-AUDIT.md",
        "plans/private.md",
        "audits/private.html",
        "migrations/private.md",
        "superpowers/private.md",
        "secret.json",
        "local-report.html",
        "css/private.css",
        "js/private.js",
        "img/private.txt",
    ):
        path = project / "docs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sentinel, encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    site = project / "site"
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    pages = list(site.rglob("*.html"))
    assert len(pages) >= 28
    assert (site / "search" / "search_index.json").is_file()
    for path in site.rglob("*"):
        if path.is_file():
            content = path.read_bytes()
            assert sentinel.encode() not in content
            assert b"/home/marcos" not in content
            assert b"/mnt/c/Users" not in content
            assert b"RKO_Python" not in content
            assert b"RKO Python" not in content
    for page in pages:
        content = page.read_text(encoding="utf-8")
        if page.name != "404.html":
            assert f"Development documentation — {version} / main" in content
            assert "not a published stable release" in content
        parser = SiteLinks(site, page)
        parser.feed(content)
        parser.verify()
    for name in ("plans", "audits", "migrations", "superpowers"):
        assert not (site / name).exists()
    assert not (site / "ROOT-LOCAL-AUDIT").exists()
    assert not (site / "secret.json").exists()
    assert not (site / "local-report.html").exists()
