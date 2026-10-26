"""Prepare publication-safe development documentation with the native MkDocs theme.

Objective:
- Publish only explicitly maintained pages and visibly preserve development status.

Technical approach:
- Derive the documentation-source allowlist from explicit navigation, not Git ignore state.
- Preserve trusted theme/search assets while dropping unlisted docs-directory inputs.
- Prepend a native admonition using the canonical project version, without a theme fork.

Flow (pseudocode):
1. Flatten navigation paths into an explicit set of public Markdown sources.
2. Remove other docs-directory files before rendering, copying or search indexing.
3. Read project version metadata and prepend development/support provenance to each page.

Important criteria:
- Globally excluding every file would also remove theme assets; filter source files only.
- This docs-only hook never imports optimizer code, executes examples or loads datasets.
- Main is not advertised as a published stable PyPI release.

Inputs/Outputs:
- Input: MkDocs source collection, page Markdown and standard event context.
- Output: Filtered public files and development-labelled page Markdown.
"""

import tomllib
from mkdocs.config.defaults import MkDocsConfig
from mkdocs.structure.files import Files
from mkdocs.structure.pages import Page
from pathlib import Path
from typing import ClassVar


class DocumentationSite:
    """Enforce the docs publication boundary and development provenance.

    Responsibilities:
    - Keep unlisted local source files out of both site output and search indexes.
    - Retain installed theme assets and read package status without importing the optimizer.

    Attributes:
        _root (ClassVar[Path]): Repository containing this hook and project metadata.
    """

    _root: ClassVar[Path] = Path(__file__).resolve().parents[1]

    @staticmethod
    def on_files(files: Files, config: MkDocsConfig) -> Files:
        """Keep only navigation-listed Markdown from docs_dir, preserving theme assets.

        Args:
            files (Files): Framework source and trusted theme file collection.
            config (MkDocsConfig): Validated configuration with explicit navigation.

        Returns:
            Files: Same collection with unlisted documentation inputs removed.
        """
        allowed: set[str] = set()
        pending: list[object] = [config.nav]
        while pending:
            item = pending.pop()
            if isinstance(item, dict):
                pending.extend(item.values())
            elif isinstance(item, list):
                pending.extend(item)
            elif isinstance(item, str) and item.endswith(".md"):
                allowed.add(item)
        source = Path(config.docs_dir).resolve()
        for file in list(files):
            if file.src_dir is not None and Path(file.src_dir).resolve() == source:
                if file.src_uri not in allowed:
                    files.remove(file)
        return files

    @staticmethod
    def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
        """Prepend development provenance without duplicating maintained source pages.

        Args:
            markdown (str): Maintained Markdown provided by MkDocs.
            page (Page): Framework page context, not modified.
            config (MkDocsConfig): Framework configuration context, not modified.
            files (Files): Framework file collection, not modified.

        Returns:
            str: Native admonition followed by the original page content.

        Raises:
            OSError: If canonical project metadata cannot be read.
            ValueError: If project TOML is invalid.
            KeyError: If the project version is missing.
        """
        metadata = DocumentationSite._root / "pyproject.toml"
        version = tomllib.loads(metadata.read_text(encoding="utf-8"))["project"]["version"]
        banner = (
            f'!!! warning "Development documentation — {version} / main"\n\n'
            "    This is the development branch, not a published stable release. "
            "Use the [source installation guide](getting-started.md). "
            "Ubuntu and Windows CI are exercised; other platforms are not certified. "
            "Read [support, release gates and known limitations](status.md) "
            "before relying on results.\n\n"
        )
        return banner + markdown


on_files = DocumentationSite.on_files
on_page_markdown = DocumentationSite.on_page_markdown
