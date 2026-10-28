# Building and publishing the documentation

[Documentation index](index.md) · [Development status](status.md) ·
[Contributor guidelines](https://github.com/MarcosPTProenca/rko-solver/blob/main/CONTRIBUTING.md)

The website uses **MkDocs' built-in theme and search**. The maintained `docs/*.md`
pages are the single source for both GitHub readers and the site. No separate app,
theme fork, custom JavaScript, backend, account service or API-doc plugin is required.
This is one **development** site, not versioned stable-release documentation.

## Preview with one command

From the repository root, with uv available:

```bash
uv run --locked --only-group docs mkdocs serve
```

The server reports its local address, normally
`http://127.0.0.1:8000/rko-solver/`. The project prefix is intentional: preview and
production use the same `/rko-solver/` path. Stop with Ctrl+C. No publishing token or
GitHub configuration is needed to preview. The docs-only group does not require
installing the optimizer, NumPy, Plotly or development test tools.

Build the static site without serving:

```bash
uv run --locked --only-group docs mkdocs build --strict
```

Output goes to ignored `site/`. Both commands use locked docs dependencies; normal
runtime installation of RKO does **not** install MkDocs. The current built-in-theme
line is MkDocs 1.6, bounded below its next major version and pinned by `uv.lock`.

## What may be published

`mkdocs.yml` navigation is the explicit publication allowlist. A docs-only hook
removes all unlisted docs-directory inputs before rendering/copying/search indexing,
while retaining assets from the installed theme. Defensive `exclude_docs` patterns
also block internal directories. Adding a public page requires updating navigation
and passing the strict build; it does not automatically publish its whole directory.

Everything else under `docs/` is excluded, including ignored root-level Markdown,
plans, audits, migrations, superpowers notes, datasets, JSON and local HTML reports.
Generated output, arbitrary local files and private working material are not site
inputs. Do not add a report demo or dataset without separately reviewing its content
and provenance. Theme/search assets come from the locked documentation dependencies.

Links to repository files outside `docs/` deliberately point to the public GitHub
repository: CONTRIBUTING, NOTICE, benchmark references, source files and workflow
files are **not copied into the website**. Avoid `../README.md` or similar escaping
site links. Use maintained documentation links or deliberate repository URLs.

The same docs-only hook adds a development warning to every built page using the
version from `pyproject.toml`. It does not import optimizer code, execute examples
or read instance data. The warning points to source installation and support/release
limitations. It must never imply that main equals a published stable PyPI package.

## Validation

```bash
uv run --locked --group docs pytest tests/test_documentation_site.py -q
uv run --locked --only-group docs mkdocs build --strict
```

The first command uses the existing dev test tools plus the explicit docs group.
Tests build a temporary project with adversarial ignored/local files and verify
that they never reach the site or search index. They check the development banner
and local links/assets under `/rko-solver/`, not merely `/`. MkDocs strict mode also
checks navigation, missing targets and document anchors. External repository links
are deliberate references, not a promise of perpetual external URL availability.

Existing library CI continues to execute the maintained quickstart/README/adapter/
validation/diagram examples. Building the website itself does **not** execute code
fences, profiling experiments, solver runs or user callbacks.

## Deployment and owner setup

The development site is published at [RKO documentation](https://marcosptproenca.github.io/rko-solver/).
Repository or Pages-path changes must keep `mkdocs.yml`, README, project metadata
and subpath tests synchronized. A successful deployment does not imply a stable
package release. Pages settings remain owner-controlled.

For a first deployment in a new repository or fork, the owner must:

1. Open **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Confirm the `github-pages` environment permits protected `main` deployments;
   add required reviewers if desired. This workflow never deploys a PR branch.
3. Merge the documentation PR, or run the **Documentation** workflow manually on
   **main** after enabling Pages. Watch its build and deployment jobs.
4. Open the successful deployment URL and verify navigation/search and the development
   warning under the configured project path. Keep the verified URL in README and
   project metadata synchronized. Do not invent a stable release or PyPI command.

The workflow builds/checks PRs with read-only repository permissions. Only its
main-only deployment job requests `pages: write` and `id-token: write`, uses the
`github-pages` environment, and configures/deploys the uploaded Pages artifact.
Reviewed Node-24-compatible official actions are pinned to commit SHAs. No personal
access token, `gh-pages` force push or long-lived publishing credential is needed.
The configuration action does not auto-enable Pages; missing owner setup should
produce an explicit deployment failure, not a falsely advertised live site.

## Recover a bad deployment

Fix or revert the offending source/configuration through a normal PR; merging to
main rebuilds and redeploys the site. Alternatively dispatch the workflow on main
after a known-good revert. Preserve Git history rather than rewriting a publishing
branch. If owner approval is enabled, approve the `github-pages` environment after
review. Owner-controlled unpublishing remains a Pages setting, not an automated
operation in this repository.

Custom domains, translated/versioned documentation, Material/mkdocstrings, a web
playground and public report demos are intentionally deferred. Package publishing,
licensing confirmation and service/Docker deployment remain separate plans.
