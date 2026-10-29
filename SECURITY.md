# Security policy

`rko-solver` is an alpha library maintained by one owner. Reports are handled on a best-effort
basis; there is no guaranteed response time.

## Supported versions

| Version | Security fixes |
| --- | --- |
| Development `main` | Yes |
| Latest published prerelease | Fixed on `main`; a new prerelease is published only when the owner approves it |
| Older prereleases | No |

Fixes are not backported. See [installation and version boundaries](docs/installation.md) for how a
published package differs from `main`.

## Reporting a vulnerability

Report privately. If the repository's **Security** tab offers **Report a vulnerability**, use it.
If it does not, open a public issue that only asks for a private channel. Do not put exploit
details, affected data, credentials or tokens in that issue.

Please include the affected version or commit, the engine and execution mode, a minimal
reproduction, and the impact you expect. Do not publish details until a fix is available or the
owner agrees to a disclosure date.

## Scope

In scope: vulnerabilities in the library code, the supervised service's authentication, request
limits and job isolation, the build and release workflows, and packaged artifacts.

Out of scope, by design (see [current limitations](docs/status.md)):

- Callbacks you supply run arbitrary Python, and process execution pickles them. Only run callbacks
  you trust.
- The FastAPI and Docker service is a private single-tenant reference for trusted callbacks. It is
  not a public service and not a sandbox for hostile code.
- The decoder-evolution pilot is experimental; evaluating generated code needs the isolation it
  documents, and its tests do not certify that isolation.
- A compiled objective that is impure, not reentrant or that raises has undefined behavior.
