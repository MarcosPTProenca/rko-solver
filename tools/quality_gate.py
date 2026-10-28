"""Deterministic structure and comment gates with a legacy-debt ratchet.

Objective:
- Keep files and functions inside objective size limits and keep explanatory comments out of
  function bodies, without forcing a rewrite of code that predates the limits.

Technical approach:
- Measure every tracked Python and Rust source file (tests and vendored code excluded) with
  `lizard` for lines, cyclomatic complexity, function length and parameter count, with
  `ast` plus `tokenize` for comment volume inside Python functions, and with `mypy` for the
  number of type errors per file.
- Store only the violations that already exist in three JSON baselines. A check passes when no
  measurement exceeds `max(limit, baselined value)`, so debt can stay but never grow, and a new
  violation always fails.
- `--tighten` rewrites the baselines keeping only entries that still violate, never above
  their recorded value, so improvements are locked in without admitting new debt.
- `--skip-types` omits the slower `mypy` pass, for fast local hooks.

Flow (pseudocode):
1. Read `[tool.quality]` from `pyproject.toml` over the built-in defaults.
2. List tracked sources, measure files, functions, comments and type errors.
3. Compare with the baselines: report each measurement above its allowance.
4. With `--write-baseline` record the current violations; with `--tighten` shrink them.

Important criteria:
- Function baseline keys are `path::qualified.name` (class and enclosing function for Python,
  the bare name for Rust) with an ordinal suffix on collisions, so a changed signature never
  discards a legacy allowance while growth is still caught by the measured values.
- A baseline is never created implicitly: a missing file means no debt is allowed.
- Pragmas such as `noqa` and `type: ignore` are not counted as explanatory comments.

Inputs/Outputs:
- Input: repository root, `--write-baseline`, `--tighten` or `--skip-types`.
- Output: a report on standard output and exit status 0 (clean) or 1 (violations).
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import lizard
import subprocess
import sys
import tokenize
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Json = dict[str, Any]


class QualityGate:
    """Measure sources and enforce limits against recorded legacy debt.

    Responsibilities:
    - Resolve limits, exclusions and baseline paths from the repository configuration.
    - Collect structure and comment metrics and compare them with the baselines.
    - Write or tighten the baselines on explicit request.

    Attributes:
        root (Path): Repository root, also the working directory of `git ls-files`.
        file_limits (dict[str, int]): Maximum lines per file extension.
        function_limits (dict[str, int]): Maximum complexity, length and parameters.
        comment_limits (dict[str, float]): Maximum comment metrics per Python function.
        exclude (frozenset[str]): Path components that exclude a file.
        structure_baseline (Path): Baseline of file and function violations.
        comment_baseline (Path): Baseline of comment violations.
        types_baseline (Path): Baseline of type errors per file.
        type_paths (list[str]): Paths handed to `mypy`.
    """

    FILE_LIMITS = {".py": 600, ".rs": 600}
    FUNCTION_LIMITS = {"cyclomatic_complexity": 10, "nloc": 100, "parameter_count": 6}
    COMMENT_LIMITS = {
        "comment_lines": 8,
        "comment_chars": 600,
        "comment_ratio": 0.5,
        "max_consecutive_comments": 5,
        "max_inline_comment_chars": 120,
    }
    EXCLUDE = frozenset(
        {".git", ".venv", "build", "dist", "node_modules", "target", "tests", "vendor"}
    )
    IGNORED_COMMENTS = ("noqa", "type: ignore", "pragma:", "nosec", "pylint:", "pyright:")

    def __init__(self, root: Path) -> None:
        """Load limits and baseline locations for a repository.

        Args:
            root (Path): Repository root containing `pyproject.toml`.
        """
        self.root = root.resolve()
        table = self._table()
        self.file_limits = {**self.FILE_LIMITS, **table.get("file-limits", {})}
        self.function_limits = {**self.FUNCTION_LIMITS, **table.get("function-limits", {})}
        self.comment_limits = {**self.COMMENT_LIMITS, **table.get("comment-limits", {})}
        self.exclude = self.EXCLUDE | frozenset(table.get("exclude", []))
        self.structure_baseline = self.root / table.get(
            "quality-baseline", ".quality/quality-baseline.json"
        )
        self.comment_baseline = self.root / table.get(
            "comment-baseline", ".quality/comment-baseline.json"
        )
        self.types_baseline = self.root / table.get("types-baseline", ".quality/mypy-baseline.json")
        self.type_paths = list(table.get("type-paths", ["src/rko_solver", "tools"]))

    def _table(self) -> Json:
        """Read the optional `[tool.quality]` table.

        Returns:
            Json: The table, or an empty mapping when absent.
        """
        path = self.root / "pyproject.toml"
        if not path.is_file():
            return {}
        section = tomllib.loads(path.read_text(encoding="utf-8")).get("tool", {}).get("quality", {})
        return section if isinstance(section, dict) else {}

    @staticmethod
    def _is_test(path: Path) -> bool:
        """Tell whether a path names a test module.

        Args:
            path (Path): Repository-relative file path.

        Returns:
            bool: True for `test_*.py`, `*_test.py` and `conftest.py`.
        """
        return (
            path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or path.name == "conftest.py"
        )

    def source_files(self) -> list[Path]:
        """List measured files: tracked or untracked-but-not-ignored, minus exclusions.

        Returns:
            list[Path]: Sorted repository-relative paths.

        Raises:
            subprocess.CalledProcessError: If `git ls-files` fails.
        """
        output = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=self.root,
        )
        return sorted(
            Path(raw)
            for raw in output.decode().split("\0")
            if raw
            and Path(raw).suffix in self.file_limits
            and not self.exclude.intersection(Path(raw).parts)
            and not self._is_test(Path(raw))
            and (self.root / raw).is_file()
        )

    @staticmethod
    def _qualified_names(source: str, filename: str) -> dict[int, str]:
        """Map each Python function's first line to `Class.method` or `outer.inner`.

        Args:
            source (str): Python source text.
            filename (str): Name used in syntax error messages.

        Returns:
            dict[int, str]: Qualified name by `def` line number.

        Raises:
            SyntaxError: If the source does not parse.
        """
        names: dict[int, str] = {}
        stack: list[tuple[ast.AST, str]] = [(ast.parse(source, filename=filename), "")]
        while stack:
            node, prefix = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qualified = f"{prefix}{child.name}"
                    if not isinstance(child, ast.ClassDef):
                        names[child.lineno] = qualified
                    stack.append((child, f"{qualified}."))
                else:
                    stack.append((child, prefix))
        return names

    @staticmethod
    def _unique(seen: dict[str, int], key: str) -> str:
        """Suffix repeated keys with their ordinal so each measurement keeps its own entry.

        Args:
            seen (dict[str, int]): Occurrence counts per key, updated in place.
            key (str): Candidate key.

        Returns:
            str: `key` for the first occurrence, `key#N` for the N-th.
        """
        seen[key] = seen.get(key, 0) + 1
        return key if seen[key] == 1 else f"{key}#{seen[key]}"

    def collect_structure(self) -> Json:
        """Measure line counts per file and complexity, length and parameters per function.

        Returns:
            Json: `files` mapping path to lines and `functions`
            mapping `path::signature` to its three metrics.

        Raises:
            OSError: If a source file cannot be read.
        """
        files: dict[str, int] = {}
        functions: dict[str, dict[str, int]] = {}
        for relative in self.source_files():
            path = self.root / relative
            source = path.read_text(encoding="utf-8")
            files[relative.as_posix()] = len(source.splitlines())
            names = self._qualified_names(source, str(relative)) if relative.suffix == ".py" else {}
            seen: dict[str, int] = {}
            for function in lizard.analyze_file(str(path)).function_list:
                name = names.get(function.start_line, function.name)
                key = self._unique(seen, f"{relative.as_posix()}::{name}")
                functions[key] = {
                    "cyclomatic_complexity": function.cyclomatic_complexity,
                    "nloc": function.nloc,
                    "parameter_count": function.parameter_count,
                }
        return {"files": files, "functions": functions}

    def _comments(self, source: str) -> dict[int, str]:
        """Collect real comments, skipping tool pragmas.

        Args:
            source (str): Python source text.

        Returns:
            dict[int, str]: Comment text by line number.
        """
        comments: dict[int, str] = {}
        stream = iter(source.splitlines(keepends=True)).__next__
        for token in tokenize.generate_tokens(stream):
            if token.type != tokenize.COMMENT:
                continue
            text = token.string[1:].strip()
            if not text.lower().startswith(self.IGNORED_COMMENTS):
                comments[token.start[0]] = text
        return comments

    @staticmethod
    def _longest_run(lines: set[int]) -> int:
        """Return the longest run of consecutive line numbers.

        Args:
            lines (set[int]): Line numbers.

        Returns:
            int: Length of the longest consecutive run.
        """
        runs = itertools.groupby(enumerate(sorted(lines)), key=lambda pair: pair[1] - pair[0])
        return max((len(list(group)) for _, group in runs), default=0)

    def _function_comments(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        lines: list[str],
        comments: dict[int, str],
    ) -> dict[str, float]:
        """Measure the comments inside one Python function.

        Args:
            node (ast.FunctionDef | ast.AsyncFunctionDef): Function definition.
            lines (list[str]): Source lines of the file.
            comments (dict[int, str]): Real comments of the file by line number.

        Returns:
            dict[str, float]: The five comment metrics of the function.
        """
        start, end = node.lineno, node.end_lineno or node.lineno
        inside = {line: text for line, text in comments.items() if start <= line <= end}
        code = sum(
            bool(lines[line - 1].strip()) and not lines[line - 1].lstrip().startswith("#")
            for line in range(start, end + 1)
        )
        inline = [
            len(text) for line, text in inside.items() if lines[line - 1].split("#", 1)[0].strip()
        ]
        return {
            "comment_lines": len(inside),
            "comment_chars": sum(len(text) for text in inside.values()),
            "comment_ratio": len(inside) / max(code, 1),
            "max_consecutive_comments": self._longest_run(set(inside)),
            "max_inline_comment_chars": max(inline, default=0),
        }

    def collect_comments(self) -> Json:
        """Measure comment volume inside every Python function.

        Returns:
            Json: `path::name` mapped to its five comment metrics.

        Raises:
            SyntaxError: If a Python source file does not parse.
            OSError: If a source file cannot be read.
        """
        functions: Json = {}
        for relative in (path for path in self.source_files() if path.suffix == ".py"):
            source = (self.root / relative).read_text(encoding="utf-8")
            lines, comments = source.splitlines(), self._comments(source)
            names = self._qualified_names(source, str(relative))
            seen: dict[str, int] = {}
            nodes = sorted(
                (
                    node
                    for node in ast.walk(ast.parse(source, filename=str(relative)))
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                ),
                key=lambda node: node.lineno,
            )
            for node in nodes:
                key = self._unique(seen, f"{relative.as_posix()}::{names[node.lineno]}")
                functions[key] = self._function_comments(node, lines, comments)
        return {"functions": functions}

    def collect_types(self) -> Json:
        """Count `mypy` errors per file, using the configuration in `pyproject.toml`.

        Returns:
            Json: `files` mapping each file under `type-paths` with errors to its error count
            (followed third-party imports never count); empty without `type-paths`.

        Raises:
            RuntimeError: If `mypy` fails to run (exit status above 1).
        """
        if not self.type_paths:
            return {"files": {}}
        completed = subprocess.run(
            [sys.executable, "-m", "mypy", *self.type_paths, "--output", "json"],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode > 1:
            raise RuntimeError(f"mypy failed to run: {completed.stderr or completed.stdout}")
        files: dict[str, int] = {}
        for line in completed.stdout.splitlines():
            if line.startswith("{"):
                name = Path(json.loads(line)["file"]).as_posix()
                if any(name == path or name.startswith(f"{path}/") for path in self.type_paths):
                    files[name] = files.get(name, 0) + 1
        return {"files": files}

    @staticmethod
    def _load(path: Path) -> Json:
        """Read a baseline, treating a missing file as no recorded debt.

        Args:
            path (Path): Baseline file.

        Returns:
            Json: Parsed baseline, or an empty mapping.
        """
        if not path.is_file():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _exceeding(entries: Json, limits: dict[str, float]) -> Json:
        """Keep, per function, only the metrics above their limit.

        Args:
            entries (Json): Function name mapped to its metrics.
            limits (dict[str, float]): Limit of each metric.

        Returns:
            Json: Functions with at least one metric above its limit, reduced to those metrics.
        """
        return {
            name: {m: v for m, v in metrics.items() if v > limits[m]}
            for name, metrics in entries.items()
            if any(v > limits[m] for m, v in metrics.items())
        }

    def legacy(self, structure: Json) -> Json:
        """Reduce a structure measurement to the entries that exceed a limit.

        Args:
            structure (Json): Output of `collect_structure`.

        Returns:
            Json: Baseline content with `version`, `files` and `functions`.
        """
        return {
            "version": 1,
            "files": {
                path: lines
                for path, lines in structure["files"].items()
                if lines > self.file_limits[Path(path).suffix]
            },
            "functions": self._exceeding(structure["functions"], self.function_limits),
        }

    def _structure_violations(self, structure: Json) -> list[str]:
        """List file and function measurements above their allowance.

        Args:
            structure (Json): Output of `collect_structure`.

        Returns:
            list[str]: Violations of the file and function limits.
        """
        errors: list[str] = []
        old = self._load(self.structure_baseline)
        for path, lines in sorted(structure["files"].items()):
            allowed = max(self.file_limits[Path(path).suffix], old.get("files", {}).get(path, 0))
            if lines > allowed:
                errors.append(f"{path}: {lines} lines exceeds {allowed}")
        for name, metrics in sorted(structure["functions"].items()):
            previous = old.get("functions", {}).get(name, {})
            for metric, limit in self.function_limits.items():
                allowed = max(limit, previous.get(metric, 0))
                if metrics[metric] > allowed:
                    errors.append(f"{name}: {metric}={metrics[metric]} exceeds {allowed}")
        return errors

    def _comment_violations(self, comments: Json) -> list[str]:
        """List comment measurements above their allowance.

        Args:
            comments (Json): Output of `collect_comments`.

        Returns:
            list[str]: Violations of the comment limits.
        """
        errors: list[str] = []
        old = self._load(self.comment_baseline).get("functions", {})
        for name, metrics in sorted(comments["functions"].items()):
            for metric, limit in self.comment_limits.items():
                allowed = max(limit, old.get(name, {}).get(metric, 0))
                if metrics[metric] > allowed:
                    errors.append(f"{name}: {metric}={metrics[metric]:g} exceeds {allowed:g}")
        return errors

    def _type_violations(self, types: Json) -> list[str]:
        """List files whose type-error count grew past the baseline.

        Args:
            types (Json): Output of `collect_types`.

        Returns:
            list[str]: Files with more type errors than recorded.
        """
        recorded = self._load(self.types_baseline).get("files", {})
        return [
            f"{path}: {count} type errors exceeds {recorded.get(path, 0)}"
            for path, count in sorted(types["files"].items())
            if count > recorded.get(path, 0)
        ]

    def violations(self, structure: Json, comments: Json, types: Json | None = None) -> list[str]:
        """List every measurement above `max(limit, baselined value)`.

        Args:
            structure (Json): Output of `collect_structure`.
            comments (Json): Output of `collect_comments`.
            types (Json | None): Output of `collect_types`, or None to skip type errors.

        Returns:
            list[str]: Human-readable violations, grouped by measurement kind.
        """
        return [
            *self._structure_violations(structure),
            *self._comment_violations(comments),
            *(self._type_violations(types) if types is not None else []),
        ]

    def tightened(
        self,
        recorded: Json,
        current: Json,
    ) -> Json:
        """Keep only recorded entries that still violate, never above their recorded value.

        Args:
            recorded (Json): Existing baseline content.
            current (Json): Current legacy entries from `legacy*`.

        Returns:
            Json: Baseline with resolved debt removed and values lowered.
        """
        result: Json = {"version": 1}
        for section, entries in current.items():
            if section == "version":
                continue
            old = recorded.get(section, {})
            kept: Json = {}
            for name, value in entries.items():
                if name not in old:
                    continue
                if isinstance(value, dict):
                    kept[name] = {
                        metric: min(number, old[name][metric])
                        for metric, number in value.items()
                        if metric in old[name]
                    }
                else:
                    kept[name] = min(value, old[name])
            result[section] = kept
        return result

    def _write(self, path: Path, content: Json) -> None:
        """Write a baseline as stable, sorted JSON.

        Args:
            path (Path): Destination file; parent directories are created.
            content (Json): Baseline content.

        Raises:
            OSError: If the file cannot be written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def run(
        self, write_baseline: bool = False, tighten: bool = False, check_types: bool = True
    ) -> int:
        """Check, record or tighten according to the requested mode.

        Args:
            write_baseline (bool): Record all current violations as accepted debt.
            tighten (bool): Remove resolved debt and lower recorded values only.
            check_types (bool): Include the `mypy` pass; the baseline modes always do.

        Returns:
            int: 0 when clean, 1 when a measurement exceeds its allowance.
        """
        structure = self.collect_structure()
        comments = self.collect_comments()
        types = (
            self.collect_types() if (check_types or write_baseline or tighten) else {"files": {}}
        )
        legacy = self.legacy(structure)
        legacy_comments = {
            "version": 1,
            "functions": self._exceeding(comments["functions"], self.comment_limits),
        }
        if write_baseline:
            self._write(self.structure_baseline, legacy)
            self._write(self.comment_baseline, legacy_comments)
            self._write(self.types_baseline, {"version": 1, **types})
            print(f"Wrote baselines in {self.structure_baseline.parent}")
            return 0
        if tighten:
            self._write(
                self.structure_baseline, self.tightened(self._load(self.structure_baseline), legacy)
            )
            self._write(
                self.comment_baseline,
                self.tightened(self._load(self.comment_baseline), legacy_comments),
            )
            self._write(self.types_baseline, self.tightened(self._load(self.types_baseline), types))
            print("Baselines tightened; review the diff before committing.")
            return 0
        errors = self.violations(structure, comments, types if check_types else None)
        if errors:
            print("Quality gate failed:\n- " + "\n- ".join(errors))
            return 1
        recorded = len(self._load(self.structure_baseline).get("functions", {}))
        remaining = len(legacy["functions"])
        note = f" ({recorded - remaining} baselined functions improved; run --tighten)"
        print("Quality gate: OK" + (note if remaining < recorded else ""))
        return 0

    @classmethod
    def main(cls, argv: Sequence[str] | None = None) -> int:
        """Parse arguments and run the gate.

        Args:
            argv (Sequence[str] | None): Command-line arguments, or the process arguments.

        Returns:
            int: Process exit status.

        Raises:
            SystemExit: If the arguments are invalid.
        """
        parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
        parser.add_argument("--root", type=Path, default=Path.cwd())
        parser.add_argument("--write-baseline", action="store_true")
        parser.add_argument("--tighten", action="store_true")
        parser.add_argument("--skip-types", action="store_true")
        args = parser.parse_args(argv)
        if args.write_baseline and args.tighten:
            parser.error("--write-baseline and --tighten are mutually exclusive.")
        return cls(args.root).run(args.write_baseline, args.tighten, not args.skip_types)


if __name__ == "__main__":
    raise SystemExit(QualityGate.main())
