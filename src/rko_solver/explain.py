"""Bounded, non-executing structural descriptions of application decoder logic.

Objective:
- Represent author-declared logic or supported Python source without tracing optimization.

Technical approach:
- Freeze validated plain-text nodes/edges with explicit provenance and disclosure metadata.
- Parse a limited statement subset through AST, treating expressions/calls as opaque.
- Represent branch joins, loop repetition and explicit terminal statements in O(V + E).

Flow (pseudocode):
1. Accept a declared graph or obtain explicitly requested Python function source.
2. Bound input size and reject unsupported control constructs before conversion.
3. Connect supported statements without executing code or evaluating annotations.
4. Validate identities, reachability and terminal structure; export detached metadata.

Important criteria:
- Static diagrams are neither observed traces nor proofs of code/domain completeness.
- No implicit exceptions, subexpression control or transitive call graphs are inferred.
- Literals/expressions are omitted by default; identifiers remain source metadata.
- Source inspection is best effort and never invokes the decoder or its objective.

Inputs/Outputs:
- Input: typed declared nodes/edges, explicit source, or an inspectable Python callable.
- Output: immutable DecoderDiagram and JSON-compatible graph/provenance metadata.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import re
import textwrap
from collections.abc import Callable
from collections.abc import Sequence
from dataclasses import asdict
from dataclasses import dataclass
from functools import partial
from typing import Any
from typing import Literal

NodeKind = Literal["start", "operation", "decision", "return", "end", "error"]


@dataclass(frozen=True)
class DiagramNode:
    """Store one bounded plain-text structural step.

    Responsibilities:
    - Validate stable identifiers, finite source locations and supported node categories.

    Attributes:
        id (str): Unique ASCII graph identifier, not executable markup.
        label (str): Plain text of at most 240 characters.
        kind (NodeKind): Structural role, not an observed runtime status.
        source_line (int | None): Optional line relative to the inspected source snippet.
    """

    id: str
    label: str
    kind: NodeKind = "operation"
    source_line: int | None = None

    def __post_init__(self) -> None:
        """Validate node fields before graph construction.

        Raises:
            ValueError: If identifiers, labels, kinds or source locations are invalid.
        """
        if not isinstance(self.id, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", self.id):
            raise ValueError("Diagram node IDs require 1..64 ASCII identifier characters.")
        if not isinstance(self.label, str) or not self.label.strip() or len(self.label) > 240:
            raise ValueError("Diagram labels require 1..240 plain-text characters.")
        if self.kind not in ("start", "operation", "decision", "return", "end", "error"):
            raise ValueError("Unsupported diagram node kind.")
        if self.source_line is not None and (
            type(self.source_line) is not int or not 1 <= self.source_line <= 1_000_000
        ):
            raise ValueError("Diagram source lines must be bounded positive integers.")


@dataclass(frozen=True)
class DiagramEdge:
    """Store a labelled directed relationship between graph nodes.

    Responsibilities:
    - Keep endpoints and edge text bounded without interpreting HTML or links.

    Attributes:
        source (str): Existing source node ID, checked by DecoderDiagram.
        target (str): Existing destination node ID, checked by DecoderDiagram.
        label (str): Plain branch/repetition label of at most 80 characters.
    """

    source: str
    target: str
    label: str = ""

    def __post_init__(self) -> None:
        """Validate bounded identifiers and branch text.

        Raises:
            ValueError: If an endpoint or label is not bounded plain text.
        """
        for endpoint in (self.source, self.target):
            if not isinstance(endpoint, str) or not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_]{0,63}", endpoint
            ):
                raise ValueError("Diagram edge endpoints require bounded ASCII identifiers.")
        if not isinstance(self.label, str) or len(self.label) > 80:
            raise ValueError("Diagram edge labels require at most 80 characters.")


@dataclass(frozen=True)
class DecoderDiagram:
    """Own an immutable bounded graph with explicit non-observational provenance.

    Responsibilities:
    - Validate declared graph topology and expose optional non-executing source conversion.
    - Export graph metadata without source files, callback arguments or complete raw source.

    Attributes:
        nodes (Sequence[DiagramNode]): Owned tuple of at most 100 reachable nodes.
        edges (Sequence[DiagramEdge]): Owned tuple of at most 200 distinct directed edges.
        title (str): Plain diagram heading of at most 160 characters.
        description (str): Plain explanatory text of at most 2000 characters.
        origin (Literal["author_declared", "source_derived"]): Provenance, never observed.
        function_name (str | None): Selected Python function name, without a file path.
        source_sha256 (str | None): Fingerprint of the inspected source, not execution proof.
        includes_expressions (bool): Whether source expressions/literals were explicitly allowed.
    """

    nodes: Sequence[DiagramNode]
    edges: Sequence[DiagramEdge]
    title: str = "Decoder logic"
    description: str = "Author-declared logic; correspondence with code is not verified."
    origin: Literal["author_declared", "source_derived"] = "author_declared"
    function_name: str | None = None
    source_sha256: str | None = None
    includes_expressions: bool = False

    def __post_init__(self) -> None:
        """Freeze input sequences and validate bounded metadata and reachable topology.

        Raises:
            ValueError: If bounds, fields, references, terminals or provenance are invalid.
        """
        if not isinstance(self.nodes, Sequence) or not 1 <= len(self.nodes) <= 100:
            raise ValueError("Decoder diagrams require 1..100 typed nodes.")
        if not isinstance(self.edges, Sequence) or len(self.edges) > 200:
            raise ValueError("Decoder diagrams support at most 200 typed edges.")
        nodes, edges = tuple(self.nodes), tuple(self.edges)
        if not all(isinstance(node, DiagramNode) for node in nodes) or not all(
            isinstance(edge, DiagramEdge) for edge in edges
        ):
            raise ValueError("Decoder diagrams require typed nodes and edges.")
        for label, limit in ((self.title, 160), (self.description, 2000)):
            if not isinstance(label, str) or len(label) > limit:
                raise ValueError("Diagram title/description exceeds its plain-text bound.")
        if not self.title.strip() or type(self.includes_expressions) is not bool:
            raise ValueError("Diagram title/disclosure metadata is invalid.")
        if self.origin == "source_derived":
            if not isinstance(self.function_name, str) or not 1 <= len(self.function_name) <= 160:
                raise ValueError("Source diagrams require a bounded function name.")
            if not isinstance(self.source_sha256, str) or not re.fullmatch(
                r"[0-9a-f]{64}", self.source_sha256
            ):
                raise ValueError("Source diagrams require a SHA-256 fingerprint.")
        elif self.origin != "author_declared" or any(
            (
                self.function_name is not None,
                self.source_sha256 is not None,
                self.includes_expressions,
            )
        ):
            raise ValueError("Declared diagrams cannot claim source/observed provenance.")
        identifiers = {node.id for node in nodes}
        starts = [node.id for node in nodes if node.kind == "start"]
        terminals = {node.id for node in nodes if node.kind in ("end", "error")}
        if len(identifiers) != len(nodes) or len(starts) != 1 or not terminals:
            raise ValueError("Diagrams require unique IDs, one start and an end/error terminal.")
        adjacency = {identifier: [] for identifier in identifiers}
        if len(set(edges)) != len(edges):
            raise ValueError("Duplicate diagram edges are not supported.")
        for edge in edges:
            if edge.source not in identifiers or edge.target not in identifiers:
                raise ValueError("Diagram edges must reference existing nodes.")
            if edge.target == starts[0] or edge.source in terminals:
                raise ValueError(
                    "Start cannot have incoming edges; terminals cannot have outgoing edges."
                )
            adjacency[edge.source].append(edge.target)
        if any(not adjacency[node.id] for node in nodes if node.id not in terminals):
            raise ValueError("Nonterminal diagram nodes require an outgoing transition.")
        reached: set[str] = set()
        pending = [starts[0]]
        while pending:
            current = pending.pop()
            if current not in reached:
                reached.add(current)
                pending.extend(adjacency[current])
        if reached != identifiers:
            raise ValueError("All diagram nodes must be reachable from the start.")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible graph with no raw source or callback arguments.

        Returns:
            dict[str, Any]: Bounded nodes, edges and provenance/disclosure fields.
        """
        data = asdict(self)
        data["nodes"] = list(data["nodes"])
        data["edges"] = list(data["edges"])
        return data

    @classmethod
    def from_source(
        cls,
        source: str,
        *,
        function_name: str | None = None,
        title: str | None = None,
        include_expressions: bool = False,
    ) -> DecoderDiagram:
        """Parse a supported function body without executing source or annotations.

        Args:
            source (str): At most 64000 characters of explicitly supplied Python source.
            function_name (str | None): Top-level function to select; required if ambiguous.
            title (str | None): Optional plain heading instead of the generated function heading.
            include_expressions (bool): Opt into abbreviated expressions, INCLUDING literals.

        Returns:
            DecoderDiagram: Source-derived structure with opaque expressions/calls.

        Raises:
            ValueError: If source is unavailable, invalid, oversized, ambiguous or unsupported.
        """
        return _SourceDiagram(source, function_name, title, include_expressions).build()

    @classmethod
    def from_callable(
        cls,
        decoder: Callable[..., object],
        *,
        title: str | None = None,
        include_expressions: bool = False,
    ) -> DecoderDiagram:
        """Inspect available Python source without invoking the callable or bound arguments.

        Args:
            decoder (Callable[..., object]): Function, bound method, partial or callable object.
            title (str | None): Optional plain heading.
            include_expressions (bool): Opt into abbreviated source expressions and their literals.

        Returns:
            DecoderDiagram: Best-effort static structure, never an evaluation witness.

        Raises:
            ValueError: If Python source is missing, unsupported, ambiguous or oversized.
        """
        candidate = decoder
        while isinstance(candidate, partial):
            candidate = candidate.func
        if inspect.ismethod(candidate):
            candidate = candidate.__func__
        elif not inspect.isfunction(candidate):
            candidate = inspect.getattr_static(type(candidate), "__call__", None)
        try:
            source = inspect.getsource(
                candidate.__code__ if inspect.isfunction(candidate) else candidate
            )
        except (TypeError, OSError, ValueError) as exc:
            raise ValueError(
                "Python decoder source is unavailable; "
                "provide explicit source or a declared diagram."
            ) from exc
        return cls.from_source(source, title=title, include_expressions=include_expressions)


class _SourceDiagram:
    """Convert the supported statement subset into bounded syntactic control edges.

    Responsibilities:
    - Reject unsupported flow before construction and keep expressions/calls opaque.
    - Route nearest-loop jumps and explicit returns without invoking user code.

    Attributes:
        source (str): Dedented source retained only during conversion.
        function (ast.FunctionDef): Selected syntax node, not an executable function.
        nodes (list[DiagramNode]): Bounded construction buffer.
        edges (list[DiagramEdge]): Bounded construction buffer.
        include_expressions (bool): Explicit literal/snippet disclosure option.
        title (str | None): Optional user heading.
    """

    def __init__(
        self,
        source: str,
        function_name: str | None,
        title: str | None,
        include_expressions: bool,
    ) -> None:
        """Validate bounded syntax and select one supported synchronous function.

        Args:
            source (str): Explicit source, never executed.
            function_name (str | None): Optional selected top-level function.
            title (str | None): Optional graph title.
            include_expressions (bool): Whether expression text may be exported.

        Raises:
            ValueError: If syntax, bounds, selection or supported constructs are invalid.
        """
        if not isinstance(source, str) or not 1 <= len(source) <= 64_000:
            raise ValueError("Python decoder source requires 1..64000 characters.")
        if type(include_expressions) is not bool:
            raise ValueError("include_expressions must be a boolean.")
        self.source = textwrap.dedent(source)
        try:
            tree = ast.parse(self.source)
        except SyntaxError, RecursionError:
            raise ValueError("Invalid or excessively nested Python decoder source.") from None
        if sum(1 for _ in ast.walk(tree)) > 4000:
            raise ValueError("Python decoder source exceeds 4000 syntax nodes.")
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        if function_name is not None:
            functions = [node for node in functions if node.name == function_name]
        if len(functions) != 1:
            raise ValueError(
                "Unsupported or ambiguous source: select one synchronous function_name."
            )
        self.function = functions[0]
        supported = (
            ast.Assign,
            ast.AnnAssign,
            ast.AugAssign,
            ast.Expr,
            ast.If,
            ast.For,
            ast.While,
            ast.Return,
            ast.Raise,
            ast.Break,
            ast.Continue,
            ast.Pass,
        )
        pending = [(self.function, False)]
        while pending:
            node, inside_loop = pending.pop()
            if node is not self.function and (
                isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await))
                or (isinstance(node, ast.stmt) and not isinstance(node, supported))
                or (isinstance(node, (ast.For, ast.While)) and node.orelse)
                or (isinstance(node, (ast.Break, ast.Continue)) and not inside_loop)
                or (isinstance(node, ast.comprehension) and node.is_async)
            ):
                raise ValueError(
                    f"Unsupported {type(node).__name__} at source line "
                    f"{getattr(node, 'lineno', self.function.lineno)}; use a declared diagram."
                )
            pending.extend(
                (child, inside_loop or isinstance(node, (ast.For, ast.While)))
                for child in ast.iter_child_nodes(node)
            )
        self.title = title
        self.include_expressions = include_expressions
        self.nodes = [
            DiagramNode("start", "Entry", "start", self.function.lineno),
            DiagramNode("end", "Normal exit", "end"),
        ]
        self.edges: list[DiagramEdge] = []

    @staticmethod
    def _reference(node: ast.AST) -> str:
        """Name identifiers/callees without exposing literal values or evaluating attributes.

        Args:
            node (ast.AST): Opaque expression/target syntax.

        Returns:
            str: Identifier chain or a generic syntax category.
        """
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{_SourceDiagram._reference(node.value)}.{node.attr}"
        if isinstance(node, ast.Call):
            return f"{_SourceDiagram._reference(node.func)}(...)"
        return type(node).__name__

    def _node(self, label: str, kind: NodeKind = "operation", line: int | None = None) -> str:
        """Append a bounded, optionally abbreviated graph step.

        Args:
            label (str): Plain generated label.
            kind (NodeKind): Structural category.
            line (int | None): Relative source location or None for synthetic joins.

        Returns:
            str: Fresh node ID.

        Raises:
            ValueError: If conversion exceeds the 100-node bound.
        """
        if len(self.nodes) >= 100:
            raise ValueError(
                "Source diagram exceeds 100 nodes; provide a smaller declared diagram."
            )
        identifier = f"n{len(self.nodes)}"
        self.nodes.append(
            DiagramNode(identifier, label[:237] + "..." if len(label) > 240 else label, kind, line)
        )
        return identifier

    def _connect(self, incoming: list[tuple[str, str]], target: str) -> None:
        """Connect pending labelled paths to a bounded destination.

        Args:
            incoming (list[tuple[str, str]]): Source IDs and pending branch labels.
            target (str): Destination ID.

        Raises:
            ValueError: If conversion exceeds the 200-edge bound.
        """
        if len(self.edges) + len(incoming) > 200:
            raise ValueError("Source diagram exceeds 200 edges.")
        self.edges.extend(DiagramEdge(source, target, label) for source, label in incoming)

    def _block(
        self,
        statements: list[ast.stmt],
        incoming: list[tuple[str, str]],
        loop: tuple[str, str] | None = None,
    ) -> list[tuple[str, str]]:
        """Connect one statement block, retaining only nonterminal outgoing paths.

        Args:
            statements (list[ast.stmt]): Supported statements in source order.
            incoming (list[tuple[str, str]]): Unfinished paths entering this block.
            loop (tuple[str, str] | None): Nearest repetition/exit targets for loop jumps.

        Returns:
            list[tuple[str, str]]: Paths that can syntactically continue after the block.

        Raises:
            ValueError: If a jump has no enclosing loop or graph bounds are exceeded.
            RecursionError: If source nesting exceeds the interpreter recursion limit.
        """
        pending = incoming
        for statement in statements:
            if not pending:
                break
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                continue
            if self.include_expressions:
                if isinstance(statement, (ast.If, ast.While)):
                    label = f"{type(statement).__name__} {ast.unparse(statement.test)}"
                elif isinstance(statement, ast.For):
                    label = f"For {ast.unparse(statement.target)} in {ast.unparse(statement.iter)}"
                else:
                    label = ast.unparse(statement)
            elif isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                target = (
                    statement.targets[0] if isinstance(statement, ast.Assign) else statement.target
                )
                action = "Update" if isinstance(statement, ast.AugAssign) else "Assign"
                if isinstance(statement, ast.AnnAssign) and statement.value is None:
                    action = "Annotate"
                label = f"{action} {self._reference(target)}"
                if isinstance(statement.value, ast.Call):
                    label += f" = {self._reference(statement.value)}"
            elif isinstance(statement, ast.Expr):
                label = (
                    f"Call {self._reference(statement.value)}"
                    if isinstance(statement.value, ast.Call)
                    else "Expression (opaque)"
                )
            elif isinstance(statement, ast.For):
                label = (
                    f"For {self._reference(statement.target)} in {self._reference(statement.iter)}"
                )
            elif isinstance(statement, ast.Return):
                label = (
                    "Return"
                    if statement.value is None
                    else f"Return {self._reference(statement.value)}"
                )
            elif isinstance(statement, ast.Raise):
                label = (
                    "Raise" if statement.exc is None else f"Raise {self._reference(statement.exc)}"
                )
            else:
                label = {
                    ast.If: "If condition",
                    ast.While: "While condition",
                    ast.Break: "Break",
                    ast.Continue: "Continue",
                    ast.Pass: "Pass",
                }[type(statement)]
            kind: NodeKind = (
                "decision" if isinstance(statement, (ast.If, ast.For, ast.While)) else "operation"
            )
            if isinstance(statement, ast.Return):
                kind = "return"
            elif isinstance(statement, ast.Raise):
                kind = "error"
            current = self._node(label, kind, statement.lineno)
            self._connect(pending, current)
            if isinstance(statement, ast.If):
                pending = self._block(statement.body, [(current, "yes")], loop) + self._block(
                    statement.orelse, [(current, "no")], loop
                )
            elif isinstance(statement, (ast.For, ast.While)):
                exit_node = self._node("Loop exit")
                self._connect(
                    [(current, "done" if isinstance(statement, ast.For) else "no")], exit_node
                )
                body = self._block(
                    statement.body,
                    [(current, "next" if isinstance(statement, ast.For) else "yes")],
                    (current, exit_node),
                )
                self._connect([(node, label or "repeat") for node, label in body], current)
                pending = [(exit_node, "")]
            elif isinstance(statement, ast.Return):
                self._connect([(current, "return")], "end")
                pending = []
            elif isinstance(statement, ast.Raise):
                pending = []
            elif isinstance(statement, (ast.Break, ast.Continue)):
                if loop is None:
                    raise ValueError(f"Unsupported {type(statement).__name__} outside a loop.")
                self._connect(
                    [(current, "break" if isinstance(statement, ast.Break) else "continue")],
                    loop[1 if isinstance(statement, ast.Break) else 0],
                )
                pending = []
            else:
                pending = [(current, "")]
        return pending

    def build(self) -> DecoderDiagram:
        """Finalize a syntax-only graph and omit an unused synthetic normal exit.

        Returns:
            DecoderDiagram: Bounded source-derived structure and disclosure metadata.

        Raises:
            ValueError: If syntax nesting, graph bounds or metadata are invalid.
        """
        try:
            pending = self._block(self.function.body, [("start", "")])
            self._connect(pending, "end")
        except RecursionError as exc:
            raise ValueError(
                "Unsupported excessive source nesting; use a declared diagram."
            ) from exc
        used_end = any(edge.target == "end" for edge in self.edges)
        nodes = [node for node in self.nodes if node.id != "end" or used_end]
        return DecoderDiagram(
            nodes,
            self.edges,
            title=self.title if self.title is not None else f"Decoder: {self.function.name}",
            description=(
                "Static syntax only. Calls/expressions are opaque; implicit exceptions are not "
                "modelled. Not an execution trace. Labels may be abbreviated."
            ),
            origin="source_derived",
            function_name=self.function.name,
            source_sha256=hashlib.sha256(self.source.encode("utf-8")).hexdigest(),
            includes_expressions=self.include_expressions,
        )
