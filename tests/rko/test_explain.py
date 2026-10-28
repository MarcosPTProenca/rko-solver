"""Structural and security checks for optional static decoder diagrams.

Objective:
- Verify bounded declared graphs and truthful source-derived control edges.

Technical approach:
- Compare graph structure rather than pixels and keep source inspection non-executing.
- Exercise unsupported syntax, private literals, missing source and immutable snapshots.

Flow (pseudocode):
1. Construct declared graphs and inspect small Python decoder fixtures.
2. Check branches, loop exits and explicit terminal statements.
3. Reject malformed/oversized graphs and unsafe assumptions.

Important criteria:
- No diagram is an original execution trace or a completeness certificate.
- Source parsing must never execute callbacks, annotations or module statements.

Inputs/Outputs:
- Input: declared nodes/edges and bounded Python source fixtures.
- Output: assertions about metadata, topology, validation and disclosure.
"""

import json
import pytest
import re
from functools import partial
from functools import wraps
from pathlib import Path
from rko_solver.explain import DecoderDiagram
from rko_solver.explain import DiagramEdge
from rko_solver.explain import DiagramNode
from unittest.mock import Mock


def declared() -> DecoderDiagram:
    """Build a small author-declared decoder graph.

    Returns:
        DecoderDiagram: Entry, operation and normal exit with two edges.
    """
    return DecoderDiagram(
        nodes=[
            DiagramNode("start", "Keys", "start"),
            DiagramNode("rank", "Rank candidates"),
            DiagramNode("end", "Solution", "end"),
        ],
        edges=[DiagramEdge("start", "rank"), DiagramEdge("rank", "end")],
        title="Priority decoder",
    )


def fixture_decoder(keys: tuple[float, ...], factor: float = 2.0) -> list[float]:
    """Collect scaled keys through an explicitly branching loop.

    Args:
        keys (tuple[float, ...]): Candidate values.
        factor (float): Scaling coefficient.

    Returns:
        list[float]: Scaled positive values.
    """
    values = []
    for key in keys:
        if key <= 0:
            continue
        values.append(key * factor)
    return values


@wraps(fixture_decoder)
def wrapped_decoder(keys: tuple[float, ...]) -> list[float]:
    """Expose a wrapper body that differs from its copied function metadata.

    Args:
        keys (tuple[float, ...]): Unused input for this non-executed fixture.

    Raises:
        RuntimeError: If this fixture is invoked instead of inspected.
    """
    raise RuntimeError("must not execute")


def test_decorated_callable_uses_actual_wrapper_code_not_copied_metadata() -> None:
    """Do not substitute __wrapped__ source for the callable's actual Python code body."""
    diagram = DecoderDiagram.from_callable(wrapped_decoder)
    assert diagram.function_name == "wrapped_decoder"
    assert any(node.kind == "error" for node in diagram.nodes)
    assert not any(node.kind == "return" for node in diagram.nodes)


def test_declared_graph_owns_sequences_and_json_snapshot() -> None:
    """Keep graph metadata immutable and detached from serialization clients."""
    diagram = declared()
    assert diagram.origin == "author_declared"
    assert isinstance(diagram.nodes, tuple)
    assert isinstance(diagram.edges, tuple)
    data = diagram.to_dict()
    data["nodes"][0]["label"] = "changed"
    assert diagram.nodes[0].label == "Keys"
    assert diagram.to_dict()["source_sha256"] is None
    json.dumps(diagram.to_dict(), allow_nan=False)


@pytest.mark.parametrize(
    "nodes,edges",
    [
        ([DiagramNode("a", "Entry", "start"), DiagramNode("a", "Exit", "end")], []),
        (
            [DiagramNode("a", "Entry", "start"), DiagramNode("b", "Exit", "end")],
            [DiagramEdge("a", "missing")],
        ),
        ([DiagramNode("a", "Entry", "start"), DiagramNode("b", "Exit", "end")], []),
        (
            [DiagramNode("a", "Not an entry"), DiagramNode("b", "Exit", "end")],
            [DiagramEdge("a", "b")],
        ),
        (
            [DiagramNode("a", "Entry", "start"), DiagramNode("b", "No terminal")],
            [DiagramEdge("a", "b")],
        ),
    ],
)
def test_invalid_topology_fails(nodes: list[DiagramNode], edges: list[DiagramEdge]) -> None:
    """Reject ambiguous identities, invalid references and unreachable nodes.

    Args:
        nodes (list[DiagramNode]): Malformed graph nodes.
        edges (list[DiagramEdge]): Malformed graph edges.
    """
    with pytest.raises(ValueError):
        DecoderDiagram(nodes, edges)


@pytest.mark.parametrize(
    "identifier,label,kind,line",
    [
        ("<script>", "Entry", "start", None),
        ("n", "", "operation", None),
        ("n", "x" * 241, "operation", None),
        ("n", "Step", "observed", None),
        ("n", "Step", "operation", True),
        ("n", "Step", "operation", 0),
    ],
)
def test_invalid_node_fields_fail(identifier: str, label: str, kind: str, line: int | None) -> None:
    """Bound plain labels, identifiers, kinds and source locations.

    Args:
        identifier (str): Candidate identifier.
        label (str): Candidate plain label.
        kind (str): Candidate node category.
        line (int | None): Candidate source line.
    """
    with pytest.raises(ValueError):
        DiagramNode(identifier, label, kind, source_line=line)


def test_dangling_operations_and_forged_provenance_fail() -> None:
    """Require explicit terminals and disallow observed/source claims on declarations."""
    diagram = declared()
    with pytest.raises(ValueError, match="Nonterminal"):
        DecoderDiagram(
            [*diagram.nodes, DiagramNode("dangling", "Unfinished")],
            [*diagram.edges, DiagramEdge("rank", "dangling")],
        )
    with pytest.raises(ValueError, match="provenance"):
        DecoderDiagram(diagram.nodes, diagram.edges, origin="observed")
    with pytest.raises(ValueError, match="function"):
        DecoderDiagram(diagram.nodes, diagram.edges, origin="source_derived")
    with pytest.raises(ValueError):
        DiagramEdge("start", "end", "x" * 81)
    with pytest.raises(ValueError, match="source"):
        DecoderDiagram.from_callable(len)


def test_nested_loop_jumps_target_the_nearest_loop() -> None:
    """Keep nested while/for break exits distinct without invented repeat edges."""
    diagram = DecoderDiagram.from_source(
        "def decode(keys):\n    for key in keys:\n        while key:\n"
        "            break\n        break\n    return keys\n"
    )
    by_line = {node.source_line: node.id for node in diagram.nodes if node.source_line}
    inner_exit = next(
        edge.target for edge in diagram.edges if edge.source == by_line[3] and edge.label == "no"
    )
    outer_exit = next(
        edge.target for edge in diagram.edges if edge.source == by_line[2] and edge.label == "done"
    )
    assert inner_exit != outer_exit
    assert any(edge.source == by_line[4] and edge.target == inner_exit for edge in diagram.edges)
    assert any(edge.source == by_line[5] and edge.target == outer_exit for edge in diagram.edges)
    assert not any(edge.label == "repeat" for edge in diagram.edges)
    with pytest.raises(ValueError, match="Unsupported"):
        DecoderDiagram.from_source("def decode(keys):\n    return keys\n    break\n")


def test_graph_and_source_size_limits_fail_before_rendering() -> None:
    """Reject oversized graphs and source before generating report assets."""
    with pytest.raises(ValueError, match="100"):
        DecoderDiagram([DiagramNode(f"n{i}", "Step") for i in range(101)], [])
    with pytest.raises(ValueError, match="200"):
        DecoderDiagram(declared().nodes, [DiagramEdge("start", "rank")] * 201)
    with pytest.raises(ValueError):
        DecoderDiagram.from_source(" " * 64_001)
    with pytest.raises(ValueError):
        DecoderDiagram.from_source("def decode(keys):\n" + "    x = 1\n" * 110 + "    return x\n")


def test_branch_returns_have_no_fallthrough_edges() -> None:
    """Join only branches that have not returned from the decoder."""
    diagram = DecoderDiagram.from_source(
        "def decode(keys: UnknownType) -> UnknownType:\n"
        "    values = []\n"
        "    if keys:\n"
        "        values.append(1)\n"
        "    else:\n"
        "        return values\n"
        "    return values\n"
    )
    by_line = {node.source_line: node for node in diagram.nodes if node.kind != "end"}
    edges = {(edge.source, edge.target, edge.label) for edge in diagram.edges}
    assert (by_line[3].id, by_line[4].id, "yes") in edges
    assert (by_line[3].id, by_line[6].id, "no") in edges
    assert (by_line[4].id, by_line[7].id, "") in edges
    assert not any(
        edge.source == by_line[6].id and edge.target == by_line[7].id for edge in diagram.edges
    )
    assert diagram.origin == "source_derived"
    assert diagram.function_name == "decode"
    assert len(diagram.source_sha256) == 64


def test_loops_break_continue_and_returns_have_explicit_targets() -> None:
    """Route loop repetition and exits without inventing fallthrough after control jumps."""
    diagram = DecoderDiagram.from_source(
        "def decode(keys):\n"
        "    values = []\n"
        "    for key in keys:\n"
        "        if key < 0:\n"
        "            continue\n"
        "        if key > 1:\n"
        "            break\n"
        "        values.append(key)\n"
        "    return values\n"
    )
    loop = next(node for node in diagram.nodes if node.source_line == 3)
    continuation = next(node for node in diagram.nodes if node.source_line == 5)
    breaking = next(node for node in diagram.nodes if node.source_line == 7)
    assert any(edge.source == continuation.id and edge.target == loop.id for edge in diagram.edges)
    target = next(edge.target for edge in diagram.edges if edge.source == breaking.id)
    assert next(node for node in diagram.nodes if node.id == target).label == "Loop exit"
    assert any(edge.target == loop.id and edge.label == "repeat" for edge in diagram.edges)
    assert all(edge.target != "start" for edge in diagram.edges)


def test_explicit_raise_is_an_error_terminal() -> None:
    """Do not connect explicitly raised exceptions to normal return paths."""
    diagram = DecoderDiagram.from_source("def decode(keys):\n    raise ValueError('PRIVATE')\n")
    error = next(node for node in diagram.nodes if node.kind == "error")
    assert not any(edge.source == error.id for edge in diagram.edges)
    assert not any(node.kind == "end" for node in diagram.nodes)
    assert "PRIVATE" not in json.dumps(diagram.to_dict())


@pytest.mark.parametrize(
    "body",
    [
        "try:\n        x = 1\n    except Exception:\n        x = 2",
        "with context():\n        x = 1",
        "yield keys",
        "for key in keys:\n        pass\n    else:\n        pass",
        "def nested():\n        return keys",
        "match keys:\n        case []:\n            pass",
        "break",
        "continue",
    ],
)
def test_unsupported_constructs_have_clear_diagnostics(body: str) -> None:
    """Reject unsupported flow rather than manufacturing a successful-looking graph.

    Args:
        body (str): Unsupported function body with relative indentation.
    """
    with pytest.raises(ValueError, match="Unsupported"):
        DecoderDiagram.from_source("def decode(keys):\n    " + body + "\n")


def test_parsing_never_executes_source_or_evaluates_annotations() -> None:
    """Parse opaque dangerous calls without importing modules or evaluating callbacks."""
    source = (
        "raise RuntimeError('module must not run')\n"
        "def decode(keys: MissingAnnotation) -> MissingAnnotation:\n"
        "    never_execute(keys)\n"
        "    return keys\n"
    )
    diagram = DecoderDiagram.from_source(source)
    assert any("never_execute" in node.label for node in diagram.nodes)
    assert "module must not run" not in json.dumps(diagram.to_dict())


def test_literal_disclosure_is_explicit() -> None:
    """Omit source literals unless the caller explicitly opts into expression labels."""
    source = "def decode(keys):\n    token = 'SENSITIVE_LITERAL_123'\n    return token\n"
    redacted = DecoderDiagram.from_source(source)
    disclosed = DecoderDiagram.from_source(source, include_expressions=True)
    assert "SENSITIVE_LITERAL_123" not in json.dumps(redacted.to_dict())
    assert "SENSITIVE_LITERAL_123" in json.dumps(disclosed.to_dict())
    assert not redacted.includes_expressions
    assert disclosed.includes_expressions


def test_callable_source_supports_bound_data_without_calling_decoder() -> None:
    """Inspect a partial's underlying function without consuming its arguments."""
    diagram = DecoderDiagram.from_callable(partial(fixture_decoder, factor=5.0))
    assert diagram.function_name == "fixture_decoder"
    assert any(node.kind == "decision" for node in diagram.nodes)
    assert not diagram.includes_expressions

    class SourceOnlyDecoder:
        """Expose inspectable application code without executing it.

        Responsibilities:
        - Fail loudly if inspection accidentally invokes the callback.

        Attributes:
            None: This fixture owns no instance data.
        """

        def __call__(self, keys: tuple[float, ...]) -> object:
            """Reject invocation of this source-only test fixture.

            Args:
                keys (tuple[float, ...]): Unused callback input.

            Raises:
                AssertionError: If inspection accidentally executes the callback.
            """
            raise AssertionError("callback must not run")

    callable_graph = DecoderDiagram.from_callable(SourceOnlyDecoder())
    assert callable_graph.function_name == "__call__"
    assert any(node.kind == "error" for node in callable_graph.nodes)


def test_documented_python_workflow_and_declared_example_execute(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Run both guide examples without optional rendering dependencies.

    Args:
        monkeypatch (pytest.MonkeyPatch): Isolates documented artifact writes.
        tmp_path (Path): Temporary application working directory.
    """
    root = Path(__file__).resolve().parents[2]
    guide = root / "docs" / "decoder-diagrams.md"
    blocks = re.findall(r"```python\n(.*?)```", guide.read_text(encoding="utf-8"), re.S)
    assert len(blocks) == 2
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, object] = {}
    for block in blocks:
        exec(compile(block, str(guide), "exec"), namespace)
    exported = json.loads((tmp_path / "decoder.json").read_text(encoding="utf-8"))
    assert exported["decoder_diagram"]["origin"] == "source_derived"
    capacity_diagram = namespace["capacity_diagram"]
    assert isinstance(capacity_diagram, DecoderDiagram)
    assert capacity_diagram.origin == "author_declared"


def test_missing_source_has_declared_diagram_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain unavailable Python source without calling the supplied callback.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces documented inspection failures.
    """
    callback = Mock()
    monkeypatch.setattr(
        "rko_solver.explain.inspect.getsource", Mock(side_effect=OSError("unavailable"))
    )
    with pytest.raises(ValueError, match="source"):
        DecoderDiagram.from_callable(callback)
    callback.assert_not_called()
    assert declared().origin == "author_declared"
