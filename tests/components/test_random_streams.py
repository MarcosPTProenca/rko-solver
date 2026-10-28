"""Check context-local generator ownership and legacy draw compatibility.

Objective:
- Protect standalone behavior, nested scopes and exception-safe restoration.

Technical approach:
- Compare scoped streams with independent Python and NumPy legacy generators.
- Reenter scopes and fail deliberately without altering caller global states.

Flow (pseudocode):
1. Resolve the standalone generator modules outside a search scope.
2. Enter nested or reused stream scopes and compare generator identities.
3. Raise inside a scope and require restoration of the enclosing context.

Important criteria:
- Context restoration must not reset an owned generator's consumed state.
- RandomState preserves legacy trajectories; Generator is not interchangeable.

Inputs/Outputs:
- Input: explicit seeds and nested activation scopes.
- Output: exact stream compatibility and ownership assertions.
"""

import ast
import numpy as np
import pickle
import pytest
import random
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from pathlib import Path
from rko_solver.components.random_streams import RandomStreams


@settings(max_examples=20, derandomize=True, deadline=None)
@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_owned_generators_match_legacy_draws(seed: int) -> None:
    """Match mixed draws and final generator states without modifying globals.

    Args:
        seed (int): Assigned valid legacy NumPy seed.
    """
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    streams = RandomStreams(seed)
    python = random.Random(seed)
    numpy = np.random.RandomState(seed)
    with streams.activate():
        for _ in range(20):
            assert RandomStreams.python_rng().random() == python.random()
            assert RandomStreams.python_rng().randint(1, 7) == python.randint(1, 7)
            assert RandomStreams.python_rng().sample(range(10), 2) == python.sample(range(10), 2)
            assert np.array_equal(RandomStreams.numpy_rng().random(64), numpy.random(64))
            assert np.array_equal(
                RandomStreams.numpy_rng().randint(0, 100, size=7), numpy.randint(0, 100, size=7)
            )
        assert streams.python.getstate() == python.getstate()
        assert pickle.dumps(streams.numpy.get_state()) == pickle.dumps(numpy.get_state())
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before


def test_nested_scopes_restore_streams_after_failure() -> None:
    """Restore enclosing identities through nested reentry and exceptional exits."""
    outer = RandomStreams(42)
    inner = RandomStreams(43)
    assert RandomStreams.python_rng() is random
    assert RandomStreams.numpy_rng() is np.random
    with outer.activate():
        assert RandomStreams.python_rng() is outer.python
        with inner.activate():
            assert RandomStreams.python_rng() is inner.python
            assert RandomStreams.numpy_rng() is inner.numpy
            with pytest.raises(ArithmeticError):
                with outer.activate():
                    assert RandomStreams.python_rng() is outer.python
                    raise ArithmeticError("deliberate scope failure")
            assert RandomStreams.python_rng() is inner.python
        assert RandomStreams.python_rng() is outer.python
        assert RandomStreams.numpy_rng() is outer.numpy
    assert RandomStreams.python_rng() is random
    assert RandomStreams.numpy_rng() is np.random


def test_library_draw_sites_resolve_owned_streams() -> None:
    """Guard known random call patterns against accidental global draw regressions.

    Raises:
        AssertionError: If production draw sites bypass the scoped providers.
    """
    root = Path(__file__).resolve().parents[2] / "src" / "rko_solver"
    for path in root.rglob("*.py"):
        if {"tests", "examples"}.intersection(path.relative_to(root).parts):
            continue
        if path.name == "random_streams.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if ast.unparse(node.func.value) == "np.random":
                    assert node.func.attr == "SeedSequence", (path, node.lineno)
            if not isinstance(node, ast.FunctionDef):
                continue
            if not any(
                isinstance(child, ast.Attribute)
                and isinstance(child.value, ast.Name)
                and child.value.id == "random"
                for child in ast.walk(node)
            ):
                continue
            assert any(
                isinstance(child, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "random"
                    for target in child.targets
                )
                and ast.unparse(child.value) == "RandomStreams.python_rng()"
                for child in ast.walk(node)
            ), (path, node.name)
