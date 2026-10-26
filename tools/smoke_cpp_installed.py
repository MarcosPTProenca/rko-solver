"""Check that an installed Linux wheel bundles a working C++ reference engine.

Objective:
- Detect a wheel built without the C++ engine or without its OpenMP runtime.

Technical approach:
- Import the installed extension, ask whether the engine is present and run one
  one-second solve on a small Knapsack through `engine="cpp"`.

Flow (pseudocode):
1. Import the extension and require `cpp_available()`.
2. Solve a 13-item Knapsack with the C++ engine.
3. Require the result to name the engine and carry a negative (profitable) cost.

Important criteria:
- Run with `python -I` outside the checkout so only the installed package is imported.
- The instance has at least 10 keys because the reference IPR needs that many.

Inputs/Outputs:
- Input: an installed rko wheel.
- Output: exit status zero and one confirmation line, or an assertion failure.
"""

from rko_solver import RKO
from rko_solver import _native


def main() -> None:
    """Solve one small Knapsack with the C++ engine and verify the result.

    Raises:
        AssertionError: If the engine is absent or returns an unusable result.
    """
    assert _native.cpp_available(), "this build has no C++ engine"
    problem = _native.KnapsackProblem(list(range(10, 62, 4)), list(range(5, 31, 2)), 60)
    result = RKO(problem, engine="cpp", random_state=1).solve(1)
    assert result.engine == "cpp" and result.cost < 0
    print("C++ engine smoke passed")


if __name__ == "__main__":
    main()
