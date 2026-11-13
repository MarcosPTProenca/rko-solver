# RKO

Canonical repository: `MarcosPTProenca/rko-solver`, branch `main`.
Run all tooling from this repository root.
Memory scope: workspace `default`, project `marcosptproenca-rko-python`.
Keep this existing memory scope despite the repository rename to preserve history.
Public limitations: `docs/status.md`. Reproducible measurements: `benchmarks/`.
Do not commit local audits, agent plans, migration diaries, generated analysis or
editor settings. Keep those ignored; retain only maintained user documentation,
benchmark code and an explicitly identified reference measurement in Git.

## DO

- Write production-grade code, prioritizing readability, consistency, and maintainability.
- Every class and every function must include a clear, objective docstring describing purpose, parameters, return values, and possible exceptions.
- Use a module docstring as a mandatory technical header at the top of every file.
- The technical header must explain, in technical terms, what the module implements and how it works (e.g., for metaheuristics, describe strategy, steps, acceptance/stop criteria, and expected complexity when applicable).
- Always include a high-level flow/pseudocode section in the header to support maintenance and review.
- Follow the fixed module docstring template below.
- Follow the fixed class and function/method docstring templates below.
- Always use complete type hints: inputs and outputs must be typed in functions, methods, and public signatures.
- Organize imports at the top of the file, one import per line, in this exact order: external libraries first, then internal imports, with no blank lines between them.
- After completing any implementation, run Ruff and only consider the task complete when lint passes with no errors.
- Keep Ruff configured and mandatory in the pre-commit workflow for automatic validation before each commit.
- For a single logical responsibility, centralize the implementation in one cohesive class, avoiding fragmentation into multiple loose functions.
- The agent output must be entirely in English (responses, explanations, summaries, and generated text).
- This project uses `uv` as the environment/package runner; always execute tooling through `uv run ...`.

### Mandatory module docstring template (header)

```python
"""
<Technical module title>

Objective:
- <what the module solves>

Technical approach:
- <how it solves it>
- <main structures/strategies>

Flow (pseudocode):
1. <step 1>
2. <step 2>
3. <step 3>

Important criteria:
- <stopping, acceptance, constraints, invariants>

Inputs/Outputs:
- Input: <types and meaning>
- Output: <types and meaning>
"""
```

### Mandatory class docstring template

```python
"""<What the class does technically>.

Responsibilities:
- <main responsibility 1>
- <main responsibility 2>

Attributes:
	attribute (type): <objective description>
"""
```

### Mandatory function/method docstring template

```python
"""<What the function/method does technically>.

Args:
	arg_name (type): <objective input description>.

Returns:
	type: <objective output description>.

Raises:
	ExceptionType: <when it can occur>.
"""
```

Additional rules for function/method docstrings:
- `Args` must list all input parameters with type and meaning.
- `Returns` is mandatory when a return value exists.
- `Raises` is mandatory when exceptions are possible; if none apply, do not invent exceptions.
- The opening description must be technical, direct, and focused on implementation behavior.

### Bash example (mandatory at the end of each implementation)

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest tests -q
```

## Rust (native engines in `engines/`)

- Native code lives in `engines/rust/rko-core` (pure Rust, no dependencies, `unsafe` forbidden) and
  `engines/rust/rko-py` (PyO3 bindings). Run `cargo` from the repository root.
- Every public item and every module carries rustdoc; a `//!` header states objective,
  approach, flow, criteria and inputs/outputs like the Python module template. No `//` comments.
- Hot loops must not allocate per evaluation: reuse caller-owned buffers, route every
  candidate through `Search::eval`/`Search::probe`, and never read the clock outside `Search`.
- Each search owns its `Rng`; never share generators or mutable state between threads except
  through `Pool` and the cancel flag.
- Mandatory after Rust changes:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --release -- -D warnings
cargo test --release
```

- A native speed or quality claim needs `benchmarks/engines/native_study.py` evidence against the
  pinned C++ reference or the Python engine, with losses and failures reported.

## DON'T

- Do not generate comments in code (neither inline comments nor block comments).
- Do not leave classes, functions, or methods without docstrings.
- Do not leave functions without input and output type hints.
- Do not skip running Ruff after finishing changes.
- Do not split a single logic into multiple separate functions unnecessarily; prefer one cohesive class.

## Code quality rules

- Always meet the project quality rules before calling a change finished: `tools/quality_gate.py`, `tools/debt_counter.py`, `lint-imports`, Ruff and the mypy ratchet. A change that fails a gate is not finished.
- Never raise a recorded baseline in `.quality/*.json` to make a gate pass, and never run `--write-baseline` for that purpose. Refactor until the gate passes.
- Do not add a parameter to a function that is already above the 6-parameter limit. Group the related values into one value object and pass that object instead.
- Do not suppress a check (`# type: ignore`, `# noqa`) except for a third-party false positive; name that false positive in the docstring of the function that carries the suppression.
- If a gate cannot be met without a design decision, stop and ask the owner with the options. Do not choose the baseline increase or the public API change yourself.

## Scientific evidence for changes

Every feature, improvement or new idea in this repository needs its own evidence, written as a short scientific report. The protocol is defined in #177.

- Write the hypothesis and the success and failure criteria before running anything.
- Run the experiment on the paper benchmark set (#178: TSP from TSPLIB, MDKP from OR-Library mknapcb, QAP from QAPLIB), at an equal evaluation budget, against the current `main` as baseline, with at least 10 paired seeds.
- Measure with the paired, interleaved design of #176: arms alternate, work is fixed by `max_evaluations`, and the host load is recorded per run. Do not require an idle host.
- Report a table (outcome, paired differences, 95% intervals, sign-test p-value, evaluation counts, load) and a plot (convergence against evaluations, distribution of paired differences), both generated by a committed script.
- Write the result as a short LaTeX section next to the experiment, using the `latex-document-skill`.
- Report negative and inconclusive results in the same format. Do not omit them.
- Self-Evolve with an LLM uses OpenAI Codex `gpt-6-luna` (owner decision). Record provider, model id, temperature and the number of calls. Use the sealed test split defined in #179.
- Tests and gates prove correctness, not improvement. A speed or quality claim without the experiment above is not accepted.

## Workflow autonomy

- Do not ask whether to open a pull request when a piece of work is finished. Open it, following the commit-pr-conventions skill, and report the URL.
- Do not ask whether to extend a measurement or experiment. When an issue needs more evidence, run the next measurement the issue calls for, including longer runs, more seeds, more families or more hosts when the issue asks for them.
- Exhaust the possibilities before closing an issue: every acceptance criterion gets evidence or an explicit "not verified" reason. Close only when those criteria are met or the issue's own rule says to close with the profile.
- Every finished piece of work ends in a pull request, not a local commit or an unpublished branch.
- When reporting the next steps to the user, give only the recommended next issue to attack. Do not list other options or ask for confirmation of it.
