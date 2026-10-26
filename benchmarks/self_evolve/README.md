# Self-Evolve research runtime

Build from the repository root with **`src` as the build context**, not the whole
checkout:

```bash
podman build --tag localhost/rko-self-evolve:dev \
  --file benchmarks/self_evolve/Containerfile src
uv run python -m rko_solver.experimental.self_evolve qualify
```

The base tag is only a build input. Campaigns pin the resulting local image ID;
archive it separately if exact replay is required. No runtime image pull or host
repository mount is permitted during candidate evaluation. The image uses Python
serial SA, not the optional Rust/C++ backend.

[User workflow and threat model](../../docs/self-evolve.md) describe explicit seed
import, development requests, reviewed candidate ingestion, hard budgets,
experimental export and offline replay. Run all three families sequentially:
knapsack, scheduling and set cover. Save campaign directories, prompts, responses
and generated reports outside Git or under ignored local audit storage.

This directory does not contain a powered performance reference or establish a
quality gain. Tiny exact oracles validate correctness; cheap development trials
are exploratory. Do not manufacture a significance claim from pilot selection.
