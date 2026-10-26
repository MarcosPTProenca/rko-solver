"""Compare owned RNG searches with a trusted pre-isolation source package.

Objective:
- Check complete fixed-work trajectories and measure RNG-routing overhead.

Technical approach:
- Import a trusted reference package under a separate module namespace.
- Alternate original global RNG searches and production owned-stream searches.

Flow (pseudocode):
1. Load the reference package, identify comparable contracts/controls and warm them.
2. Run numerical and ordering objectives with fixed evaluation counts.
3. Require identical results, elite pools and both final RNG states.

Important criteria:
- The reference package executes Python code and must be trusted.
- Intentional contract or unsupported-control changes are not RNG-only comparisons.
- Adaptive clocks are frozen; real timed cooperation is not being compared.
- Timings exclude imports, stream construction and process startup.

Inputs/Outputs:
- Input: --reference pointing to a trusted pre-isolation src/rko_solver directory.
- Output: JSON replay checks/ratios and explicit not_comparable mode records.
"""

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
import numpy as np
import pickle
import random
import statistics
import sys
from contextlib import nullcontext
from fixed_work import _run
from itertools import product
from pathlib import Path
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.solution_pool import ElitePool
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.orchestration.configuration import MH_SPECS
from rko_solver.orchestration.types import CostFn
from rko_solver.orchestration.types import DecoderFn


def replay_modes(current: type, reference: type) -> tuple[int, ...]:
    """Identify controls whose search contract can support an RNG-only comparison.

    Args:
        current (type): Current registered algorithm constructor.
        reference (type): Trusted historical constructor for the same algorithm.

    Returns:
        tuple[int, ...]: Shared controls, or no modes after an intentional lattice change.

    Raises:
        ValueError: If a trusted constructor has no inspectable signature.
    """
    if getattr(current, "_replay_contract", None) != getattr(reference, "_replay_contract", None):
        return ()
    if all(
        "control" in inspect.signature(algorithm).parameters for algorithm in (current, reference)
    ):
        return (0, 1)
    return (0,)


def replay_type(algorithm: type, control: int) -> type:
    """Wrap either implementation with the same fixed adaptive clock.

    Args:
        algorithm (type): Reference or production algorithm class.
        control (int): Offline or adaptive setting when supported.

    Returns:
        type: Compatible search constructor with a frozen policy clock.

    Raises:
        ValueError: If adaptive control is requested from an offline-only baseline.
    """
    supports_control = "control" in inspect.signature(algorithm).parameters
    if control == 1 and not supports_control:
        raise ValueError("Reference constructor does not implement adaptive control.")

    class Replay(algorithm):
        """Use real search operations with deterministic policy time.

        Responsibilities:
        - Select the requested control mode without changing numeric kernels.
        """

        def __init__(
            self, pool: ElitePool, decoder: DecoderFn, cost: CostFn, dimension: int
        ) -> None:
            """Forward constructor arguments and select adaptive control.

            Args:
                pool (ElitePool): Elite pool shared with the fixed-work runner.
                decoder (DecoderFn): Numerical or permutation decoder.
                cost (CostFn): Fixed-work objective.
                dimension (int): Number of keys.
            """
            settings = {"control": control} if supports_control else {}
            super().__init__(pool, decoder, cost, dimension, **settings)

        def elapsed(self) -> float:
            """Freeze policy time while retaining callback-boundary stopping.

            Returns:
                float: Constant adaptive policy timestamp.
            """
            return 0.0

    return Replay


def seeded_pool(workload: str, seed: int, dimension: int) -> SolutionPool:
    """Prepare the same IPR elites regardless of reference class identity.

    Args:
        workload (str): Sphere, rugged or ordering objective.
        seed (int): Search seed whose independent elite generator is offset.
        dimension (int): Number of keys.

    Returns:
        SolutionPool: Ten workload-consistent initial elites.
    """
    pool = SolutionPool()
    generator = np.random.RandomState(seed + 1000)
    for _ in range(10):
        keys = generator.random_sample(dimension)
        if workload == "ordering":
            cost = float(np.dot(np.argsort(keys, kind="stable"), np.arange(1, dimension + 1)))
        elif workload == "rugged":
            values = 10 * keys - 5
            cost = float(10 * dimension + np.sum(values**2 - 10 * np.cos(2 * np.pi * values)))
        else:
            cost = float(np.dot(keys, keys))
        pool.insert((cost, keys.tolist()), "seed")
    return pool


def main() -> None:
    """Replay all registered searches against an explicit trusted baseline.

    Raises:
        ValueError: If the reference package cannot be loaded.
        AssertionError: If any search or RNG fingerprint differs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "_rko_rng_reference",
        args.reference / "__init__.py",
        submodule_search_locations=[str(args.reference)],
    )
    if spec is None or spec.loader is None:
        raise ValueError("Reference must identify a trusted src/rko_solver package.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    saved = random.getstate(), np.random.get_state()
    try:
        for name, settings in MH_SPECS.items():
            current: type[MetaheuristicBase] = settings.constructor
            reference = getattr(
                importlib.import_module(
                    f"{spec.name}.metaheuristics.{current.__module__.rsplit('.', 1)[1]}"
                ),
                current.__name__,
            )
            modes = replay_modes(current, reference)
            for control in (0, 1) if settings.supports_control else (0,):
                if control not in modes:
                    print(
                        json.dumps(
                            {
                                "algorithm": name,
                                "control": control,
                                "status": "not_comparable",
                                "reason": "reference search contract or adaptive control differs",
                            }
                        ),
                        flush=True,
                    )
                    continue
                classes = {
                    "global": replay_type(reference, control),
                    "owned": replay_type(current, control),
                }
                for workload in ("sphere", "rugged", "ordering"):
                    for mode in ("global", "owned"):
                        streams = RandomStreams(199) if mode == "owned" else None
                        with streams.activate() if streams is not None else nullcontext():
                            _run(
                                classes[mode],
                                workload,
                                199,
                                dimension=64,
                                evaluation_limit=100,
                                pool=seeded_pool(workload, 199, 64)
                                if name == "IPR" and mode == "global"
                                else None,
                            )
                    ratios = []
                    cpu_ratios = []
                    timings = []
                    for repeat, seed in product(range(2), (200, 201, 202)):
                        results = {}
                        for mode in (
                            ("global", "owned") if (repeat + seed) % 2 == 0 else ("owned", "global")
                        ):
                            streams = RandomStreams(seed) if mode == "owned" else None
                            with streams.activate() if streams is not None else nullcontext():
                                result = _run(
                                    classes[mode],
                                    workload,
                                    seed,
                                    dimension=64,
                                    evaluation_limit=1000,
                                    pool=seeded_pool(workload, seed, 64)
                                    if name == "IPR" and mode == "global"
                                    else None,
                                )
                            if streams is not None:
                                result["rng_sha256"] = hashlib.sha256(
                                    pickle.dumps(
                                        (streams.python.getstate(), streams.numpy.get_state())
                                    )
                                ).hexdigest()
                            results[mode] = result
                        assert all(
                            results["global"][key] == results["owned"][key]
                            for key in results["global"]
                            if key not in ("seconds", "cpu_seconds")
                        ), (name, control, workload, seed, results)
                        timings.append(
                            {
                                "seed": seed,
                                "repeat": repeat,
                                "global_wall": results["global"]["seconds"],
                                "owned_wall": results["owned"]["seconds"],
                                "global_cpu": results["global"]["cpu_seconds"],
                                "owned_cpu": results["owned"]["cpu_seconds"],
                            }
                        )
                        ratios.append(results["global"]["seconds"] / results["owned"]["seconds"])
                        cpu_ratios.append(
                            results["global"]["cpu_seconds"] / results["owned"]["cpu_seconds"]
                        )
                    print(
                        json.dumps(
                            {
                                "algorithm": name,
                                "control": control,
                                "workload": workload,
                                "pairs": len(ratios),
                                "timings": timings,
                                "wall_ratio": statistics.median(ratios),
                                "cpu_ratio": statistics.median(cpu_ratios),
                            }
                        ),
                        flush=True,
                    )
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])


if __name__ == "__main__":
    main()
