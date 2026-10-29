"""Source-identified serial fixed-work search diagnostics.

Objective:
- Establish reproducible baseline records before choosing a native optimization.

Technical approach:
- Reuse fixed_work's real search, callback protection and evaluation cap.
- Record source/configuration hashes and explicit environment observations.
- Optionally collect cProfile call counts and exclusive/inclusive function time.

Flow (pseudocode):
1. Validate the requested bounded workload and capture its provenance.
2. Save caller RNG state and execute the existing fixed-work search.
3. Restore RNG state and emit one JSON diagnostic record.

Important criteria:
- Profiled times diagnose costs; they are not acceptance timing evidence.
- Search timing excludes imports, provenance, construction, startup and shutdown.
- CPU observations do not certify quotas or actual native team sizes.
- No whole-solver, memory, quality or C++ parity claim follows.

Inputs/Outputs:
- Input: algorithm, workload, dimension, completed evaluation cap and seed.
- Output: JSON containing provenance, replay fingerprints and optional profile.
"""

import argparse
import cProfile
import hashlib
import json
import numpy as np
import os
import platform
import pstats
import random
import rko_solver
import subprocess
import sys
from benchmarks.runtime.fixed_work import _run
from pathlib import Path
from rko_solver.metaheuristics.base import MetaheuristicBase
from rko_solver.metaheuristics.brkga import BRKGA
from rko_solver.metaheuristics.ga import GeneticAlgorithm
from rko_solver.metaheuristics.grasp import GRASP
from rko_solver.metaheuristics.ipr import IPR
from rko_solver.metaheuristics.multi_start import MultiStart
from rko_solver.metaheuristics.sa import SA


class PerformanceProbe:
    """Collect restricted fixed-work evidence without changing solver semantics.

    Responsibilities:
    - Identify the source, configuration and observed execution environment.
    - Preserve caller RNG state around the existing serial diagnostic runner.
    - Distinguish profiling evidence from unprofiled search timing.

    Attributes:
        algorithms (dict[str, type[MetaheuristicBase]]): Supported diagnostic searches.
    """

    algorithms: dict[str, type[MetaheuristicBase]] = {
        "SA": SA,
        "GA": GeneticAlgorithm,
        "BRKGA": BRKGA,
        "GRASP": GRASP,
        "MultiStart": MultiStart,
        "IPR": IPR,
    }

    @staticmethod
    def provenance() -> dict[str, object]:
        """Identify executed sources and observed environment without claiming isolation.

        Returns:
            dict[str, object]: Source hashes, Git identity and environment observations.

        Raises:
            OSError: If available source/configuration files cannot be read.
        """
        root = Path(__file__).resolve().parents[2]
        package = Path(rko_solver.__file__).resolve().parent
        paths = {
            "rko/" + path.relative_to(package).as_posix(): path
            for path in package.rglob("*")
            if path.suffix in (".py", ".toml") and "tests" not in path.relative_to(package).parts
        }
        paths.update(
            {
                name: root / name
                for name in (
                    "benchmarks/runtime/fixed_work.py",
                    "benchmarks/runtime/performance_profile.py",
                    "pyproject.toml",
                    "uv.lock",
                    ".python-version",
                )
                if (root / name).is_file()
            }
        )
        hashes = {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()
        }
        git: dict[str, str | None] = {}
        for label, arguments in (
            ("revision", ["rev-parse", "HEAD"]),
            ("status", ["status", "--porcelain", "--untracked-files=normal"]),
            ("diff", ["diff", "HEAD", "--binary"]),
        ):
            try:
                completed = subprocess.run(
                    ["git", "-C", str(root), *arguments],
                    capture_output=True,
                    check=True,
                    timeout=10,
                )
            except OSError, subprocess.SubprocessError:
                git[label] = None
            else:
                git[label] = (
                    hashlib.sha256(completed.stdout).hexdigest()
                    if label == "diff"
                    else completed.stdout.decode("utf-8", errors="replace").strip()
                )
        quota = Path("/sys/fs/cgroup/cpu.max")
        return {
            "python": sys.version,
            "executable": sys.executable,
            "numpy": np.__version__,
            "numpy_source": str(Path(np.__file__).resolve()),
            "rko_source": str(package),
            "platform": platform.platform(),
            "processor": platform.processor() or None,
            "logical_cpus": os.cpu_count(),
            "process_cpus": os.process_cpu_count(),
            "affinity": (
                sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
            ),
            "cgroup_v2_cpu_max": quota.read_text().strip() if quota.is_file() else None,
            "native_thread_environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "git": git,
            "files_sha256": hashes,
            "source_sha256": hashlib.sha256(
                json.dumps(hashes, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }

    @classmethod
    def run(
        cls,
        algorithm: str = "SA",
        workload: str = "sphere",
        dimension: int = 20,
        evaluations: int = 64,
        seed: int = 200,
        *,
        profile: bool = False,
    ) -> dict[str, object]:
        """Execute one real fixed-work diagnostic and preserve caller random streams.

        Args:
            algorithm (str): Supported algorithm name; IPR receives ten seeded elites.
            workload (str): Sphere, rugged or stable ordering objective.
            dimension (int): Positive chromosome length.
            evaluations (int): Positive number of completed search objectives.
            seed (int): Nonnegative NumPy-compatible seed, including IPR's offset.
            profile (bool): Whether to collect function-level diagnostic timings.

        Returns:
            dict[str, object]: Versioned record with provenance, result and profile.

        Raises:
            ValueError: If the workload or integer inputs are unsupported.
            AssertionError: If the existing fixed-work runner violates its contracts.
            OSError: If source provenance cannot be read.
        """
        if algorithm not in cls.algorithms or workload not in ("sphere", "rugged", "ordering"):
            raise ValueError("Unsupported diagnostic algorithm or workload.")
        if type(profile) is not bool:
            raise ValueError("profile must be a boolean.")
        if any(type(value) is not int or value < 1 for value in (dimension, evaluations)):
            raise ValueError("dimension and evaluations must be positive integers.")
        maximum_seed = 2**32 - 1 - (1000 if algorithm == "IPR" else 0)
        if type(seed) is not int or not 0 <= seed <= maximum_seed:
            raise ValueError(f"seed must be an integer between 0 and {maximum_seed}.")
        provenance = cls.provenance()
        saved_python, saved_numpy = random.getstate(), np.random.get_state()
        profiler = cProfile.Profile() if profile else None
        try:
            arguments = (cls.algorithms[algorithm], workload, seed)
            keywords = {"dimension": dimension, "evaluation_limit": evaluations}
            result = (
                _run(*arguments, **keywords)
                if profiler is None
                else profiler.runcall(_run, *arguments, **keywords)
            )
        finally:
            random.setstate(saved_python)
            np.random.set_state(saved_numpy)
        functions = []
        total_seconds = None
        if profiler is not None:
            statistics = pstats.Stats(profiler)
            total_seconds = statistics.total_tt
            for (filename, line, name), values in statistics.stats.items():
                primitive, calls, exclusive, cumulative, _ = values
                functions.append(
                    {
                        "file": filename,
                        "line": line,
                        "function": name,
                        "primitive_calls": primitive,
                        "calls": calls,
                        "exclusive_seconds": exclusive,
                        "inclusive_seconds": cumulative,
                    }
                )
            functions.sort(key=lambda row: row["exclusive_seconds"], reverse=True)
        return {
            "schema_version": 1,
            "scope": "serial_fixed_work_search_only",
            "profiled": profile,
            "profile_total_seconds": total_seconds,
            "profile_scope": (
                "fixed_work_runner_including_construction_and_seed_pool" if profile else None
            ),
            "seed_pool_evaluations": 10 if algorithm == "IPR" else 0,
            "provenance": provenance,
            "result": result,
            "profile": functions[:20],
        }


def main() -> None:
    """Validate CLI inputs and emit one source-identified diagnostic JSON record.

    Raises:
        SystemExit: If arguments or workload bounds are invalid.
        OSError: If provenance cannot be read or output cannot be written.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=PerformanceProbe.algorithms, default="SA")
    parser.add_argument("--workload", choices=("sphere", "rugged", "ordering"), default="sphere")
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--evaluations", type=int, default=64)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--profile", action="store_true")
    arguments = vars(parser.parse_args())
    try:
        result = PerformanceProbe.run(**arguments)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
