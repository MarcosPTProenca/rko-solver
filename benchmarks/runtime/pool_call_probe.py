"""Benchmark-only timing probe for elite-pool calls made by spawned search workers.

Objective:
- Measure how much wall time each worker spends inside elite-pool manager calls, without changing
  the shipped pool, so issue 136 can be judged on process-mode evidence.

Technical approach:
- Wrap `PoolProxy._callmethod` so every manager round trip is timed by method name.
- Each process keeps running totals and writes them to one JSON file per process at most once per
  second; the final partial interval can be lost when a worker is terminated, which is recorded.
- Import this module before the solver so spawned children install the same wrapper.

Flow (pseudocode):
1. At import, record the process start time and patch `PoolProxy._callmethod`.
2. On every call, add its elapsed seconds to the method's total.
3. When a second has passed, rewrite this process's JSON file with totals and alive seconds.

Important criteria:
- The probe adds two clock reads per call; it is an observer, not part of the algorithm.
- Totals include manager-side contention, because the client waits for the server.
- Output directory comes from the environment; without it, nothing is written.

Inputs/Outputs:
- Input: environment variable `RKO_POOL_PROBE_DIR` (directory, must exist).
- Output: `<pid>.json` files with per-method call counts, seconds and process alive seconds.
"""

from __future__ import annotations

import json
import os
import time
from rko_solver.orchestration.processes import PoolProxy

DIRECTORY = os.environ.get("RKO_POOL_PROBE_DIR")
STARTED = time.perf_counter()
TOTALS: dict[str, list[float]] = {}
LAST_WRITE = [0.0]
_ORIGINAL = PoolProxy._callmethod


def _write() -> None:
    """Rewrite this process's totals file with the current counts and alive seconds."""
    if DIRECTORY is None:
        return
    record = {
        "pid": os.getpid(),
        "alive_seconds": time.perf_counter() - STARTED,
        "methods": {name: {"calls": int(c), "seconds": s} for name, (c, s) in TOTALS.items()},
    }
    path = os.path.join(DIRECTORY, f"{os.getpid()}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle)


def _timed(self: PoolProxy, methodname: str, args: tuple = (), kwds: dict | None = None) -> object:
    """Call the manager method and add its wall time to the per-method totals."""
    began = time.perf_counter()
    try:
        return _ORIGINAL(self, methodname, args, kwds or {})
    finally:
        elapsed = time.perf_counter() - began
        bucket = TOTALS.setdefault(methodname, [0, 0.0])
        bucket[0] += 1
        bucket[1] += elapsed
        now = time.perf_counter()
        if now - LAST_WRITE[0] >= 1.0:
            LAST_WRITE[0] = now
            _write()


if DIRECTORY is not None:
    PoolProxy._callmethod = _timed  # type: ignore[method-assign]
