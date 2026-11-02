"""Measure how decoded decisions respond to the library's own shaking intensity.

Objective:
- Test whether the Q-RKO proposal's "plateau-then-chaos" finding (small key perturbations change
  almost no decoded decisions, larger ones change nearly all of them, with an abrupt transition
  rather than a gradual one) describes the decoders already shipped in this repository, at the
  shaking intensities RKO's own metaheuristics use by default.

Technical approach:
- The proposal measured its own untrained neural quantile codec, never implemented as a library
  feature. This instead measures `rko_solver.components.shaking.random.shake`, the real
  perturbation operator every offline and Q-learning-controlled `SA`/`ILS`/`VNS`/`LNS` run calls,
  against the sort decoders this repository already ships: the plain permutation sort (TSP,
  `TspDecoder` with no prior) and the plain greedy fill (packing, `PackingDecoder` with no prior),
  plus each decoder's
  residual-decoder form with the hand-written prior from the decoder-form study (PR 59/60).
- For each instance, draw one random key vector, decode it, then call `shake` at a grid of
  intensities spanning and extending the `beta_min`/`beta_max` defaults already hardcoded in
  `sa.py` (0.05-0.25), `ils.py` (0.10-0.40), `vns.py` and `lns.py` (0.10-0.30). At each intensity,
  decode the shaken keys and record the fraction of decoded positions that changed: the selected
  set for packing, the visited order for TSP.
- Repeat with independent key draws and shake seeds (`RandomStreams`) to report a mean and a 95
  percent bootstrap interval per intensity, not a single noisy point.
- A single-call operator breakdown (`OPERATORS`) isolates which of `shake`'s three named-function
  moves drives the aggregate curve: `mirror` reflects one key across `0.5`, which can jump that
  item from one end of a full-vector sort order to the other; `swap` and `neighbor_swap` exchange
  two positions and move at most those two items. `shake`'s fourth move, an inline single-index
  reassignment with no named function, is not isolated here. A greedy-fill decoder (packing) only
  changes the selected set when a moved item crosses the capacity margin; a full permutation-sort
  decoder (TSP) changes the relative order of every item strictly between the old and new position
  of the reflected key.

Flow (pseudocode):
1. For each decoder (plain and residual, each problem), for each test instance:
2. draw one key vector, decode it;
3. for each intensity in the grid, for each repeat: shake the same keys at that intensity with a
   fresh seed, decode the result, and record the changed-position fraction;
4. aggregate mean and bootstrap interval per decoder and intensity;
5. separately, for each of the three named-function shaking operators in isolation, apply one call
   and record the same changed fraction, to attribute the aggregate curve to a mechanism.

Important criteria:
- Decision fraction is 0.0 at intensity 0 and at most 1.0; the grid brackets every shipped
  `beta_min`/`beta_max` default so the plot answers the risk the proposal raises, not an arbitrary
  range.
- This is a measurement, not a kill-or-continue comparison: there is no pre-registered rule and no
  `advance` verdict, because no design decision follows directly from either finding. A gradual
  profile is reassuring; an abrupt one documents a known risk for later shaking-operator tuning.
- Every decoded output is independently re-decoded from the exact perturbed keys written to the
  report, so the measurement is reproducible from the report alone.

Inputs/Outputs:
- Input: design settings (instance counts, sizes, intensity grid, repeats) and an output path.
- Output: a JSON file with the intensity grid, per-decoder change-fraction tables with bootstrap
  intervals, and the raw per-trial records.
"""

from __future__ import annotations

import argparse
import json
import numpy as np
from benchmarks.engines.packing_problems import PackingDecoder
from benchmarks.engines.packing_problems import PackingFamily
from benchmarks.engines.tsp_problems import TspDecoder
from benchmarks.engines.tsp_problems import TspFamily
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from rko_solver.components.random_streams import RandomStreams
from rko_solver.components.shaking.mirror import mirror_shake
from rko_solver.components.shaking.neighbor_swap import neighbor_swap_shake
from rko_solver.components.shaking.random import shake
from rko_solver.components.shaking.swap import swap_shake
from typing import Any

PACKING_KIND = "mkp"
TSP_KIND = "uniform"
INTENSITY_GRID: tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0)
DECODERS = ("packing_plain", "packing_resid", "tsp_plain", "tsp_resid")
OPERATORS: tuple[tuple[str, Callable[[np.ndarray], np.ndarray]], ...] = (
    ("mirror", mirror_shake),
    ("swap", swap_shake),
    ("neighbor_swap", neighbor_swap_shake),
)


@dataclass(slots=True)
class SensitivityDesign:
    """Fixed numerical design of the sensitivity measurement.

    Responsibilities:
    - Keep instance size, test instance count, the intensity grid and the repeat count together.

    Attributes:
        n_items (int): Items (packing) or cities (TSP) per instance.
        test_instances (int): Instances measured per decoder.
        test_base (int): First test seed.
        repeats (int): Independent shake draws per instance and intensity.
        resamples (int): Bootstrap resamples per intensity.
        resid_spread (float): Residual-decoder spread, held fixed rather than tuned.
    """

    n_items: int = 200
    test_instances: int = 20
    test_base: int = 9000
    repeats: int = 8
    operator_repeats: int = 20
    resamples: int = 2000
    resid_spread: float = 0.1


class ShakeSensitivityStudy:
    """Measure the decoded change fraction of `shake` across intensities, per decoder.

    Responsibilities:
    - Build the four decoders, run every test instance through the intensity grid and summarize.

    Attributes:
        design (SensitivityDesign): Fixed numerical design.
    """

    def __init__(self, design: SensitivityDesign) -> None:
        """Store the design.

        Args:
            design (SensitivityDesign): Fixed numerical design.
        """
        self.design = design

    def decoders(self, seed: int) -> dict[str, tuple[Callable[[np.ndarray], Any], int]]:
        """Build the four named decoders and the key-vector length each one takes.

        Args:
            seed (int): Instance seed.

        Returns:
            dict[str, tuple[Callable[[np.ndarray], Any], int]]: Decoder and key count per name.
        """
        packing = PackingFamily.make(PACKING_KIND, self.design.n_items, seed)
        tsp = TspFamily.make(TSP_KIND, self.design.n_items, seed)
        tour = TspFamily.two_opt(tsp, TspFamily.nearest_neighbour(tsp))
        return {
            "packing_plain": (PackingDecoder(packing), packing.n_items),
            "packing_resid": (
                PackingDecoder(
                    packing, PackingFamily.hand_scores(packing), self.design.resid_spread
                ),
                packing.n_items,
            ),
            "tsp_plain": (TspDecoder(tsp), tsp.n_cities),
            "tsp_resid": (TspDecoder(tsp, tour, self.design.resid_spread), tsp.n_cities),
        }

    @staticmethod
    def changed_fraction(before: np.ndarray, after: np.ndarray) -> float:
        """Fraction of positions that differ between two decoded outputs of equal length.

        Args:
            before (np.ndarray): Decoded output before shaking.
            after (np.ndarray): Decoded output after shaking, same shape as `before`.

        Returns:
            float: Fraction of differing positions in `[0, 1]`.
        """
        return float(np.mean(np.asarray(before) != np.asarray(after)))

    def trial(
        self, name: str, decoder: Callable[[np.ndarray], Any], n_keys: int, seed: int
    ) -> list[dict[str, Any]]:
        """Measure the change fraction at every intensity, repeated, for one instance.

        Args:
            name (str): Decoder name, recorded in each row.
            decoder (Callable[[np.ndarray], Any]): Decoder under test.
            n_keys (int): Key-vector length.
            seed (int): Instance seed, recorded in each row.

        Returns:
            list[dict[str, Any]]: One row per intensity and repeat.
        """
        rows: list[dict[str, Any]] = []
        keys = np.random.default_rng(seed).random(n_keys)
        before = decoder(keys)
        for level, intensity in enumerate(INTENSITY_GRID):
            for repeat in range(self.design.repeats):
                shake_seed = int(np.random.SeedSequence([seed, level, repeat]).generate_state(1)[0])
                with RandomStreams(shake_seed).activate():
                    shaken = shake(keys.copy(), intensity, intensity)
                after = decoder(shaken)
                rows.append(
                    {
                        "decoder": name,
                        "instance": seed,
                        "intensity": intensity,
                        "repeat": repeat,
                        "changed_fraction": self.changed_fraction(before, after),
                    }
                )
        return rows

    def operator_trial(
        self, name: str, decoder: Callable[[np.ndarray], Any], n_keys: int, seed: int
    ) -> list[dict[str, Any]]:
        """Apply each shaking operator once in isolation, repeated, for one instance.

        Args:
            name (str): Decoder name, recorded in each row.
            decoder (Callable[[np.ndarray], Any]): Decoder under test.
            n_keys (int): Key-vector length.
            seed (int): Instance seed, recorded in each row.

        Returns:
            list[dict[str, Any]]: One row per operator and repeat.
        """
        rows: list[dict[str, Any]] = []
        keys = np.random.default_rng(seed).random(n_keys)
        before = decoder(keys)
        for op_index, (op_name, operator) in enumerate(OPERATORS):
            for repeat in range(self.design.operator_repeats):
                op_seed = int(
                    np.random.SeedSequence([seed, op_index, repeat, 1]).generate_state(1)[0]
                )
                with RandomStreams(op_seed).activate():
                    after_keys = operator(keys.copy())
                after = decoder(after_keys)
                rows.append(
                    {
                        "decoder": name,
                        "instance": seed,
                        "operator": op_name,
                        "repeat": repeat,
                        "changed_fraction": self.changed_fraction(before, after),
                    }
                )
        return rows

    def run(self) -> dict[str, Any]:
        """Measure every decoder on every test instance and summarize per intensity and operator.

        Returns:
            dict[str, Any]: Intensity grid, raw trial rows, operator-breakdown rows and summaries.
        """
        rows: list[dict[str, Any]] = []
        operator_rows: list[dict[str, Any]] = []
        for offset in range(self.design.test_instances):
            seed = self.design.test_base + offset
            built = self.decoders(seed)
            for name in DECODERS:
                decoder, n_keys = built[name]
                rows.extend(self.trial(name, decoder, n_keys, seed))
                operator_rows.extend(self.operator_trial(name, decoder, n_keys, seed))
        return {
            "intensities": INTENSITY_GRID,
            "operators": [name for name, _ in OPERATORS],
            "trials": rows,
            "operator_trials": operator_rows,
            "summary": self.summarize(rows, "intensity", INTENSITY_GRID),
            "operator_summary": self.summarize(
                operator_rows, "operator", tuple(name for name, _ in OPERATORS)
            ),
        }

    def summarize(
        self, rows: list[dict[str, Any]], level_field: str, levels: tuple[Any, ...]
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Compute the mean and a bootstrap interval of the changed fraction per decoder/level.

        Args:
            rows (list[dict[str, Any]]): Raw trial rows, each tagged by `decoder` and
                `level_field`.
            level_field (str): Row key that selects one level, e.g. `intensity` or `operator`.
            levels (tuple[Any, ...]): Every level to summarize, in the table's column order.

        Returns:
            dict[str, dict[str, dict[str, float]]]: `summary[decoder][str(level)]` with `mean`,
            `ci_low` and `ci_high` of the changed fraction.
        """
        summary: dict[str, dict[str, dict[str, float]]] = {}
        rng = np.random.default_rng(0)
        for name in DECODERS:
            summary[name] = {}
            for level in levels:
                values = np.array(
                    [
                        row["changed_fraction"]
                        for row in rows
                        if row["decoder"] == name and row[level_field] == level
                    ]
                )
                picks = rng.integers(0, len(values), (self.design.resamples, len(values)))
                means = values[picks].mean(axis=1)
                summary[name][str(level)] = {
                    "mean": float(values.mean()),
                    "ci_low": float(np.percentile(means, 2.5)),
                    "ci_high": float(np.percentile(means, 97.5)),
                }
        return summary


def main(argv: list[str] | None = None) -> int:
    """Run the shake-sensitivity study and write its JSON report.

    Args:
        argv (list[str] | None): Arguments, or None for `sys.argv`.

    Returns:
        int: Process exit code.
    """
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--n-items", type=int, default=200)
    parser.add_argument("--test-instances", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/shake-sensitivity-study.json"))
    options = parser.parse_args(argv)
    design = SensitivityDesign(
        n_items=options.n_items, test_instances=options.test_instances, repeats=options.repeats
    )
    report = {"design": asdict(design)} | ShakeSensitivityStudy(design).run()
    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
