"""Orchestration unit test suite.

Objective:
- Validate configuration parsing and metaheuristic factory behavior in isolation.

Technical approach:
- Exercise ``parse_run_config`` with valid and invalid dictionaries.
- Exercise ``build_metaheuristic`` with real parameter files and controlled temporary TOML fixtures.
- Assert both success paths and explicit error contracts.

Flow (pseudocode):
1. Build representative config dictionaries.
2. Parse configuration and validate normalization/validation results.
3. Instantiate metaheuristics through the factory in offline and online modes.
4. Inject malformed offline TOML and verify raised exceptions.

Important criteria:
- Alias normalization must map ``GeneticAlgorithm`` to ``GA``.
- Invalid control values and unknown active names must be rejected.
- Missing required offline parameters must raise ``ValueError``.

Inputs/Outputs:
- Input: in-memory config dictionaries and temporary TOML files.
- Output: unittest assertions over objects and raised exceptions.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from rko_solver.components.solution_pool import SolutionPool
from rko_solver.metaheuristics.ga import GeneticAlgorithm
from rko_solver.metaheuristics.sa import SA
from rko_solver.orchestration.configuration import parse_run_config
from rko_solver.orchestration.factory import build_metaheuristic
from tempfile import TemporaryDirectory


class TestParseRunConfig(unittest.TestCase):
    """Test suite for orchestration run-configuration validation."""

    def test_parse_run_config_normalizes_aliases(self) -> None:
        """Validate alias normalization and typed output for valid config data."""
        cfg: dict[str, object] = {
            "run": {"control": 1, "pool_size": 5},
            "active_metaheuristics": {"list": ["GeneticAlgorithm", "SA"]},
        }

        parsed = parse_run_config(cfg)

        self.assertEqual(parsed.control, 1)
        self.assertEqual(parsed.pool_size, 5)
        self.assertEqual(parsed.active_metaheuristics, ("GA", "SA"))

    def test_parse_run_config_rejects_invalid_control(self) -> None:
        """Ensure unsupported control modes raise a configuration validation error."""
        cfg: dict[str, object] = {
            "run": {"control": 3, "pool_size": 5},
            "active_metaheuristics": {"list": ["SA"]},
        }

        with self.assertRaises(ValueError):
            _ = parse_run_config(cfg)

    def test_parse_run_config_rejects_unknown_metaheuristic(self) -> None:
        """Ensure unknown active metaheuristic names are rejected."""
        cfg: dict[str, object] = {
            "run": {"control": 0, "pool_size": 5},
            "active_metaheuristics": {"list": ["UNKNOWN_MH"]},
        }

        with self.assertRaises(ValueError):
            _ = parse_run_config(cfg)


class TestBuildMetaheuristic(unittest.TestCase):
    """Test suite for metaheuristic factory construction contracts."""

    def setUp(self) -> None:
        """Initialize common decoder/cost callbacks and a shared solution pool."""
        self.pool = SolutionPool(capacity=8)

    def _decoder(self, keys: object) -> object:
        """Return keys unchanged so cost can be evaluated directly in tests."""
        return keys

    def _cost(self, decoded: object) -> float:
        """Return additive objective value for numeric vectors used in tests."""
        if isinstance(decoded, list):
            return float(sum(decoded))
        return 0.0

    def test_build_metaheuristic_builds_sa_offline(self) -> None:
        """Build SA in offline mode and validate type and configured hyperparameters."""
        mh = build_metaheuristic(
            mh_name="SA",
            pool=self.pool,
            decoder=self._decoder,
            cost=self._cost,
            tam_solution=12,
            control=0,
            params_dir=None,
        )

        self.assertIsInstance(mh, SA)
        self.assertEqual(mh.tam_solution, 12)
        self.assertGreater(mh.sa_max, 0)

    def test_build_metaheuristic_maps_genetic_algorithm_alias(self) -> None:
        """Build GA using alias name and validate canonical constructor resolution."""
        mh = build_metaheuristic(
            mh_name="GeneticAlgorithm",
            pool=self.pool,
            decoder=self._decoder,
            cost=self._cost,
            tam_solution=10,
            control=0,
            params_dir=None,
        )

        self.assertIsInstance(mh, GeneticAlgorithm)

    def test_build_metaheuristic_injects_online_grid_when_control_is_one(self) -> None:
        """Ensure online mode forwards both control flag and parameter grid."""
        mh = build_metaheuristic(
            mh_name="GA",
            pool=self.pool,
            decoder=self._decoder,
            cost=self._cost,
            tam_solution=10,
            control=1,
            params_dir=None,
        )

        self.assertIsInstance(mh, GeneticAlgorithm)
        self.assertEqual(mh.control, 1)
        self.assertGreater(len(mh.parameters), 0)

    def test_build_metaheuristic_rejects_unknown_name(self) -> None:
        """Ensure unregistered metaheuristic names raise ``ValueError``."""
        with self.assertRaises(ValueError):
            _ = build_metaheuristic(
                mh_name="NOPE",
                pool=self.pool,
                decoder=self._decoder,
                cost=self._cost,
                tam_solution=10,
                control=0,
                params_dir=None,
            )

    def test_build_metaheuristic_raises_when_offline_param_is_missing(self) -> None:
        """Ensure missing required offline fields produce explicit factory errors."""
        with TemporaryDirectory() as temp_dir:
            params_dir = Path(temp_dir)
            params_dir.joinpath("ga.toml").write_text(
                "[offline]\nprob_cros = 0.8\nprob_mut = 0.02\n\n"
                "[online]\nsize_pop = [30.0]\nprob_cros = [0.8]\nprob_mut = [0.02]\n",
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                _ = build_metaheuristic(
                    mh_name="GA",
                    pool=self.pool,
                    decoder=self._decoder,
                    cost=self._cost,
                    tam_solution=10,
                    control=0,
                    params_dir=params_dir,
                )


if __name__ == "__main__":
    unittest.main()
