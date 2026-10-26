"""Conservative runtime setup and cleanup lifecycle accounting.

Objective:
- Durably reserve startup before constructing a runtime and retain unknown setup charges.
Technical approach:
- Track one setup token, conservative reservation and measured setup/cleanup consumption.
- Commit pending, ready and failed state transitions under the study controller lock.
Flow (pseudocode):
1. Reject prior unknown work and reserve setup plus another bounded job.
2. Replace the pending marker only after bounded successful qualification.
3. Charge measured cleanup or retain the full reservation and an integrity stop.
Important criteria:
- Cleanup overruns are charged, never clipped, and prohibit later collection.
- Interrupted setup and failed cleanup never release durable authorization silently.
Inputs/Outputs:
- Input: immutable study state and one runtime setup token.
- Output: durable lifecycle markers and conservative setup resource charges.
"""

from __future__ import annotations

import contextlib
import sqlite3
import time
from .sandbox import JOB_GRACE_SECONDS
from .study_state import StudyState
from dataclasses import dataclass


@dataclass
class StudyRuntimeSession:
    """Account for one reserved runtime setup and its cleanup transition.

    Responsibilities:
    - Reserve startup before any external runtime construction.
    - Record successful qualification and retain unknown or failed lifecycle stops.

    Attributes:
        study (StudyState): Owning controller and frozen budgets.
        token (str): Unique identity of this setup.
        reservation (float): Conservative full setup reservation.
        cleanup_reserve (float): Reserved cleanup allowance.
        setup (float): Previously accumulated setup charge.
        elapsed (float): Measured setup time, or full reservation while unknown.
        ready (bool): Whether qualification and authorization completed.
    """

    study: StudyState
    token: str
    reservation: float = 900.0
    cleanup_reserve: float = 60.0
    setup: float = 0.0
    elapsed: float = 900.0
    ready: bool = False

    def reserve(self) -> None:
        """Admit and commit bounded startup before runtime construction.

        Raises:
            ValueError: If prior state, pending work or budgets cannot admit setup and a job.
        """
        with (
            self.study._lock(),
            contextlib.closing(sqlite3.connect(self.study.path / "measurements.sqlite")) as db,
        ):
            self.study._admit_collection(db)
            if not db.execute("SELECT 1 FROM jobs WHERE status='pending' LIMIT 1").fetchone():
                raise ValueError("No pending jobs require runtime setup.")
            row = db.execute("SELECT value FROM state WHERE key='setup_charge'").fetchone()
            self.setup = float(row[0]) if row else 0.0
            charged = (
                db.execute("SELECT COALESCE(SUM(charge),0) FROM jobs").fetchone()[0] + self.setup
            )
            started = db.execute("SELECT value FROM state WHERE key='started'").fetchone()
            elapsed = time.time() - float(started[0]) if started else 0
            required = self.reservation + self.study.spec["seconds"] + JOB_GRACE_SECONDS
            if (
                elapsed < 0
                or charged + required > self.study.spec["cpu_budget"]
                or elapsed + required >= self.study.spec["elapsed_budget"]
            ):
                raise ValueError("Insufficient budget for runtime setup and another job.")
            db.execute(
                "INSERT OR REPLACE INTO state VALUES ('setup_charge',?)",
                (str(self.setup + self.reservation),),
            )
            db.execute("INSERT INTO state VALUES ('setup_pending',?)", (self.token,))
            db.execute("INSERT OR IGNORE INTO state VALUES ('started',?)", (str(time.time()),))
            db.commit()

    def qualified(self, elapsed: float) -> None:
        """Commit the ready marker after bounded successful runtime qualification.

        Args:
            elapsed (float): Measured setup wall time including the conservative margin.

        Raises:
            ValueError: If qualification exhausted the setup and cleanup reservation.
        """
        self.elapsed = elapsed
        if self.elapsed + self.cleanup_reserve > self.reservation:
            raise ValueError("Runtime setup exhausted its conservative reservation.")
        with (
            self.study._lock(),
            contextlib.closing(sqlite3.connect(self.study.path / "measurements.sqlite")) as db,
        ):
            db.execute(
                "UPDATE state SET value=? WHERE key='setup_charge'",
                (str(self.setup + self.elapsed + self.cleanup_reserve),),
            )
            db.execute("DELETE FROM state WHERE key='setup_pending'")
            db.execute("INSERT INTO state VALUES ('setup_ready',?)", (self.token,))
            db.commit()

    def finish(self, cleaned: bool, closing: float) -> None:
        """Record cleanup consumption or a durable unknown/failed lifecycle stop.

        Args:
            cleaned (bool): Whether runtime context cleanup returned successfully.
            closing (float): Monotonic timestamp immediately before cleanup began.
        """
        with (
            self.study._lock(),
            contextlib.closing(sqlite3.connect(self.study.path / "measurements.sqlite")) as db,
        ):
            if self.ready and cleaned:
                cleanup = time.monotonic() - closing + 0.1
                if cleanup > self.cleanup_reserve:
                    db.execute(
                        "INSERT OR REPLACE INTO state VALUES ('setup_failed',?)", (self.token,)
                    )
                db.execute(
                    "UPDATE state SET value=? WHERE key='setup_charge'",
                    (str(self.setup + self.elapsed + cleanup),),
                )
                db.execute("DELETE FROM state WHERE key='setup_ready'")
            else:
                if not self.ready:
                    db.execute(
                        "UPDATE state SET value=? WHERE key='setup_charge'",
                        (str(self.setup + self.reservation),),
                    )
                db.execute("INSERT OR REPLACE INTO state VALUES ('setup_failed',?)", (self.token,))
            db.commit()
