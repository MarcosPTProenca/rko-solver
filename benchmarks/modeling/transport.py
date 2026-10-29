"""Bounded generated-evidence serialization and transport.

Objective:
- Preserve reproducible component measurements and explicit controls.

Technical approach:
- Separate clock-bound measured work from setup and source bookkeeping.

Flow (pseudocode):
1. Prepare the unchanged original inputs and control settings.
2. Execute the original fixed work or capture exact source identity.
3. Return complete observations without claiming an unmeasured speedup.

Important criteria:
- Preserve paired order, warmups, fixed work, cleanup and failure propagation.

Inputs/Outputs:
- Input: original workload and preplanned measurement policy.
- Output: complete source-identified component observations.
"""

from __future__ import annotations

import multiprocessing as mp
import pickle
import random
import time
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from .overhead import ModelOverheadProbe


class ModelTransportProbe:
    """Measure bounded evidence serialization and spawned transport separately.

    Responsibilities:
    - Keep measured work and clock boundaries unchanged.
    - Preserve reproducible ordering, controls and failure reporting.
    """

    def __init__(self, probe: type[ModelOverheadProbe], repeats: int) -> None:
        """Own the original transport trial policy.

        Args:
            probe (type[ModelOverheadProbe]): Original shared probe operations.
            repeats (int): Preplanned paired repetition count.

        """
        self.probe = probe
        self.repeats = repeats

    def _measure(
        self, parent: Connection, dimension: int, rows: int, searches: int
    ) -> dict[str, Any]:
        """Measure one original bounded payload shape.

        Args:
            parent (Connection): Live owned duplex connection.
            dimension (int): Original witness dimension.
            rows (int): Original row count.
            searches (int): Original worker session count.

        Returns:
            dict[str, Any]: Original measured observations or validated settings.

        Raises:
            TimeoutError: If transport does not complete.
        """
        variants = self.probe.messages(dimension, rows, searches)
        encoded = [
            [pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL) for message in messages]
            for messages in variants
        ]
        for messages in variants:
            self.probe.transport_sample(parent, messages, 5)
        count = 100
        observations = []
        for _ in range(self.repeats):
            order = [0, 1]
            self.generator.shuffle(order)
            elapsed = {}
            for index in order:
                elapsed[index] = self.probe.transport_sample(parent, variants[index], count)
            observations.append({"control_seconds": elapsed[0], "candidate_seconds": elapsed[1]})
        pickle_times = self._pickle_times(variants, encoded)
        return {
            "dimension": dimension,
            "rows": rows,
            "worker_searches": searches,
            "logical_rounds_per_trial": count,
            "pairs": observations,
            "summary": self.probe.paired_summary(observations),
            "control_bytes": sum(map(len, encoded[0])),
            "candidate_bytes": sum(map(len, encoded[1])),
            "pickle_per_round": pickle_times,
            "receiver_startup_seconds_excluded": self.startup,
        }

    def _pickle_times(
        self, variants: tuple[list[object], list[object]], encoded: list[list[bytes]]
    ) -> list[dict[str, float]]:
        """Measure encoding and decoding in unchanged fixed-work clock boundaries.

        Args:
            variants (tuple[list[object], list[object]]): Original paired payloads.
            encoded (list[list[bytes]]): Original serialized payloads.

        Returns:
            list[dict[str, float]]: Original measured observations or validated settings.
        """
        pickle_times = []
        for messages, blobs in zip(variants, encoded, strict=True):
            start = time.perf_counter()
            for _ in range(100):
                for message in messages:
                    pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
            encoding = (time.perf_counter() - start) / 100
            start = time.perf_counter()
            for _ in range(100):
                for blob in blobs:
                    pickle.loads(blob)
            decoding = (time.perf_counter() - start) / 100
            pickle_times.append(
                {"encode_seconds_per_round": encoding, "decode_seconds_per_round": decoding}
            )
        return pickle_times

    def run(self) -> list[dict[str, Any]]:
        """Run and reliably reap the original spawned receiver.

        Returns:
            list[dict[str, Any]]: Original measured observations or validated settings.

        Raises:
            TimeoutError: If receiver startup or transport does not complete.
            RuntimeError: If the spawned receiver exits unexpectedly.
        """
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        worker = context.Process(target=self.probe.echo, args=(child,))
        began = time.perf_counter()
        worker.start()
        child.close()
        try:
            if not parent.poll(60) or parent.recv() != "ready":
                raise TimeoutError("Spawned transport receiver did not become ready.")
            self.startup = time.perf_counter() - began
            records = []
            self.generator = random.Random(20261025)
            for dimension, rows, searches in ((8, 4, 1), (128, 64, 1), (128, 64, 4), (128, 64, 12)):
                records.append(self._measure(parent, dimension, rows, searches))
            return records
        finally:
            try:
                parent.send(None)
            except BrokenPipeError, EOFError, OSError:
                pass
            parent.close()
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
            worker.close()
