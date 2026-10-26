"""Bounded original-evaluation decision recording and identity-preserving numeric references.

Objective:
- Preserve explicitly emitted decoder decisions without replaying callbacks or caching payloads.

Technical approach:
- Use a context-local bounded recorder activated only around the original decoder.
- Carry a lightweight observation reference on an internal float subclass through existing caches.
- Evict event payloads from bounded session storage; detach only retained pool/result evidence.
- Verify a selected reference's binding, never infer identity from equal keys/cost.

Flow (pseudocode):
1. Allocate a bounded observation cell before an opted-in decoder invocation.
2. Capture authored text events while that invocation is active.
3. Seal the cell only after its original finite objective completes.
4. Carry its reference with the cost through unchanged numerical comparisons.
5. Snapshot selected evidence or explicitly report lost identity/retention/binding.

Important criteria:
- Disabled recording allocates no event store; explicit emit calls are otherwise no-ops.
- Numeric arithmetic/coercion may discard a reference and must not trigger guessed recovery.
- No decoded payload, input value, source, filesystem path or application RNG is read.
- Retained event storage is bounded per session plus existing elite/result occupancy.

Inputs/Outputs:
- Input: opt-in configuration, authored text and genuine callback observations.
- Output: immutable bounded decision evidence or an explicit unavailable status.
"""

import hashlib
import json
import math
import secrets
import sys
from array import array
from collections import deque
from collections.abc import Callable
from collections.abc import Sequence
from contextvars import ContextVar
from dataclasses import asdict
from dataclasses import dataclass
from typing import Literal
from typing import Self


@dataclass(frozen=True, slots=True)
class TraceConfig:
    """Limit explicitly enabled decision recording without a permanent candidate archive.

    Responsibilities:
    - Bound event count, encoded content and live session observation payloads.

    Attributes:
        max_events (int): Maximum events retained for one observation.
        max_bytes (int): Maximum UTF-8 JSON bytes of that observation's event array.
        max_records (int): Maximum live session payload records, including the pinned incumbent.
    """

    max_events: int = 128
    max_bytes: int = 65_536
    max_records: int = 8

    def __post_init__(self) -> None:
        """Reject unsafe or nonintegral tracing caps before any callback executes.

        Raises:
            ValueError: If any limit is outside the bounded recording envelope.
        """
        for value, lower, upper in (
            (self.max_events, 1, 1024),
            (self.max_bytes, 64, 1_048_576),
            (self.max_records, 2, 64),
        ):
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(
                    "Trace limits must be bounded positive integers; records require at least two."
                )


@dataclass(frozen=True, slots=True)
class DecisionEvent:
    """Retain one explicit authored decoder decision, not inferred program semantics.

    Responsibilities:
    - Keep bounded plain text without arbitrary objects, links or private-value capture.

    Attributes:
        action (str): Nonblank decision/step label, at most 256 characters.
        reason (str): Optional authored explanation, at most 1024 characters.
    """

    action: str
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate primitive bounded text without evaluating user objects.

        Raises:
            ValueError: If labels are not bounded plain strings or cannot encode as UTF-8.
        """
        if (
            type(self.action) is not str
            or not self.action.strip()
            or len(self.action) > 256
            or type(self.reason) is not str
            or len(self.reason) > 1024
        ):
            raise ValueError("Decision labels must be bounded plain text.")
        try:
            self.action.encode("utf-8")
            self.reason.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("Decision labels must be valid UTF-8 text.") from exc


@dataclass(frozen=True, slots=True)
class DecisionTrace:
    """Describe a selected original observation or explicitly unavailable evidence.

    Responsibilities:
    - Bind identity to its original finite keys/cost without disclosing a decoded payload.
    - Distinguish captured, empty, truncated and unavailable observations.

    Attributes:
        state (str): recorded, no_events, truncated or unavailable.
        evaluation_id (str | None): Opaque invocation identity, not a coordinate/cost lookup key.
        source (str): Producing search or initial_pool namespace.
        keys_digest (str | None): Canonical key-binding checksum, not a provenance authenticator.
        cost (float | None): Original finite objective.
        events (tuple[DecisionEvent, ...]): Immutable authored events in emission order.
        dropped_events (int): Events omitted by configured limits.
        reason (str | None): Safe unavailable category.
        max_events (int): Requested observation count cap.
        max_bytes (int): Requested encoded event-array cap.
        provenance (str): original_evaluation, never reconstructed execution.
    """

    state: Literal["recorded", "no_events", "truncated", "unavailable"]
    evaluation_id: str | None = None
    source: str = ""
    keys_digest: str | None = None
    cost: float | None = None
    events: tuple[DecisionEvent, ...] = ()
    dropped_events: int = 0
    reason: str | None = None
    max_events: int = 128
    max_bytes: int = 65_536
    provenance: Literal["original_evaluation"] = "original_evaluation"

    def __post_init__(self) -> None:
        """Validate immutable primitive evidence and bounded public construction.

        Raises:
            ValueError: If evidence, binding, status or limits are malformed/nonfinite.
        """
        TraceConfig(self.max_events, self.max_bytes)
        if self.state not in ("recorded", "no_events", "truncated", "unavailable"):
            raise ValueError("Invalid decision trace state.")
        if self.provenance != "original_evaluation":
            raise ValueError("Decision traces cannot claim reconstructed execution.")
        if type(self.events) is not tuple or any(
            type(event) is not DecisionEvent for event in self.events
        ):
            raise ValueError("Trace events must be an immutable tuple of DecisionEvent records.")
        if (
            len(self.events) > self.max_events
            or len(
                json.dumps(
                    [asdict(event) for event in self.events],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            )
            > self.max_bytes
        ):
            raise ValueError("Decision evidence exceeds its declared limits.")
        if type(self.dropped_events) is not int or self.dropped_events < 0:
            raise ValueError("Dropped-event counts must be nonnegative integers.")
        if type(self.source) is not str or len(self.source) > 128:
            raise ValueError("Trace source must be bounded text.")
        if self.reason not in (None, "identity_unavailable", "retention_limit", "binding_mismatch"):
            raise ValueError("Invalid unavailable trace category.")
        if self.evaluation_id is not None and (
            type(self.evaluation_id) is not str or not 1 <= len(self.evaluation_id) <= 128
        ):
            raise ValueError("Invalid evaluation identity.")
        if self.keys_digest is not None and (
            type(self.keys_digest) is not str
            or len(self.keys_digest) != 64
            or any(value not in "0123456789abcdef" for value in self.keys_digest)
        ):
            raise ValueError("Invalid trace binding checksum.")
        if self.cost is not None and (
            type(self.cost) not in (int, float) or not math.isfinite(self.cost)
        ):
            raise ValueError("Trace cost must be finite.")
        if self.state != "unavailable" and (
            self.evaluation_id is None
            or self.keys_digest is None
            or self.cost is None
            or self.reason is not None
        ):
            raise ValueError("Observed evidence requires an original invocation binding.")
        if (
            (self.state == "recorded" and not self.events)
            or (self.state in ("no_events", "unavailable") and self.events)
            or (self.state == "truncated" and not self.dropped_events)
            or (self.state != "truncated" and self.dropped_events)
        ):
            raise ValueError("Trace content contradicts its completeness state.")
        if self.state == "unavailable" and self.reason is None:
            raise ValueError("Unavailable evidence requires an explicit reason.")

    @staticmethod
    def digest(keys: Sequence[float]) -> str:
        """Bind canonical IEEE-754 coordinates without storing source arrays/payloads.

        Args:
            keys (Sequence[float]): Original detached chromosome coordinates.

        Returns:
            str: SHA-256 binding checksum, not cryptographic proof of execution.
        """
        values = array("d", keys)
        if sys.byteorder == "little":
            values.byteswap()
        return hashlib.sha256(values.tobytes()).hexdigest()

    def matches(self, keys: Sequence[float], cost: float) -> bool:
        """Verify a retained identity has not been paired with different numerical output.

        Args:
            keys (Sequence[float]): Selected final chromosome.
            cost (float): Selected original objective.

        Returns:
            bool: Whether this evidence's existing binding is intact.
        """
        return self.cost == cost and self.keys_digest == self.digest(keys)

    def to_dict(self) -> dict[str, object]:
        """Export only explicitly authored evidence and bounded primitive provenance.

        Returns:
            dict[str, object]: JSON-compatible evidence, with no decoder payload/source.
        """
        return asdict(self)

    @classmethod
    def selected(cls, value: float, keys: Sequence[float]) -> DecisionTrace:
        """Read the selected numeric reference, never recover identity by matching coordinates.

        Args:
            value (float): Selected tagged or ordinary cost.
            keys (Sequence[float]): Selected chromosome for binding validation only.

        Returns:
            DecisionTrace: Existing observation evidence or explicit lost identity/binding.
        """
        if not isinstance(value, _ObservedCost):
            return cls("unavailable", reason="identity_unavailable")
        trace = value._cell.snapshot()
        if not trace.matches(keys, value):
            return cls("unavailable", reason="binding_mismatch")
        return trace


class _TraceCell:
    """Hold one lightweight identity while its bounded event payload may be evicted.

    Responsibilities:
    - Clear evicted content even when population caches retain their numeric reference.
    - Snapshot only completed finite observations for owned process transport.

    Attributes:
        identity (str): Invocation identity independent of keys/cost equality.
        source (str): Public producing phase/search label.
        config (TraceConfig): Fixed observation limits.
        pinned (bool): Protect the current session incumbent from eviction.
    """

    def __init__(self, identity: str, source: str, config: TraceConfig) -> None:
        """Allocate an empty bounded observation buffer, not a candidate history.

        Args:
            identity (str): Unique per-session invocation identity.
            source (str): Producing phase/search label.
            config (TraceConfig): Trusted checked limits.
        """
        self.identity, self.source, self.config = identity, source, config
        self.pinned = False
        self.events: list[DecisionEvent] | None = []
        self.encoded_bytes = 2
        self.dropped = 0
        self.witness: DecisionTrace | None = None
        self.keys_digest: str | None = None
        self.cost: float | None = None

    def emit(self, action: str, reason: str) -> None:
        """Retain authored text or count omitted events without unbounded accumulation.

        Args:
            action (str): Explicit domain step/decision.
            reason (str): Explicit optional reason.

        Raises:
            ValueError: If an active non-evicted event contains malformed text.
        """
        if self.events is None:
            return
        event = DecisionEvent(action, reason)
        encoded = len(json.dumps(asdict(event), ensure_ascii=False, separators=(",", ":")).encode())
        size = encoded + bool(self.events)
        if (
            self.dropped
            or len(self.events) >= self.config.max_events
            or self.encoded_bytes + size > self.config.max_bytes
        ):
            self.dropped += 1
            return
        self.events.append(event)
        self.encoded_bytes += size

    def seal(self, keys: Sequence[float], cost: float) -> None:
        """Associate emitted events with that same invocation's finite objective.

        Args:
            keys (Sequence[float]): Original detached keys.
            cost (float): Completed finite original objective.
        """
        self.keys_digest, self.cost = DecisionTrace.digest(keys), cost
        if self.events is not None:
            state = "truncated" if self.dropped else ("recorded" if self.events else "no_events")
            self.witness = DecisionTrace(
                state,
                self.identity,
                self.source,
                self.keys_digest,
                cost,
                tuple(self.events),
                self.dropped,
                max_events=self.config.max_events,
                max_bytes=self.config.max_bytes,
            )
        self.events = None

    def evict(self) -> None:
        """Release event content while retaining a small honest unavailable identity."""
        self.events = None
        self.witness = None

    def snapshot(self) -> DecisionTrace:
        """Snapshot the original completed evidence or its explicit retention loss.

        Returns:
            DecisionTrace: Immutable payload suitable for owned process transport.
        """
        if self.witness is not None:
            return self.witness
        return DecisionTrace(
            "unavailable",
            self.identity,
            self.source,
            self.keys_digest,
            self.cost,
            reason="retention_limit",
            max_events=self.config.max_events,
            max_bytes=self.config.max_bytes,
        )


class _TraceStore:
    """Bound live session event buffers independently of algorithm population/cache size.

    Responsibilities:
    - Retain at most max_records payloads and protect one current incumbent.

    Attributes:
        config (TraceConfig): Per-session record/content envelope.
        source (str): Algorithm or initial_pool label.
        cells (deque[_TraceCell]): Bounded payload-owning cells, not all evaluations.
    """

    def __init__(self, config: TraceConfig, source: str) -> None:
        """Create one private namespace without consuming optimizer/application RNG streams.

        Args:
            config (TraceConfig): Checked per-session caps.
            source (str): Producing phase/search label.
        """
        self.config, self.source = config, source
        self.namespace = secrets.token_hex(16)
        self.sequence = 0
        self.cells: deque[_TraceCell] = deque()

    def admit(self, trace: DecisionTrace) -> _TraceCell:
        """Bound imported elite payloads by the same session budget as local observations.

        Args:
            trace (DecisionTrace): Immutable original evidence received from an owned pool.

        Returns:
            _TraceCell: Budget-owned reference, with explicit loss if foreign limits exceed policy.
        """
        cell = self.begin()
        cell.identity = trace.evaluation_id or "unavailable"
        cell.source = trace.source
        cell.keys_digest, cell.cost = trace.keys_digest, trace.cost
        cell.events = None
        if trace.max_events <= self.config.max_events and trace.max_bytes <= self.config.max_bytes:
            cell.witness = trace
        return cell

    def begin(self) -> _TraceCell:
        """Allocate an invocation cell after evicting the oldest unpinned event payload.

        Returns:
            _TraceCell: New bounded decoder recording context.
        """
        if len(self.cells) >= self.config.max_records:
            old = next(cell for cell in self.cells if not cell.pinned)
            self.cells.remove(old)
            old.evict()
        self.sequence += 1
        cell = _TraceCell(f"{self.namespace}:{self.sequence}", self.source, self.config)
        self.cells.append(cell)
        return cell


class _ObservedCost(float):
    """Carry an observation reference through unchanged numeric caches and comparisons.

    Responsibilities:
    - Preserve evaluation identity without attaching an event archive to every float.
    - Freeze only bounded selected/elite evidence when crossing owned process channels.

    Attributes:
        _cell (_TraceCell): Small reference; content can be evicted by its session store.
    """

    def __new__(cls, value: float, cell: _TraceCell) -> Self:
        """Construct a normal-valued float with an internal original-observation reference.

        Args:
            value (float): Completed finite objective.
            cell (_TraceCell): Corresponding sealed observation.

        Returns:
            Self: Identity-preserving internal numeric value.
        """
        result = super().__new__(cls, value)
        result._cell = cell
        return result

    def __reduce__(self) -> tuple[Callable[..., _ObservedCost], tuple[float, DecisionTrace]]:
        """Transport only a bounded immutable snapshot, never a recorder/store or callback.

        Returns:
            tuple: Pickle reconstruction for trusted internal process channels.
        """
        return self.restore, (float(self), self._cell.snapshot())

    @staticmethod
    def restore(value: float, trace: DecisionTrace, register: bool = True) -> _ObservedCost:
        """Rebuild detached process/elite evidence without retaining the producing session.

        Args:
            value (float): Unchanged numeric objective.
            trace (DecisionTrace): Bounded original observation snapshot.
            register (bool): Register process-reader copies in the current session budget.

        Returns:
            _ObservedCost: Detached numeric reference.
        """
        store = _ACTIVE_STORE.get() if register else None
        if store is not None:
            return _ObservedCost(value, store.admit(trace))
        config = TraceConfig(trace.max_events, trace.max_bytes)
        cell = _TraceCell(trace.evaluation_id or "unavailable", trace.source, config)
        cell.events = None
        cell.witness = trace
        return _ObservedCost(value, cell)

    @staticmethod
    def retain(value: float) -> float:
        """Freeze admitted elite evidence while leaving ordinary costs completely unchanged.

        Args:
            value (float): Tagged or ordinary numeric objective.

        Returns:
            float: Detached evidence-bearing value, or the exact ordinary input.
        """
        return (
            _ObservedCost.restore(float(value), value._cell.snapshot(), register=False)
            if isinstance(value, _ObservedCost)
            else value
        )

    @staticmethod
    def for_reader(value: float) -> float:
        """Share ordinary numbers or budget an elite snapshot inside an active search.

        Args:
            value (float): Original pool observation.

        Returns:
            float: Bounded reader reference, without duplicating an unbounded event archive.
        """
        if isinstance(value, _ObservedCost) and _ACTIVE_STORE.get() is not None:
            return _ObservedCost.restore(float(value), value._cell.snapshot())
        return value

    @staticmethod
    def cast(value: float) -> float:
        """Normalize ordinary scalar inputs without discarding a genuine observation reference.

        Args:
            value (float): Original numeric scalar or identity-bearing cost.

        Returns:
            float: Float-compatible normalized cost.
        """
        return value if isinstance(value, _ObservedCost) else float(value)


_ACTIVE_STORE: ContextVar[_TraceStore | None] = ContextVar("rko_trace_store", default=None)
_CURRENT: ContextVar[_TraceCell | None] = ContextVar("rko_decision_recorder", default=None)


def record_decision(action: str, reason: str = "") -> None:
    """Emit explicit authored text only inside the current traced original decoder call.

    Args:
        action (str): Domain decision/step, not automatically inferred from Python code.
        reason (str): Optional authored reason; never include secrets or private values by default.

    Raises:
        ValueError: If active recording receives malformed/oversized text.
    """
    cell = _CURRENT.get()
    if cell is not None:
        cell.emit(action, reason)
