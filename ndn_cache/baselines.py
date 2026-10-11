from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Sequence

from .metrics import SimulationMetrics
from .names import split_components
from .packets import DataPacket, Interest, RequestEvent
from .store import CacheEntry, ContentStore
from .agent import PendingInterest, RequestOutcome


class BaselinePolicy(str, Enum):
    LRU = "lru"
    LFU = "lfu"
    FIFO = "fifo"
    RANDOM = "random"
    BELADY = "belady"
    NO_CACHE = "no_cache"

    @classmethod
    def parse(cls, value: "BaselinePolicy | str") -> "BaselinePolicy":
        if isinstance(value, cls):
            return value
        normalised = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "oracle": cls.BELADY,
            "offline": cls.BELADY,
            "optimal": cls.BELADY,
            "none": cls.NO_CACHE,
            "nocache": cls.NO_CACHE,
            "no_cache": cls.NO_CACHE,
        }
        if normalised in aliases:
            return aliases[normalised]
        try:
            return cls(normalised)
        except ValueError as exc:
            choices = ", ".join(policy.value for policy in cls)
            raise ValueError(f"unknown baseline {value!r}; choose one of {choices}") from exc


@dataclass(frozen=True, slots=True)
class BaselineResult:
    policy: BaselinePolicy
    metrics: SimulationMetrics
    per_node: dict[str, SimulationMetrics]
    final_cache: dict[str, tuple[str, ...]]
    max_items: int | None
    max_bytes: int | None
    seed: int

    @property
    def hit_ratio(self) -> float:
        return self.metrics.hit_ratio

    @property
    def byte_hit_ratio(self) -> float:
        return self.metrics.byte_hit_ratio

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy.value,
            "max_items": self.max_items,
            "max_bytes": self.max_bytes,
            "seed": self.seed,
            "metrics": self.metrics.as_dict(),
            "per_node": {
                node_id: metrics.as_dict()
                for node_id, metrics in sorted(self.per_node.items())
            },
            "final_cache": {
                node_id: list(names)
                for node_id, names in sorted(self.final_cache.items())
            },
        }

    to_dict = as_dict


@dataclass(slots=True)
class _PolicyRecord:
    inserted: int
    last_access: int
    frequency: int = 1


@dataclass(slots=True)
class _NodeState:
    store: ContentStore
    metrics: SimulationMetrics = field(default_factory=SimulationMetrics)
    records: dict[str, _PolicyRecord] = field(default_factory=dict)
    tick: int = 0
    rng: random.Random = field(default_factory=random.Random)

    def touch(self, name: str) -> None:
        record = self.records.get(name)
        if record is None:
            self.records[name] = _PolicyRecord(
                inserted=self.tick,
                last_access=self.tick,
                frequency=1,
            )
        else:
            record.last_access = self.tick
            record.frequency += 1

    def inserted(self, name: str) -> None:
        self.records[name] = _PolicyRecord(
            inserted=self.tick,
            last_access=self.tick,
            frequency=1,
        )


def _stable_node_seed(seed: int, node_id: str) -> int:
    digest = hashlib.blake2b(
        node_id.encode("utf-8"), digest_size=8, person=b"ndn-base"
    ).digest()
    return int(seed) ^ int.from_bytes(digest, "big")


def _entry_packet(entry: CacheEntry) -> DataPacket:
    packet = getattr(entry, "packet", None)
    if packet is None:
        packet = getattr(entry, "data", None)
    if packet is None:
        raise TypeError("ContentStore entries must expose .packet or .data")
    return packet


def _packet_size(packet: DataPacket) -> int:
    size = getattr(packet, "size_bytes", None)
    if size is None:
        content = getattr(packet, "content", b"")
        size = len(content)
    return int(size)


def _packet_is_fresh(packet: DataPacket, now: float) -> bool:
    method = getattr(packet, "is_fresh", None)
    if callable(method):
        return bool(method(now))
    period = getattr(packet, "freshness_period", None)
    if period is None:
        return True
    produced = float(getattr(packet, "produced_at", 0.0))
    return now < produced + float(period)


def _make_interest(event: RequestEvent) -> Interest:
    return Interest(
        name=event.name,
        timestamp=event.timestamp,
        must_be_fresh=event.must_be_fresh,
        can_be_prefix=event.can_be_prefix,
    )


def _make_data(event: RequestEvent) -> DataPacket:
    return DataPacket(
        name=event.name,
        size_bytes=event.size_bytes,
        freshness_period=event.freshness_period,
        produced_at=event.timestamp,
    )


class BaselineSimulator:

    def __init__(
        self,
        policy: BaselinePolicy | str,
        *,
        max_items: int | None = None,
        max_bytes: int | None = None,
        seed: int = 0,
    ) -> None:
        self.policy = BaselinePolicy.parse(policy)
        if max_items is None and max_bytes is None:
            max_items = 100
        if max_items is not None and max_items < 0:
            raise ValueError("max_items cannot be negative")
        if max_bytes is not None and max_bytes < 0:
            raise ValueError("max_bytes cannot be negative")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.seed = int(seed)
        self._nodes: dict[str, _NodeState] = {}
        self._events: list[RequestEvent] = []
        self._future: dict[tuple[str, str], deque[int]] = {}
        self._current_index = -1

    def _node(self, node_id: str) -> _NodeState:
        state = self._nodes.get(node_id)
        if state is None:
            state = _NodeState(
                store=ContentStore(
                    max_items=self.max_items,
                    max_bytes=self.max_bytes,
                ),
                rng=random.Random(_stable_node_seed(self.seed, node_id)),
            )
            self._nodes[node_id] = state
        return state

    def _prepare_future(self) -> None:
        future: dict[tuple[str, str], deque[int]] = defaultdict(deque)
        for index, event in enumerate(self._events):
            future[(event.node_id, event.name)].append(index)
        self._future = dict(future)

    def _advance_future(self, event: RequestEvent, index: int) -> None:
        if self.policy is not BaselinePolicy.BELADY:
            return
        positions = self._future[(event.node_id, event.name)]
        if not positions or positions[0] != index:
            raise RuntimeError("Belady future-use index became inconsistent")
        positions.popleft()

    def _next_eligible_use(
        self,
        node_id: str,
        packet: DataPacket,
    ) -> float:
        packet_components = split_components(packet.name)
        for next_index in range(self._current_index + 1, len(self._events)):
            future_event = self._events[next_index]
            if future_event.node_id != node_id:
                continue
            matches = future_event.name == packet.name
            if future_event.can_be_prefix:
                prefix = split_components(future_event.name)
                matches = packet_components[: len(prefix)] == prefix
            if not matches:
                continue
            if future_event.must_be_fresh and not _packet_is_fresh(
                packet,
                future_event.timestamp,
            ):

                return math.inf
            return float(next_index)
        return math.inf
    def _victim_name(self, node_id: str, state: _NodeState) -> str:
        entries = list(state.store.entries())
        if not entries:
            raise RuntimeError("cannot select a victim from an empty cache")
        names = sorted(_entry_packet(entry).name for entry in entries)

        if self.policy is BaselinePolicy.RANDOM:
            return names[state.rng.randrange(len(names))]
        if self.policy is BaselinePolicy.BELADY:
            next_uses: dict[str, float] = {}
            for entry in entries:
                packet = _entry_packet(entry)
                next_uses[packet.name] = self._next_eligible_use(node_id, packet)
            farthest = max(next_uses.values())
            return min(name for name, position in next_uses.items() if position == farthest)

        def record(name: str) -> _PolicyRecord:
            try:
                return state.records[name]
            except KeyError as exc:
                raise RuntimeError(f"missing policy metadata for {name}") from exc

        if self.policy is BaselinePolicy.LRU:
            return min(names, key=lambda name: (record(name).last_access, name))
        if self.policy is BaselinePolicy.LFU:
            return min(
                names,
                key=lambda name: (
                    record(name).frequency,
                    record(name).last_access,
                    record(name).inserted,
                    name,
                ),
            )
        if self.policy is BaselinePolicy.FIFO:
            return min(names, key=lambda name: (record(name).inserted, name))
        raise RuntimeError(f"policy {self.policy.value} does not select victims")

    def _remove_stale_exact(
        self,
        event: RequestEvent,
        state: _NodeState,
        hit_entry: CacheEntry | None,
    ) -> None:

        if hit_entry is not None:
            return
        incoming = _make_data(event)
        if not state.store.is_cacheable(incoming):
            return
        existing = state.store.get(event.name)
        if existing is None:
            return
        packet = _entry_packet(existing)
        if event.must_be_fresh and not _packet_is_fresh(packet, event.timestamp):
            removed = state.store.remove(event.name)
            if removed is not None:
                state.records.pop(event.name, None)
                state.metrics.record_expiration(_packet_size(packet))

    def _admit(
        self,
        event: RequestEvent,
        state: _NodeState,
        packet: DataPacket,
    ) -> None:
        if self.policy is BaselinePolicy.NO_CACHE or not state.store.is_cacheable(
            packet
        ):
            state.metrics.record_rejection(event.size_bytes)
            return

        while not state.store.has_capacity(packet):
            if len(state.store) == 0:
                state.metrics.record_rejection(event.size_bytes)
                return
            victim_name = self._victim_name(event.node_id, state)
            removed = state.store.remove(victim_name)
            if removed is None:
                raise RuntimeError("selected baseline victim disappeared")
            removed_packet = _entry_packet(removed)
            state.records.pop(victim_name, None)
            state.metrics.record_eviction(_packet_size(removed_packet))

        state.store.insert(packet, now=event.timestamp)
        state.inserted(packet.name)
        state.metrics.record_insertion()

    def _process(self, event: RequestEvent, index: int) -> None:
        self._current_index = index
        state = self._node(event.node_id)
        state.tick += 1
        self._advance_future(event, index)

        hit_entry = state.store.lookup(_make_interest(event))
        hit = hit_entry is not None
        response_size = (
            _packet_size(_entry_packet(hit_entry))
            if hit_entry is not None
            else event.size_bytes
        )
        state.metrics.record_request(hit, response_size)
        if hit_entry is not None:
            state.touch(_entry_packet(hit_entry).name)
            return

        self._remove_stale_exact(event, state, hit_entry)
        self._admit(event, state, _make_data(event))

    def run(self, events: Iterable[RequestEvent]) -> BaselineResult:

        self._nodes.clear()
        self._events = list(events)
        previous: float | None = None
        for index, event in enumerate(self._events):
            timestamp = float(event.timestamp)
            if not math.isfinite(timestamp):
                raise ValueError(f"event {index}: timestamp must be finite")
            if previous is not None and timestamp < previous:
                raise ValueError("events must be ordered by non-decreasing timestamp")
            if int(event.size_bytes) <= 0:
                raise ValueError(f"event {index}: size_bytes must be positive")
            previous = timestamp
        if self.policy is BaselinePolicy.BELADY:
            self._prepare_future()
        else:
            self._future = {}

        for index, event in enumerate(self._events):
            self._process(event, index)

        per_node = {
            node_id: state.metrics.copy()
            for node_id, state in sorted(self._nodes.items())
        }
        aggregate = SimulationMetrics()
        for metrics in per_node.values():
            aggregate += metrics
        final_cache = {
            node_id: tuple(
                sorted(_entry_packet(entry).name for entry in state.store.entries())
            )
            for node_id, state in sorted(self._nodes.items())
        }
        return BaselineResult(
            policy=self.policy,
            metrics=aggregate,
            per_node=per_node,
            final_cache=final_cache,
            max_items=self.max_items,
            max_bytes=self.max_bytes,
            seed=self.seed,
        )


class BaselineCacheController:

    SUPPORTED_POLICIES = (
        BaselinePolicy.LRU,
        BaselinePolicy.LFU,
        BaselinePolicy.FIFO,
        BaselinePolicy.RANDOM,
        BaselinePolicy.NO_CACHE,
    )

    def __init__(
        self,
        policy: BaselinePolicy | str,
        *,
        cache_items: int | None,
        cache_bytes: int | None = None,
        seed: int = 0,
    ) -> None:
        parsed = BaselinePolicy.parse(policy)
        if parsed not in self.SUPPORTED_POLICIES:
            raise ValueError(
                "Belady is not a valid topology baseline: future per-router "
                "lookups depend on earlier cache decisions"
            )
        self.policy = parsed
        self.cache_items = cache_items
        self.cache_bytes = cache_bytes
        self.seed = int(seed)
        self._simulator = BaselineSimulator(
            parsed,
            max_items=cache_items,
            max_bytes=cache_bytes,
            seed=seed,
        )

    @property
    def nodes(self) -> dict[str, _NodeState]:
        return self._simulator._nodes

    def reset_episode(self) -> None:
        self._simulator._nodes.clear()

    def begin_interest(self, event: RequestEvent) -> PendingInterest:
        state = self._simulator._node(event.node_id)
        state.tick += 1
        hit_entry = state.store.lookup(_make_interest(event))
        if hit_entry is not None:
            state.touch(_entry_packet(hit_entry).name)
        return PendingInterest(event=event, hit_entry=hit_entry)

    @staticmethod
    def _validate_response(event: RequestEvent, packet: DataPacket) -> None:
        if packet.name == event.name:
            return
        if event.can_be_prefix:
            prefix = split_components(event.name)
            components = split_components(packet.name)
            if components[: len(prefix)] == prefix:
                return
        raise ValueError(
            f"Data {packet.name!r} does not satisfy Interest {event.name!r}"
        )

    def _admit(
        self,
        event: RequestEvent,
        state: _NodeState,
        packet: DataPacket,
    ) -> tuple[bool, list[str], int]:
        if self.policy is BaselinePolicy.NO_CACHE or not state.store.is_cacheable(
            packet
        ):
            return False, [], 0

        evicted: list[str] = []
        evicted_bytes = 0
        while not state.store.has_capacity(packet):
            if len(state.store) == 0:
                return False, evicted, evicted_bytes
            victim_name = self._simulator._victim_name(event.node_id, state)
            removed = state.store.remove(victim_name)
            if removed is None:
                raise RuntimeError("selected baseline victim disappeared")
            removed_packet = _entry_packet(removed)
            state.records.pop(victim_name, None)
            evicted.append(victim_name)
            evicted_bytes += _packet_size(removed_packet)

        state.store.insert(packet, now=event.timestamp)
        state.inserted(packet.name)
        return True, evicted, evicted_bytes

    def finish_interest(
        self,
        pending: PendingInterest,
        *,
        response: DataPacket | None,
        training: bool,
        future_reuse: set[str] | None = None,
        defer_update: bool = False,
    ) -> RequestOutcome:

        del future_reuse, defer_update
        if training:
            raise ValueError("baseline topology controllers do not train")
        if pending.completed:
            raise RuntimeError("Interest has already been completed")
        event = pending.event
        state = self._simulator._node(event.node_id)
        if pending.hit:
            if response is not None:
                raise ValueError("a cache hit must not receive a second response")
            assert pending.hit_entry is not None
            data = _entry_packet(pending.hit_entry)
        else:
            if response is None:
                raise ValueError("a cache miss requires returned Data")
            self._validate_response(event, response)
            data = response

        expired_bytes = 0
        if not pending.hit and event.must_be_fresh:
            stale_exact = state.store.get(data.name)
            if (
                stale_exact is not None
                and state.store.is_cacheable(data)
                and not stale_exact.is_fresh(event.timestamp)
            ):
                expired_bytes = _packet_size(_entry_packet(stale_exact))
                state.store.remove(data.name)
                state.records.pop(data.name, None)

        admitted = True
        evicted: list[str] = []
        evicted_bytes = 0
        if not pending.hit:
            admitted, evicted, evicted_bytes = self._admit(event, state, data)

        pending.completed = True
        return RequestOutcome(
            node_id=event.node_id,
            name=event.name,
            hit=pending.hit,
            reward=pending.reward,
            data=data,
            evicted_names=evicted,
            evicted_bytes=evicted_bytes,
            expired_bytes=expired_bytes,
            admission_rejected=not admitted,
        )

    def abandon_interest(self, pending: PendingInterest) -> None:
        if pending.completed:
            raise RuntimeError("Interest has already been completed")
        if pending.hit:
            raise ValueError("a cache hit cannot be abandoned")
        pending.completed = True


def _resolve_capacities(
    *,
    cache_items: int | None,
    cache_bytes: int | None,
    max_items: int | None,
    max_bytes: int | None,
) -> tuple[int | None, int | None]:
    if cache_items is not None and max_items is not None and cache_items != max_items:
        raise ValueError("cache_items and max_items disagree")
    if cache_bytes is not None and max_bytes is not None and cache_bytes != max_bytes:
        raise ValueError("cache_bytes and max_bytes disagree")
    items = max_items if max_items is not None else cache_items
    byte_limit = max_bytes if max_bytes is not None else cache_bytes
    if items is None and byte_limit is None:
        items = 100
    return items, byte_limit


def evaluate_baseline(
    events: Iterable[RequestEvent],
    policy: BaselinePolicy | str,
    *,
    cache_items: int | None = None,
    cache_bytes: int | None = None,
    max_items: int | None = None,
    max_bytes: int | None = None,
    seed: int = 0,
) -> BaselineResult:

    items, byte_limit = _resolve_capacities(
        cache_items=cache_items,
        cache_bytes=cache_bytes,
        max_items=max_items,
        max_bytes=max_bytes,
    )
    return BaselineSimulator(
        policy,
        max_items=items,
        max_bytes=byte_limit,
        seed=seed,
    ).run(events)


def evaluate_baselines(
    events: Iterable[RequestEvent],
    *,
    policies: Sequence[BaselinePolicy | str] = (
        BaselinePolicy.LRU,
        BaselinePolicy.LFU,
        BaselinePolicy.FIFO,
        BaselinePolicy.RANDOM,
        BaselinePolicy.BELADY,
    ),
    cache_items: int | None = None,
    cache_bytes: int | None = None,
    max_items: int | None = None,
    max_bytes: int | None = None,
    seed: int = 0,
) -> dict[str, BaselineResult]:
    materialised = list(events)
    results: dict[str, BaselineResult] = {}
    for raw_policy in policies:
        policy = BaselinePolicy.parse(raw_policy)
        results[policy.value] = evaluate_baseline(
            materialised,
            policy,
            cache_items=cache_items,
            cache_bytes=cache_bytes,
            max_items=max_items,
            max_bytes=max_bytes,
            seed=seed,
        )
    return results


run_baseline = evaluate_baseline


__all__ = [
    "BaselineCacheController",
    "BaselinePolicy",
    "BaselineResult",
    "BaselineSimulator",
    "evaluate_baseline",
    "evaluate_baselines",
    "run_baseline",
]
