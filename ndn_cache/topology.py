from __future__ import annotations

import copy
import math
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields, replace
from enum import Enum
from typing import Iterable, Protocol

import torch

from .agent import (
    NeuralCacheAgent,
    PendingInterest,
    RequestOutcome,
    UpdateSummary,
)
from .metrics import SimulationMetrics
from .names import canonicalize_name, split_components
from .packets import DataPacket, RequestEvent
from .training import validate_trace


class TopologyKind(str, Enum):

    LINE = "line"
    RING = "ring"
    STAR = "star"
    MESH = "mesh"

    @classmethod
    def parse(cls, value: "TopologyKind | str") -> "TopologyKind":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        if normalized in {"full_mesh", "fullmesh"}:
            normalized = "mesh"
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"unknown topology {value!r}; choose one of {choices}"
            ) from exc


def _is_component_prefix(prefix: str, name: str) -> bool:
    prefix_components = split_components(prefix)
    name_components = split_components(name)
    return name_components[: len(prefix_components)] == prefix_components


@dataclass(frozen=True, slots=True)
class Route:

    prefix: str
    producer: str
    path: tuple[str, ...]

    @property
    def hops(self) -> int:
        return max(0, len(self.path) - 1)


@dataclass(slots=True)
class NetworkTopology:

    kind: TopologyKind
    adjacency: dict[str, frozenset[str]]
    producers: dict[str, str]
    fib: dict[str, dict[str, str]] = field(init=False)

    def __post_init__(self) -> None:
        self.kind = TopologyKind.parse(self.kind)
        if not self.adjacency:
            raise ValueError("topology must contain at least one router")
        normalized: dict[str, frozenset[str]] = {}
        for raw_node, raw_neighbors in self.adjacency.items():
            node = str(raw_node).strip()
            if not node:
                raise ValueError("router identifiers cannot be empty")
            if node in normalized:
                raise ValueError(f"duplicate router identifier {node!r}")
            normalized[node] = frozenset(str(item).strip() for item in raw_neighbors)
        nodes = set(normalized)
        for node, neighbors in normalized.items():
            if "" in neighbors:
                raise ValueError("router identifiers cannot be empty")
            if node in neighbors:
                raise ValueError(f"self-link at router {node!r}")
            unknown = neighbors - nodes
            if unknown:
                raise ValueError(
                    f"router {node!r} links to unknown routers {sorted(unknown)!r}"
                )
            for neighbor in neighbors:
                if node not in normalized[neighbor]:
                    raise ValueError(
                        f"link {node!r}<->{neighbor!r} is not bidirectional"
                    )
        self.adjacency = normalized
        self._validate_connected()

        normalized_producers: dict[str, str] = {}
        for raw_prefix, raw_node in self.producers.items():
            prefix = canonicalize_name(raw_prefix)
            node = str(raw_node).strip()
            if node not in self.adjacency:
                raise ValueError(
                    f"producer for {prefix!r} uses unknown router {node!r}"
                )
            if prefix in normalized_producers:
                raise ValueError(f"duplicate producer prefix {prefix!r}")
            normalized_producers[prefix] = node
        if not normalized_producers:
            raise ValueError("topology must register at least one producer prefix")
        self.producers = normalized_producers
        self.fib = self._build_fib()

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(sorted(self.adjacency))

    @property
    def edges(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (node, neighbor)
            for node in self.nodes
            for neighbor in sorted(self.adjacency[node])
            if node < neighbor
        )

    def _validate_connected(self) -> None:
        start = min(self.adjacency)
        reached = {start}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(self.adjacency[node]):
                if neighbor not in reached:
                    reached.add(neighbor)
                    queue.append(neighbor)
        if reached != set(self.adjacency):
            missing = sorted(set(self.adjacency) - reached)
            raise ValueError(f"topology is disconnected; unreachable routers: {missing}")

    def shortest_path(self, source: str, target: str) -> tuple[str, ...]:

        if source not in self.adjacency:
            raise ValueError(f"unknown source router {source!r}")
        if target not in self.adjacency:
            raise ValueError(f"unknown target router {target!r}")
        if source == target:
            return (source,)
        parents: dict[str, str | None] = {source: None}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor in sorted(self.adjacency[node]):
                if neighbor in parents:
                    continue
                parents[neighbor] = node
                if neighbor == target:
                    path = [target]
                    cursor = target
                    while parents[cursor] is not None:
                        cursor = parents[cursor]
                        path.append(cursor)
                    return tuple(reversed(path))
                queue.append(neighbor)
        raise RuntimeError(f"no path from {source!r} to {target!r}")

    def _build_fib(self) -> dict[str, dict[str, str]]:
        fib: dict[str, dict[str, str]] = {node: {} for node in self.nodes}
        for prefix, producer in sorted(self.producers.items()):
            for node in self.nodes:
                if node == producer:
                    continue
                path = self.shortest_path(node, producer)
                next_hop = path[1]
                if next_hop not in self.adjacency[node]:
                    raise AssertionError("FIB next hop is not a physical neighbor")
                fib[node][prefix] = next_hop
        return fib

    def resolve_producer(self, name: str) -> tuple[str, str]:

        canonical = canonicalize_name(name)
        matches = [
            (len(split_components(prefix)), prefix, producer)
            for prefix, producer in self.producers.items()
            if _is_component_prefix(prefix, canonical)
        ]
        if not matches:
            raise ValueError(f"no producer route for Name {canonical!r}")
        _, prefix, producer = max(matches, key=lambda item: (item[0], item[1]))
        return prefix, producer

    def route(self, source: str, name: str) -> Route:

        if source not in self.adjacency:
            raise ValueError(f"unknown requester router {source!r}")
        prefix, producer = self.resolve_producer(name)
        path = [source]
        visited = {source}
        current = source
        while current != producer:
            next_hop = self.fib[current].get(prefix)
            if next_hop is None:
                raise RuntimeError(
                    f"router {current!r} has no FIB next hop for {prefix!r}"
                )
            if next_hop not in self.adjacency[current]:
                raise RuntimeError("FIB attempted to bypass a physical link")
            if next_hop in visited:
                raise RuntimeError(f"routing loop detected at {next_hop!r}")
            path.append(next_hop)
            visited.add(next_hop)
            current = next_hop
            if len(path) > len(self.adjacency):
                raise RuntimeError("route exceeded topology hop limit")
        return Route(prefix=prefix, producer=producer, path=tuple(path))

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "nodes": list(self.nodes),
            "edges": [list(edge) for edge in self.edges],
            "adjacency": {
                node: sorted(self.adjacency[node]) for node in self.nodes
            },
            "producers": dict(sorted(self.producers.items())),
            "fib": {
                node: dict(sorted(self.fib[node].items())) for node in self.nodes
            },
        }


def build_topology(
    kind: TopologyKind | str,
    *,
    num_nodes: int,
    producer_prefix: str = "/dataset/item",
    producer_node: str = "node-0",
    star_center: str | None = None,
) -> NetworkTopology:

    parsed = TopologyKind.parse(kind)
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int):
        raise TypeError("num_nodes must be an integer")
    if num_nodes < 2:
        raise ValueError("a network topology requires at least two routers")
    if parsed is TopologyKind.RING and num_nodes < 3:
        raise ValueError("a ring topology requires at least three routers")
    nodes = tuple(f"node-{index}" for index in range(num_nodes))
    if producer_node not in nodes:
        raise ValueError(
            f"producer_node must be one of {', '.join(nodes)}"
        )
    resolved_star_center = producer_node
    if parsed is TopologyKind.STAR and star_center is not None:
        candidate = str(star_center).strip()
        if candidate.lower() == "auto":
            resolved_star_center = next(
                node for node in nodes if node != producer_node
            )
        elif candidate in nodes:
            resolved_star_center = candidate
        else:
            raise ValueError(
                f"star_center must be 'auto' or one of {', '.join(nodes)}"
            )
    mutable: dict[str, set[str]] = {node: set() for node in nodes}

    def connect(left: str, right: str) -> None:
        if left == right:
            raise ValueError("self-links are not allowed")
        mutable[left].add(right)
        mutable[right].add(left)

    if parsed is TopologyKind.LINE:
        for index in range(num_nodes - 1):
            connect(nodes[index], nodes[index + 1])
    elif parsed is TopologyKind.RING:
        for index in range(num_nodes):
            connect(nodes[index], nodes[(index + 1) % num_nodes])
    elif parsed is TopologyKind.STAR:
        for node in nodes:
            if node != resolved_star_center:
                connect(resolved_star_center, node)
    elif parsed is TopologyKind.MESH:
        for left_index in range(num_nodes):
            for right_index in range(left_index + 1, num_nodes):
                connect(nodes[left_index], nodes[right_index])
    else:
        raise AssertionError(f"unhandled topology {parsed}")

    return NetworkTopology(
        kind=parsed,
        adjacency={node: frozenset(neighbors) for node, neighbors in mutable.items()},
        producers={producer_prefix: producer_node},
    )


@dataclass(slots=True)
class NodeForwardingStats:
    interests_received: int = 0
    interests_forwarded: int = 0
    data_received: int = 0
    data_forwarded: int = 0
    producer_responses: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "interests_received": self.interests_received,
            "interests_forwarded": self.interests_forwarded,
            "data_received": self.data_received,
            "data_forwarded": self.data_forwarded,
            "producer_responses": self.producer_responses,
        }


@dataclass(slots=True)
class NetworkMetrics:

    consumer_requests: int = 0
    satisfied_requests: int = 0
    cache_hit_requests: int = 0
    producer_fetches: int = 0
    requested_bytes: int = 0
    cache_hit_bytes: int = 0
    interest_hops: int = 0
    data_hops: int = 0
    producer_path_hops: int = 0
    router_lookups: int = 0
    router_cache_hits: int = 0
    router_cache_misses: int = 0

    @property
    def unsatisfied_requests(self) -> int:
        return self.consumer_requests - self.satisfied_requests

    @property
    def cache_hit_ratio(self) -> float:
        return (
            self.cache_hit_requests / self.consumer_requests
            if self.consumer_requests
            else 0.0
        )

    @property
    def byte_cache_hit_ratio(self) -> float:
        return (
            self.cache_hit_bytes / self.requested_bytes
            if self.requested_bytes
            else 0.0
        )

    @property
    def router_cache_hit_ratio(self) -> float:
        return (
            self.router_cache_hits / self.router_lookups
            if self.router_lookups
            else 0.0
        )

    @property
    def mean_retrieval_hops(self) -> float:
        return (
            self.interest_hops / self.satisfied_requests
            if self.satisfied_requests
            else 0.0
        )

    @property
    def hop_savings_ratio(self) -> float:
        return (
            (self.producer_path_hops - self.interest_hops)
            / self.producer_path_hops
            if self.producer_path_hops
            else 0.0
        )

    def validate(self) -> None:
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{item.name} must be an integer")
            if value < 0:
                raise ValueError(f"{item.name} must be non-negative")
        if self.satisfied_requests > self.consumer_requests:
            raise AssertionError("satisfied requests exceed consumer requests")
        if self.satisfied_requests != (
            self.cache_hit_requests + self.producer_fetches
        ):
            raise AssertionError("network request source accounting is inconsistent")
        if self.router_lookups != self.router_cache_hits + self.router_cache_misses:
            raise AssertionError("router lookup accounting is inconsistent")
        if self.cache_hit_bytes > self.requested_bytes:
            raise AssertionError("cache-hit bytes exceed requested bytes")
        if self.interest_hops > self.producer_path_hops:
            raise AssertionError("actual Interest path exceeds producer route")

    def as_dict(self) -> dict[str, int | float]:
        self.validate()
        return {
            "consumer_requests": self.consumer_requests,
            "satisfied_requests": self.satisfied_requests,
            "unsatisfied_requests": self.unsatisfied_requests,
            "cache_hit_requests": self.cache_hit_requests,
            "producer_fetches": self.producer_fetches,
            "requested_bytes": self.requested_bytes,
            "cache_hit_bytes": self.cache_hit_bytes,
            "cache_hit_ratio": self.cache_hit_ratio,
            "byte_cache_hit_ratio": self.byte_cache_hit_ratio,
            "interest_hops": self.interest_hops,
            "data_hops": self.data_hops,
            "producer_path_hops": self.producer_path_hops,
            "mean_retrieval_hops": self.mean_retrieval_hops,
            "hop_savings_ratio": self.hop_savings_ratio,
            "router_lookups": self.router_lookups,
            "router_cache_hits": self.router_cache_hits,
            "router_cache_misses": self.router_cache_misses,
            "router_cache_hit_ratio": self.router_cache_hit_ratio,
        }


@dataclass(frozen=True, slots=True)
class NetworkRequestResult:
    index: int
    timestamp: float
    requester: str
    interest_name: str
    source_kind: str
    source_node: str | None
    producer_path: tuple[str, ...] | None
    interest_path: tuple[str, ...]
    response_name: str | None
    response_size: int

    @property
    def hops(self) -> int:
        return max(0, len(self.interest_path) - 1)

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "requester": self.requester,
            "interest_name": self.interest_name,
            "source_kind": self.source_kind,
            "source_node": self.source_node,
            "producer_path": (
                list(self.producer_path) if self.producer_path is not None else None
            ),
            "interest_path": list(self.interest_path),
            "hops": self.hops,
            "response_name": self.response_name,
            "response_size": self.response_size,
        }


@dataclass(slots=True)
class TopologyRunResult:
    topology: NetworkTopology
    metrics: NetworkMetrics
    per_node_cache: dict[str, SimulationMetrics]
    per_node_forwarding: dict[str, NodeForwardingStats]
    final_cache: dict[str, tuple[str, ...]]
    requests: list[NetworkRequestResult]
    controller_efficiency: dict[str, int | float] = field(default_factory=dict)
    updates: list[UpdateSummary] = field(default_factory=list)

    def as_dict(self, *, include_requests: bool = False) -> dict[str, object]:
        result: dict[str, object] = {
            "topology": self.topology.as_dict(),
            "network_metrics": self.metrics.as_dict(),
            "per_node": {
                node: {
                    "cache": self.per_node_cache[node].as_dict(),
                    "forwarding": self.per_node_forwarding[node].as_dict(),
                    "final_cache": list(self.final_cache[node]),
                }
                for node in self.topology.nodes
            },
            "controller_efficiency": dict(self.controller_efficiency),
            "updates": [asdict(update) for update in self.updates],
        }
        if include_requests:
            result["request_results"] = [item.as_dict() for item in self.requests]
        return result


class CacheController(Protocol):

    nodes: dict[str, object]

    def reset_episode(self) -> None: ...

    def begin_interest(self, event: RequestEvent) -> PendingInterest: ...

    def finish_interest(
        self,
        pending: PendingInterest,
        *,
        response: DataPacket | None,
        training: bool,
        defer_update: bool = False,
    ) -> RequestOutcome: ...

    def abandon_interest(self, pending: PendingInterest) -> None: ...


class TopologySimulator:

    def __init__(
        self,
        topology: NetworkTopology,
        agent: NeuralCacheAgent | CacheController,
    ) -> None:
        self.topology = topology
        self.agent = agent
        self.network_metrics = NetworkMetrics()
        self.per_node_cache: dict[str, SimulationMetrics] = {}
        self.per_node_forwarding: dict[str, NodeForwardingStats] = {}
        self.updates: list[UpdateSummary] = []

    def _reset(self, *, training: bool) -> None:
        self.agent.reset_episode()
        model = getattr(self.agent, "model", None)
        if model is not None:
            model.train(training)
        self.network_metrics = NetworkMetrics()
        self.per_node_cache = {
            node: SimulationMetrics() for node in self.topology.nodes
        }
        self.per_node_forwarding = {
            node: NodeForwardingStats() for node in self.topology.nodes
        }
        self.updates = []

    def _record_local(self, outcome: RequestOutcome) -> None:
        metrics = self.per_node_cache[outcome.node_id]
        metrics.record_request(outcome.hit, outcome.data.size_bytes)
        if outcome.expired_bytes:
            metrics.record_expiration(outcome.expired_bytes)
        if not outcome.hit:
            if outcome.admission_rejected:
                metrics.record_rejection(outcome.data.size_bytes)
            else:
                metrics.record_insertion()
            if outcome.evicted_names:
                metrics.record_eviction(
                    outcome.evicted_bytes,
                    count=len(outcome.evicted_names),
                )
        self.network_metrics.router_lookups += 1
        if outcome.hit:
            self.network_metrics.router_cache_hits += 1
        else:
            self.network_metrics.router_cache_misses += 1
        if outcome.update_loss is not None:
            last_update = getattr(self.agent, "last_update", None)
            if last_update is not None:
                self.updates.append(last_update)

    def _process_one(
        self,
        event: RequestEvent,
        index: int,
        *,
        training: bool,
    ) -> NetworkRequestResult:
        pending_misses: list[PendingInterest] = []
        visited: list[str] = [event.node_id]
        response = None
        source_kind = "cache"
        source_node = event.node_id
        self.per_node_forwarding[event.node_id].interests_received += 1

        ingress_pending = self.agent.begin_interest(event)
        if ingress_pending.hit:
            outcome = self.agent.finish_interest(
                ingress_pending,
                response=None,
                training=training,
                defer_update=training,
            )
            self._record_local(outcome)
            response = outcome.data
            try:
                producer_path = self.topology.route(
                    event.node_id, event.name
                ).path
            except ValueError:
                producer_path = None
        else:
            pending_misses.append(ingress_pending)
            try:
                route = self.topology.route(event.node_id, event.name)
            except ValueError:
                if training:
                    raise ValueError(
                        "topology training requires every Interest to have a route"
                    )
                self.agent.abandon_interest(ingress_pending)
                local_metrics = self.per_node_cache[event.node_id]
                local_metrics.record_request(False, event.size_bytes)
                self.network_metrics.router_lookups += 1
                self.network_metrics.router_cache_misses += 1
                self.network_metrics.consumer_requests += 1
                self.network_metrics.requested_bytes += event.size_bytes
                return NetworkRequestResult(
                    index=index,
                    timestamp=event.timestamp,
                    requester=event.node_id,
                    interest_name=event.name,
                    source_kind="no_route",
                    source_node=None,
                    producer_path=None,
                    interest_path=(event.node_id,),
                    response_name=None,
                    response_size=0,
                )
            producer_path = route.path
            source_kind = "producer"
            source_node = route.producer
            if event.node_id == route.producer:
                response = event.to_data_packet()
                self.per_node_forwarding[event.node_id].producer_responses += 1

        route_nodes = () if response is not None else producer_path[1:]
        for node_id in route_nodes:
            visited.append(node_id)
            self.per_node_forwarding[node_id].interests_received += 1
            local_event = replace(event, node_id=node_id)
            pending = self.agent.begin_interest(local_event)
            if pending.hit:
                outcome = self.agent.finish_interest(
                    pending,
                    response=None,
                    training=training,
                    defer_update=training,
                )
                self._record_local(outcome)
                response = outcome.data
                source_kind = "cache"
                source_node = node_id
                break
            pending_misses.append(pending)
            if node_id == source_node:
                response = event.to_data_packet()
                self.per_node_forwarding[node_id].producer_responses += 1
                break

        if response is None:
            raise RuntimeError(f"producer did not return Data for {event.name!r}")

        for pending in reversed(pending_misses):
            outcome = self.agent.finish_interest(
                pending,
                response=response,
                training=training,
                defer_update=training,
            )
            self._record_local(outcome)

        if training and isinstance(self.agent, NeuralCacheAgent):
            update = self.agent.maybe_update()
            if update is not None:
                self.updates.append(update)

        for position in range(len(visited) - 1):
            upstream = visited[position]
            self.per_node_forwarding[upstream].interests_forwarded += 1
        for position in range(len(visited) - 1, 0, -1):
            upstream = visited[position]
            downstream = visited[position - 1]
            self.per_node_forwarding[upstream].data_forwarded += 1
            self.per_node_forwarding[downstream].data_received += 1

        hops = len(visited) - 1
        metrics = self.network_metrics
        metrics.consumer_requests += 1
        metrics.satisfied_requests += 1
        metrics.requested_bytes += response.size_bytes
        metrics.interest_hops += hops
        metrics.data_hops += hops
        if producer_path is not None:
            metrics.producer_path_hops += max(0, len(producer_path) - 1)
        if source_kind == "cache":
            metrics.cache_hit_requests += 1
            metrics.cache_hit_bytes += response.size_bytes
        else:
            metrics.producer_fetches += 1

        return NetworkRequestResult(
            index=index,
            timestamp=event.timestamp,
            requester=event.node_id,
            interest_name=event.name,
            source_kind=source_kind,
            source_node=source_node,
            producer_path=producer_path,
            interest_path=tuple(visited),
            response_name=response.name,
            response_size=response.size_bytes,
        )

    def run(
        self,
        events: Iterable[RequestEvent],
        *,
        training: bool = False,
    ) -> TopologyRunResult:

        trace = validate_trace(events)
        if training and not isinstance(self.agent, NeuralCacheAgent):
            raise TypeError("only NeuralCacheAgent supports topology training")
        self._reset(training=training)
        for event in trace:
            if event.node_id not in self.topology.adjacency:
                raise ValueError(
                    f"requester {event.node_id!r} is not a topology router"
                )

        results: list[NetworkRequestResult] = []
        gradient_context = nullcontext() if training else torch.no_grad()
        with gradient_context:
            for index, event in enumerate(trace):
                results.append(
                    self._process_one(event, index, training=training)
                )
        if training:
            final_update = self.agent.flush()
            if final_update is not None:
                self.updates.append(final_update)

        self.network_metrics.validate()
        for metrics in self.per_node_cache.values():
            metrics.validate()
        final_cache = {
            node: tuple(
                entry.packet.name
                for entry in self.agent.nodes[node].store.entries()
            )
            if node in self.agent.nodes
            else ()
            for node in self.topology.nodes
        }
        return TopologyRunResult(
            topology=self.topology,
            metrics=self.network_metrics,
            per_node_cache={
                node: metrics.copy()
                for node, metrics in self.per_node_cache.items()
            },
            per_node_forwarding={
                node: replace(stats)
                for node, stats in self.per_node_forwarding.items()
            },
            final_cache=final_cache,
            requests=results,
            controller_efficiency=(
                self.agent.replacement_efficiency.as_dict()
                if hasattr(self.agent, "replacement_efficiency")
                else {}
            ),
            updates=list(self.updates),
        )


@dataclass(slots=True)
class TopologyEpochResult:
    epoch: int
    train_cache_hit_ratio: float
    train_byte_cache_hit_ratio: float
    validation_cache_hit_ratio: float | None
    mean_loss: float | None
    updates: int

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(slots=True)
class TopologyFitResult:
    epochs: list[TopologyEpochResult]
    best_epoch: int
    best_validation_cache_hit_ratio: float
    initial_validation_cache_hit_ratio: float | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": [epoch.as_dict() for epoch in self.epochs],
            "best_epoch": self.best_epoch,
            "best_validation_cache_hit_ratio": (
                self.best_validation_cache_hit_ratio
            ),
            "initial_validation_cache_hit_ratio": (
                self.initial_validation_cache_hit_ratio
            ),
        }


def fit_topology(
    agent: NeuralCacheAgent,
    topology: NetworkTopology,
    train_events: Iterable[RequestEvent],
    *,
    validation_events: Iterable[RequestEvent] = (),
    epochs: int = 3,
    select_initial: bool = False,
) -> TopologyFitResult:

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    train_trace = validate_trace(train_events)
    validation_trace = validate_trace(validation_events)
    if not train_trace:
        raise ValueError("topology training trace is empty")

    history: list[TopologyEpochResult] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    best_baseline = 0.0
    best_baseline_initialized = False
    best_epoch = 0
    best_score = -math.inf
    initial_validation_score: float | None = None

    if select_initial and validation_trace:
        initial_validation = TopologySimulator(topology, agent).run(
            validation_trace,
            training=False,
        )
        initial_validation_score = (
            initial_validation.metrics.cache_hit_ratio
        )
        best_score = initial_validation_score
        best_state = copy.deepcopy(agent.model.state_dict())
        best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())
        best_baseline = agent.running_return_baseline
        best_baseline_initialized = agent._baseline_initialized

    for epoch in range(1, epochs + 1):
        train_result = TopologySimulator(topology, agent).run(
            train_trace,
            training=True,
        )
        validation_result = (
            TopologySimulator(topology, agent).run(
                validation_trace,
                training=False,
            )
            if validation_trace
            else None
        )
        score = (
            validation_result.metrics.cache_hit_ratio
            if validation_result is not None
            else train_result.metrics.cache_hit_ratio
        )
        losses = [update.loss for update in train_result.updates]
        history.append(
            TopologyEpochResult(
                epoch=epoch,
                train_cache_hit_ratio=train_result.metrics.cache_hit_ratio,
                train_byte_cache_hit_ratio=(
                    train_result.metrics.byte_cache_hit_ratio
                ),
                validation_cache_hit_ratio=(
                    validation_result.metrics.cache_hit_ratio
                    if validation_result is not None
                    else None
                ),
                mean_loss=(
                    sum(losses) / len(losses) if losses else None
                ),
                updates=len(losses),
            )
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(agent.model.state_dict())
            best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())
            best_baseline = agent.running_return_baseline
            best_baseline_initialized = agent._baseline_initialized

    assert best_state is not None
    assert best_optimizer_state is not None
    agent.model.load_state_dict(best_state)
    agent.optimizer.load_state_dict(best_optimizer_state)
    agent.running_return_baseline = best_baseline
    agent._baseline_initialized = best_baseline_initialized
    agent.model.eval()
    return TopologyFitResult(
        epochs=history,
        best_epoch=best_epoch,
        best_validation_cache_hit_ratio=best_score,
        initial_validation_cache_hit_ratio=initial_validation_score,
    )


def collect_router_observations(
    topology: NetworkTopology,
    events: Iterable[RequestEvent],
    *,
    behavior_policy: str = "lru",
    cache_items: int | None,
    cache_bytes: int | None = None,
    seed: int = 0,
) -> list[RequestEvent]:

    from .baselines import BaselineCacheController

    trace = validate_trace(events)
    controller = BaselineCacheController(
        behavior_policy,
        cache_items=cache_items,
        cache_bytes=cache_bytes,
        seed=seed,
    )
    replay = TopologySimulator(topology, controller).run(trace)
    observations: list[RequestEvent] = []
    for event, request in zip(trace, replay.requests, strict=True):
        observations.extend(
            replace(event, node_id=node_id)
            for node_id in request.interest_path
        )
    return observations


__all__ = [
    "CacheController",
    "NetworkMetrics",
    "NetworkRequestResult",
    "NetworkTopology",
    "NodeForwardingStats",
    "Route",
    "TopologyKind",
    "TopologyEpochResult",
    "TopologyFitResult",
    "TopologyRunResult",
    "TopologySimulator",
    "build_topology",
    "collect_router_observations",
    "fit_topology",
]
