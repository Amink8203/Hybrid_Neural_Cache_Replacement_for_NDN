
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .packets import DataPacket
    from .store import CacheEntry, ContentStore


@dataclass(slots=True)
class NameStatistics:
    requests: int = 0
    last_request: float | None = None
    last_request_tick: int | None = None
    mean_interval: float | None = None
    decayed_requests: float = 0.0
    decay_tick: int = 0


class RequestStatistics:

    def __init__(self, *, popularity_decay: float = 32.0) -> None:
        if popularity_decay <= 0:
            raise ValueError("popularity_decay must be positive")
        self.popularity_decay = float(popularity_decay)
        self.total_requests = 0
        self.by_name: dict[str, NameStatistics] = {}

    def observe(self, name: str, now: float) -> None:
        stats = self.by_name.setdefault(name, NameStatistics())
        next_tick = self.total_requests + 1
        stats.decayed_requests *= math.exp(
            -(next_tick - stats.decay_tick) / self.popularity_decay
        )
        stats.decayed_requests += 1.0
        stats.decay_tick = next_tick
        if stats.last_request is not None:
            interval = max(0.0, now - stats.last_request)
            if stats.mean_interval is None:
                stats.mean_interval = interval
            else:
                stats.mean_interval = 0.8 * stats.mean_interval + 0.2 * interval
        stats.requests += 1
        stats.last_request = now
        stats.last_request_tick = next_tick
        self.total_requests += 1

    def get(self, name: str) -> NameStatistics:
        return self.by_name.get(name, NameStatistics())

    def decayed_frequency(self, name: str) -> float:

        stats = self.by_name.get(name)
        if stats is None:
            return 0.0
        return stats.decayed_requests * math.exp(
            -(self.total_requests - stats.decay_tick) / self.popularity_decay
        )


class CandidateFeatureBuilder:

    FEATURE_NAMES = (
        "decayed_request_frequency",
        "recency",
        "regularity",
        "cache_hit_fraction",
        "residency_recency",
        "freshness_remaining",
        "relative_size",
        "is_incoming",
    )

    def __init__(self, recency_scale: float = 100.0) -> None:
        if recency_scale <= 0:
            raise ValueError("recency_scale must be positive")
        self.recency_scale = recency_scale

    def build(
        self,
        *,
        name: str,
        packet: "DataPacket",
        entry: "CacheEntry | None",
        statistics: RequestStatistics,
        store: "ContentStore",
        now: float,
        is_incoming: bool,
    ) -> list[float]:
        stats = statistics.get(name)
        decayed_count = statistics.decayed_frequency(name)
        frequency = decayed_count / (1.0 + decayed_count)
        if stats.last_request is None:
            recency = 0.0
        else:
            recency = math.exp(-max(0.0, now - stats.last_request) / self.recency_scale)
        regularity = (
            0.0
            if stats.mean_interval is None
            else math.exp(-stats.mean_interval / self.recency_scale)
        )
        entry_hits = 0 if entry is None else entry.hit_count
        cache_hit_fraction = entry_hits / max(1, stats.requests)
        residency_recency = (
            1.0
            if entry is None
            else math.exp(
                -max(0.0, now - entry.inserted_at) / self.recency_scale
            )
        )
        if packet.freshness_period is None:
            freshness = 1.0
        elif packet.freshness_period <= 0:
            freshness = 0.0
        else:
            freshness_origin = (
                now if entry is None else entry.inserted_at
            )
            remaining = (
                freshness_origin + packet.freshness_period - now
            ) / packet.freshness_period
            freshness = min(1.0, max(0.0, remaining))
        max_bytes = store.max_bytes
        if max_bytes is not None and max_bytes > 0:
            relative_size = min(1.0, packet.size_bytes / max_bytes)
        else:
            relative_size = min(1.0, math.log1p(packet.size_bytes) / math.log1p(65536))
        return [
            min(1.0, max(0.0, frequency)),
            recency,
            regularity,
            min(1.0, max(0.0, cache_hit_fraction)),
            residency_recency,
            freshness,
            relative_size,
            1.0 if is_incoming else 0.0,
        ]

    def batch(
        self,
        candidates: list[tuple[str, "DataPacket", "CacheEntry | None", bool]],
        *,
        statistics: RequestStatistics,
        store: "ContentStore",
        now: float,
        device: torch.device,
    ) -> Tensor:
        rows = [
            self.build(
                name=name,
                packet=packet,
                entry=entry,
                statistics=statistics,
                store=store,
                now=now,
                is_incoming=is_incoming,
            )
            for name, packet, entry, is_incoming in candidates
        ]
        if not rows:
            return torch.empty(
                (0, len(self.FEATURE_NAMES)),
                dtype=torch.float32,
                device=device,
            )
        popularity = [
            statistics.decayed_frequency(name)
            for name, _, _, _ in candidates
        ]
        last_access_tick = []
        for name, _, _, is_incoming in candidates:
            tick = statistics.get(name).last_request_tick
            if tick is None:
                tick = statistics.total_requests if is_incoming else 0
            last_access_tick.append(float(tick))

        def minmax(values: list[float]) -> list[float]:
            low = min(values)
            high = max(values)
            if high - low <= 1e-15:
                return [1.0] * len(values)
            return [(value - low) / (high - low) for value in values]

        for row, popularity_rank, recency_rank in zip(
            rows,
            minmax(popularity),
            minmax(last_access_tick),
            strict=True,
        ):
            row[0] = popularity_rank
            row[1] = recency_rank
        return torch.tensor(rows, dtype=torch.float32, device=device)
