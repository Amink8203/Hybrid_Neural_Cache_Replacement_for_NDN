
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping


@dataclass(slots=True)
class SimulationMetrics:


    requests: int = 0
    hits: int = 0
    misses: int = 0
    bytes_requested: int = 0
    bytes_hit: int = 0
    insertions: int = 0
    evictions: int = 0
    evicted_bytes: int = 0
    rejections: int = 0
    rejected_bytes: int = 0
    expirations: int = 0
    expired_bytes: int = 0

    @staticmethod
    def _non_negative_int(value: int, name: str) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool")
        integer = int(value)
        if integer != value:
            raise ValueError(f"{name} must be an integer")
        if integer < 0:
            raise ValueError(f"{name} must be non-negative")
        return integer

    def record_request(self, hit: bool, size_bytes: int = 0) -> None:

        size = self._non_negative_int(size_bytes, "size_bytes")
        self.requests += 1
        self.bytes_requested += size
        if hit:
            self.hits += 1
            self.bytes_hit += size
        else:
            self.misses += 1

    def record_hit(self, size_bytes: int = 0) -> None:
        self.record_request(True, size_bytes)

    def record_miss(self, size_bytes: int = 0) -> None:
        self.record_request(False, size_bytes)

    def record_insertion(self, count: int = 1) -> None:
        self.insertions += self._non_negative_int(count, "count")

    def record_eviction(self, size_bytes: int = 0, count: int = 1) -> None:
        self.evictions += self._non_negative_int(count, "count")
        self.evicted_bytes += self._non_negative_int(size_bytes, "size_bytes")

    def record_rejection(self, size_bytes: int = 0, count: int = 1) -> None:
        self.rejections += self._non_negative_int(count, "count")
        self.rejected_bytes += self._non_negative_int(size_bytes, "size_bytes")

    def record_expiration(self, size_bytes: int = 0, count: int = 1) -> None:
        self.expirations += self._non_negative_int(count, "count")
        self.expired_bytes += self._non_negative_int(size_bytes, "size_bytes")

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.requests if self.requests else 0.0

    @property
    def hit_rate(self) -> float:

        return self.hit_ratio

    @property
    def miss_ratio(self) -> float:
        return self.misses / self.requests if self.requests else 0.0

    @property
    def byte_hit_ratio(self) -> float:
        return self.bytes_hit / self.bytes_requested if self.bytes_requested else 0.0

    @property
    def byte_hit_rate(self) -> float:
        return self.byte_hit_ratio

    @property
    def eviction_rate(self) -> float:

        return self.evictions / self.requests if self.requests else 0.0

    @property
    def rejection_rate(self) -> float:

        return self.rejections / self.misses if self.misses else 0.0

    def validate(self) -> None:

        for field in fields(self):
            self._non_negative_int(getattr(self, field.name), field.name)
        if self.hits + self.misses != self.requests:
            raise ValueError("hits + misses must equal requests")
        if self.bytes_hit > self.bytes_requested:
            raise ValueError("bytes_hit cannot exceed bytes_requested")

    def as_dict(self) -> dict[str, int | float]:
        self.validate()
        result: dict[str, int | float] = {
            field.name: getattr(self, field.name) for field in fields(self)
        }
        result.update(
            {
                "hit_ratio": self.hit_ratio,
                "miss_ratio": self.miss_ratio,
                "byte_hit_ratio": self.byte_hit_ratio,
                "eviction_rate": self.eviction_rate,
                "rejection_rate": self.rejection_rate,
            }
        )
        return result

    to_dict = as_dict

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "SimulationMetrics":
        counter_names = {field.name for field in fields(cls)}
        metrics = cls(
            **{
                name: cls._non_negative_int(values.get(name, 0), name)
                for name in counter_names
            }
        )
        metrics.validate()
        return metrics

    def merge(self, other: "SimulationMetrics") -> "SimulationMetrics":
        if not isinstance(other, SimulationMetrics):
            raise TypeError("can only merge SimulationMetrics")
        for field in fields(self):
            setattr(
                self,
                field.name,
                getattr(self, field.name) + getattr(other, field.name),
            )
        self.validate()
        return self

    def copy(self) -> "SimulationMetrics":
        return SimulationMetrics(
            **{field.name: getattr(self, field.name) for field in fields(self)}
        )

    def __iadd__(self, other: "SimulationMetrics") -> "SimulationMetrics":
        return self.merge(other)

    def __add__(self, other: "SimulationMetrics") -> "SimulationMetrics":
        return self.copy().merge(other)


__all__ = ["SimulationMetrics"]
