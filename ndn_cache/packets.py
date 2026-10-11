
from __future__ import annotations

import math
from dataclasses import dataclass

from .names import canonicalize_name


def _finite_time(value: float, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("size_bytes must be an integer")
    if value <= 0:
        raise ValueError("size_bytes must be positive")
    return value


def _freshness(value: float | None) -> float | None:
    if value is None:
        return None
    result = _finite_time(value, "freshness_period")
    if result < 0:
        raise ValueError("freshness_period cannot be negative")
    return result


@dataclass(frozen=True, slots=True)
class Interest:

    name: str
    timestamp: float = 0.0
    must_be_fresh: bool = True
    can_be_prefix: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", canonicalize_name(self.name))
        object.__setattr__(
            self, "timestamp", _finite_time(self.timestamp, "timestamp")
        )
        if not isinstance(self.must_be_fresh, bool):
            raise TypeError("must_be_fresh must be a bool")
        if not isinstance(self.can_be_prefix, bool):
            raise TypeError("can_be_prefix must be a bool")


@dataclass(frozen=True, slots=True)
class DataPacket:

    name: str
    size_bytes: int = 1
    freshness_period: float | None = None
    produced_at: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", canonicalize_name(self.name))
        object.__setattr__(self, "size_bytes", _positive_size(self.size_bytes))
        object.__setattr__(
            self, "freshness_period", _freshness(self.freshness_period)
        )
        object.__setattr__(
            self, "produced_at", _finite_time(self.produced_at, "produced_at")
        )

    @property
    def freshness_period_ms(self) -> float | None:

        return (
            None
            if self.freshness_period is None
            else self.freshness_period * 1000.0
        )


@dataclass(frozen=True, slots=True)
class RequestEvent:

    timestamp: float
    node_id: str
    name: str
    size_bytes: int = 1
    freshness_period: float | None = None
    must_be_fresh: bool = True
    can_be_prefix: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "timestamp", _finite_time(self.timestamp, "timestamp")
        )
        if not isinstance(self.node_id, str):
            raise TypeError("node_id must be a string")
        if not self.node_id:
            raise ValueError("node_id cannot be empty")
        object.__setattr__(self, "name", canonicalize_name(self.name))
        object.__setattr__(self, "size_bytes", _positive_size(self.size_bytes))
        object.__setattr__(
            self, "freshness_period", _freshness(self.freshness_period)
        )
        if not isinstance(self.must_be_fresh, bool):
            raise TypeError("must_be_fresh must be a bool")
        if not isinstance(self.can_be_prefix, bool):
            raise TypeError("can_be_prefix must be a bool")

    def to_interest(self) -> Interest:

        return Interest(
            self.name,
            self.timestamp,
            self.must_be_fresh,
            self.can_be_prefix,
        )

    def to_data_packet(self) -> DataPacket:

        return DataPacket(
            self.name,
            self.size_bytes,
            self.freshness_period,
            self.timestamp,
        )

    @property
    def interest(self) -> Interest:
        return self.to_interest()

    @property
    def packet(self) -> DataPacket:
        return self.to_data_packet()


__all__ = ["DataPacket", "Interest", "RequestEvent"]
