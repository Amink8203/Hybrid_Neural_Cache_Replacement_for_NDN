
from __future__ import annotations

from dataclasses import dataclass

from .names import split_components
from .packets import DataPacket, Interest


@dataclass(slots=True)
class CacheEntry:

    packet: DataPacket
    inserted_at: float
    last_access: float
    hit_count: int = 0

    def is_fresh(self, now: float) -> bool:
        period = self.packet.freshness_period
        if period is None:
            return True
        return now < self.inserted_at + period


class ContentStore:
    def __init__(self, max_items: int | None, max_bytes: int | None = None) -> None:
        if max_items is not None:
            if isinstance(max_items, bool) or not isinstance(max_items, int):
                raise TypeError("max_items must be an integer or None")
            if max_items < 0:
                raise ValueError("max_items cannot be negative")
        if max_bytes is not None:
            if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
                raise TypeError("max_bytes must be an integer or None")
            if max_bytes < 0:
                raise ValueError("max_bytes cannot be negative")
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._entries: dict[str, CacheEntry] = {}
        self.current_bytes = 0

    def __len__(self) -> int:
        return len(self._entries)

    def entries(self) -> tuple[CacheEntry, ...]:

        return tuple(self._entries[name] for name in sorted(self._entries))

    def get(self, name: str) -> CacheEntry | None:

        from .names import canonicalize_name

        return self._entries.get(canonicalize_name(name))

    def lookup(self, interest: Interest) -> CacheEntry | None:

        candidates: list[CacheEntry] = []
        exact = self._entries.get(interest.name)
        if exact is not None:
            candidates.append(exact)
        if interest.can_be_prefix:
            prefix = split_components(interest.name)
            for name in sorted(self._entries):
                if name == interest.name:
                    continue
                components = split_components(name)
                if components[: len(prefix)] == prefix:
                    candidates.append(self._entries[name])
        for entry in candidates:
            if interest.must_be_fresh and not entry.is_fresh(interest.timestamp):
                continue
            entry.last_access = interest.timestamp
            entry.hit_count += 1
            return entry
        return None

    def is_cacheable(self, packet: DataPacket) -> bool:
        if self.max_items == 0:
            return False
        if self.max_bytes == 0:
            return False
        if self.max_bytes is not None and packet.size_bytes > self.max_bytes:
            return False
        return True

    def has_capacity(self, packet: DataPacket) -> bool:

        if not self.is_cacheable(packet):
            return False
        previous = self._entries.get(packet.name)
        item_delta = 0 if previous is not None else 1
        byte_delta = packet.size_bytes - (
            previous.packet.size_bytes if previous is not None else 0
        )
        if (
            self.max_items is not None
            and len(self._entries) + item_delta > self.max_items
        ):
            return False
        if (
            self.max_bytes is not None
            and self.current_bytes + byte_delta > self.max_bytes
        ):
            return False
        return True

    def insert(self, packet: DataPacket, *, now: float) -> CacheEntry:

        if not self.has_capacity(packet):
            raise OverflowError("Content Store capacity exceeded")
        previous = self._entries.get(packet.name)
        if previous is not None:
            self.current_bytes -= previous.packet.size_bytes
        entry = CacheEntry(
            packet=packet,
            inserted_at=float(now),
            last_access=float(now),
            hit_count=0,
        )
        self._entries[packet.name] = entry
        self.current_bytes += packet.size_bytes
        return entry

    def remove(self, name: str) -> CacheEntry | None:
        from .names import canonicalize_name

        entry = self._entries.pop(canonicalize_name(name), None)
        if entry is not None:
            self.current_bytes -= entry.packet.size_bytes
        return entry

    def clear(self) -> None:
        self._entries.clear()
        self.current_bytes = 0

    def validate(self) -> None:
        if self.max_items is not None and len(self) > self.max_items:
            raise AssertionError("Content Store item capacity violated")
        expected_bytes = sum(
            entry.packet.size_bytes for entry in self._entries.values()
        )
        if self.current_bytes != expected_bytes:
            raise AssertionError("Content Store byte accounting is inconsistent")
        if self.max_bytes is not None and self.current_bytes > self.max_bytes:
            raise AssertionError("Content Store byte capacity violated")


__all__ = ["CacheEntry", "ContentStore"]
