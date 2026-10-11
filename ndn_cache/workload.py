from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .packets import RequestEvent


TRACE_COLUMNS = (
    "timestamp",
    "node_id",
    "name",
    "size_bytes",
    "freshness_period",
    "must_be_fresh",
    "can_be_prefix",
)


@dataclass(frozen=True, slots=True)
class SyntheticWorkloadConfig:
    num_requests: int = 10_000
    catalog_size: int = 1_000
    num_nodes: int = 4
    zipf_alpha: float = 1.1
    temporal_locality: float = 0.35
    locality_window: int = 32
    locality_decay: float = 0.82
    drift_interval: int | None = 1_000
    drift_fraction: float = 0.10
    request_rate: float = 20.0
    start_time: float = 0.0
    min_object_size: int = 512
    max_object_size: int = 64 * 1024
    fixed_object_size: int | None = None
    min_freshness_period: float = 5.0
    max_freshness_period: float = 120.0
    name_prefix: str = "/dataset/item"
    num_name_categories: int = 1
    node_ids: tuple[str, ...] | None = None
    node_weights: tuple[float, ...] | None = None
    seed: int = 0
    must_be_fresh: bool = True

    def __post_init__(self) -> None:
        if self.num_requests < 0:
            raise ValueError("num_requests must be non-negative")
        if self.catalog_size <= 0:
            raise ValueError("catalog_size must be positive")
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if not math.isfinite(self.zipf_alpha) or self.zipf_alpha <= 0:
            raise ValueError("zipf_alpha must be finite and positive")
        if not 0.0 <= self.temporal_locality <= 1.0:
            raise ValueError("temporal_locality must be in [0, 1]")
        if self.locality_window <= 0:
            raise ValueError("locality_window must be positive")
        if not 0.0 < self.locality_decay <= 1.0:
            raise ValueError("locality_decay must be in (0, 1]")
        if self.drift_interval is not None and self.drift_interval <= 0:
            raise ValueError("drift_interval must be positive or None")
        if not 0.0 <= self.drift_fraction <= 1.0:
            raise ValueError("drift_fraction must be in [0, 1]")
        if not math.isfinite(self.request_rate) or self.request_rate <= 0:
            raise ValueError("request_rate must be finite and positive")
        if not math.isfinite(self.start_time):
            raise ValueError("start_time must be finite")
        if self.min_object_size <= 0:
            raise ValueError("min_object_size must be positive")
        if self.max_object_size < self.min_object_size:
            raise ValueError("max_object_size must be >= min_object_size")
        if self.fixed_object_size is not None:
            if (
                isinstance(self.fixed_object_size, bool)
                or not isinstance(self.fixed_object_size, int)
            ):
                raise TypeError("fixed_object_size must be an integer or None")
            if self.fixed_object_size <= 0:
                raise ValueError("fixed_object_size must be positive")
        if (
            not math.isfinite(self.min_freshness_period)
            or self.min_freshness_period < 0
        ):
            raise ValueError("min_freshness_period must be finite and non-negative")
        if (
            not math.isfinite(self.max_freshness_period)
            or self.max_freshness_period < self.min_freshness_period
        ):
            raise ValueError(
                "max_freshness_period must be finite and >= min_freshness_period"
            )
        if not self.name_prefix or not self.name_prefix.startswith("/"):
            raise ValueError("name_prefix must be a non-empty absolute NDN name")
        if self.num_name_categories <= 0:
            raise ValueError("num_name_categories must be positive")
        if self.node_ids is not None:
            if len(self.node_ids) != self.num_nodes:
                raise ValueError("node_ids length must equal num_nodes")
            if any(not str(node_id).strip() for node_id in self.node_ids):
                raise ValueError("node_ids cannot contain empty identifiers")
            if len(set(self.node_ids)) != len(self.node_ids):
                raise ValueError("node_ids must be unique")
        if self.node_weights is not None:
            if len(self.node_weights) != self.num_nodes:
                raise ValueError("node_weights length must equal num_nodes")
            if any(
                (not math.isfinite(weight) or weight < 0)
                for weight in self.node_weights
            ):
                raise ValueError("node_weights must be finite and non-negative")
            if sum(self.node_weights) <= 0:
                raise ValueError("node_weights must have a positive sum")

    @property
    def effective_node_ids(self) -> tuple[str, ...]:
        if self.node_ids is not None:
            return self.node_ids
        return tuple(f"node-{index}" for index in range(self.num_nodes))


def _sample_cdf(rng: np.random.Generator, cdf: np.ndarray) -> int:

    return min(int(np.searchsorted(cdf, rng.random(), side="right")), len(cdf) - 1)


def generate_synthetic_trace(
    config: SyntheticWorkloadConfig | None = None,
    **overrides: object,
) -> list[RequestEvent]:

    if config is None:
        config = SyntheticWorkloadConfig(**overrides)
    elif overrides:
        config = replace(config, **overrides)
    if not isinstance(config, SyntheticWorkloadConfig):
        raise TypeError("config must be a SyntheticWorkloadConfig")

    rng = np.random.default_rng(config.seed)
    node_ids = config.effective_node_ids

    if config.node_weights is None:
        node_cdf = np.linspace(
            1.0 / config.num_nodes, 1.0, config.num_nodes, dtype=np.float64
        )
    else:
        weights = np.asarray(config.node_weights, dtype=np.float64)
        node_cdf = np.cumsum(weights / weights.sum())
        node_cdf[-1] = 1.0

    ranks = np.arange(1, config.catalog_size + 1, dtype=np.float64)
    zipf_weights = np.power(ranks, -config.zipf_alpha)
    zipf_cdf = np.cumsum(zipf_weights / zipf_weights.sum())
    zipf_cdf[-1] = 1.0

    if config.fixed_object_size is not None:
        object_sizes = np.full(
            config.catalog_size,
            config.fixed_object_size,
            dtype=np.int64,
        )
    else:
        object_sizes = rng.integers(
            config.min_object_size,
            config.max_object_size + 1,
            size=config.catalog_size,
            dtype=np.int64,
        )
    if config.min_freshness_period == config.max_freshness_period:
        freshness = np.full(
            config.catalog_size,
            config.min_freshness_period,
            dtype=np.float64,
        )
    else:
        freshness = rng.uniform(
            config.min_freshness_period,
            config.max_freshness_period,
            size=config.catalog_size,
        )

    recent: list[deque[int]] = [
        deque(maxlen=config.locality_window) for _ in range(config.num_nodes)
    ]
    locality_weights = np.power(
        config.locality_decay,
        np.arange(config.locality_window, dtype=np.float64),
    )
    locality_cdfs: dict[int, np.ndarray] = {}

    prefix = config.name_prefix.rstrip("/")
    width = max(6, len(str(config.catalog_size - 1)))
    drift_stride = int(round(config.catalog_size * config.drift_fraction))
    if config.drift_fraction > 0 and drift_stride == 0:
        drift_stride = 1

    events: list[RequestEvent] = []
    timestamp = float(config.start_time)
    for index in range(config.num_requests):
        if index:
            timestamp += float(rng.exponential(1.0 / config.request_rate))

        node_index = _sample_cdf(rng, node_cdf)
        history = recent[node_index]
        use_local = (
            bool(history)
            and config.temporal_locality > 0
            and rng.random() < config.temporal_locality
        )
        if use_local:
            history_length = len(history)
            cdf = locality_cdfs.get(history_length)
            if cdf is None:
                weights = locality_weights[:history_length]
                cdf = np.cumsum(weights / weights.sum())
                cdf[-1] = 1.0
                locality_cdfs[history_length] = cdf
            age = _sample_cdf(rng, cdf)
            content_index = history[-1 - age]
        else:
            rank = _sample_cdf(rng, zipf_cdf)
            epoch = (
                0
                if config.drift_interval is None
                else index // config.drift_interval
            )
            content_index = (
                rank + (epoch * drift_stride)
            ) % config.catalog_size

        history.append(content_index)
        object_component = f"{content_index:0{width}d}"
        if config.num_name_categories > 1:
            category = content_index % config.num_name_categories
            object_name = (
                f"{prefix}/category-{category:02d}/{object_component}"
            )
        else:
            object_name = f"{prefix}/{object_component}"
        events.append(
            RequestEvent(
                timestamp=timestamp,
                node_id=node_ids[node_index],
                name=object_name,
                size_bytes=int(object_sizes[content_index]),
                freshness_period=float(freshness[content_index]),
                must_be_fresh=config.must_be_fresh,
                can_be_prefix=False,
            )
        )

    return events


generate_trace = generate_synthetic_trace


_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "time", "request_time"),
    "node_id": ("node_id", "node", "router_id", "consumer_id"),
    "name": ("name", "content_name", "ndn_name", "address"),
    "size_bytes": ("size_bytes", "size", "bytes", "content_size"),
    "freshness_period": (
        "freshness_period",
        "freshness",
        "freshness_seconds",
        "ttl",
    ),
    "freshness_period_ms": ("freshness_period_ms", "freshness_ms", "ttl_ms"),
    "must_be_fresh": ("must_be_fresh", "mbf"),
    "can_be_prefix": ("can_be_prefix", "prefix"),
}


def _normalise_column(name: str) -> str:
    return name.lstrip("\ufeff").strip().lower().replace("-", "_").replace(" ", "_")


def _resolve_columns(fieldnames: Sequence[str] | None) -> dict[str, str]:
    if not fieldnames:
        raise ValueError("trace CSV has no header")
    available = {_normalise_column(field): field for field in fieldnames}
    resolved: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in available:
                resolved[canonical] = available[alias]
                break
    missing = [
        field for field in ("timestamp", "node_id", "name") if field not in resolved
    ]
    if missing:
        raise ValueError(
            "trace CSV is missing required column(s): " + ", ".join(missing)
        )
    return resolved


def _parse_bool(value: str | None, *, default: bool, field: str, row: int) -> bool:
    if value is None or not value.strip():
        return default
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "y", "on"}:
        return True
    if normalised in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"row {row}: invalid {field} boolean {value!r}")


def _optional_float(value: str | None, *, field: str, row: int) -> float | None:
    if value is None or not value.strip():
        return None
    if value.strip().lower() in {"none", "null"}:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row}: invalid {field} {value!r}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"row {row}: {field} must be finite and non-negative")
    return parsed


def iter_trace_csv(
    path: str | os.PathLike[str],
    *,
    require_monotonic: bool = True,
) -> Iterator[RequestEvent]:

    previous_timestamp: float | None = None
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = _resolve_columns(reader.fieldnames)
        for row_number, row in enumerate(reader, start=2):
            if row is None or not any((value or "").strip() for value in row.values()):
                continue
            try:
                timestamp = float(row[columns["timestamp"]])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"row {row_number}: invalid timestamp "
                    f"{row.get(columns['timestamp'])!r}"
                ) from exc
            if not math.isfinite(timestamp):
                raise ValueError(f"row {row_number}: timestamp must be finite")
            if (
                require_monotonic
                and previous_timestamp is not None
                and timestamp < previous_timestamp
            ):
                raise ValueError(
                    f"row {row_number}: timestamp {timestamp} precedes "
                    f"previous timestamp {previous_timestamp}"
                )

            node_id = (row.get(columns["node_id"]) or "").strip()
            name = (row.get(columns["name"]) or "").strip()
            if not node_id:
                raise ValueError(f"row {row_number}: node_id cannot be empty")
            if not name:
                raise ValueError(f"row {row_number}: name cannot be empty")

            size_column = columns.get("size_bytes")
            raw_size = row.get(size_column, "1") if size_column else "1"
            try:
                size_bytes = int(raw_size or "1")
            except ValueError as exc:
                raise ValueError(
                    f"row {row_number}: invalid size_bytes {raw_size!r}"
                ) from exc
            if size_bytes <= 0:
                raise ValueError(f"row {row_number}: size_bytes must be positive")

            if "freshness_period" in columns:
                freshness_period = _optional_float(
                    row.get(columns["freshness_period"]),
                    field="freshness_period",
                    row=row_number,
                )
            elif "freshness_period_ms" in columns:
                freshness_ms = _optional_float(
                    row.get(columns["freshness_period_ms"]),
                    field="freshness_period_ms",
                    row=row_number,
                )
                freshness_period = (
                    None if freshness_ms is None else freshness_ms / 1_000.0
                )
            else:
                freshness_period = None

            must_be_fresh = _parse_bool(
                row.get(columns["must_be_fresh"])
                if "must_be_fresh" in columns
                else None,
                default=True,
                field="must_be_fresh",
                row=row_number,
            )
            can_be_prefix = _parse_bool(
                row.get(columns["can_be_prefix"])
                if "can_be_prefix" in columns
                else None,
                default=False,
                field="can_be_prefix",
                row=row_number,
            )
            previous_timestamp = timestamp
            yield RequestEvent(
                timestamp=timestamp,
                node_id=node_id,
                name=name,
                size_bytes=size_bytes,
                freshness_period=freshness_period,
                must_be_fresh=must_be_fresh,
                can_be_prefix=can_be_prefix,
            )


def load_trace_csv(
    path: str | os.PathLike[str],
    *,
    require_monotonic: bool = True,
) -> list[RequestEvent]:

    return list(iter_trace_csv(path, require_monotonic=require_monotonic))


def _event_field(event: RequestEvent, name: str, default: object = None) -> object:
    return getattr(event, name, default)


def save_trace_csv(
    events: Iterable[RequestEvent],
    path: str | os.PathLike[str],
    *,
    atomic: bool = True,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    if atomic:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            text=True,
        )
        os.close(descriptor)
        output_path = Path(raw_path)
        temporary_path = output_path
    else:
        output_path = destination

    try:
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRACE_COLUMNS)
            writer.writeheader()
            previous_timestamp: float | None = None
            for index, event in enumerate(events, start=1):
                timestamp = float(_event_field(event, "timestamp"))
                node_id = str(_event_field(event, "node_id", "")).strip()
                name = str(_event_field(event, "name", "")).strip()
                size_bytes = int(_event_field(event, "size_bytes", 1))
                freshness_period = _event_field(event, "freshness_period")
                if not math.isfinite(timestamp):
                    raise ValueError(f"event {index}: timestamp must be finite")
                if previous_timestamp is not None and timestamp < previous_timestamp:
                    raise ValueError(f"event {index}: timestamps are not monotonic")
                if not node_id or not name:
                    raise ValueError(f"event {index}: node_id and name are required")
                if size_bytes <= 0:
                    raise ValueError(f"event {index}: size_bytes must be positive")
                if freshness_period is not None:
                    freshness_period = float(freshness_period)
                    if (
                        not math.isfinite(freshness_period)
                        or freshness_period < 0
                    ):
                        raise ValueError(
                            f"event {index}: freshness_period must be "
                            "finite and non-negative"
                        )
                writer.writerow(
                    {
                        "timestamp": format(timestamp, ".17g"),
                        "node_id": node_id,
                        "name": name,
                        "size_bytes": size_bytes,
                        "freshness_period": (
                            ""
                            if freshness_period is None
                            else format(freshness_period, ".17g")
                        ),
                        "must_be_fresh": int(
                            bool(_event_field(event, "must_be_fresh", True))
                        ),
                        "can_be_prefix": int(
                            bool(_event_field(event, "can_be_prefix", False))
                        ),
                    }
                )
                previous_timestamp = timestamp
        if atomic:
            os.replace(output_path, destination)
            temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
    return destination


__all__ = [
    "TRACE_COLUMNS",
    "SyntheticWorkloadConfig",
    "generate_synthetic_trace",
    "generate_trace",
    "iter_trace_csv",
    "load_trace_csv",
    "save_trace_csv",
]
