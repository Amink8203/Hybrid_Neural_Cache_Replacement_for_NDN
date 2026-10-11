
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import torch

from . import __version__
from .agent import NeuralCacheAgent
from .baselines import (
    BaselineCacheController,
    BaselinePolicy,
    evaluate_baselines,
)
from .checkpoint import load_checkpoint, save_checkpoint
from .config import ModelConfig, TrainingConfig
from .model import NeuralCacheModel
from .packets import RequestEvent
from .training import (
    fit,
    pretrain_reuse_model,
    run_neural_trace,
    split_trace,
)
from .topology import (
    TopologyKind,
    TopologySimulator,
    build_topology,
    collect_router_observations,
    fit_topology,
)
from .utils import resolve_device, seed_everything, write_json
from .workload import (
    SyntheticWorkloadConfig,
    generate_synthetic_trace,
    load_trace_csv,
)


def _add_workload_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("request trace")
    group.add_argument(
        "--trace",
        type=Path,
        help="CSV trace; omit to generate a deterministic synthetic trace",
    )
    group.add_argument("--requests", type=int, default=2_000)
    group.add_argument("--catalog-size", type=int, default=400)
    group.add_argument(
        "--nodes",
        type=int,
        default=2,
        help=(
            "number of logical nodes; topology mode interprets this as total "
            "routers including the producer"
        ),
    )
    group.add_argument("--zipf-alpha", type=float, default=1.1)
    group.add_argument("--temporal-locality", type=float, default=0.35)
    group.add_argument("--locality-window", type=int, default=32)
    group.add_argument("--locality-decay", type=float, default=0.82)
    group.add_argument("--drift-interval", type=int, default=500)
    group.add_argument("--drift-fraction", type=float, default=0.10)
    group.add_argument("--request-rate", type=float, default=20.0)
    group.add_argument("--min-object-size", type=int, default=512)
    group.add_argument("--max-object-size", type=int, default=64 * 1024)
    group.add_argument(
        "--packet-size",
        type=int,
        default=None,
        help=(
            "force every Data packet to the same positive size; applies to "
            "synthetic and imported traces"
        ),
    )
    group.add_argument("--min-freshness", type=float, default=5.0)
    group.add_argument("--max-freshness", type=float, default=120.0)
    group.add_argument("--name-categories", type=int, default=1)
    group.add_argument(
        "--node-skew-alpha",
        type=float,
        default=0.0,
        help="consumer demand skew; zero gives uniform node selection",
    )
    group.add_argument("--seed", type=int, default=7)


def _add_cache_arguments(
    parser: argparse.ArgumentParser,
    *,
    defaults: bool,
) -> None:
    group = parser.add_argument_group("cache")
    group.add_argument(
        "--cache-size",
        type=int,
        default=32 if defaults else None,
        help="maximum Data packets per node",
    )
    group.add_argument(
        "--byte-only",
        action="store_true",
        help="disable the item limit; requires a byte capacity",
    )
    group.add_argument(
        "--cache-bytes",
        type=int,
        default=None,
        help="optional byte limit per node in addition to --cache-size",
    )


def _add_neural_training_arguments(parser: argparse.ArgumentParser) -> None:

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--vocab-size", type=int, default=4096)
    model_group.add_argument("--max-name-components", type=int, default=16)
    model_group.add_argument("--embedding-dim", type=int, default=32)
    model_group.add_argument("--name-hidden-dim", type=int, default=48)
    model_group.add_argument("--context-hidden-dim", type=int, default=64)
    model_group.add_argument("--head-hidden-dim", type=int, default=64)
    model_group.add_argument("--dropout", type=float, default=0.0)
    model_group.add_argument("--popularity-prior-weight", type=float, default=1.0)
    model_group.add_argument("--recency-prior-weight", type=float, default=0.0)
    model_group.add_argument("--residual-scale", type=float, default=0.05)
    rl_group = parser.add_argument_group("learning")
    rl_group.add_argument("--learning-rate", type=float, default=3e-4)
    rl_group.add_argument("--gamma", type=float, default=0.99)
    rl_group.add_argument("--rollout-length", type=int, default=256)
    rl_group.add_argument("--temperature", type=float, default=0.7)
    rl_group.add_argument("--entropy-coefficient", type=float, default=0.0)
    rl_group.add_argument("--auxiliary-coefficient", type=float, default=0.25)
    rl_group.add_argument("--prediction-horizon", type=int, default=32)
    rl_group.add_argument("--calibration-samples", type=int, default=16)
    rl_group.add_argument(
        "--replacement-shortlist-size",
        type=int,
        default=32,
        help="score only the K least-recently-used residents per replacement",
    )
    rl_group.add_argument("--gradient-clip", type=float, default=1.0)
    rl_group.add_argument("--recency-scale", type=float, default=100.0)
    rl_group.add_argument("--popularity-decay", type=float, default=32.0)
    rl_group.add_argument(
        "--allow-bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="let the policy reject newly returned Data instead of evicting",
    )


def _parse_topology_recency_prior(value: str) -> tuple[str, float]:

    if value.count("=") != 1:
        raise argparse.ArgumentTypeError(
            "expected KIND=FLOAT, for example mesh=0.5"
        )
    raw_kind, raw_weight = (part.strip() for part in value.split("=", 1))
    if not raw_kind or not raw_weight:
        raise argparse.ArgumentTypeError(
            "expected KIND=FLOAT, for example mesh=0.5"
        )
    try:
        kind = TopologyKind.parse(raw_kind).value
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid recency-prior weight {raw_weight!r}"
        ) from exc
    if not math.isfinite(weight) or weight < 0.0:
        raise argparse.ArgumentTypeError(
            "recency-prior weight must be a finite nonnegative number"
        )
    return kind, weight


def _force_packet_size(
    events: Sequence[RequestEvent],
    packet_size: int | None,
) -> list[RequestEvent]:
    if packet_size is None:
        return list(events)
    if isinstance(packet_size, bool) or packet_size <= 0:
        raise ValueError("--packet-size must be a positive integer")
    return [
        replace(event, size_bytes=packet_size)
        for event in events
    ]


def _load_or_generate_trace(args: argparse.Namespace):
    if args.trace is not None:
        return _force_packet_size(
            load_trace_csv(args.trace),
            args.packet_size,
        )
    drift_interval = None if args.drift_interval == 0 else args.drift_interval
    config = SyntheticWorkloadConfig(
        num_requests=args.requests,
        catalog_size=args.catalog_size,
        num_nodes=args.nodes,
        zipf_alpha=args.zipf_alpha,
        temporal_locality=args.temporal_locality,
        locality_window=args.locality_window,
        locality_decay=args.locality_decay,
        drift_interval=drift_interval,
        drift_fraction=args.drift_fraction,
        request_rate=args.request_rate,
        min_object_size=args.min_object_size,
        max_object_size=args.max_object_size,
        fixed_object_size=args.packet_size,
        min_freshness_period=args.min_freshness,
        max_freshness_period=args.max_freshness,
        num_name_categories=args.name_categories,
        node_weights=(
            None
            if args.node_skew_alpha == 0
            else tuple(
                (index + 1) ** (-args.node_skew_alpha)
                for index in range(args.nodes)
            )
        ),
        seed=args.seed,
    )
    return generate_synthetic_trace(config)


def _load_or_generate_topology_trace(
    args: argparse.Namespace,
    *,
    consumer_nodes: tuple[str, ...],
):

    if args.trace is not None:
        return (
            _force_packet_size(
                load_trace_csv(args.trace),
                args.packet_size,
            ),
            None,
        )
    if not consumer_nodes:
        raise ValueError("topology requires at least one non-producer requester")
    drift_interval = None if args.drift_interval == 0 else args.drift_interval
    config = SyntheticWorkloadConfig(
        num_requests=args.requests,
        catalog_size=args.catalog_size,
        num_nodes=len(consumer_nodes),
        node_ids=consumer_nodes,
        zipf_alpha=args.zipf_alpha,
        temporal_locality=args.temporal_locality,
        locality_window=args.locality_window,
        locality_decay=args.locality_decay,
        drift_interval=drift_interval,
        drift_fraction=args.drift_fraction,
        request_rate=args.request_rate,
        min_object_size=args.min_object_size,
        max_object_size=args.max_object_size,
        fixed_object_size=args.packet_size,
        min_freshness_period=args.min_freshness,
        max_freshness_period=args.max_freshness,
        num_name_categories=args.name_categories,
        node_weights=(
            None
            if args.node_skew_alpha == 0
            else tuple(
                (index + 1) ** (-args.node_skew_alpha)
                for index in range(len(consumer_nodes))
            )
        ),
        name_prefix=args.producer_prefix,
        seed=args.seed,
    )
    return generate_synthetic_trace(config), config


def _model_config_from_args(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        vocab_size=args.vocab_size,
        max_name_components=args.max_name_components,
        embedding_dim=args.embedding_dim,
        name_hidden_dim=args.name_hidden_dim,
        context_hidden_dim=args.context_hidden_dim,
        head_hidden_dim=args.head_hidden_dim,
        dropout=args.dropout,
        popularity_prior_weight=args.popularity_prior_weight,
        recency_prior_weight=args.recency_prior_weight,
        residual_scale=args.residual_scale,
    )


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        rollout_length=args.rollout_length,
        temperature=args.temperature,
        entropy_coefficient=args.entropy_coefficient,
        auxiliary_coefficient=args.auxiliary_coefficient,
        prediction_horizon=args.prediction_horizon,
        calibration_samples=args.calibration_samples,
        replacement_shortlist_size=args.replacement_shortlist_size,
        gradient_clip=args.gradient_clip,
        allow_bypass=args.allow_bypass,
        recency_scale=args.recency_scale,
        popularity_decay=args.popularity_decay,
    )


def _plot_history(history: dict[str, object], destination: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = history["epochs"]
    assert isinstance(epochs, list)
    x = [row["epoch"] for row in epochs]
    train = [row["train_hit_ratio"] for row in epochs]
    validation = [row["validation_hit_ratio"] for row in epochs]
    losses = [row["mean_loss"] for row in epochs]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(x, train, marker="o", label="train hit ratio")
    if any(value is not None for value in validation):
        axis.plot(x, validation, marker="s", label="validation hit ratio")
    axis.set_xlabel("epoch")
    axis.set_ylabel("hit ratio")
    axis.set_ylim(0.0, 1.0)
    axis.grid(alpha=0.25)
    second = axis.twinx()
    second.plot(x, losses, color="tab:red", alpha=0.5, label="mean loss")
    second.set_ylabel("training loss")
    lines = axis.lines + second.lines
    axis.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=150)
    plt.close(figure)


def _comparison_rows(
    neural: dict[str, object],
    baselines: dict[str, object],
) -> list[dict[str, object]]:
    neural_metrics = neural["metrics"]
    assert isinstance(neural_metrics, dict)
    rows = [
        {
            "policy": "neural",
            "hit_ratio": neural_metrics["hit_ratio"],
            "byte_hit_ratio": neural_metrics["byte_hit_ratio"],
            "evictions": neural_metrics["evictions"],
            "rejections": neural_metrics["rejections"],
        }
    ]
    for policy, raw_result in baselines.items():
        result = raw_result
        metrics = result.metrics.as_dict()
        rows.append(
            {
                "policy": policy,
                "hit_ratio": metrics["hit_ratio"],
                "byte_hit_ratio": metrics["byte_hit_ratio"],
                "evictions": metrics["evictions"],
                "rejections": metrics["rejections"],
            }
        )
    return rows


def _print_rows(rows: Sequence[dict[str, object]]) -> None:
    print("policy       hit ratio   byte hit   evictions   rejections")
    print("-----------  ----------  ---------  ----------  ----------")
    for row in rows:
        print(
            f"{str(row['policy']):<11}  "
            f"{float(row['hit_ratio']):>10.4f}  "
            f"{float(row['byte_hit_ratio']):>9.4f}  "
            f"{int(row['evictions']):>10d}  "
            f"{int(row['rejections']):>10d}"
        )

def _ensure_distinct_paths(*paths: Path | None) -> None:
    resolved = [path.resolve() for path in paths if path is not None]
    if len(resolved) != len(set(resolved)):
        raise ValueError("input and output paths must be distinct")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_sha256(events: Sequence[RequestEvent]) -> str:

    digest = hashlib.sha256()
    for event in events:
        row = [
            event.timestamp,
            event.node_id,
            event.name,
            event.size_bytes,
            event.freshness_period,
            event.must_be_fresh,
            event.can_be_prefix,
        ]
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()

def command_train(args: argparse.Namespace) -> int:
    _ensure_distinct_paths(args.trace, args.checkpoint, args.output, args.plot)
    seed_everything(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    trace = _load_or_generate_trace(args)
    cache_items = None if args.byte_only else args.cache_size
    if cache_items is None and args.cache_bytes is None:
        raise ValueError("byte-only mode requires --cache-bytes")
    train_trace, validation_trace, test_trace = split_trace(
        trace,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    if not test_trace:
        raise ValueError(
            "trace split has no held-out test requests; lower "
            "--train-fraction/--validation-fraction or use a longer trace"
        )
    model_config = _model_config_from_args(args)
    training_config = _training_config_from_args(args)
    model = NeuralCacheModel(model_config)
    agent = NeuralCacheAgent(
        model,
        training_config,
        cache_items=cache_items,
        cache_bytes=args.cache_bytes,
        device=device,
        seed=args.seed,
    )
    fit_result = fit(
        agent,
        train_trace,
        validation_events=validation_trace,
        epochs=args.epochs,
    )
    test_result = run_neural_trace(
        agent,
        test_trace,
        training=False,
        collect_calibration=True,
    )
    baseline_results = evaluate_baselines(
        test_trace,
        policies=args.baselines,
        cache_items=cache_items,
        cache_bytes=args.cache_bytes,
        seed=args.seed,
    )
    report = {
        "trace": {
            "source": str(args.trace) if args.trace else "synthetic",
            "total_requests": len(trace),
            "train_requests": len(train_trace),
            "validation_requests": len(validation_trace),
            "test_requests": len(test_trace),
            "seed": args.seed,
        },
        "cache": {
            "items_per_node": cache_items,
            "bytes_per_node": args.cache_bytes,
        },
        "model_config": model_config.to_dict(),
        "training_config": training_config.to_dict(),
        "training": fit_result.as_dict(),
        "test": test_result.as_dict(),
        "baselines": {
            name: result.as_dict() for name, result in baseline_results.items()
        },
    }
    rows = _comparison_rows(test_result.as_dict(), baseline_results)
    report["comparison"] = rows
    save_checkpoint(
        args.checkpoint,
        agent.model,
        training_config=training_config,
        optimizer=agent.optimizer,
        metadata={
            "cache_items": cache_items,
            "cache_bytes": args.cache_bytes,
            "seed": args.seed,
            "training": fit_result.as_dict(),
            "test": test_result.as_dict(),
        },
    )
    write_json(args.output, report)
    if args.plot is not None:
        _plot_history(fit_result.as_dict(), args.plot)
    _print_rows(rows)
    print(f"\ncheckpoint: {args.checkpoint}")
    print(f"report:     {args.output}")
    if args.plot is not None:
        print(f"plot:       {args.plot}")
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    _ensure_distinct_paths(args.trace, args.checkpoint, args.output)
    seed_everything(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    model, training_config, metadata = load_checkpoint(
        args.checkpoint,
        device=device,
    )
    if training_config is None:
        training_config = TrainingConfig()
    if args.replacement_shortlist_size is not None:
        training_config.replacement_shortlist_size = (
            args.replacement_shortlist_size
        )
        training_config.validate()
    if args.byte_only:
        cache_items = None
    else:
        cache_items = args.cache_size
        if cache_items is None:
            saved_items = metadata.get("cache_items", 32)
            cache_items = (
                None if saved_items is None else int(saved_items)
            )
    cache_bytes = (
        args.cache_bytes
        if args.cache_bytes is not None
        else metadata.get("cache_bytes")
    )
    if cache_items is None and cache_bytes is None:
        raise ValueError("byte-only mode requires --cache-bytes")
    trace = _load_or_generate_trace(args)
    agent = NeuralCacheAgent(
        model,
        training_config,
        cache_items=cache_items,
        cache_bytes=cache_bytes,
        device=device,
        seed=args.seed,
    )
    neural_result = run_neural_trace(
        agent,
        trace,
        training=False,
        collect_calibration=True,
    )
    baseline_results = evaluate_baselines(
        trace,
        policies=args.baselines,
        cache_items=cache_items,
        cache_bytes=cache_bytes,
        seed=args.seed,
    )
    rows = _comparison_rows(neural_result.as_dict(), baseline_results)
    report = {
        "checkpoint": str(args.checkpoint),
        "requests": len(trace),
        "cache": {
            "items_per_node": cache_items,
            "bytes_per_node": cache_bytes,
            "neural_replacement_shortlist": (
                training_config.replacement_shortlist_size
            ),
        },
        "neural": neural_result.as_dict(),
        "baselines": {
            name: result.as_dict() for name, result in baseline_results.items()
        },
        "comparison": rows,
    }
    write_json(args.output, report)
    _print_rows(rows)
    print(f"\nreport: {args.output}")
    return 0


def command_topology(args: argparse.Namespace) -> int:

    _ensure_distinct_paths(args.trace, args.checkpoint, args.output)
    if len(args.topologies) != len(set(args.topologies)):
        raise ValueError("--topologies cannot contain duplicates")
    seed_everything(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    model, training_config, metadata = load_checkpoint(
        args.checkpoint,
        device=device,
    )
    if training_config is None:
        training_config = TrainingConfig()
    if args.replacement_shortlist_size is not None:
        training_config.replacement_shortlist_size = (
            args.replacement_shortlist_size
        )
        training_config.validate()

    if args.byte_only:
        cache_items = None
    else:
        cache_items = args.cache_size
        if cache_items is None:
            saved_items = metadata.get("cache_items", 32)
            cache_items = None if saved_items is None else int(saved_items)
    cache_bytes = (
        args.cache_bytes
        if args.cache_bytes is not None
        else metadata.get("cache_bytes")
    )
    if cache_items is None and cache_bytes is None:
        raise ValueError("byte-only mode requires --cache-bytes")

    prototype = build_topology(
        args.topologies[0],
        num_nodes=args.nodes,
        producer_prefix=args.producer_prefix,
        producer_node=args.producer_node,
        star_center=args.star_center,
    )
    consumer_nodes = tuple(
        node for node in prototype.nodes if node != args.producer_node
    )
    trace, workload_config = _load_or_generate_topology_trace(
        args,
        consumer_nodes=consumer_nodes,
    )

    topology_results: dict[str, object] = {}
    topology_baselines: dict[str, dict[str, object]] = {}
    comparison: list[dict[str, object]] = []
    for raw_kind in args.topologies:
        topology = build_topology(
            raw_kind,
            num_nodes=args.nodes,
            producer_prefix=args.producer_prefix,
            producer_node=args.producer_node,
            star_center=args.star_center,
        )
        run_model = copy.deepcopy(model)
        agent = NeuralCacheAgent(
            run_model,
            training_config,
            cache_items=cache_items,
            cache_bytes=cache_bytes,
            device=device,
            seed=args.seed,
        )
        result = TopologySimulator(topology, agent).run(trace)
        kind = topology.kind.value
        topology_results[kind] = result.as_dict(
            include_requests=args.include_requests
        )
        metrics = result.metrics
        evictions = sum(
            item.evictions for item in result.per_node_cache.values()
        )
        comparison.append(
            {
                "topology": kind,
                "policy": "neural",
                "cache_hit_ratio": metrics.cache_hit_ratio,
                "byte_cache_hit_ratio": metrics.byte_cache_hit_ratio,
                "producer_fetches": metrics.producer_fetches,
                "mean_retrieval_hops": metrics.mean_retrieval_hops,
                "hop_savings_ratio": metrics.hop_savings_ratio,
                "router_lookups": metrics.router_lookups,
                "evictions": evictions,
            }
        )
        topology_baselines[kind] = {}
        for raw_policy in args.baselines:
            controller = BaselineCacheController(
                raw_policy,
                cache_items=cache_items,
                cache_bytes=cache_bytes,
                seed=args.seed,
            )
            baseline_result = TopologySimulator(topology, controller).run(trace)
            policy = controller.policy.value
            topology_baselines[kind][policy] = baseline_result.as_dict(
                include_requests=args.include_requests
            )
            baseline_metrics = baseline_result.metrics
            baseline_evictions = sum(
                item.evictions for item in baseline_result.per_node_cache.values()
            )
            comparison.append(
                {
                    "topology": kind,
                    "policy": policy,
                    "cache_hit_ratio": baseline_metrics.cache_hit_ratio,
                    "byte_cache_hit_ratio": baseline_metrics.byte_cache_hit_ratio,
                    "producer_fetches": baseline_metrics.producer_fetches,
                    "mean_retrieval_hops": baseline_metrics.mean_retrieval_hops,
                    "hop_savings_ratio": baseline_metrics.hop_savings_ratio,
                    "router_lookups": baseline_metrics.router_lookups,
                    "evictions": baseline_evictions,
                }
            )

    report = {
        "package_version": __version__,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _file_sha256(args.checkpoint),
        },
        "trace": {
            "source": str(args.trace) if args.trace else "synthetic",
            "consumer_requests": len(trace),
            "seed": args.seed,
            "sha256": _trace_sha256(trace),
            "input_file_sha256": (
                _file_sha256(args.trace) if args.trace is not None else None
            ),
            "synthetic_config": (
                asdict(workload_config) if workload_config is not None else None
            ),
        },
        "network": {
            "routers": args.nodes,
            "producer_node": args.producer_node,
            "producer_prefix": args.producer_prefix,
            "star_center": args.star_center,
            "placement": "leave-copy-everywhere",
            "routing": "component-prefix shortest-path FIB",
            "pit_aggregation": False,
        },
        "cache": {
            "items_per_router": cache_items,
            "bytes_per_router": cache_bytes,
            "neural_replacement_shortlist": (
                training_config.replacement_shortlist_size
            ),
        },
        "topologies": topology_results,
        "baselines": topology_baselines,
        "comparison": comparison,
    }
    write_json(args.output, report)

    print(
        "topology  policy    cache hit  byte hit  producer  mean hops  hop saved  "
        "lookups  evictions"
    )
    print(
        "--------  --------  ---------  --------  --------  ---------  ---------  "
        "-------  ---------"
    )
    for row in comparison:
        print(
            f"{str(row['topology']):<8}  "
            f"{str(row['policy']):<8}  "
            f"{float(row['cache_hit_ratio']):>9.6f}  "
            f"{float(row['byte_cache_hit_ratio']):>8.6f}  "
            f"{int(row['producer_fetches']):>8d}  "
            f"{float(row['mean_retrieval_hops']):>9.3f}  "
            f"{float(row['hop_savings_ratio']):>9.6f}  "
            f"{int(row['router_lookups']):>7d}  "
            f"{int(row['evictions']):>9d}"
        )
    print(f"\nreport: {args.output}")
    return 0


def command_topology_train(args: argparse.Namespace) -> int:

    _ensure_distinct_paths(args.trace, args.output)
    if len(args.topologies) != len(set(args.topologies)):
        raise ValueError("--topologies cannot contain duplicates")
    recency_prior_overrides: dict[str, float] = {}
    for kind, weight in args.topology_recency_prior:
        if kind in recency_prior_overrides:
            raise ValueError(
                "--topology-recency-prior cannot override the same "
                f"topology twice: {kind}"
            )
        recency_prior_overrides[kind] = weight
    unselected_overrides = (
        set(recency_prior_overrides) - set(args.topologies)
    )
    if unselected_overrides:
        names = ", ".join(sorted(unselected_overrides))
        raise ValueError(
            "topology recency-prior overrides must also appear in "
            f"--topologies: {names}"
        )
    torch.set_num_threads(args.torch_threads)
    device = resolve_device(args.device)
    cache_items = None if args.byte_only else args.cache_size
    if cache_items is None and args.cache_bytes is None:
        raise ValueError("byte-only mode requires --cache-bytes")

    prototype = build_topology(
        args.topologies[0],
        num_nodes=args.nodes,
        producer_prefix=args.producer_prefix,
        producer_node=args.producer_node,
        star_center=args.star_center,
    )
    consumer_nodes = tuple(
        node for node in prototype.nodes if node != args.producer_node
    )
    seed_everything(args.seed)
    trace, workload_config = _load_or_generate_topology_trace(
        args,
        consumer_nodes=consumer_nodes,
    )
    train_trace, validation_trace, test_trace = split_trace(
        trace,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
    )
    if not test_trace:
        raise ValueError(
            "trace split has no held-out test requests; use a longer trace "
            "or lower the training/validation fractions"
        )
    if not validation_trace:
        raise ValueError(
            "topology training requires a nonempty validation split"
        )
    packet_sizes = sorted({event.size_bytes for event in trace})
    uniform_packet_size = (
        packet_sizes[0] if len(packet_sizes) == 1 else None
    )

    model_config = _model_config_from_args(args)
    training_config_template = _training_config_from_args(args)
    if training_config_template.calibration_samples <= 0:
        raise ValueError(
            "topology pretraining requires --calibration-samples > 0"
        )
    online_training_config = TrainingConfig.from_dict(
        training_config_template.to_dict()
    )
    online_training_config.auxiliary_coefficient = 0.0
    online_training_config.calibration_samples = 0
    topology_results: dict[str, object] = {}
    model_configs_by_topology: dict[str, dict[str, object]] = {}
    comparison: list[dict[str, object]] = []
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    trace_digest = _trace_sha256(trace)
    partition_digests = {
        "train": _trace_sha256(train_trace),
        "validation": _trace_sha256(validation_trace),
        "test": _trace_sha256(test_trace),
    }

    for raw_kind in args.topologies:
        seed_everything(args.seed)
        topology = build_topology(
            raw_kind,
            num_nodes=args.nodes,
            producer_prefix=args.producer_prefix,
            producer_node=args.producer_node,
            star_center=args.star_center,
        )
        kind = topology.kind.value
        topology_model_config = ModelConfig.from_dict(
            model_config.to_dict()
        )
        topology_model_config.recency_prior_weight = (
            recency_prior_overrides.get(
                kind,
                model_config.recency_prior_weight,
            )
        )
        topology_model_config.validate()
        model_configs_by_topology[kind] = (
            topology_model_config.to_dict()
        )
        experiment_spec = {
            "schema": "preaction-ranking-decayed-popularity-v1",
            "package_version": __version__,
            "topology": topology.as_dict(),
            "model_config": topology_model_config.to_dict(),
            "training_config": training_config_template.to_dict(),
            "cache_items": cache_items,
            "cache_bytes": args.cache_bytes,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "validation_fraction": args.validation_fraction,
            "partition_digests": partition_digests,
            "pretrain_epochs": args.pretrain_epochs,
            "pretrain_batch_length": args.pretrain_batch_length,
            "pretrain_learning_rate": args.pretrain_learning_rate,
            "pretrain_behavior": args.pretrain_behavior,
            "rl_epochs": args.epochs,
        }
        experiment_fingerprint = hashlib.sha256(
            json.dumps(
                experiment_spec,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        checkpoint = args.checkpoint_dir / f"{kind}.pt"
        pretrained_checkpoint = args.checkpoint_dir / f"{kind}-pretrained.pt"
        resumed = False
        final_metadata: dict[str, object] = {}
        pretraining_metadata: dict[str, object] = {}
        loaded_model = None
        loaded_config = None
        if args.resume and checkpoint.is_file() and pretrained_checkpoint.is_file():
            try:
                loaded_model, loaded_config, final_metadata = load_checkpoint(
                    checkpoint,
                    device=device,
                )
                pretrained_model, _, pretraining_metadata = load_checkpoint(
                    pretrained_checkpoint,
                    device=device,
                )
                resumed = (
                    final_metadata.get("experiment_fingerprint")
                    == experiment_fingerprint
                    and pretraining_metadata.get("experiment_fingerprint")
                    == experiment_fingerprint
                    and isinstance(final_metadata.get("fit"), dict)
                    and isinstance(
                        pretraining_metadata.get("pretraining"), dict
                    )
                    and loaded_model.config.to_dict()
                    == topology_model_config.to_dict()
                    and pretrained_model.config.to_dict()
                    == topology_model_config.to_dict()
                )
            except (KeyError, RuntimeError, TypeError, ValueError):
                loaded_model = None
                loaded_config = None
                final_metadata = {}
                pretraining_metadata = {}
                resumed = False

        if resumed:
            assert loaded_model is not None
            training_config = loaded_config or TrainingConfig.from_dict(
                online_training_config.to_dict()
            )
            agent = NeuralCacheAgent(
                loaded_model,
                training_config,
                cache_items=cache_items,
                cache_bytes=args.cache_bytes,
                device=device,
                seed=args.seed,
            )
            optimizer_state = final_metadata.get("_optimizer_state")
            if optimizer_state is not None:
                agent.optimizer.load_state_dict(optimizer_state)
            pretraining_report = dict(
                pretraining_metadata["pretraining"]
            )
            router_observations = int(
                pretraining_metadata["router_observations"]
            )
            fit_report = dict(final_metadata["fit"])
        else:
            training_config = TrainingConfig.from_dict(
                training_config_template.to_dict()
            )
            agent = NeuralCacheAgent(
                NeuralCacheModel(
                    ModelConfig.from_dict(
                        topology_model_config.to_dict()
                    )
                ),
                training_config,
                cache_items=cache_items,
                cache_bytes=args.cache_bytes,
                device=device,
                seed=args.seed,
            )
            for parameter_group in agent.optimizer.param_groups:
                parameter_group["lr"] = args.pretrain_learning_rate
            pretraining_observations = collect_router_observations(
                topology,
                train_trace,
                behavior_policy=args.pretrain_behavior,
                cache_items=cache_items,
                cache_bytes=args.cache_bytes,
                seed=args.seed,
            )
            pretraining_result = pretrain_reuse_model(
                agent,
                pretraining_observations,
                epochs=args.pretrain_epochs,
                batch_length=args.pretrain_batch_length,
                validation_score=lambda: TopologySimulator(
                    topology,
                    agent,
                ).run(
                    validation_trace,
                    training=False,
                ).metrics.cache_hit_ratio,
                select_initial=True,
            )
            pretraining_report = pretraining_result.as_dict()
            router_observations = len(pretraining_observations)
            save_checkpoint(
                pretrained_checkpoint,
                agent.model,
                training_config=training_config,
                optimizer=agent.optimizer,
                metadata={
                    "training_mode": (
                        "topology-specific-supervised-pretraining"
                    ),
                    "topology": topology.as_dict(),
                    "behavior_policy": args.pretrain_behavior,
                    "external_training_requests": len(train_trace),
                    "router_observations": router_observations,
                    "pretraining": pretraining_report,
                    "cache_items": cache_items,
                    "cache_bytes": args.cache_bytes,
                    "seed": args.seed,
                    "trace_sha256": trace_digest,
                    "pretrain_learning_rate": args.pretrain_learning_rate,
                    "partition_digests": partition_digests,
                    "experiment_fingerprint": experiment_fingerprint,
                    "experiment_spec": experiment_spec,
                },
            )

            agent.optimizer = torch.optim.Adam(
                agent.model.parameters(),
                lr=training_config.learning_rate,
            )
            agent.config.auxiliary_coefficient = 0.0
            agent.config.calibration_samples = 0
            agent.config.validate()
            fit_result = fit_topology(
                agent,
                topology,
                train_trace,
                validation_events=validation_trace,
                epochs=args.epochs,
                select_initial=True,
            )
            fit_report = fit_result.as_dict()
            save_checkpoint(
                checkpoint,
                agent.model,
                training_config=training_config,
                optimizer=agent.optimizer,
                metadata={
                    "training_mode": "topology-specific-reinforce",
                    "pretrained_checkpoint": str(pretrained_checkpoint),
                    "pretraining": pretraining_report,
                    "topology": topology.as_dict(),
                    "cache_items": cache_items,
                    "cache_bytes": args.cache_bytes,
                    "seed": args.seed,
                    "fit": fit_report,
                    "auxiliary_future_reuse_labels": False,
                    "trace_sha256": trace_digest,
                    "partition_digests": partition_digests,
                    "experiment_fingerprint": experiment_fingerprint,
                    "experiment_spec": experiment_spec,
                },
            )

        neural_test = TopologySimulator(topology, agent).run(
            test_trace,
            training=False,
        )

        baseline_results: dict[str, object] = {}
        for raw_policy in args.baselines:
            controller = BaselineCacheController(
                raw_policy,
                cache_items=cache_items,
                cache_bytes=args.cache_bytes,
                seed=args.seed,
            )
            baseline_results[controller.policy.value] = TopologySimulator(
                topology,
                controller,
            ).run(test_trace)

        if checkpoint.resolve() == args.output.resolve():
            raise ValueError("checkpoint and report paths must be distinct")

        topology_results[kind] = {
            "topology": topology.as_dict(),
            "model_config": topology_model_config.to_dict(),
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": _file_sha256(checkpoint),
            },
            "pretrained_checkpoint": {
                "path": str(pretrained_checkpoint),
                "sha256": _file_sha256(pretrained_checkpoint),
            },
            "pretraining": {
                **pretraining_report,
                "behavior_policy": args.pretrain_behavior,
                "external_training_requests": len(train_trace),
                "router_observations": router_observations,
            },
            "training": fit_report,
            "resumed": resumed,
            "neural_test": neural_test.as_dict(
                include_requests=args.include_requests
            ),
            "baselines": {
                policy: result.as_dict(
                    include_requests=args.include_requests
                )
                for policy, result in baseline_results.items()
            },
        }

        policies = {"neural": neural_test, **baseline_results}
        for policy, result in policies.items():
            metrics = result.metrics
            if uniform_packet_size is not None and not math.isclose(
                metrics.cache_hit_ratio,
                metrics.byte_cache_hit_ratio,
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise AssertionError(
                    "uniform packet sizes must produce equal request and "
                    "byte cache-hit ratios"
                )
            comparison.append(
                {
                    "topology": kind,
                    "policy": policy,
                    "cache_hit_ratio": metrics.cache_hit_ratio,
                    "byte_cache_hit_ratio": metrics.byte_cache_hit_ratio,
                    "producer_fetches": metrics.producer_fetches,
                    "mean_retrieval_hops": metrics.mean_retrieval_hops,
                    "hop_savings_ratio": metrics.hop_savings_ratio,
                    "router_lookups": metrics.router_lookups,
                    "evictions": sum(
                        item.evictions for item in result.per_node_cache.values()
                    ),
                }
            )

    report = {
        "package_version": __version__,
        "training_mode": (
            "topology-specific-supervised-pretraining-then-online-reinforce"
        ),
        "initialization": (
            "identical seed and neural residual initial weights per topology"
        ),
        "pretraining_behavior_policy": args.pretrain_behavior,
        "online_auxiliary_future_reuse_labels": False,
        "trace": {
            "source": str(args.trace) if args.trace else "synthetic",
            "total_requests": len(trace),
            "train_requests": len(train_trace),
            "validation_requests": len(validation_trace),
            "test_requests": len(test_trace),
            "seed": args.seed,
            "sha256": _trace_sha256(trace),
            "partition_digests": partition_digests,
            "input_file_sha256": (
                _file_sha256(args.trace) if args.trace is not None else None
            ),
            "synthetic_config": (
                asdict(workload_config) if workload_config is not None else None
            ),
            "unique_packet_sizes": packet_sizes,
            "uniform_packet_size": uniform_packet_size,
        },
        "network": {
            "routers": args.nodes,
            "producer_node": args.producer_node,
            "producer_prefix": args.producer_prefix,
            "star_center": args.star_center,
            "placement": "leave-copy-everywhere",
            "routing": "component-prefix shortest-path FIB",
        },
        "cache": {
            "items_per_router": cache_items,
            "bytes_per_router": args.cache_bytes,
            "capacity_unit": (
                "packets"
                if args.cache_bytes is None
                else ("bytes" if cache_items is None else "packets-and-bytes")
            ),
            "equal_packet_size_experiment": uniform_packet_size is not None,
            "neural_replacement_shortlist": (
                training_config_template.replacement_shortlist_size
            ),
        },
        "model_config": model_config.to_dict(),
        "topology_recency_prior_overrides": recency_prior_overrides,
        "model_config_by_topology": model_configs_by_topology,
        "pretraining_config": training_config_template.to_dict(),
        "online_training_config": online_training_config.to_dict(),
        "topologies": topology_results,
        "comparison": comparison,
    }
    write_json(args.output, report)

    print(
        "topology  policy    cache hit  byte hit  producer  mean hops  "
        "best epoch"
    )
    print(
        "--------  --------  ---------  --------  --------  ---------  "
        "----------"
    )
    for row in comparison:
        topology_result = topology_results[str(row["topology"])]
        training = topology_result["training"]
        best_epoch = (
            int(training["best_epoch"])
            if row["policy"] == "neural"
            else 0
        )
        epoch_label = (
            "pretrained"
            if row["policy"] == "neural" and best_epoch == 0
            else (str(best_epoch) if best_epoch else "-")
        )
        print(
            f"{str(row['topology']):<8}  "
            f"{str(row['policy']):<8}  "
            f"{float(row['cache_hit_ratio']):>9.6f}  "
            f"{float(row['byte_cache_hit_ratio']):>8.6f}  "
            f"{int(row['producer_fetches']):>8d}  "
            f"{float(row['mean_retrieval_hops']):>9.3f}  "
            f"{epoch_label:>10}"
        )
    print(f"\ncheckpoints: {args.checkpoint_dir}")
    print(f"report:      {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ndn-neural-cache",
        description="LSTM + policy-gradient NDN Content Store simulator",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train and compare the policy")
    _add_workload_arguments(train)
    _add_cache_arguments(train, defaults=True)
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--train-fraction", type=float, default=0.70)
    train.add_argument("--validation-fraction", type=float, default=0.15)
    train.add_argument("--device", default="cpu")
    train.add_argument("--torch-threads", type=int, default=1)
    train.add_argument("--checkpoint", type=Path, default=Path("artifacts/model.pt"))
    train.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/training_report.json"),
    )
    train.add_argument(
        "--plot",
        type=Path,
        default=Path("artifacts/training.png"),
    )
    train.add_argument(
        "--baselines",
        nargs="+",
        default=["lru", "lfu", "fifo", "random", "belady"],
        choices=[policy.value for policy in BaselinePolicy],
    )
    model_group = train.add_argument_group("model")
    model_group.add_argument("--vocab-size", type=int, default=4096)
    model_group.add_argument("--max-name-components", type=int, default=16)
    model_group.add_argument("--embedding-dim", type=int, default=32)
    model_group.add_argument("--name-hidden-dim", type=int, default=48)
    model_group.add_argument("--context-hidden-dim", type=int, default=64)
    model_group.add_argument("--head-hidden-dim", type=int, default=64)
    model_group.add_argument("--dropout", type=float, default=0.0)
    model_group.add_argument("--popularity-prior-weight", type=float, default=1.0)
    model_group.add_argument("--recency-prior-weight", type=float, default=0.0)
    model_group.add_argument("--residual-scale", type=float, default=0.05)
    rl_group = train.add_argument_group("learning")
    rl_group.add_argument("--learning-rate", type=float, default=3e-4)
    rl_group.add_argument("--gamma", type=float, default=0.99)
    rl_group.add_argument("--rollout-length", type=int, default=256)
    rl_group.add_argument("--temperature", type=float, default=0.7)
    rl_group.add_argument("--entropy-coefficient", type=float, default=0.0)
    rl_group.add_argument("--auxiliary-coefficient", type=float, default=0.25)
    rl_group.add_argument("--prediction-horizon", type=int, default=32)
    rl_group.add_argument("--calibration-samples", type=int, default=16)
    rl_group.add_argument(
        "--replacement-shortlist-size",
        type=int,
        default=32,
        help="score only the K least-recently-used residents per replacement",
    )
    rl_group.add_argument("--gradient-clip", type=float, default=1.0)
    rl_group.add_argument("--recency-scale", type=float, default=100.0)
    rl_group.add_argument("--popularity-decay", type=float, default=32.0)
    rl_group.add_argument(
        "--allow-bypass",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="let the policy reject newly returned Data instead of evicting",
    )
    train.set_defaults(handler=command_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate an existing checkpoint on a trace",
    )
    _add_workload_arguments(evaluate)
    _add_cache_arguments(evaluate, defaults=False)
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--torch-threads", type=int, default=1)
    evaluate.add_argument(
        "--replacement-shortlist-size",
        type=int,
        default=None,
        help="override the checkpoint's LRU neural shortlist size",
    )
    evaluate.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation_report.json"),
    )
    evaluate.add_argument(
        "--baselines",
        nargs="+",
        default=["lru", "lfu", "fifo", "random", "belady"],
        choices=[policy.value for policy in BaselinePolicy],
    )
    evaluate.set_defaults(handler=command_evaluate)

    topology = subparsers.add_parser(
        "topology",
        help="evaluate a checkpoint over multi-hop NDN router topologies",
    )
    _add_workload_arguments(topology)
    _add_cache_arguments(topology, defaults=False)
    topology.add_argument("checkpoint", type=Path)
    topology.add_argument(
        "--topologies",
        nargs="+",
        choices=[kind.value for kind in TopologyKind],
        default=[kind.value for kind in TopologyKind],
        help="router graphs to compare using the same request trace",
    )
    topology.add_argument(
        "--producer-node",
        default="node-0",
        help="router hosting the producer (default: node-0)",
    )
    topology.add_argument(
        "--producer-prefix",
        default="/dataset/item",
        help="NDN Name prefix originated by the producer",
    )
    topology.add_argument(
        "--star-center",
        default="auto",
        help=(
            "star hub router, or 'auto' for the first non-producer router; "
            "independent placement prevents star from collapsing into mesh"
        ),
    )
    topology.add_argument(
        "--include-requests",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="include every Interest path in the JSON report",
    )
    topology.add_argument(
        "--baselines",
        nargs="+",
        default=["lru", "lfu", "fifo", "random"],
        choices=[
            policy.value
            for policy in BaselineCacheController.SUPPORTED_POLICIES
        ],
        help=(
            "topology-aware policies evaluated on every graph "
            "(Belady is not valid for policy-dependent router streams)"
        ),
    )
    topology.add_argument("--device", default="cpu")
    topology.add_argument("--torch-threads", type=int, default=1)
    topology.add_argument(
        "--replacement-shortlist-size",
        type=int,
        default=None,
        help="override the checkpoint's per-router LRU neural shortlist size",
    )
    topology.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/topology_report.json"),
    )
    topology.set_defaults(handler=command_topology, nodes=6)

    topology_train = subparsers.add_parser(
        "topology-train",
        help="train a separate neural model for every router topology",
    )
    _add_workload_arguments(topology_train)
    _add_cache_arguments(topology_train, defaults=True)
    _add_neural_training_arguments(topology_train)
    topology_train.add_argument(
        "--pretrain-epochs",
        type=int,
        default=2,
        help="supervised future-reuse pretraining epochs per topology",
    )
    topology_train.add_argument(
        "--pretrain-batch-length",
        type=int,
        default=64,
    )
    topology_train.add_argument(
        "--pretrain-learning-rate",
        type=float,
        default=3e-4,
        help="learning rate for pairwise ranking before RL optimizer reset",
    )
    topology_train.add_argument(
        "--pretrain-behavior",
        choices=["lru", "lfu", "fifo", "random"],
        default="lru",
        help="fixed cache policy used to collect router-local training streams",
    )
    topology_train.add_argument("--epochs", type=int, default=3)
    topology_train.add_argument("--train-fraction", type=float, default=0.70)
    topology_train.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )
    topology_train.add_argument(
        "--topologies",
        nargs="+",
        choices=[kind.value for kind in TopologyKind],
        default=[kind.value for kind in TopologyKind],
        help="independently trained router graphs",
    )
    topology_train.add_argument(
        "--topology-recency-prior",
        action="append",
        type=_parse_topology_recency_prior,
        default=[],
        metavar="KIND=FLOAT",
        help=(
            "override --recency-prior-weight for one selected topology; "
            "repeat for multiple graphs"
        ),
    )
    topology_train.add_argument("--producer-node", default="node-0")
    topology_train.add_argument(
        "--producer-prefix",
        default="/dataset/item",
    )
    topology_train.add_argument("--star-center", default="auto")
    topology_train.add_argument(
        "--baselines",
        nargs="+",
        default=["lru", "lfu", "fifo", "random"],
        choices=[
            policy.value
            for policy in BaselineCacheController.SUPPORTED_POLICIES
        ],
    )
    topology_train.add_argument(
        "--include-requests",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    topology_train.add_argument("--device", default="cpu")
    topology_train.add_argument("--torch-threads", type=int, default=1)
    topology_train.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("artifacts/topology_models"),
    )
    topology_train.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "reuse completed topology checkpoints whose graph, seed, cache, "
            "training-count, and trace digest match this run"
        ),
    )
    topology_train.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/topology_training_report.json"),
    )
    topology_train.set_defaults(handler=command_topology_train, nodes=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
