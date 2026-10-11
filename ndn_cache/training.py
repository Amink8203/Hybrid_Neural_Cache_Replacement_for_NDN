
from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable

import torch

from .agent import NeuralCacheAgent, UpdateSummary
from .metrics import SimulationMetrics
from .packets import RequestEvent


@dataclass(slots=True)
class NeuralRunResult:
    metrics: SimulationMetrics
    per_node: dict[str, SimulationMetrics]
    updates: list[UpdateSummary] = field(default_factory=list)
    brier_score: float | None = None
    calibration_examples: int = 0
    replacement_efficiency: dict[str, int | float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.as_dict(),
            "per_node": {
                node_id: metrics.as_dict()
                for node_id, metrics in sorted(self.per_node.items())
            },
            "updates": [asdict(update) for update in self.updates],
            "brier_score": self.brier_score,
            "calibration_examples": self.calibration_examples,
            "replacement_efficiency": dict(self.replacement_efficiency),
        }

class FutureReuseWindow(set[str]):

    def __init__(self, events: Iterable[RequestEvent] = ()) -> None:
        super().__init__(event.name for event in events)

@dataclass(slots=True)
class EpochResult:
    epoch: int
    train_hit_ratio: float
    train_byte_hit_ratio: float
    validation_hit_ratio: float | None
    mean_loss: float | None
    updates: int

    def as_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(slots=True)
class FitResult:
    epochs: list[EpochResult]
    best_epoch: int
    best_validation_hit_ratio: float

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": [epoch.as_dict() for epoch in self.epochs],
            "best_epoch": self.best_epoch,
            "best_validation_hit_ratio": self.best_validation_hit_ratio,
        }


@dataclass(slots=True)
class PretrainingEpochResult:
    epoch: int
    mean_ranking_loss: float
    calibration_batches: int
    calibration_examples: int
    ranking_pairs: int
    validation_hit_ratio: float | None = None

    def as_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(slots=True)
class PretrainingResult:
    epochs: list[PretrainingEpochResult]
    observation_requests: int
    best_epoch: int
    best_validation_hit_ratio: float | None
    initial_validation_hit_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "epochs": [epoch.as_dict() for epoch in self.epochs],
            "observation_requests": self.observation_requests,
            "best_epoch": self.best_epoch,
            "best_validation_hit_ratio": self.best_validation_hit_ratio,
            "initial_validation_hit_ratio": self.initial_validation_hit_ratio,
        }


def validate_trace(events: Iterable[RequestEvent]) -> list[RequestEvent]:
    trace = list(events)
    previous_timestamp: float | None = None
    for index, event in enumerate(trace):
        if not isinstance(event, RequestEvent):
            raise TypeError(f"trace item {index} is not a RequestEvent")
        if previous_timestamp is not None and event.timestamp < previous_timestamp:
            raise ValueError("trace must be ordered by non-decreasing timestamp")
        previous_timestamp = event.timestamp
    return trace


def split_trace(
    events: Iterable[RequestEvent],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> tuple[list[RequestEvent], list[RequestEvent], list[RequestEvent]]:

    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if train_fraction + validation_fraction > 1.0:
        raise ValueError("train and validation fractions cannot exceed 1")
    trace = validate_trace(events)
    train_end = int(len(trace) * train_fraction)
    if trace and train_end == 0:
        train_end = 1
    validation_end = min(
        len(trace),
        train_end + int(len(trace) * validation_fraction),
    )
    return (
        trace[:train_end],
        trace[train_end:validation_end],
        trace[validation_end:],
    )


def future_reuse_labels(
    events: Iterable[RequestEvent],
    horizon: int,
) -> list[set[str]]:

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    trace = list(events)
    by_node: dict[str, list[int]] = {}
    for global_index, event in enumerate(trace):
        by_node.setdefault(event.node_id, []).append(global_index)
    labels = [FutureReuseWindow() for _ in trace]
    for indices in by_node.values():
        local_events = [trace[index] for index in indices]
        for local_index, global_index in enumerate(indices):
            labels[global_index] = FutureReuseWindow(
                local_events[
                    local_index + 1 : local_index + 1 + horizon
                ]
            )
    return labels


def future_reuse_scores(
    events: Iterable[RequestEvent],
    horizon: int,
    *,
    censor_tail: bool = True,
) -> list[dict[str, float] | None]:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    trace = list(events)
    by_node: dict[str, list[int]] = {}
    for global_index, event in enumerate(trace):
        by_node.setdefault(event.node_id, []).append(global_index)
    scores: list[dict[str, float] | None] = [None] * len(trace)
    for indices in by_node.values():
        for local_index, global_index in enumerate(indices):
            remaining = len(indices) - local_index - 1
            if censor_tail and remaining < horizon:
                continue
            utilities: dict[str, float] = {}
            for distance, future_index in enumerate(
                indices[local_index + 1 : local_index + 1 + horizon],
                start=1,
            ):
                name = trace[future_index].name
                utilities.setdefault(name, 1.0 / distance)
            scores[global_index] = utilities
    return scores


def pretrain_reuse_model(
    agent: NeuralCacheAgent,
    events: Iterable[RequestEvent],
    *,
    epochs: int = 2,
    batch_length: int = 64,
    validation_score: Callable[[], float] | None = None,
    select_initial: bool = True,
) -> PretrainingResult:

    if epochs <= 0:
        raise ValueError("pretraining epochs must be positive")
    if batch_length <= 0:
        raise ValueError("pretraining batch_length must be positive")
    if agent.config.calibration_samples <= 0:
        raise ValueError("pretraining requires positive calibration_samples")
    trace = validate_trace(events)
    if not trace:
        raise ValueError("pretraining observation trace is empty")
    labels = future_reuse_scores(trace, agent.config.prediction_horizon)
    history: list[PretrainingEpochResult] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    best_epoch = 0
    best_score = -math.inf
    initial_score: float | None = None
    if validation_score is not None and select_initial:
        initial_score = float(validation_score())
        best_score = initial_score
        best_state = copy.deepcopy(agent.model.state_dict())
        best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())

    for epoch in range(1, epochs + 1):
        agent.reset_episode()
        agent.model.train()
        agent.optimizer.zero_grad(set_to_none=True)
        pending_losses: list[tuple[torch.Tensor, int]] = []
        loss_total = 0.0
        pair_total = 0
        batches = 0
        examples = 0

        def apply_batch() -> None:
            nonlocal batches
            if not pending_losses:
                return
            pending_pairs = sum(count for _, count in pending_losses)
            loss = sum(
                item * count for item, count in pending_losses
            ) / pending_pairs
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite supervised pretraining loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.model.parameters(),
                agent.config.gradient_clip,
                error_if_nonfinite=True,
            )
            agent.optimizer.step()
            agent.optimizer.zero_grad(set_to_none=True)
            pending_losses.clear()
            batches += 1
            agent._finish_update_boundary()

        for index, event in enumerate(trace):
            target_scores = labels[index]
            outcome = agent.process(
                event,
                training=False,
                future_reuse_scores=target_scores,
                supervise_decision=target_scores is not None,
            )
            if outcome.supervised_loss is None:
                continue
            pair_count = outcome.supervised_pairs
            pending_losses.append((outcome.supervised_loss, pair_count))
            loss_total += float(outcome.supervised_loss.detach().cpu()) * pair_count
            pair_total += pair_count
            examples += 1
            if len(pending_losses) >= batch_length:
                apply_batch()
        apply_batch()
        epoch_validation = (
            float(validation_score()) if validation_score is not None else None
        )
        history.append(
            PretrainingEpochResult(
                epoch=epoch,
                mean_ranking_loss=(
                    loss_total / pair_total if pair_total else 0.0
                ),
                calibration_batches=batches,
                calibration_examples=examples,
                ranking_pairs=pair_total,
                validation_hit_ratio=epoch_validation,
            )
        )
        if validation_score is None:
            best_epoch = epoch
            best_state = copy.deepcopy(agent.model.state_dict())
            best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())
        elif epoch_validation is not None and epoch_validation > best_score:
            best_score = epoch_validation
            best_epoch = epoch
            best_state = copy.deepcopy(agent.model.state_dict())
            best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())

    assert best_state is not None
    assert best_optimizer_state is not None
    agent.model.load_state_dict(best_state)
    agent.optimizer.load_state_dict(best_optimizer_state)
    agent.model.eval()
    return PretrainingResult(
        epochs=history,
        observation_requests=len(trace),
        best_epoch=best_epoch,
        best_validation_hit_ratio=(
            best_score if validation_score is not None else None
        ),
        initial_validation_hit_ratio=initial_score,
    )


def run_neural_trace(
    agent: NeuralCacheAgent,
    events: Iterable[RequestEvent],
    *,
    training: bool,
    collect_calibration: bool = False,
) -> NeuralRunResult:

    trace = validate_trace(events)
    labels = future_reuse_labels(trace, agent.config.prediction_horizon)
    agent.reset_episode()
    agent.model.train(training)
    aggregate = SimulationMetrics()
    per_node: dict[str, SimulationMetrics] = {}
    updates: list[UpdateSummary] = []
    squared_error = 0.0
    calibration_count = 0

    gradient_context = nullcontext() if training else torch.no_grad()
    with gradient_context:
        for index, event in enumerate(trace):
            outcome = agent.process(
                event,
                training=training,
                future_reuse=labels[index],
            )
            node_metrics = per_node.setdefault(event.node_id, SimulationMetrics())
            for metrics in (aggregate, node_metrics):
                metrics.record_request(outcome.hit, outcome.data.size_bytes)
                if outcome.expired_bytes:
                    metrics.record_expiration(outcome.expired_bytes)
                if not outcome.hit:
                    if outcome.admission_rejected:
                        metrics.record_rejection(event.size_bytes)
                    else:
                        metrics.record_insertion()
                    if outcome.evicted_names:
                        metrics.record_eviction(
                            outcome.evicted_bytes,
                            count=len(outcome.evicted_names),
                        )
            if outcome.update_loss is not None and agent.last_update is not None:
                updates.append(agent.last_update)

            if collect_calibration:
                scores = agent.score_cache(event.node_id, event.timestamp)
                for name, score in scores.items():
                    target = 1.0 if name in labels[index] else 0.0
                    squared_error += (score - target) ** 2
                    calibration_count += 1

    if training:
        final_update = agent.flush()
        if final_update is not None:
            updates.append(final_update)
    aggregate.validate()
    for metrics in per_node.values():
        metrics.validate()
    return NeuralRunResult(
        metrics=aggregate,
        per_node=per_node,
        updates=updates,
        brier_score=(
            squared_error / calibration_count if calibration_count else None
        ),
        calibration_examples=calibration_count,
        replacement_efficiency=agent.replacement_efficiency.as_dict(),
    )


def fit(
    agent: NeuralCacheAgent,
    train_events: Iterable[RequestEvent],
    *,
    validation_events: Iterable[RequestEvent] = (),
    epochs: int = 5,
) -> FitResult:

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    train_trace = validate_trace(train_events)
    validation_trace = validate_trace(validation_events)
    if not train_trace:
        raise ValueError("training trace is empty")

    history: list[EpochResult] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    best_baseline = 0.0
    best_baseline_initialized = False
    best_epoch = 0
    best_score = -math.inf
    for epoch in range(1, epochs + 1):
        train_result = run_neural_trace(agent, train_trace, training=True)
        validation_result = (
            run_neural_trace(agent, validation_trace, training=False)
            if validation_trace
            else None
        )
        score = (
            validation_result.metrics.hit_ratio
            if validation_result is not None
            else train_result.metrics.hit_ratio
        )
        losses = [update.loss for update in train_result.updates]
        history.append(
            EpochResult(
                epoch=epoch,
                train_hit_ratio=train_result.metrics.hit_ratio,
                train_byte_hit_ratio=train_result.metrics.byte_hit_ratio,
                validation_hit_ratio=(
                    validation_result.metrics.hit_ratio
                    if validation_result is not None
                    else None
                ),
                mean_loss=sum(losses) / len(losses) if losses else None,
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
    return FitResult(
        epochs=history,
        best_epoch=best_epoch,
        best_validation_hit_ratio=best_score,
    )


__all__ = [
    "EpochResult",
    "FitResult",
    "FutureReuseWindow",
    "NeuralRunResult",
    "PretrainingEpochResult",
    "PretrainingResult",
    "fit",
    "future_reuse_labels",
    "future_reuse_scores",
    "pretrain_reuse_model",
    "run_neural_trace",
    "split_trace",
    "validate_trace",
]
