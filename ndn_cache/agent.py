
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import TrainingConfig
from .features import CandidateFeatureBuilder, RequestStatistics
from .model import NeuralCacheModel, RecurrentState
from .names import split_components
from .packets import DataPacket, Interest, RequestEvent
from .store import CacheEntry, ContentStore


@dataclass(slots=True)
class NodeState:
    store: ContentStore
    statistics: RequestStatistics = field(default_factory=RequestStatistics)
    recurrent_state: RecurrentState | None = None
    context: Tensor | None = None


@dataclass(slots=True)
class RolloutStep:
    reward: float
    log_probabilities: list[Tensor] = field(default_factory=list)
    entropies: list[Tensor] = field(default_factory=list)
    auxiliary_loss: Tensor | None = None


@dataclass(slots=True)
class RequestOutcome:
    node_id: str
    name: str
    hit: bool
    reward: int
    data: DataPacket
    evicted_names: list[str] = field(default_factory=list)
    evicted_bytes: int = 0
    expired_bytes: int = 0
    admission_rejected: bool = False
    scores: dict[str, float] = field(default_factory=dict)
    update_loss: float | None = None
    supervised_loss: Tensor | None = None
    supervised_pairs: int = 0


@dataclass(slots=True)
class PendingInterest:
    event: RequestEvent
    hit_entry: CacheEntry | None
    completed: bool = False

    @property
    def hit(self) -> bool:
        return self.hit_entry is not None

    @property
    def reward(self) -> int:
        return int(self.hit)


@dataclass(slots=True)
class UpdateSummary:
    loss: float
    policy_loss: float
    auxiliary_loss: float
    entropy: float
    decision_steps: int
    calibration_batches: int
    gradient_norm: float


@dataclass(slots=True)
class ReplacementEfficiencyStats:
    replacement_decisions: int = 0
    shortlist_activations: int = 0
    resident_candidates_available: int = 0
    resident_candidates_scored: int = 0
    total_candidates_scored: int = 0
    max_resident_candidates_scored: int = 0

    @property
    def resident_scoring_reduction_ratio(self) -> float:
        if not self.resident_candidates_available:
            return 0.0
        return 1.0 - (
            self.resident_candidates_scored
            / self.resident_candidates_available
        )

    def record(
        self,
        *,
        available: int,
        residents_scored: int,
        total_scored: int,
    ) -> None:
        self.replacement_decisions += 1
        self.resident_candidates_available += available
        self.resident_candidates_scored += residents_scored
        self.total_candidates_scored += total_scored
        self.max_resident_candidates_scored = max(
            self.max_resident_candidates_scored,
            residents_scored,
        )
        if residents_scored < available:
            self.shortlist_activations += 1

    def as_dict(self) -> dict[str, int | float]:
        return {
            "replacement_decisions": self.replacement_decisions,
            "shortlist_activations": self.shortlist_activations,
            "resident_candidates_available": self.resident_candidates_available,
            "resident_candidates_scored": self.resident_candidates_scored,
            "total_candidates_scored": self.total_candidates_scored,
            "max_resident_candidates_scored": (
                self.max_resident_candidates_scored
            ),
            "resident_scoring_reduction_ratio": (
                self.resident_scoring_reduction_ratio
            ),
        }


class NeuralCacheAgent:
    def __init__(
        self,
        model: NeuralCacheModel,
        training_config: TrainingConfig,
        *,
        cache_items: int | None,
        cache_bytes: int | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
    ) -> None:
        training_config.validate()
        if cache_items is not None and cache_items < 0:
            raise ValueError("cache_items cannot be negative")
        if cache_bytes is not None and cache_bytes < 0:
            raise ValueError("cache_bytes cannot be negative")
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = training_config
        self.cache_items = cache_items
        self.cache_bytes = cache_bytes
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=training_config.learning_rate,
        )
        self.feature_builder = CandidateFeatureBuilder(
            recency_scale=training_config.recency_scale
        )
        if len(self.feature_builder.FEATURE_NAMES) != model.config.feature_dim:
            raise ValueError(
                "model feature_dim must match CandidateFeatureBuilder features"
            )
        self.nodes: dict[str, NodeState] = {}
        self.rollouts: dict[str, list[RolloutStep]] = {}
        self.pending_requests = 0
        self.running_return_baseline = 0.0
        self._baseline_initialized = False
        self._random = random.Random(seed)
        self._torch_generator = torch.Generator(device=self.device)
        self._torch_generator.manual_seed(seed)
        self.last_update: UpdateSummary | None = None
        self.replacement_efficiency = ReplacementEfficiencyStats()
        self._deferred_update_due = False

    def reset_episode(self) -> None:
        self.nodes.clear()
        self.rollouts.clear()
        self.pending_requests = 0
        self.last_update = None
        self.replacement_efficiency = ReplacementEfficiencyStats()
        self._deferred_update_due = False

    def _node(self, node_id: str) -> NodeState:
        if node_id not in self.nodes:
            self.nodes[node_id] = NodeState(
                store=ContentStore(
                    max_items=self.cache_items,
                    max_bytes=self.cache_bytes,
                ),
                statistics=RequestStatistics(
                    popularity_decay=self.config.popularity_decay
                ),
            )
        return self.nodes[node_id]

    @staticmethod
    def _data_for_event(event: RequestEvent) -> DataPacket:
        return DataPacket(
            name=event.name,
            size_bytes=event.size_bytes,
            freshness_period=event.freshness_period,
            produced_at=event.timestamp,
        )

    def process(
        self,
        event: RequestEvent,
        *,
        training: bool,
        future_reuse: set[str] | None = None,
        future_reuse_scores: Mapping[str, float] | None = None,
        supervise_decision: bool = False,
        defer_update: bool = False,
    ) -> RequestOutcome:
        pending = self.begin_interest(event)
        response = None if pending.hit else self._data_for_event(event)
        return self.finish_interest(
            pending,
            response=response,
            training=training,
            future_reuse=future_reuse,
            future_reuse_scores=future_reuse_scores,
            supervise_decision=supervise_decision,
            defer_update=defer_update,
        )

    def begin_interest(self, event: RequestEvent) -> PendingInterest:
        node = self._node(event.node_id)
        interest = Interest(
            name=event.name,
            timestamp=event.timestamp,
            must_be_fresh=event.must_be_fresh,
            can_be_prefix=event.can_be_prefix,
        )
        hit_entry = node.store.lookup(interest)
        context, recurrent_state = self.model.advance_context(
            event.name,
            node.recurrent_state,
        )
        node.context = context
        node.recurrent_state = recurrent_state
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

    def finish_interest(
        self,
        pending: PendingInterest,
        *,
        response: DataPacket | None,
        training: bool,
        future_reuse: set[str] | None = None,
        future_reuse_scores: Mapping[str, float] | None = None,
        supervise_decision: bool = False,
        defer_update: bool = False,
    ) -> RequestOutcome:
        if pending.completed:
            raise RuntimeError("Interest has already been completed")
        event = pending.event
        node = self._node(event.node_id)
        hit = pending.hit
        reward = pending.reward
        if hit:
            if response is not None:
                raise ValueError("a cache hit must not receive a second response")
            assert pending.hit_entry is not None
            data = pending.hit_entry.packet
        else:
            if response is None:
                raise ValueError("a cache miss requires returned Data")
            self._validate_response(event, response)
            data = response
        node.statistics.observe(data.name, event.timestamp)

        expired_bytes = 0
        if not hit and event.must_be_fresh:
            stale_exact = node.store.get(data.name)
            if (
                stale_exact is not None
                and node.store.is_cacheable(data)
                and not stale_exact.is_fresh(event.timestamp)
            ):
                expired_bytes = stale_exact.packet.size_bytes
                node.store.remove(stale_exact.packet.name)

        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        evicted: list[str] = []
        evicted_bytes = 0
        rejected = False
        detached_scores: dict[str, float] = {}
        supervised_loss: Tensor | None = None
        supervised_pairs = 0
        if not hit:
            resident_sizes = {
                entry.packet.name: entry.packet.size_bytes
                for entry in node.store.entries()
            }
            (
                admitted,
                evicted,
                log_probabilities,
                entropies,
                detached_scores,
                supervised_loss,
                supervised_pairs,
            ) = self._admit(
                node,
                data,
                now=event.timestamp,
                training=training,
                allow_bypass=False if expired_bytes else None,
                future_reuse_scores=(
                    future_reuse_scores if supervise_decision else None
                ),
            )
            rejected = not admitted
            evicted_bytes = sum(
                resident_sizes[name] for name in evicted
            )

        auxiliary_loss = None
        if (
            training
            and self.config.auxiliary_coefficient > 0
            and self.config.calibration_samples > 0
            and future_reuse is not None
            and len(node.store) > 0
        ):
            auxiliary_loss = self._calibration_loss(
                node,
                now=event.timestamp,
                future_reuse=future_reuse,
            )

        update_loss = None
        if training:
            self.rollouts.setdefault(event.node_id, []).append(
                RolloutStep(
                    reward=float(reward),
                    log_probabilities=log_probabilities,
                    entropies=entropies,
                    auxiliary_loss=auxiliary_loss,
                )
            )
            self.pending_requests += 1
            tails_have_future_reward = all(
                not steps[-1].log_probabilities
                for steps in self.rollouts.values()
                if steps
            )
            normal_boundary = (
                self.pending_requests >= self.config.rollout_length
                and tails_have_future_reward
            )
            hard_memory_boundary = (
                self.pending_requests >= 4 * self.config.rollout_length
            )
            if normal_boundary or hard_memory_boundary:
                if defer_update:
                    self._deferred_update_due = True
                else:
                    summary = self.update(terminal=False)
                    update_loss = summary.loss if summary is not None else None

        pending.completed = True
        return RequestOutcome(
            node_id=event.node_id,
            name=event.name,
            hit=hit,
            reward=reward,
            data=data,
            evicted_names=evicted,
            evicted_bytes=evicted_bytes,
            expired_bytes=expired_bytes,
            admission_rejected=rejected,
            scores=detached_scores,
            update_loss=update_loss,
            supervised_loss=supervised_loss,
            supervised_pairs=supervised_pairs,
        )

    def abandon_interest(self, pending: PendingInterest) -> None:
        if pending.completed:
            raise RuntimeError("Interest has already been completed")
        if pending.hit:
            raise ValueError("a cache hit cannot be abandoned")
        event = pending.event
        node = self._node(event.node_id)
        node.statistics.observe(event.name, event.timestamp)
        pending.completed = True

    def _candidate_rows(
        self,
        node: NodeState,
        *,
        incoming: DataPacket | None,
        include_incoming: bool,
        now: float,
        resident_entries: Iterable[CacheEntry] | None = None,
    ) -> tuple[list[str], list[tuple[str, DataPacket, CacheEntry | None, bool]], Tensor]:
        entries = sorted(
            (
                node.store.entries()
                if resident_entries is None
                else resident_entries
            ),
            key=lambda entry: entry.packet.name,
        )
        candidates: list[tuple[str, DataPacket, CacheEntry | None, bool]] = [
            (entry.packet.name, entry.packet, entry, False) for entry in entries
        ]
        if include_incoming:
            if incoming is None:
                raise ValueError("incoming packet required when include_incoming=True")
            candidates.append((incoming.name, incoming, None, True))
        names = [candidate[0] for candidate in candidates]
        features = self.feature_builder.batch(
            candidates,
            statistics=node.statistics,
            store=node.store,
            now=now,
            device=self.device,
        )
        if node.context is None:
            raise RuntimeError("node context has not been initialized")
        logits = self.model.score_logits(names, features, node.context)
        return names, candidates, logits

    def _replacement_shortlist(self, node: NodeState) -> list[CacheEntry]:
        entries = sorted(
            node.store.entries(),
            key=lambda entry: (
                entry.last_access,
                entry.inserted_at,
                entry.packet.name,
            ),
        )
        return entries[: self.config.replacement_shortlist_size]

    def _select_victim(
        self,
        names: list[str],
        logits: Tensor,
        *,
        training: bool,
    ) -> tuple[int, Tensor | None, Tensor | None]:
        if not names:
            raise RuntimeError("cannot select from an empty candidate set")
        if len(names) != logits.numel():
            raise ValueError("candidate/logit size mismatch")
        if training and len(names) > 1:
            victim_logits = -logits / self.config.temperature
            log_distribution = torch.log_softmax(victim_logits, dim=0)
            probabilities = log_distribution.exp()
            selected_tensor = torch.multinomial(
                probabilities.detach(),
                1,
                generator=self._torch_generator,
            )
            selected = int(selected_tensor.item())
            log_probability = log_distribution[selected]
            entropy = -(probabilities * log_distribution).sum()
            return selected, log_probability, entropy
        selected = min(
            range(len(names)),
            key=lambda index: (float(logits[index].detach().cpu()), names[index]),
        )
        return selected, None, None

    def _admit(
        self,
        node: NodeState,
        packet: DataPacket,
        *,
        now: float,
        training: bool,
        allow_bypass: bool | None = None,
        future_reuse_scores: Mapping[str, float] | None = None,
    ) -> tuple[
        bool,
        list[str],
        list[Tensor],
        list[Tensor],
        dict[str, float],
        Tensor | None,
        int,
    ]:
        store = node.store
        bypass_enabled = (
            self.config.allow_bypass
            if allow_bypass is None
            else allow_bypass
        )
        if not store.is_cacheable(packet):
            return False, [], [], [], {}, None, 0

        evicted: list[str] = []
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        reported_scores: dict[str, float] = {}
        supervised_terms: list[tuple[Tensor, int]] = []
        first_decision = True
        while not store.has_capacity(packet):
            if len(store) == 0:
                return (
                    False,
                    evicted,
                    log_probabilities,
                    entropies,
                    reported_scores,
                    self._combine_supervised_terms(supervised_terms),
                    sum(count for _, count in supervised_terms),
                )
            include_incoming = bypass_enabled and first_decision
            available = len(store)
            resident_entries = self._replacement_shortlist(node)
            names, candidates, logits = self._candidate_rows(
                node,
                incoming=packet,
                include_incoming=include_incoming,
                now=now,
                resident_entries=resident_entries,
            )
            self.replacement_efficiency.record(
                available=available,
                residents_scored=len(resident_entries),
                total_scored=len(names),
            )
            scores = torch.sigmoid(logits.detach()).cpu().tolist()
            reported_scores.update(zip(names, (float(value) for value in scores)))
            if future_reuse_scores is not None:
                ranking = self._candidate_ranking_loss(
                    names,
                    logits,
                    future_reuse_scores,
                )
                if ranking is not None:
                    supervised_terms.append(ranking)
            selected, log_probability, entropy = self._select_victim(
                names,
                logits,
                training=training,
            )
            if log_probability is not None:
                log_probabilities.append(log_probability)
            if entropy is not None:
                entropies.append(entropy)
            _, _, selected_entry, is_incoming = candidates[selected]
            if is_incoming:
                return (
                    False,
                    evicted,
                    log_probabilities,
                    entropies,
                    reported_scores,
                    self._combine_supervised_terms(supervised_terms),
                    sum(count for _, count in supervised_terms),
                )
            assert selected_entry is not None
            removed = store.remove(selected_entry.packet.name)
            if removed is None:
                raise RuntimeError("selected victim disappeared from the store")
            evicted.append(selected_entry.packet.name)
            first_decision = False

        store.insert(packet, now=now)
        return (
            True,
            evicted,
            log_probabilities,
            entropies,
            reported_scores,
            self._combine_supervised_terms(supervised_terms),
            sum(count for _, count in supervised_terms),
        )

    @staticmethod
    def _candidate_ranking_loss(
        names: list[str],
        logits: Tensor,
        future_reuse_scores: Mapping[str, float],
    ) -> tuple[Tensor, int] | None:
        if len(names) < 2:
            return None
        targets = torch.tensor(
            [float(future_reuse_scores.get(name, 0.0)) for name in names],
            dtype=logits.dtype,
            device=logits.device,
        )
        preferred = targets[:, None] > targets[None, :] + 1e-12
        pair_count = int(preferred.sum().item())
        if pair_count == 0:
            return None
        margins = logits[:, None] - logits[None, :]
        return F.softplus(-margins[preferred]).mean(), pair_count

    @staticmethod
    def _combine_supervised_terms(
        terms: list[tuple[Tensor, int]],
    ) -> Tensor | None:
        if not terms:
            return None
        total = sum(count for _, count in terms)
        return sum(loss * count for loss, count in terms) / total

    def _calibration_loss(
        self,
        node: NodeState,
        *,
        now: float,
        future_reuse: set[str],
    ) -> Tensor:
        entries = sorted(node.store.entries(), key=lambda entry: entry.packet.name)
        limit = self.config.calibration_samples
        if len(entries) > limit:
            entries = self._random.sample(entries, limit)
            entries.sort(key=lambda entry: entry.packet.name)
        candidates = [
            (entry.packet.name, entry.packet, entry, False) for entry in entries
        ]
        names = [candidate[0] for candidate in candidates]
        features = self.feature_builder.batch(
            candidates,
            statistics=node.statistics,
            store=node.store,
            now=now,
            device=self.device,
        )
        assert node.context is not None
        logits = self.model.score_logits(names, features, node.context)
        targets = torch.tensor(
            [1.0 if name in future_reuse else 0.0 for name in names],
            dtype=torch.float32,
            device=self.device,
        )
        return F.binary_cross_entropy_with_logits(logits, targets)

    def score_cache(self, node_id: str, now: float) -> dict[str, float]:
        node = self._node(node_id)
        if node.context is None or len(node.store) == 0:
            return {}
        with torch.no_grad():
            names, _, logits = self._candidate_rows(
                node,
                incoming=None,
                include_incoming=False,
                now=now,
            )
            values = torch.sigmoid(logits).cpu().tolist()
        return dict(zip(names, (float(value) for value in values)))

    def update(self, *, terminal: bool = False) -> UpdateSummary | None:
        if self.pending_requests == 0:
            return None

        action_terms: list[tuple[Tensor, float]] = []
        entropy_terms: list[Tensor] = []
        auxiliary_terms: list[Tensor] = []
        for steps in self.rollouts.values():
            future_return = (
                0.0
                if terminal or not self._baseline_initialized
                else self.running_return_baseline
            )
            returns = [0.0] * len(steps)
            for index in range(len(steps) - 1, -1, -1):
                returns[index] = future_return
                future_return = (
                    steps[index].reward + self.config.gamma * future_return
                )
            for step, reward_to_go in zip(steps, returns):
                if step.log_probabilities:
                    action_terms.append(
                        (torch.stack(step.log_probabilities).sum(), reward_to_go)
                    )
                entropy_terms.extend(step.entropies)
                if step.auxiliary_loss is not None:
                    auxiliary_terms.append(step.auxiliary_loss)

        if not action_terms and not auxiliary_terms:
            self._finish_update_boundary()
            return None

        policy_loss = torch.zeros((), device=self.device)
        if action_terms:
            returns_tensor = torch.tensor(
                [value for _, value in action_terms],
                dtype=torch.float32,
                device=self.device,
            )
            baseline = (
                self.running_return_baseline if self._baseline_initialized else 0.0
            )
            advantages = returns_tensor - baseline
            if len(action_terms) > 1:
                std = advantages.std(unbiased=False)
                if float(std) > 1e-6:
                    advantages = (advantages - advantages.mean()) / (std + 1e-8)
            policy_loss = torch.stack(
                [
                    -log_probability * advantage.detach()
                    for (log_probability, _), advantage in zip(
                        action_terms,
                        advantages,
                    )
                ]
            ).mean()
            mean_return = float(returns_tensor.mean().detach().cpu())
            if not self._baseline_initialized:
                self.running_return_baseline = mean_return
                self._baseline_initialized = True
            else:
                momentum = self.config.baseline_momentum
                self.running_return_baseline = (
                    momentum * self.running_return_baseline
                    + (1.0 - momentum) * mean_return
                )

        entropy = (
            torch.stack(entropy_terms).mean()
            if entropy_terms
            else torch.zeros((), device=self.device)
        )
        auxiliary_loss = (
            torch.stack(auxiliary_terms).mean()
            if auxiliary_terms
            else torch.zeros((), device=self.device)
        )
        loss = (
            policy_loss
            - self.config.entropy_coefficient * entropy
            + self.config.auxiliary_coefficient * auxiliary_loss
        )
        if not bool(torch.isfinite(loss)):
            self._finish_update_boundary()
            raise FloatingPointError("non-finite neural cache loss")

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip,
            error_if_nonfinite=True,
        )
        self.optimizer.step()
        summary = UpdateSummary(
            loss=float(loss.detach().cpu()),
            policy_loss=float(policy_loss.detach().cpu()),
            auxiliary_loss=float(auxiliary_loss.detach().cpu()),
            entropy=float(entropy.detach().cpu()),
            decision_steps=len(action_terms),
            calibration_batches=len(auxiliary_terms),
            gradient_norm=float(gradient_norm_tensor.detach().cpu()),
        )
        self.last_update = summary
        self._finish_update_boundary()
        return summary

    def _finish_update_boundary(self) -> None:
        for node in self.nodes.values():
            node.recurrent_state = self.model.detach_state(node.recurrent_state)
            if node.context is not None:
                node.context = node.context.detach()
        self.rollouts.clear()
        self.pending_requests = 0

    def flush(self) -> UpdateSummary | None:
        return self.update(terminal=True)

    def maybe_update(self) -> UpdateSummary | None:
        if not self._deferred_update_due:
            return None
        self._deferred_update_due = False
        return self.update(terminal=False)

    def parameters_are_finite(self) -> bool:
        return all(
            bool(torch.isfinite(parameter).all())
            for parameter in self.model.parameters()
        )
