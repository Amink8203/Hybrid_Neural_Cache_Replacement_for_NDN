
from __future__ import annotations

from typing import TypeAlias

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .names import NameTokenizer

RecurrentState: TypeAlias = tuple[Tensor, Tensor]


class NeuralCacheModel(nn.Module):

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.tokenizer = NameTokenizer(
            vocab_size=config.vocab_size,
            max_components=config.max_name_components,
        )
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            padding_idx=NameTokenizer.PAD_ID,
        )
        self.name_lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.name_hidden_dim,
            batch_first=True,
        )
        context_dropout = config.dropout if config.context_layers > 1 else 0.0
        self.context_lstm = nn.LSTM(
            input_size=config.name_hidden_dim,
            hidden_size=config.context_hidden_dim,
            num_layers=config.context_layers,
            dropout=context_dropout,
            batch_first=True,
        )
        head_input = (
            config.name_hidden_dim
            + config.context_hidden_dim
            + config.feature_dim
        )
        self.score_head = nn.Sequential(
            nn.Linear(head_input, config.head_hidden_dim),
            nn.LayerNorm(config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden_dim, 1),
        )
        self.feature_prior = nn.Linear(config.feature_dim, 1, bias=False)
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def reset_parameters(self) -> None:

        for name, parameter in self.named_parameters():
            if parameter.ndim >= 2:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)
        final = self.score_head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        nn.init.zeros_(self.feature_prior.weight)
        with torch.no_grad():
            self.feature_prior.weight[0, 0] = (
                self.config.popularity_prior_weight
            )
            if self.config.feature_dim > 1:
                self.feature_prior.weight[0, 1] = (
                    self.config.recency_prior_weight
                )
        self.feature_prior.weight.requires_grad_(False)

    def encode_names(self, names: list[str]) -> Tensor:

        if not names:
            return torch.empty(
                (0, self.config.name_hidden_dim),
                dtype=torch.float32,
                device=self.device,
            )
        tokens, lengths = self.tokenizer.batch_encode(names, device=self.device)
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.name_lstm(packed)
        return hidden[-1]

    def advance_context(
        self,
        requested_name: str,
        state: RecurrentState | None,
    ) -> tuple[Tensor, RecurrentState]:

        name_vector = self.encode_names([requested_name]).unsqueeze(1)
        output, next_state = self.context_lstm(name_vector, state)
        return output[:, -1, :], next_state

    def score_logits(
        self,
        names: list[str],
        features: Tensor,
        context: Tensor,
    ) -> Tensor:

        if features.ndim != 2:
            raise ValueError("features must have shape [candidates, feature_dim]")
        if features.shape != (len(names), self.config.feature_dim):
            raise ValueError(
                "features shape does not match candidate count/model feature_dim"
            )
        if context.ndim != 2 or context.shape != (
            1,
            self.config.context_hidden_dim,
        ):
            raise ValueError(
                "context must have shape [1, context_hidden_dim] for one node"
            )
        if not names:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        name_vectors = self.encode_names(names)
        expanded_context = context.expand(len(names), -1)
        inputs = torch.cat(
            (
                name_vectors,
                expanded_context,
                features.to(device=self.device, dtype=torch.float32),
            ),
            dim=1,
        )
        numeric = features.to(device=self.device, dtype=torch.float32)
        residual = self.score_head(inputs).squeeze(-1)
        return (
            self.feature_prior(numeric).squeeze(-1)
            + self.config.residual_scale * torch.tanh(residual)
        )

    def score(
        self,
        names: list[str],
        features: Tensor,
        context: Tensor,
    ) -> Tensor:

        return torch.sigmoid(self.score_logits(names, features, context))

    @staticmethod
    def detach_state(state: RecurrentState | None) -> RecurrentState | None:

        if state is None:
            return None
        return state[0].detach(), state[1].detach()
