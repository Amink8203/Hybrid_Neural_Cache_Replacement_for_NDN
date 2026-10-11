
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class ModelConfig:

    vocab_size: int = 4096
    max_name_components: int = 16
    embedding_dim: int = 32
    name_hidden_dim: int = 48
    context_hidden_dim: int = 64
    head_hidden_dim: int = 64
    feature_dim: int = 8
    context_layers: int = 1
    dropout: float = 0.0
    popularity_prior_weight: float = 1.0
    recency_prior_weight: float = 0.0
    residual_scale: float = 0.05

    def validate(self) -> None:
        if self.vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        for field_name in (
            "max_name_components",
            "embedding_dim",
            "name_hidden_dim",
            "context_hidden_dim",
            "head_hidden_dim",
            "feature_dim",
            "context_layers",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.popularity_prior_weight < 0:
            raise ValueError("popularity_prior_weight cannot be negative")
        if self.recency_prior_weight < 0:
            raise ValueError("recency_prior_weight cannot be negative")
        if self.residual_scale < 0:
            raise ValueError("residual_scale cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        config = cls(**values)
        config.validate()
        return config


@dataclass(slots=True)
class TrainingConfig:

    learning_rate: float = 3e-4
    gamma: float = 0.99
    rollout_length: int = 256
    temperature: float = 0.7
    entropy_coefficient: float = 0.0
    auxiliary_coefficient: float = 0.25
    prediction_horizon: int = 32
    calibration_samples: int = 16
    replacement_shortlist_size: int = 32
    gradient_clip: float = 1.0
    baseline_momentum: float = 0.95
    allow_bypass: bool = False
    recency_scale: float = 100.0
    popularity_decay: float = 32.0

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.rollout_length <= 1:
            raise ValueError("rollout_length must be greater than 1")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if self.entropy_coefficient < 0 or self.auxiliary_coefficient < 0:
            raise ValueError("loss coefficients cannot be negative")
        if self.prediction_horizon <= 0:
            raise ValueError("prediction_horizon must be positive")
        if self.calibration_samples < 0:
            raise ValueError("calibration_samples cannot be negative")
        if self.replacement_shortlist_size <= 0:
            raise ValueError("replacement_shortlist_size must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if not 0.0 <= self.baseline_momentum < 1.0:
            raise ValueError("baseline_momentum must be in [0, 1)")
        if self.recency_scale <= 0:
            raise ValueError("recency_scale must be positive")
        if self.popularity_decay <= 0:
            raise ValueError("popularity_decay must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainingConfig":
        config = cls(**values)
        config.validate()
        return config
