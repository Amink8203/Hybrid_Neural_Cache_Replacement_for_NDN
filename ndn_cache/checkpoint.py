
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, TrainingConfig
from .model import NeuralCacheModel


def save_checkpoint(
    path: str | Path,
    model: NeuralCacheModel,
    *,
    training_config: TrainingConfig | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "training_config": (
            training_config.to_dict() if training_config is not None else None
        ),
        "metadata": metadata or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, destination)
    return destination


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[NeuralCacheModel, TrainingConfig | None, dict[str, Any]]:

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    model_config = ModelConfig.from_dict(payload["model_config"])
    model = NeuralCacheModel(model_config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    training_values = payload.get("training_config")
    training_config = (
        TrainingConfig.from_dict(training_values)
        if training_values is not None
        else None
    )
    metadata = dict(payload.get("metadata", {}))
    if "optimizer_state" in payload:
        metadata["_optimizer_state"] = payload["optimizer_state"]
    return model, training_config, metadata
