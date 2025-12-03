"""Configuration helpers for experimentation scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List


@dataclass(slots=True)
class ExperimentConfig:
    """Full configuration describing a single training run."""

    id: str
    hidden_layer_sizes: List[int]
    seed: int = 0
    gamma: float = 0.99
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.999
    num_episodes: int = 3000
    batch_size: int = 128
    replay_buffer_size: int = 50_000
    target_update_freq: int = 500
    eval_interval: int = 250
    eval_episodes: int = 5
    moving_avg_window: int = 50
    reward_clip: float = 5.0
    max_grad_norm: float = 1.0
    notes: str | None = None

    def with_overrides(
        self,
        *,
        episodes: int | None = None,
        eval_interval: int | None = None,
        batch_size: int | None = None,
        eval_episodes: int | None = None,
    ) -> "ExperimentConfig":
        """Return a copy overriding frequently tweaked attributes."""
        return replace(
            self,
            num_episodes=episodes if episodes is not None else self.num_episodes,
            eval_interval=(
                eval_interval if eval_interval is not None else self.eval_interval
            ),
            batch_size=batch_size if batch_size is not None else self.batch_size,
            eval_episodes=(
                eval_episodes if eval_episodes is not None else self.eval_episodes
            ),
        )


@dataclass(slots=True)
class SweepConfig:
    """Collection of experiment definitions loaded from JSON/YAML."""

    experiments: List[ExperimentConfig] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "SweepConfig":
        """Parse a JSON file into ExperimentConfig objects."""
        data = cls._load_file(path)
        if isinstance(data, dict) and "experiments" in data:
            raw_experiments = data["experiments"]
        elif isinstance(data, list):
            raw_experiments = data
        else:
            raise ValueError("Configuration must be a list or contain 'experiments'")
        experiments = [cls._parse_item(item) for item in raw_experiments]
        return cls(experiments=experiments)

    @staticmethod
    def _load_file(path: str | Path) -> Any:
        payload = Path(path).read_text()
        return json.loads(payload)

    @staticmethod
    def _parse_item(item: Dict[str, Any]) -> ExperimentConfig:
        if "hidden_layers" in item and "hidden_layer_sizes" not in item:
            item = dict(item)
            item["hidden_layer_sizes"] = item.pop("hidden_layers")
        if "id" not in item:
            raise ValueError("Each experiment must define an 'id'")
        if "hidden_layer_sizes" not in item:
            raise ValueError("Each experiment must define 'hidden_layer_sizes'")
        return ExperimentConfig(**item)
