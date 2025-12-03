"""Neural network and buffer utilities for SteakEnv DQN experiments."""

from __future__ import annotations

import random
from collections import deque
from typing import Deque, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Fully-connected Q-network with configurable hidden layers."""

    def __init__(
        self, state_dim: int, n_actions: int, hidden_layer_sizes: Iterable[int]
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, n_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class ReplayBuffer:
    """Fixed-size experience buffer supporting random sampling."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, float]] = deque(
            maxlen=capacity
        )

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, float(done)))

    def sample(
        self, sample_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, sample_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            states.astype(np.float32),
            actions.astype(np.int64),
            rewards.astype(np.float32),
            next_states.astype(np.float32),
            dones.astype(np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)
