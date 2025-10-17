"""Minimal DQN agent for the steak environment."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from perfect_steak.steak_env import SteakEnv


@dataclass
class DQNConfig:
    episodes: int = 250
    max_steps_per_episode: int = 300
    buffer_size: int = 50_000
    batch_size: int = 128
    gamma: float = 0.99
    lr: float = 3e-4
    target_update_interval: int = 500
    warmup_steps: int = 2000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 25_000
    seed: int = 37


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
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


def linear_epsilon(step: int, config: DQNConfig) -> float:
    slope = (config.epsilon_end - config.epsilon_start) / max(config.epsilon_decay, 1)
    epsilon = config.epsilon_start + slope * step
    return float(np.clip(epsilon, config.epsilon_end, config.epsilon_start))


def optimize_model(
    policy_net: QNetwork,
    target_net: QNetwork,
    optimizer: optim.Optimizer,
    replay: ReplayBuffer,
    batch_size: int,
    gamma: float,
    device: torch.device,
) -> float:
    if len(replay) < batch_size:
        return 0.0

    states, actions, rewards, next_states, dones = replay.sample(batch_size)
    states_t = torch.from_numpy(states).to(device)
    actions_t = torch.from_numpy(actions).unsqueeze(-1).to(device)
    rewards_t = torch.from_numpy(rewards).unsqueeze(-1).to(device)
    next_states_t = torch.from_numpy(next_states).to(device)
    dones_t = torch.from_numpy(dones).unsqueeze(-1).to(device)

    q_values = policy_net(states_t).gather(1, actions_t)
    with torch.no_grad():
        next_q = target_net(next_states_t).max(dim=1, keepdim=True)[0]
        target_q = rewards_t + gamma * (1.0 - dones_t) * next_q

    loss = nn.functional.mse_loss(q_values, target_q)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def train(config: DQNConfig = DQNConfig()) -> None:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SteakEnv()
    obs_dim = env.observation_space.shape[0]  # type: ignore[union-attr]
    action_dim = env.action_space.n  # type: ignore[union-attr]

    policy_net = QNetwork(obs_dim, action_dim).to(device)
    target_net = QNetwork(obs_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.lr)
    replay = ReplayBuffer(config.buffer_size)

    global_step = 0
    recent_returns: Deque[float] = deque(maxlen=20)

    for episode in range(1, config.episodes + 1):
        obs, _ = env.reset()
        total_reward = 0.0
        epsilon_display = 1.0

        for step in range(config.max_steps_per_episode):
            epsilon = linear_epsilon(global_step, config) if global_step > config.warmup_steps else 1.0
            epsilon_display = epsilon

            if random.random() < epsilon:
                action = env.action_space.sample()  # type: ignore[union-attr]
            else:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
                    action = int(policy_net(obs_t).argmax(dim=1).item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.push(obs, action, reward, next_obs, done)

            obs = next_obs
            total_reward += reward
            global_step += 1

            if global_step > config.warmup_steps:
                loss = optimize_model(
                    policy_net, target_net, optimizer, replay, config.batch_size, config.gamma, device
                )
                if global_step % config.target_update_interval == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            if done:
                break

        recent_returns.append(total_reward)
        if episode % 10 == 0:
            mean_return = np.mean(recent_returns) if recent_returns else 0.0
            print(
                f"Episode {episode:03d} | "
                f"last_reward={total_reward:.3f} | "
                f"mean_20={mean_return:.3f} | "
                f"epsilon={epsilon_display:.3f}"
            )

    print("Training complete.")


if __name__ == "__main__":
    train()
