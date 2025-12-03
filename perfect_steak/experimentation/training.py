"""Training harness for DQN SteakEnv experiments."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from perfect_steak.steak_env import SteakEnv

from .config import ExperimentConfig
from .dqn_agent import QNetwork, ReplayBuffer


OPTIMAL_SCORE = 2.7


@dataclass(slots=True)
class EpisodeMetrics:
    episode: int
    reward: float
    epsilon: float
    timestamp_s: float
    moving_avg: float


@dataclass(slots=True)
class EvaluationStep:
    step: int
    action: int
    action_name: str
    reward: float
    observation: List[float]
    core_temp_c: float
    top_surface_temp_c: float
    bottom_surface_temp_c: float
    browning_top: float
    browning_bottom: float
    time_elapsed_s: float
    layer_temps: List[float] | None = None


@dataclass(slots=True)
class EvaluationEpisode:
    episode_index: int
    total_reward: float
    steps: List[EvaluationStep] = field(default_factory=list)
    final_info: Dict[str, float] | None = None


@dataclass(slots=True)
class TrainingRunResult:
    config: ExperimentConfig
    episode_metrics: List[EpisodeMetrics]
    evaluation_episodes: List[EvaluationEpisode]
    best_eval_score: float
    best_eval_episode: int
    best_model_state: Dict[str, torch.Tensor]
    time_to_optimal_episode: int | None
    total_training_seconds: float


class DQNTrainer:
    """Train a configurable DQN on SteakEnv while logging metrics."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.env = SteakEnv()
        self.eval_env = SteakEnv()
        obs, _ = self.env.reset(seed=self.config.seed)
        self.eval_env.reset(seed=self.config.seed + 1)
        self.state_dim = (
            self.env.observation_space.shape[0]
            if getattr(self.env, "observation_space", None) is not None
            else len(obs)
        )
        if getattr(self.env, "action_space", None) is not None:
            self.n_actions = self.env.action_space.n  # type: ignore[assignment]
        else:
            self.n_actions = len(self.env.ACTIONS)
        self._eval_seed_counter = 0

    def train(self) -> TrainingRunResult:
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        q_net = QNetwork(
            self.state_dim, self.n_actions, self.config.hidden_layer_sizes
        ).to(self.device)
        q_target = QNetwork(
            self.state_dim, self.n_actions, self.config.hidden_layer_sizes
        ).to(self.device)
        q_target.load_state_dict(q_net.state_dict())
        q_target.eval()

        optimizer = optim.Adam(q_net.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.SmoothL1Loss()
        replay_buffer = ReplayBuffer(self.config.replay_buffer_size)

        episode_logs: List[EpisodeMetrics] = []
        epsilon = self.config.epsilon_start
        total_steps = 0
        best_eval_score = -math.inf
        best_eval_episode = -1
        best_state_dict: Dict[str, torch.Tensor] | None = None
        time_to_optimal: int | None = None
        moving_window: List[float] = []
        start_time = time.time()

        for episode in range(1, self.config.num_episodes + 1):
            state, _ = self.env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action = self._epsilon_greedy(q_net, state, epsilon)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                clipped_reward = float(
                    np.clip(reward, -self.config.reward_clip, self.config.reward_clip)
                )
                replay_buffer.append(state, action, clipped_reward, next_state, done)
                self._train_step(q_net, q_target, optimizer, loss_fn, replay_buffer)

                total_reward += clipped_reward
                state = next_state
                total_steps += 1

                if total_steps % self.config.target_update_freq == 0:
                    q_target.load_state_dict(q_net.state_dict())

            epsilon = max(epsilon * self.config.epsilon_decay, self.config.epsilon_min)
            moving_window.append(total_reward)
            if len(moving_window) > self.config.moving_avg_window:
                moving_window.pop(0)
            moving_avg = float(np.mean(moving_window))
            timestamp_s = time.time() - start_time
            episode_logs.append(
                EpisodeMetrics(
                    episode=episode,
                    reward=total_reward,
                    epsilon=epsilon,
                    timestamp_s=timestamp_s,
                    moving_avg=moving_avg,
                )
            )
            if moving_avg >= OPTIMAL_SCORE and time_to_optimal is None:
                time_to_optimal = episode

            if episode % self.config.eval_interval == 0:
                eval_score = self._evaluate_policy(q_net)
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_eval_episode = episode
                    best_state_dict = {
                        k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()
                    }

        total_training_seconds = time.time() - start_time
        final_evaluations = self._evaluate_policy(q_net, record_actions=True)

        if best_state_dict is None:
            best_state_dict = {
                k: v.detach().cpu().clone() for k, v in q_net.state_dict().items()
            }
            best_eval_score = final_evaluations[0].total_reward if final_evaluations else 0.0
            best_eval_episode = self.config.num_episodes

        return TrainingRunResult(
            config=self.config,
            episode_metrics=episode_logs,
            evaluation_episodes=final_evaluations,
            best_eval_score=best_eval_score,
            best_eval_episode=best_eval_episode,
            best_model_state=best_state_dict,
            time_to_optimal_episode=time_to_optimal,
            total_training_seconds=total_training_seconds,
        )

    def _epsilon_greedy(
        self, q_net: QNetwork, state: np.ndarray, epsilon: float
    ) -> int:
        if np.random.rand() < epsilon:
            return int(np.random.randint(0, self.n_actions))
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(q_net(state_tensor).argmax(dim=1).item())

    def _train_step(
        self,
        q_net: QNetwork,
        q_target: QNetwork,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        replay_buffer: ReplayBuffer,
    ) -> None:
        if len(replay_buffer) < self.config.batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(
            self.config.batch_size
        )
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        current_q = q_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = q_target(next_states_t).max(dim=1)[0]
            targets = rewards_t + self.config.gamma * next_q * (1 - dones_t)
        loss = loss_fn(current_q, targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), self.config.max_grad_norm)
        optimizer.step()

    def _evaluate_policy(
        self,
        q_net: QNetwork,
        *,
        record_actions: bool = False,
    ) -> List[EvaluationEpisode] | float:
        episodes: List[EvaluationEpisode] = []
        total_rewards: List[float] = []
        for episode_idx in range(self.config.eval_episodes):
            eval_seed = self.config.seed + 10_000 + self._eval_seed_counter
            self._eval_seed_counter += 1
            state, _ = self.eval_env.reset(seed=eval_seed)
            done = False
            episode_reward = 0.0
            step_logs: List[EvaluationStep] = []
            while not done:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = int(q_net(state_tensor).argmax(dim=1).item())
                next_state, reward, terminated, truncated, info = self.eval_env.step(
                    action
                )
                done = terminated or truncated
                episode_reward += reward

                if record_actions:
                    step_logs.append(
                        EvaluationStep(
                            step=len(step_logs),
                            action=action,
                            action_name=self.eval_env.ACTIONS.get(action, str(action)),
                            reward=float(reward),
                            observation=state.tolist(),
                            core_temp_c=float(info.get("core_temp_c", 0.0)),
                            top_surface_temp_c=float(info.get("top_surface_temp_c", 0.0)),
                            bottom_surface_temp_c=float(info.get("bottom_surface_temp_c", 0.0)),
                            browning_top=float(info.get("browning_top", 0.0)),
                            browning_bottom=float(info.get("browning_bottom", 0.0)),
                            time_elapsed_s=float(info.get("time_elapsed_s", 0.0)),
                            layer_temps=(
                                info.get("layer_temps").tolist()
                                if isinstance(info.get("layer_temps"), np.ndarray)
                                else info.get("layer_temps")
                            ),
                        )
                    )

                state = next_state

            total_rewards.append(episode_reward)
            if record_actions:
                episodes.append(
                    EvaluationEpisode(
                        episode_index=episode_idx,
                        total_reward=float(episode_reward),
                        steps=step_logs,
                        final_info=self._final_info_from_steps(step_logs),
                    )
                )

        if record_actions:
            return episodes
        return float(np.mean(total_rewards)) if total_rewards else 0.0

    @staticmethod
    def _final_info_from_steps(
        steps: Sequence[EvaluationStep],
    ) -> Dict[str, float] | None:
        if not steps:
            return None
        last = steps[-1]
        return {
            "core_temp_c": last.core_temp_c,
            "top_surface_temp_c": last.top_surface_temp_c,
            "bottom_surface_temp_c": last.bottom_surface_temp_c,
            "browning_top": last.browning_top,
            "browning_bottom": last.browning_bottom,
            "time_elapsed_s": last.time_elapsed_s,
        }
