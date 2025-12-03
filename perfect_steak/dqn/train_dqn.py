"""DQN agent for the steak environment using the Lab 6 notebook structure."""

import random
from collections import deque
from typing import Deque, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from perfect_steak.steak_env import SteakEnv

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
env = SteakEnv()

if getattr(env, "action_space", None) is not None:
    n_actions = env.action_space.n  # type: ignore
else:
    n_actions = len(env.ACTIONS)

# Reset to get correct state_dim (now 6 instead of 7)
obs_reset, _ = env.reset()
if getattr(env, "observation_space", None) is not None:
    state_dim = env.observation_space.shape[0]  # type: ignore
else:
    state_dim = len(obs_reset)

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
gamma = 0.99
alpha = 1e-4  # Slight increase back to 1e-4 since episodes are shorter/cleaner
epsilon = 1.0
epsilon_min = 0.02
epsilon_decay = 0.999  # Slower decay to ensure exploration in new state space
num_episodes = 5000 
batch_size = 128
replay_buffer_size = 50_000
target_update_freq = 500  # More frequent updates due to shorter episodes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------------------------------------------------------
# Function approximator
# -----------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


q_net = QNetwork().to(device)
q_net_target = QNetwork().to(device)
q_net_target.load_state_dict(q_net.state_dict())
q_net_target.eval()

optimizer = optim.Adam(q_net.parameters(), lr=alpha)
loss_fn = nn.SmoothL1Loss()


# -----------------------------------------------------------------------------
# Replay buffer
# -----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
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


replay_buffer = ReplayBuffer(replay_buffer_size)
eval_env = SteakEnv()


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def epsilon_greedy(state: np.ndarray, eps: float) -> int:
    if random.random() < eps:
        return random.randrange(n_actions)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        return int(q_net(state_tensor).argmax(dim=1).item())


def train_step() -> None:
    if len(replay_buffer) < batch_size:
        return

    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    current_q = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_q = q_net_target(next_states).max(dim=1)[0]
        targets = rewards + gamma * next_q * (1 - dones)

    loss = loss_fn(current_q, targets)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()


def update_target_network() -> None:
    q_net_target.load_state_dict(q_net.state_dict())
    q_net_target.eval()


def evaluate_policy(episodes: int = 5) -> float:
    scores = []
    for _ in range(episodes):
        state, _ = eval_env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = int(q_net(state_tensor).argmax(dim=1).item())
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
        scores.append(episode_reward)
    return float(np.mean(scores))


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------
episode_returns: list[float] = []
total_steps = 0

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0.0
    last_info = {}

    while not done:
        action = epsilon_greedy(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        if done:
            last_info = info
        
        clipped_reward = float(np.clip(reward, -5.0, 5.0))
        replay_buffer.append(state, action, clipped_reward, next_state, done)
        train_step()

        total_steps += 1
        if total_steps % target_update_freq == 0:
            update_target_network()

        state = next_state
        total_reward += clipped_reward

    episode_returns.append(total_reward)
    epsilon = max(epsilon * epsilon_decay, epsilon_min)

    if (episode + 1) % 100 == 0 and last_info:
        print(
            "Final steak snapshot -> "
            f"episode {episode + 1}, "
            f"reward={total_reward:.3f}, "
            f"core={last_info.get('core_temp_c', 0):.1f}°C, "
            f"top_brown={last_info.get('browning_top', 0):.1f}, "
            f"bottom_brown={last_info.get('browning_bottom', 0):.1f}, "
            f"time={last_info.get('time_elapsed_s', 0):.1f}s"
        )
    if (episode + 1) % 250 == 0:
        eval_score = evaluate_policy()
        print(f"[Eval @ episode {episode + 1}] average return={eval_score:.3f}")


# -----------------------------------------------------------------------------
# Plot training performance
# -----------------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(episode_returns)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("DQN Training Performance (SteakEnv)")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# Evaluation with greedy policy
# -----------------------------------------------------------------------------
eval_episodes = 5
eval_scores: list[float] = []

for ep in range(eval_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0.0

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = int(q_net(state_tensor).argmax(dim=1).item())
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward

    eval_scores.append(total_reward)
    print(f"Evaluation episode {ep + 1}: return={total_reward:.2f}")

print(f"\nAverage evaluation return over {eval_episodes} episodes: {np.mean(eval_scores):.2f}")