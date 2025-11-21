"""Simplified steak-cooking reinforcement learning environment.

The module implements a one-dimensional heat-transfer simulation inspired by the
project brief. Each time step models conduction between discrete layers of the
steak as well as high-level Maillard browning dynamics. The resulting
environment exposes a Gymnasium-compatible interface and can be used both for
heuristic rollouts and learning with DQN.

The implementation intentionally favors transparency over physical accuracy so
extensive inline comments explain every approximation that is being made.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - fallback if gymnasium missing
    gym = None  # type: ignore
    spaces = None  # type: ignore


@dataclass
class SteakEnvConfig:
    """Container for every tunable constant in the environment.

    The defaults follow the narrative from the proposal 
    (1.25 inch steak, medium-rare target). Additional coefficients such as
    ``surface_coupling_bottom`` are provided so the thermal response can be
    calibrated without rewriting the solver.
    """

    n_layers: int = 21  # Number of finite-difference layers through the steak.
    steak_thickness_m: float = 0.03175  # Total thickness (1.25 in) represented by the grid.
    thermal_diffusivity: float = 2.0e-7  # Thermal diffusivity (m^2/s) of thawed beef.
    time_step_s: float = 1.0  # Simulation step in seconds.
    initial_temp_c: float = 21.1  # Starting temperature of every layer (~70 °F).
    ambient_temp_c: float = 23.0  # Room temperature that the top surface exchanges with.
    heat_settings_c: Dict[str, float] = field(
        default_factory=lambda: {"low": 121.0, "medium": 176.0, "high": 232.0}
    )  # Burner set-points (°C) for low/medium/high heat settings.
    browning_threshold_c: float = 140.0  # Temperature where Maillard reactions begin.
    browning_burn_c: float = 180.0  # Surface temperature beyond which burning accelerates.
    browning_gain: float = 0.0006  # Quadratic coefficient (k) from the browning model.
    browning_cap: float = 180.0  # Upper limit to keep browning values bounded.
    browning_target: float = 100.0  # Idealized brownness score (used in reward shaping).
    browning_sigma: float = 20.0  # Spread of the Gaussian sear preference.
    core_target_c: float = 56.0  # Desired core temp (medium-rare) for the reward peak.
    core_sigma: float = 3.0  # Spread of the Gaussian doneness preference.
    burn_core_c: float = 74.0  # Internal temperature that triggers a burn penalty.
    burn_penalty: float = 2.5  # Penalty subtracted when the steak burns.
    raw_core_c: float = 40.0  # Below this temperature the steak is considered raw.
    raw_penalty: float = 1.0  # Penalty for removing a raw steak.
    encourage_core_c: float = 30.0  # Threshold to discourage instant removal.
    remove_reward_raw_penalty: float = 0.1  # Penalty when removing before encourage_core_c.
    remove_reward_cooked_bonus: float = 0.2  # Bonus when removing after threshold + margin.
    heat_progress_reward: float = 0.001  # Reward per degree increase each step.
    maillard_progress_reward: float = 0.0005  # Reward per unit browning progress.
    doneness_weight: float = 1.5  # Importance of internal temperature in the final reward.
    sear_weight: float = 1.0  # Importance of surface browning in the final reward.
    time_penalty_per_second: float = 0.001  # Per-second penalty to encourage faster cooking.
    max_duration_s: float = 900.0  # Hard cap on episode length (15 minutes).
    surface_coupling_bottom: float = 3.0  # Conductive coupling coefficient for the pan side.
    surface_coupling_top: float = 1.5  # Conductive coupling for the top interior layer.
    ambient_cooling_top: float = 0.25  # How aggressively ambient air cools the exposed surface.

    def __post_init__(self) -> None:
        """Validate configuration state."""
        if self.n_layers < 3:
            raise ValueError("n_layers must be at least 3 for finite difference")


class SteakEnv(gym.Env if gym else object):
    """One-dimensional steak cooking environment with Maillard browning.

    When Gymnasium is available, the class derives from ``gym.Env`` so it can
    plug directly into standard RL tooling. Otherwise it still behaves like a
    regular Python object offering ``reset``/``step`` but without registered
    spaces.
    """

    metadata = {"render_modes": []}

    ACTIONS = {
        0: "wait",
        1: "flip",
        2: "set_low",
        3: "set_medium",
        4: "set_high",
        5: "remove",
    }

    def __init__(self, config: SteakEnvConfig | None = None) -> None:
        """Prepare environment state and calculate finite-difference constants."""
        self.config = config or SteakEnvConfig()
        self.dx = self.config.steak_thickness_m / (self.config.n_layers - 1)
        # ``ratio`` captures (alpha * dt / dx^2) and governs stability of the explicit solver.
        self.ratio = (
            self.config.thermal_diffusivity
            * self.config.time_step_s
            / (self.dx**2)
        )
        # Register discrete action space if Gym is available; otherwise keep a
        # ``None`` sentinel so downstream callers can branch gracefully.
        self.action_space = (
            spaces.Discrete(len(self.ACTIONS)) if spaces else None  # type: ignore[assignment]
        )

        # Observations expose the key indicators available to the chef-like agent:
        # core temperature, sear levels, current burner setting, and elapsed time.
        high = np.array(
            [
                250.0,  # core temperature upper bound
                5.0,  # browning top
                5.0,  # browning bottom
                max(self.config.heat_settings_c.values()),
                self.config.max_duration_s,
            ],
            dtype=np.float32,
        )
        low = np.zeros_like(high)
        low[0] = 0.0  # core temp cannot be negative
        self.observation_space = (
            spaces.Box(low=low, high=high, dtype=np.float32) if spaces else None  # type: ignore[assignment]
        )

        # Internal simulation buffers are initialized during ``reset``.
        self.layer_temps: np.ndarray | None = None  # Discretized temperature profile (°C).
        self.browning_top: float = 0.0  # Accumulated Maillard score for the top surface.
        self.browning_bottom: float = 0.0  # Accumulated Maillard score for the bottom surface.
        self.pan_temp: float = self.config.heat_settings_c["medium"]  # Active burner temperature.
        self.time_elapsed: float = 0.0  # Total simulated seconds since the episode began.
        self.cumulative_penalty: float = 0.0  # Aggregated time penalties used in final reward.
        self.last_action: str = "wait"  # Human-readable name of the most recent action.
        self._rng = np.random.default_rng()  # RNG instance used for sampling actions/seeds.

    # -- core environment API -------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, dict]:
        """Reinitialize temperature grid and brownness metrics.

        Parameters mirror Gymnasium's extended signature so training code can
        specify seeds or custom episode options.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.layer_temps = np.full(
            shape=(self.config.n_layers,), fill_value=self.config.initial_temp_c
        )
        self.browning_top = 0.0
        self.browning_bottom = 0.0
        self.pan_temp = self.config.heat_settings_c["medium"]
        self.time_elapsed = 0.0
        self.cumulative_penalty = 0.0
        self.last_action = "wait"
        obs = self._get_observation()
        info = {
            "layer_temps": self.layer_temps.copy(),
            "browning_top": self.browning_top,
            "browning_bottom": self.browning_bottom,
            "pan_temperature_c": self.pan_temp,
            "time_elapsed_s": self.time_elapsed,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:  # obs, reward, terminated, truncated, info
        """Advance the environment by one time step.

        Actions mutate high-level controls (heat level, flip, remove) while the
        solver handles conduction and browning. Rewards are only provided when
        the steak is removed or the episode is forcibly truncated.
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action {action}")
        assert self.layer_temps is not None, "reset() must be called before step()"

        self.last_action = self.ACTIONS[action]
        terminated = False
        truncated = False

        if action == 1:  # flip
            self._flip()
        elif action in (2, 3, 4):  # set heat
            levels = ["low", "medium", "high"]
            self.pan_temp = self.config.heat_settings_c[levels[action - 2]]
        elif action == 5:  # remove
            core_temp = float(self.layer_temps[len(self.layer_temps) // 2])
            if core_temp < 30.0 or self.time_elapsed < 30.0:
                # Pretend remove is a no-op until it is meaningful.
                reward = -self.config.remove_reward_raw_penalty
                obs = self._get_observation()
                info = self._build_info(reward, terminated, truncated)
                return obs, reward, terminated, truncated, info
            terminated = True
            reward = self._final_reward(removed=True)
            obs = self._get_observation()
            info = self._build_info(reward, terminated, truncated)
            return obs, reward, terminated, truncated, info

        # integrate dynamics
        prev_core = float(self.layer_temps[len(self.layer_temps) // 2])
        prev_browning = self.browning_top + self.browning_bottom
        self._apply_heat_step()
        curr_core = float(self.layer_temps[len(self.layer_temps) // 2])
        curr_browning = self.browning_top + self.browning_bottom
        incremental_reward = 0.0
        if curr_core > prev_core:
            incremental_reward += self.config.heat_progress_reward * (curr_core - prev_core)
        if curr_browning > prev_browning:
            incremental_reward += self.config.maillard_progress_reward * (curr_browning - prev_browning)
        if 40.0 <= curr_core <= 60.0:
            incremental_reward += 0.005
        self.time_elapsed += self.config.time_step_s
        self.cumulative_penalty -= (
            self.config.time_penalty_per_second * self.config.time_step_s
        )

        if self.time_elapsed >= self.config.max_duration_s:
            truncated = True
            terminated = True

        reward = incremental_reward
        if terminated:
            reward = self._final_reward(removed=False)

        obs = self._get_observation()
        info = self._build_info(reward, terminated, truncated)
        return obs, reward, terminated, truncated, info

    # -- helpers --------------------------------------------------------------
    def sample_action(self) -> int:
        """Draw a random action regardless of Gymnasium availability."""
        if self.action_space is None:
            return self._rng.integers(0, len(self.ACTIONS))  # type: ignore[return-value]
        return int(self.action_space.sample())  # type: ignore[return-value]

    def _flip(self) -> None:
        """Reverse temperature layers and swap browning scores."""
        assert self.layer_temps is not None
        self.layer_temps = self.layer_temps[::-1].copy()
        self.browning_top, self.browning_bottom = (
            self.browning_bottom,
            self.browning_top,
        )

    def _apply_heat_step(self) -> None:
        """Update layer temperatures using a simple explicit scheme."""
        assert self.layer_temps is not None
        new_temps = self.layer_temps.copy()

        # Interior finite-difference stencil (second derivative).
        for idx in range(1, self.config.n_layers - 1):
            laplacian = (
                self.layer_temps[idx - 1]
                - 2 * self.layer_temps[idx]
                + self.layer_temps[idx + 1]
            )
            new_temps[idx] += self.ratio * laplacian

        # Boundary layer for the bottom surface (pan contact). We blend the
        # conduction term with an empirical coupling factor to simulate direct
        # heat transfer from the burner.
        bottom_exchange = (
            self.config.surface_coupling_bottom * (self.pan_temp - self.layer_temps[0])
            + (self.layer_temps[1] - self.layer_temps[0])
        )
        new_temps[0] += self.ratio * bottom_exchange

        # Boundary for the top surface which exchanges heat with the ambient
        # environment. The ambient cooling term is intentionally gentle so the
        # steak does not behave as if it were exposed to freezing air.
        top_exchange = (
            self.config.ambient_cooling_top * (self.config.ambient_temp_c - self.layer_temps[-1])
            + self.config.surface_coupling_top * (self.layer_temps[-2] - self.layer_temps[-1])
        )
        new_temps[-1] += self.ratio * top_exchange

        self.layer_temps = new_temps
        self._update_browning()

    def _update_browning(self) -> None:
        """Grow Maillard browning levels based on surface temperatures."""
        assert self.layer_temps is not None

        dt = self.config.time_step_s
        for surface, idx in (("bottom", 0), ("top", -1)):
            surface_temp = self.layer_temps[idx]
            excess = max(0.0, surface_temp - self.config.browning_threshold_c)
            if excess <= 0.0:
                continue

            current = self.browning_bottom if surface == "bottom" else self.browning_top
            rate = self.config.browning_gain * (excess**2)
            if surface_temp >= self.config.browning_burn_c:
                rate *= 1.4
            delta = rate * dt

            if surface == "bottom":
                self.browning_bottom = min(
                    self.config.browning_cap, max(0.0, self.browning_bottom + delta)
                )
            else:
                self.browning_top = min(
                    self.config.browning_cap, max(0.0, self.browning_top + delta)
                )

    def _final_reward(self, *, removed: bool) -> float:
        """Evaluate the steak when the agent decides to finish cooking."""
        assert self.layer_temps is not None
        core_temp = float(self.layer_temps[len(self.layer_temps) // 2])
        avg_brown = (self.browning_top + self.browning_bottom) / 2.0

        # Doneness and sear terms rely on Gaussian-shaped preferences centered
        # around the ideal medium-rare temperature and a target brownness level.
        doneness = math.exp(
            -((core_temp - self.config.core_target_c) ** 2)
            / (2 * self.config.core_sigma**2)
        )
        sear = math.exp(
            -((avg_brown - self.config.browning_target) ** 2)
            / (2 * self.config.browning_sigma**2)
        )

        reward = (
                self.config.doneness_weight * doneness
                + self.config.sear_weight * sear
                + self.cumulative_penalty
        )

        # Heavy penalty when the steak overcooks or the browning exceeds the
        # maximum recommended cap. This encodes "burnt steak" outcomes.
        burn_limit = self.config.browning_target + 50.0
        burn = core_temp >= self.config.burn_core_c or (
            self.browning_top > burn_limit or self.browning_bottom > burn_limit
        )
        if burn:
            reward -= self.config.burn_penalty
        if core_temp <= self.config.raw_core_c:
            reward -= self.config.raw_penalty
        elif core_temp < self.config.encourage_core_c:
            reward -= self.config.remove_reward_raw_penalty
        elif core_temp >= self.config.encourage_core_c + 10.0:
            reward += self.config.remove_reward_cooked_bonus
        if not removed:
            # If truncated, treat as suboptimal finish.
            reward -= 0.5
        return reward

    def _get_observation(self) -> np.ndarray:
        """Construct the observation vector exposed to the agent."""
        assert self.layer_temps is not None
        core_temp = float(self.layer_temps[len(self.layer_temps) // 2])
        obs_raw = np.array(
            [
                core_temp,
                self.browning_top,
                self.browning_bottom,
                self.pan_temp,
                self.time_elapsed,
            ],
            dtype=np.float32,
        )
        # Normalize features to roughly 0-1 to help learning stability.
        normalization_bounds = np.array(
            [250.0, 150.0, 150.0, 250.0, self.config.max_duration_s], dtype=np.float32
        )
        obs = np.clip(obs_raw / normalization_bounds, 0.0, 1.0)
        return obs

    def _build_info(self, reward: float, terminated: bool, truncated: bool) -> dict:
        """Gather diagnostic information for logging or visualization."""
        assert self.layer_temps is not None
        info = {
            "core_temp_c": float(self.layer_temps[len(self.layer_temps) // 2]),
            "top_surface_temp_c": float(self.layer_temps[-1]),
            "bottom_surface_temp_c": float(self.layer_temps[0]),
            "browning_top": self.browning_top,
            "browning_bottom": self.browning_bottom,
            "pan_temperature_c": self.pan_temp,
            "time_elapsed_s": self.time_elapsed,
            "last_action": self.last_action,
            "terminated": terminated,
            "truncated": truncated,
            "reward": reward,
            "layer_temps": self.layer_temps.copy(),
        }
        return info


def _demo_rollout(steps: int = 120) -> None:
    """Standalone smoke test when the module is executed directly."""
    env = SteakEnv()
    obs, info = env.reset()
    print("Initial obs:", obs)
    print("Initial info:", {k: round(v, 2) if isinstance(v, float) else v for k, v in info.items()})
    for step in range(steps):
        action = env.sample_action()
        obs, reward, terminated, truncated, info = env.step(action)
        if (step + 1) % 10 == 0 or terminated:
            print(
                f"Step {step+1:03d} | action={info['last_action']:<10} | "
                f"core={info['core_temp_c']:.1f}°C | "
                f"brown(top/bot)=({info['browning_top']:.2f}, {info['browning_bottom']:.2f}) | "
                f"reward={reward:.3f}"
            )
        if terminated:
            break


if __name__ == "__main__":
    _demo_rollout()
