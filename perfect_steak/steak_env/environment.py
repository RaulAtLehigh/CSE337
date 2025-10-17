from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:  # pragma: no cover - fallback for environments without gymnasium
    gym = None  # type: ignore
    spaces = None  # type: ignore


@dataclass
class SteakEnvConfig:
    """Parameter bundle matching the project proposal."""

    n_layers: int = 21
    steak_thickness_m: float = 0.03175  # 1.25 inches
    thermal_diffusivity: float = 9.0e-7  # m^2 / s, sped up for simulation
    time_step_s: float = 1.0
    initial_temp_c: float = 5.0
    ambient_temp_c: float = 23.0
    heat_settings_c: Dict[str, float] = field(
        default_factory=lambda: {"low": 121.0, "medium": 176.0, "high": 232.0}
    )
    browning_threshold_c: float = 140.0
    browning_burn_c: float = 180.0
    browning_scale: float = 1.2e-5  # tuned for second-scale steps
    browning_target: float = 1.0
    browning_sigma: float = 0.6
    core_target_c: float = 56.0  # medium-rare
    core_sigma: float = 3.0
    burn_core_c: float = 71.0
    burn_penalty: float = 2.5
    doneness_weight: float = 1.5
    sear_weight: float = 1.0
    time_penalty_per_second: float = 0.01
    max_duration_s: float = 900.0
    surface_coupling_bottom: float = 4.0
    surface_coupling_top: float = 1.5

    def __post_init__(self) -> None:
        if self.n_layers < 3:
            raise ValueError("n_layers must be at least 3 for finite difference")


class SteakEnv(gym.Env if gym else object):
    """One-dimensional steak cooking environment with Maillard browning."""

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
        self.config = config or SteakEnvConfig()
        self.dx = self.config.steak_thickness_m / (self.config.n_layers - 1)
        self.ratio = (
            self.config.thermal_diffusivity
            * self.config.time_step_s
            / (self.dx**2)
        )
        self.action_space = (
            spaces.Discrete(len(self.ACTIONS)) if spaces else None  # type: ignore[assignment]
        )
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

        self.layer_temps: np.ndarray | None = None
        self.browning_top: float = 0.0
        self.browning_bottom: float = 0.0
        self.pan_temp: float = self.config.heat_settings_c["medium"]
        self.time_elapsed: float = 0.0
        self.cumulative_penalty: float = 0.0
        self.last_action: str = "wait"
        self._rng = np.random.default_rng()

    # -- core environment API -------------------------------------------------
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> Tuple[np.ndarray, dict]:
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
        info = {"layer_temps": self.layer_temps.copy()}
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:  # obs, reward, terminated, truncated, info
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
            terminated = True
            reward = self._final_reward(removed=True)
            obs = self._get_observation()
            info = self._build_info(reward, terminated, truncated)
            return obs, reward, terminated, truncated, info

        # integrate dynamics
        self._apply_heat_step()
        self.time_elapsed += self.config.time_step_s
        self.cumulative_penalty -= (
            self.config.time_penalty_per_second * self.config.time_step_s
        )

        if self.time_elapsed >= self.config.max_duration_s:
            truncated = True
            terminated = True

        reward = 0.0
        if terminated:
            reward = self._final_reward(removed=False)

        obs = self._get_observation()
        info = self._build_info(reward, terminated, truncated)
        return obs, reward, terminated, truncated, info

    # -- helpers --------------------------------------------------------------
    def sample_action(self) -> int:
        """Convenience wrapper for random actions."""
        if self.action_space is None:
            return self._rng.integers(0, len(self.ACTIONS))  # type: ignore[return-value]
        return int(self.action_space.sample())  # type: ignore[return-value]

    def _flip(self) -> None:
        assert self.layer_temps is not None
        self.layer_temps = self.layer_temps[::-1].copy()
        self.browning_top, self.browning_bottom = (
            self.browning_bottom,
            self.browning_top,
        )

    def _apply_heat_step(self) -> None:
        assert self.layer_temps is not None
        new_temps = self.layer_temps.copy()

        # interior finite difference
        for idx in range(1, self.config.n_layers - 1):
            laplacian = (
                self.layer_temps[idx - 1]
                - 2 * self.layer_temps[idx]
                + self.layer_temps[idx + 1]
            )
            new_temps[idx] += self.ratio * laplacian

        # boundary: bottom surface (index 0) in contact with the pan
        bottom_exchange = (
            self.config.surface_coupling_bottom * (self.pan_temp - self.layer_temps[0])
            + (self.layer_temps[1] - self.layer_temps[0])
        )
        new_temps[0] += self.ratio * bottom_exchange

        # boundary: top surface (index -1) exposed to ambient air
        top_exchange = (
            self.config.surface_coupling_top * (self.config.ambient_temp_c - self.layer_temps[-1])
            + (self.layer_temps[-2] - self.layer_temps[-1])
        )
        new_temps[-1] += self.ratio * top_exchange

        self.layer_temps = new_temps
        self._update_browning()

    def _update_browning(self) -> None:
        assert self.layer_temps is not None

        for surface, idx in (("bottom", 0), ("top", -1)):
            surface_temp = self.layer_temps[idx]
            excess = max(0.0, surface_temp - self.config.browning_threshold_c)
            if excess <= 0.0:
                continue

            current = self.browning_bottom if surface == "bottom" else self.browning_top
            saturation_cap = self.config.browning_target * 1.8
            headroom = max(0.0, 1.0 - (current / saturation_cap))
            delta = (
                self.config.browning_scale
                * (excess**2)
                * self.config.time_step_s
                * headroom
            )

            if surface == "bottom":
                self.browning_bottom = max(0.0, self.browning_bottom + delta)
                if surface_temp >= self.config.browning_burn_c:
                    burn_excess = surface_temp - self.config.browning_burn_c
                    self.browning_bottom += (
                        self.config.browning_scale
                        * burn_excess
                        * self.config.time_step_s
                        * 0.5
                    )
            else:
                self.browning_top = max(0.0, self.browning_top + delta)
                if surface_temp >= self.config.browning_burn_c:
                    burn_excess = surface_temp - self.config.browning_burn_c
                    self.browning_top += (
                        self.config.browning_scale
                        * burn_excess
                        * self.config.time_step_s
                        * 0.5
                    )

    def _final_reward(self, *, removed: bool) -> float:
        assert self.layer_temps is not None
        core_temp = float(self.layer_temps[len(self.layer_temps) // 2])
        avg_brown = (self.browning_top + self.browning_bottom) / 2.0

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

        burn = core_temp >= self.config.burn_core_c or (
            self.browning_top > self.config.browning_target * 2.5
            or self.browning_bottom > self.config.browning_target * 2.5
        )
        if burn:
            reward -= self.config.burn_penalty

        if not removed:
            # If truncated, treat as suboptimal finish.
            reward -= 0.5
        return reward

    def _get_observation(self) -> np.ndarray:
        assert self.layer_temps is not None
        core_temp = float(self.layer_temps[len(self.layer_temps) // 2])
        obs = np.array(
            [
                core_temp,
                self.browning_top,
                self.browning_bottom,
                self.pan_temp,
                self.time_elapsed,
            ],
            dtype=np.float32,
        )
        return obs

    def _build_info(self, reward: float, terminated: bool, truncated: bool) -> dict:
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
        }
        return info


def _demo_rollout(steps: int = 120) -> None:
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
