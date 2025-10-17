"""MuJoCo viewer that visualizes the steak environment."""

from __future__ import annotations

import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    from mujoco import viewer
except Exception as exc:  # pragma: no cover - fail fast with guidance
    raise SystemExit(
        "MuJoCo is required for visualization. "
        "Install `mujoco` (and `mujoco-python-viewer`) inside your venv."
    ) from exc

from perfect_steak.steak_env import SteakEnv, SteakEnvConfig

SCENE_PATH = Path(__file__).with_name("steak.xml")
_RELAUNCHED_FLAG = "PERFECT_STEAK_MJPYTHON"


def _ensure_mjpython_on_macos() -> None:
    """Re-launch under mjpython on macOS where required."""
    if platform.system() != "Darwin":
        return

    # When the viewer is already managed by mjpython the viewer module exposes _MJPYTHON.
    is_mjpython = isinstance(
        getattr(viewer, "_MJPYTHON", None),
        getattr(viewer, "_MjPythonBase", ()),
    )
    if is_mjpython or os.environ.get(_RELAUNCHED_FLAG) == "1":
        return

    mjpython_path = shutil.which("mjpython")
    if not mjpython_path:
        raise SystemExit(
            "MuJoCo's interactive viewer on macOS must run under `mjpython`. "
            "Install the `mujoco` wheel in your environment (it ships mjpython) "
            "and execute `mjpython -m perfect_steak.mujoco_scene.run_viewer`."
        )

    os.environ[_RELAUNCHED_FLAG] = "1"
    try:
        os.execvp(mjpython_path, [mjpython_path, "-m", "perfect_steak.mujoco_scene.run_viewer"])
    except OSError as exc:
        raise SystemExit(
            "Failed to relaunch under `mjpython`. "
            "Ensure MuJoCo is installed correctly and try running:\n"
            f"  {mjpython_path} -m perfect_steak.mujoco_scene.run_viewer"
        ) from exc


def _browning_to_rgba(value: float) -> np.ndarray:
    value = float(np.clip(value, 0.0, 2.5))
    raw = np.array([0.82, 0.36, 0.36])
    seared = np.array([0.45, 0.25, 0.12])
    charred = np.array([0.2, 0.12, 0.08])
    if value < 1.0:
        t = value / 1.0
        color = (1 - t) * raw + t * seared
    else:
        t = min((value - 1.0) / 1.5, 1.0)
        color = (1 - t) * seared + t * charred
    return np.append(color, 1.0)


def _heat_color(pan_temp: float, config: SteakEnvConfig) -> np.ndarray:
    low = config.heat_settings_c["low"]
    high = config.heat_settings_c["high"]
    t = np.clip((pan_temp - low) / (high - low + 1e-6), 0.0, 1.0)
    base = np.array([0.4, 0.05, 0.02, 0.7])
    hot = np.array([1.0, 0.3, 0.1, 0.9])
    return (1 - t) * base + t * hot


def _heuristic_policy(
    obs: np.ndarray,
    env: SteakEnv,
    flip_state: dict[str, bool],
) -> int:
    core_temp, browning_top, browning_bottom, pan_temp, time_elapsed = obs
    if time_elapsed < 45.0:
        if pan_temp < env.config.heat_settings_c["high"]:
            return 4  # crank heat up early
        return 0
    if not flip_state["performed"]:
        flip_state["performed"] = True
        return 1  # flip once after searing the first side
    if core_temp >= env.config.core_target_c - 0.5:
        return 5  # remove when target reached
    if pan_temp != env.config.heat_settings_c["medium"]:
        return 3  # settle to medium heat
    return 0  # wait


def main() -> None:
    if not SCENE_PATH.exists():
        raise SystemExit(f"Missing scene file at {SCENE_PATH}")

    _ensure_mjpython_on_macos()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    steak_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "steak_core"
    )
    burner_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "burner")

    env = SteakEnv()
    obs, info = env.reset()
    flip_state = {"performed": False}

    with viewer.launch_passive(model, data) as cam:
        cam.cam.azimuth = 120
        cam.cam.elevation = -25
        cam.cam.distance = 0.6

        while cam.is_running():
            action = _heuristic_policy(obs, env, flip_state)
            obs, reward, terminated, truncated, step_info = env.step(action)

            model.geom_rgba[steak_geom_id] = _browning_to_rgba(
                (step_info["browning_top"] + step_info["browning_bottom"]) / 2.0
            )
            model.geom_rgba[burner_geom_id] = _heat_color(env.pan_temp, env.config)

            mujoco.mj_forward(model, data)
            cam_text = [
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Steak Stats",
                    "\n".join(
                        [
                            f"Core temp: {step_info['core_temp_c']:.1f} °C",
                            f"Top brown: {step_info['browning_top']:.2f}",
                            f"Bottom brown: {step_info['browning_bottom']:.2f}",
                            f"Pan temp: {env.pan_temp:.1f} °C",
                            f"Time: {step_info['time_elapsed_s']:.1f} s",
                            f"Action: {step_info['last_action']}",
                            f"Reward: {reward:.3f}",
                        ]
                    ),
                )
            ]
            if terminated or truncated:
                cam_text.append(
                    (
                        mujoco.mjtFontScale.mjFONTSCALE_200,
                        mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                        "Episode finished",
                        "Close the window or press ESC to exit.",
                    )
                )
            cam.set_texts(cam_text)
            cam.sync()

            if terminated or truncated:
                time.sleep(2.0)
                break

            time.sleep(0.1)


if __name__ == "__main__":
    main()
