"""Interactive MuJoCo viewer for the steak environment.

This script couples the reinforcement-learning environment to a MuJoCo scene so
we can watch the steak sear in real time. The code is heavily commented because
the viewer API requires several moving pieces (mjpython re-launch, color
mapping, and overlay text updates).
"""

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
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap

SCENE_PATH = Path(__file__).with_name("steak.xml")
_RELAUNCHED_FLAG = "PERFECT_STEAK_MJPYTHON"

# Browning color stops derived from the provided palette.
_B_STOPS = np.array([0.0, 50.0, 100.0, 150.0], dtype=np.float32)
_B_COLORS = np.array(
    [
        [220 / 255.0, 160 / 255.0, 150 / 255.0, 1.0],  # Raw
        [210 / 255.0, 180 / 255.0, 140 / 255.0, 1.0],  # Tan
        [139 / 255.0, 69 / 255.0, 19 / 255.0, 1.0],  # Ideal
        [40 / 255.0, 20 / 255.0, 10 / 255.0, 1.0],  # Burnt
    ],
    dtype=np.float32,
)

_TEMP_CM_STOPS = np.array([0.0, 0.5, 0.7, 1.0], dtype=np.float32)
_TEMP_COLORS = np.array(
    [
        [0.0, 0.0, 0.3, 1.0],  # Deep blue for cold
        [0.2, 0.4, 1.0, 1.0],  # Mid blue
        [1.0, 1.0, 0.0, 1.0],  # Yellow near target
        [1.0, 0.0, 0.0, 1.0],  # Red for very hot
    ],
    dtype=np.float32,
)
_TEMP_CMAP = LinearSegmentedColormap.from_list(
    "steak_temp",
    list(zip(_TEMP_CM_STOPS, _TEMP_COLORS[:, :3])),
)


def _ensure_mjpython_on_macos() -> None:
    """Re-launch under mjpython on macOS where required.

    MuJoCo's Python viewer depends on Cocoa event loops that are only available
    through the ``mjpython`` shim. When the user ran ``python -m`` directly we
    replace the current process with an ``mjpython`` invocation so the GUI opens
    correctly.
    """
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


def _heat_color(pan_temp: float, config: SteakEnvConfig) -> np.ndarray:
    """Blend between cool and hot colors based on the burner temperature."""
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
    """Hand-crafted policy so the visualization shows plausible behavior."""
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


def _browning_value_to_rgba(value: float) -> np.ndarray:
    """Map a browning score (0-150) to the provided hex palette."""
    value = float(np.clip(value, _B_STOPS[0], _B_STOPS[-1]))
    rgba = np.empty(4, dtype=np.float32)
    for idx in range(4):
        rgba[idx] = np.interp(value, _B_STOPS, _B_COLORS[:, idx])
    return rgba


def _showcase_steak(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam: viewer.Handle,
    steak_body_id: int,
    height_offset: float = 0.25,
    spin_duration: float = 4.0,
    steps_per_second: int = 60,
) -> None:
    """Raise the steak and spin it slowly so the viewer can inspect both sides."""

    qpos0 = data.qpos.copy()
    slide_idx = 0
    yaw_idx = 1

    base_height = data.qpos[slide_idx]
    target_height = base_height + height_offset
    total_steps = int(spin_duration * steps_per_second)

    for step in range(total_steps):
        t = step / max(total_steps - 1, 1)
        # Smooth height interpolation using ease-in/out.
        height = base_height + (target_height - base_height) * (0.5 - 0.5 * np.cos(np.pi * min(t * 2, 1.0)))
        data.qpos[slide_idx] = height

        # Simple yaw rotation around the vertical axis.
        yaw = 2 * np.pi * t
        data.qpos[yaw_idx] = yaw

        mujoco.mj_forward(model, data)
        cam.sync()
        time.sleep(1.0 / steps_per_second)

    data.qpos[:] = qpos0
    mujoco.mj_forward(model, data)


def _update_heatmap(layer_temps: np.ndarray, image: np.ndarray) -> None:
    """Render a temperature heatmap with embedded text into the given image buffer."""
    temp_min = 20.0
    temp_max = 210.0
    height, width, _ = image.shape

    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    heat_data = layer_temps[::-1][:, None]
    ax.imshow(
        heat_data,
        aspect="auto",
        cmap=_TEMP_CMAP,
        vmin=temp_min,
        vmax=temp_max,
        interpolation="nearest",
    )
    ax.set_axis_off()

    num_layers = len(layer_temps)
    for idx, temp in enumerate(layer_temps[::-1]):
        layer_id = num_layers - 1 - idx
        y = (idx + 0.5) / num_layers
        ax.text(
            0.5,
            y,
            f"L{layer_id:02d}: {temp:.1f}°C",
            color="white",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
            path_effects=[patheffects.withStroke(linewidth=1, foreground="black")],
        )

    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
    rgb = rgba[:, :, :3]
    np.copyto(image, rgb)


def main() -> None:
    """Entry point used by both python and mjpython launches."""
    if not SCENE_PATH.exists():
        raise SystemExit(f"Missing scene file at {SCENE_PATH}")

    _ensure_mjpython_on_macos()

    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)

    steak_bottom_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "steak_bottom"
    )
    steak_top_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "steak_top"
    )
    burner_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "burner")

    env = SteakEnv()
    obs, info = env.reset()
    flip_state = {"performed": False}
    steak_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "steak")
    heatmap_viewport = mujoco.MjrRect(0, 0, 0, 0)
    heatmap_image = np.zeros((450, 160, 3), dtype=np.uint8)

    with viewer.launch_passive(model, data) as cam:
        cam.cam.azimuth = 120
        cam.cam.elevation = -25
        cam.cam.distance = 0.6

        while cam.is_running():
            action = _heuristic_policy(obs, env, flip_state)
            obs, reward, terminated, truncated, step_info = env.step(action)

            model.geom_rgba[steak_bottom_geom_id] = _browning_value_to_rgba(
                step_info["browning_bottom"]
            )
            model.geom_rgba[steak_top_geom_id] = _browning_value_to_rgba(
                step_info["browning_top"]
            )
            model.geom_rgba[burner_geom_id] = _heat_color(env.pan_temp, env.config)

            # ``mj_forward`` updates MuJoCo's internal state so the viewer
            # reflects the latest asset color and transforms.
            mujoco.mj_forward(model, data)
            cam_text = [
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    "Steak Stats",
                    "\n".join(
                        [
                            f"Core temp: {step_info['core_temp_c']:.1f} C",
                            f"Top browning B: {step_info['browning_top']:.1f}",
                            f"Bottom browning B: {step_info['browning_bottom']:.1f}",
                            f"Pan temp: {env.pan_temp:.1f} C",
                            f"Time: {step_info['time_elapsed_s']:.1f} s",
                            f"Action: {step_info['last_action']}",
                            f"Reward: {reward:.3f}",
                        ]
                    ),
                )
            ]
            cam.set_texts(cam_text)
            _update_heatmap(step_info["layer_temps"], heatmap_image)
            with cam.lock():
                viewport = cam.viewport
                heatmap_viewport.left = viewport.width - heatmap_image.shape[1]
                heatmap_viewport.bottom = viewport.height - heatmap_image.shape[0]
                heatmap_viewport.width = heatmap_image.shape[1]
                heatmap_viewport.height = heatmap_image.shape[0]
                cam.set_images([(heatmap_viewport, heatmap_image)])
            cam.sync()

            if terminated or truncated:
                _showcase_steak(
                    model,
                    data,
                    cam,
                    steak_body_id=steak_body_id,
                )
                break

            time.sleep(0.1)


if __name__ == "__main__":
    main()
