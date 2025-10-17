# Perfect Steak RL Project

This project models the problem statement from _The Perfect Steak_ CSE 337 proposal. It includes:
- a finite-difference heat conduction environment with Maillard browning dynamics
- a MuJoCo visualization scaffold
- a minimal Deep Q-Network training script

## Project Structure
- `steak_env/environment.py` – reinforcement learning environment capturing steak physics and rewards
- `mujoco_scene/steak.xml` – MuJoCo asset that visualizes a steak slab and pan heater
- `mujoco_scene/run_viewer.py` – helper to drive the MuJoCo scene from the environment state
- `dqn/train_dqn.py` – lightweight DQN implementation using PyTorch

## Requirements
Activate the provided virtual environment (`source venv/bin/activate`) and install:
```bash
pip install -r perfect_steak/requirements.txt
```
Uncomment the MuJoCo entries in the requirements file if you plan to use the visualizer.

Key dependencies:
- `numpy`
- `torch`
- `mujoco` and `mujoco-python-viewer` (optional, only for visualization)

## Running the Environment
```bash
source venv/bin/activate
python -m perfect_steak.steak_env.environment
```
This runs a single random rollout and prints the trajectory for inspection.

## MuJoCo Visualization
```bash
source venv/bin/activate
mjpython -m perfect_steak.mujoco_scene.run_viewer  # macOS requires mjpython
# Linux/Windows users can use `python -m ...` if preferred
```
The script expects `mujoco` and `mujoco-python-viewer`. It opens a viewer that animates the steak layers, surface browning, and current heat setting while stepping the environment with a simple policy.

## Training the DQN
```bash
source venv/bin/activate
python -m perfect_steak.dqn.train_dqn
```
This kicks off a short training loop that periodically reports episode rewards. Hyperparameters are intentionally conservative so you can verify end-to-end wiring before longer runs.

## Next Steps
- Tune the physical constants in `steak_env/environment.py` to match empirical data.
- Experiment with reward shaping (e.g., earlier feedback for Maillard progress).
- Replace the scripted MuJoCo policy with the learned DQN agent for live playback.
