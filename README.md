# Raul Contreras
# Perfect Steak RL Project

This project follows the final project details of "The Perfect steak". 

Included are:
- a finite-difference heat conduction environment with Maillard browning dynamics
- a MuJoCo visualization
- a Deep Q-Network training script

## Project Structure
- `steak_env/environment.py` – reinforcement learning environment capturing steak physics and rewards
- `mujoco_scene/steak.xml` – MuJoCo asset that visualizes a steak slab and pan heater
- `mujoco_scene/run_viewer.py` – helper to drive the MuJoCo scene from the environment state
- `dqn/train_dqn.py` – lightweight DQN implementation using PyTorch
- `experimentation/training.py` - configurable large scale experimentation to generate different DQNs

## Requirements
(I recommend you create a vertual environment and then you can load the required dependencies using):
```bash
pip install -r perfect_steak/requirements.txt
```
To use the MuJoCo visualizer, install the optional packages **(have to uncomment)**:
```bash
pip install "mujoco>=3.1" "mujoco-python-viewer==0.1.4"
```

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

## Running Experiments
Architecture sweeps and individual experiments live under `perfect_steak/experimentation`.

- Quick functional test of the lightweight notebook-style DQN:
  ```bash
  python -m perfect_steak.dqn.train_dqn
  ```
- Full experiment setup with logging, plots, and checkpoints:
  ```bash
  python -m perfect_steak.experimentation.scripts.run_dqn_sweep \
      --config-file perfect_steak/experimentation/configs/architecture_grid.json \
      --results-dir perfect_steak/experimentation/results
  ```
  Use `--dry-run` for a short sanity check, or `--resume` to skip runs that already have metadata. Each run folder (e.g. `results/l4_h128`) collects `summary.txt`, `reward_curve.png`, `actions_eval.json`, `metadata.json`, and `best_model.pt`. A comparison table is written to `results/comparison.md` for graders to review.

## MuJoCo Visualization
```bash
mjpython -m perfect_steak.mujoco_scene.run_viewer \
    --checkpoint perfect_steak/experimentation/results/l4_h128/best_model.pt
```
The viewer animates the steak layers, browning, and current action. If `--checkpoint` is omitted the script falls back to the default policy, otherwise it loads the provided `.pt` file (and `metadata.json` if present) to replay a trained network.

Ensure `mujoco` and `mujoco-python-viewer` are installed in the active environment and invoke the command above from the project root.

## Training one DQN
```bash
python -m perfect_steak.dqn.train_dqn
```
This kicks off a short training loop that periodically reports episode rewards. Hyperparameters are intentionally conservative so you can verify end-to-end wiring before longer runs.


--
### IMPORTANT: make sure you have all dependencies downloaded to visualize and run experiments. Refer to: 
```bash
perfect_steak/requirements.txt
```
