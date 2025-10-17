import gymnasium as gym
import random
import numpy as np

# You can use the following code for tiling

import numpy as np


class TileCoderXY:
    """
    A TileCoder for function approximation that applies tile coding on the x and y coordinates
    of a 3D state. Instead of providing tile widths, the user provides the number of tiles per
    dimension. The tile widths are computed based on the state bounds and the number of tiles.
    The z coordinate is not used.
    """

    def __init__(self, num_tilings, tiles_per_dim, state_low, state_high):
        """
        Initialize the TileCoderXY.

        Parameters:
            num_tilings (int): Number of overlapping tilings.
            tiles_per_dim (array-like of 2 ints): Number of tiles along the x and y dimensions.
            state_low (array-like of 2 floats): Lower bounds for the x and y dimensions.
            state_high (array-like of 2 floats): Upper bounds for the x and y dimensions.
        """
        self.num_tilings = num_tilings
        self.tiles_per_dim = np.array(tiles_per_dim, dtype=int)
        self.state_low = np.array(state_low, dtype=float)
        self.state_high = np.array(state_high, dtype=float)

        # Compute the tile width for each dimension.
        # We assume that the grid spans exactly from state_low to state_high.
        # When there are N tiles, there are N-1 intervals between the boundaries.
        self.tile_width = (self.state_high - self.state_low) / (self.tiles_per_dim - 1)

        # Precompute an offset for each tiling to create overlapping grids.
        # self.offsets = [(i / self.num_tilings) * self.tile_width for i in range(self.num_tilings)]
        # self.offsets = self.compute_8_offsets()
        # self.offsets = np.stack(self._compute_offsets(), axis=0)  # shape: (num_tilings, dims)
        # Precompute offsets for each tiling.
        # For tiling i:
        #   offset_x = (((i + 0) % num_tilings) / num_tilings) * tile_width[0]
        #   offset_y = (((i + 1) % num_tilings) / num_tilings) * tile_width[1]
        offsets = np.empty((self.num_tilings, 2))
        for i in range(self.num_tilings):
            offsets[i, 0] = (((i + 0) % self.num_tilings) / self.num_tilings) * self.tile_width[0]
            offsets[i, 1] = (((i + 1) % self.num_tilings) / self.num_tilings) * self.tile_width[1]
        self.offsets = offsets


        # Precompute multiplier for flattening a 2D index.
        # For grid shape (N, M), flat index = x_index * M + y_index.
        self.multiplier = self.tiles_per_dim[1]

        # Initialize a weight vector for each tiling.
        num_tiles = np.prod(self.tiles_per_dim)
        self.weights = [np.zeros(num_tiles) for _ in range(self.num_tilings)]

    def save(self, file_name):
        np.savez(file_name + ".npz", weights=self.weights)

    def load(self, file_name):
        self.weights = np.load(file_name+".npz")["weights"]


    def compute_8_offsets(self):
        """
        Compute a list of offsets using a combination of cardinal and diagonal directions.
        The offsets include:
          - Center: [0, 0]
          - Cardinal: right, left, up, down (half-tile shifts)
          - Diagonal: up-right, up-left, down-right, down-left (half-tile shifts)

        If the number of tilings exceeds the number of unique offsets (9), the list is repeated.

        Returns:
            List of 2-element numpy arrays representing the offset for each tiling.
        """
        half_tile = self.tile_width / 8.0
        base_offsets = [
            np.array([0.0, 0.0]),  # Center (no shift)
            np.array([half_tile[0], 0.0]),  # Right
            np.array([-half_tile[0], 0.0]),  # Left
            np.array([0.0, half_tile[1]]),  # Up
            np.array([0.0, -half_tile[1]]),  # Down
            np.array([half_tile[0], half_tile[1]]),  # Up-right
            np.array([-half_tile[0], half_tile[1]]),  # Up-left
            np.array([half_tile[0], -half_tile[1]]),  # Down-right
            np.array([-half_tile[0], -half_tile[1]])  # Down-left
        ]
        offsets = []
        for i in range(self.num_tilings):
            offsets.append(base_offsets[i % len(base_offsets)])
        return offsets

    def get_tile_indices(self, state):
        """
        Compute the active tile indices for all tilings given a 2D state.

        Parameters:
            state (array-like of length 2): The input state [x, y].

        Returns:
            List of tuples (tiling_index, flat_tile_index) for each tiling.
        """
        state = np.array(state, dtype=float)  # shape: (2,)
        # Compute shifted states for all tilings in one vectorized operation.
        # Shape of shifted: (num_tilings, 2)
        shifted = (state - self.state_low) + self.offsets

        # Compute tile coordinates (integer indices) for each tiling.
        # Division is broadcasted over the offsets.
        tile_coords = (shifted / self.tile_width).astype(int)  # shape: (num_tilings, 2)

        # Clip to ensure indices are within bounds.
        tile_coords[:, 0] = np.clip(tile_coords[:, 0], 0, self.tiles_per_dim[0] - 1)
        tile_coords[:, 1] = np.clip(tile_coords[:, 1], 0, self.tiles_per_dim[1] - 1)

        # Compute flat indices for each tiling.
        # flat_index = x_index * (tiles_per_dim[1]) + y_index
        flat_indices = tile_coords[:, 0] * self.tiles_per_dim[1] + tile_coords[:, 1]

        # Return a list of (tiling_index, flat_index) tuples.
        return list(zip(range(self.num_tilings), flat_indices))


    def predict(self, state):
        """
        Compute the approximated function value for a given 3D state using tile coding on x and y.

        Parameters:
            state (array-like): The input state [x, y, z].

        Returns:
            float: The function approximation (sum of weights for the active tiles).
        """
        active_tiles = self.get_tile_indices(state)
        return sum(self.weights[tiling][idx] for tiling, idx in active_tiles)

    def update(self, state, target, alpha):
        """
        Update the weights given a state and target value.

        Parameters:
            state (array-like): The input state [x, y, z].
            target (float): The target function value.
            alpha (float): The overall learning rate.
        """

        prediction = self.predict(state)
        error = target - prediction
        # Distribute the learning rate equally among all tilings.
        alpha_per_tiling = alpha / self.num_tilings

        active_tiles = self.get_tile_indices(state)
        for tiling, idx in active_tiles:
            self.weights[tiling][idx] += alpha_per_tiling * error


"""
For each episode:

Initialize state s.
Choose action a using ε-greedy based on Q(s,a).
For each step:
Take action a, observe (s_next, r, done).
Choose next action a_next using ε-greedy from s_next.
Compute TD target:
target = r + gamma * Q(s_next, a_next)
(if s_next is terminal, then target = r).
Compute TD error:
delta = target - Q(s,a)
Update weights:
w <- w + alpha * delta * x(s,a)
Update s = s_next, a = a_next.
End episode when the goal is reached or step limit is hit.
"""
import gymnasium as gym
import random
import numpy as np

# Create the MountainCar environment
env = gym.make('MountainCar-v0', max_episode_steps = 500)

num_episodes = 500
tile_dim = 4
# Box: [-1.2 -0.07], [0.6 0.07]
# initializing the tiles for each action, each one has its own weight vector
left_tiles = TileCoderXY(tile_dim, [tile_dim, tile_dim], [-1.2, -0.07], [0.6, 0.07])
right_tiles = TileCoderXY(tile_dim, [tile_dim, tile_dim], [-1.2, -0.07], [0.6, 0.07])
stay_tiles = TileCoderXY(tile_dim, [tile_dim, tile_dim], [-1.2, -0.07], [0.6, 0.07])
epsilon = .05 # randomness rate
alpha = .125 # learning rate
gamma = 1 # discount rate
n_actions = 3

def epsilon_greedy(state):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
      possibleActions = [left_tiles.predict(state), stay_tiles.predict(state), right_tiles.predict(state)]
      action = np.argmax(possibleActions)
      return action

for episode in range(num_episodes):
  state, info = env.reset()
  done = False
  total_reward = 0
  action = epsilon_greedy(state)
  
  while not done:
      # Take action a, observe (s_next, r, done).
      next_state, reward, terminated, truncated, info = env.step(action)
      total_reward += reward
      # Choose next action a_next using ε-greedy from s_next.
      next_action = epsilon_greedy(next_state)
      # target = r + gamma * Q(s_next, a_next)
      done = terminated or truncated
      q_next = [left_tiles.predict(next_state),
          stay_tiles.predict(next_state),
          right_tiles.predict(next_state)][next_action]
      target = reward if done else reward + gamma * q_next
      if action == 0: ##left
          left_tiles.update(state, target, alpha)
      elif action == 1: ## no push
          stay_tiles.update(state, target, alpha)
      else: ## right
          right_tiles.update(state, target, alpha)
      state = next_state
      action = next_action


  print(f"Episode {episode + 1}: Total Reward = {total_reward}")

