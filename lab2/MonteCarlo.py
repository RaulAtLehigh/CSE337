import gymnasium as gym
import gym_simplegrid
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


START_LOC = 15
GOAL_LOC = 3
# Define the initial location and the goal location in the grid.
# Each grid cell has an a number from 0 to 63.
options ={
	'start_loc': START_LOC,
	'goal_loc': GOAL_LOC
}


V = np.array([random.randint(-100, -1) for _ in range(8*8)], dtype=float)
V[START_LOC] = -101
V[GOAL_LOC] = 0
returns = [[] for _ in range(8*8)]
gamma = 0.95

for t in range(2):
	env = gym.make('SimpleGrid-8x8-v0', render_mode='human')
	obs, info = env.reset(seed=1, options=options) 
	done = env.unwrapped.done # variable that indicates when we reached the goal

	counter = 0
	observations = []
	rewards = []
	while not done: # epsiode that loops until we reach the goal 
		action = env.action_space.sample()   # Random action
		obs, reward, done, _, info = env.step(action)
		observations.append(obs) 
		rewards.append(reward)
		counter += 1 # Keeping track of the number of steps
		grid_size = 8
		row = obs // grid_size
		col = obs % grid_size
		print("Location (", row, ",", col, ") t=",t, " reward=", reward, sep="")
	
	G = 0 # keeps track of total Reward
	for t in range(counter - 2, -1, -1):
		G = gamma * G + rewards[t + 1]
		if(observations[t] not in observations[0:t-1]):
			index = ((observations[t] // grid_size) + 1) * (observations[t] % grid_size)
			returns[index].append(G)
			V[index] = sum(returns[index]) / len(returns[index])

# Plotting code remains the same
# Reshape the 1D value function array into a 4x4 grid
V_grid = V.reshape((8, 8))

print("here")
# Plot the value function as a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(V_grid, annot=True, cmap="viridis", fmt=".1f", linewidths=.5)
plt.title("Value Function")
plt.xlabel("Column")
plt.ylabel("Row")
plt.show()
	


	
