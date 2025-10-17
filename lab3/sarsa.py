import gymnasium as gym
import numpy as np

# Create the environment
env = gym.make("CliffWalking-v1", render_mode="ansi")

n_episodes = 1000      # number of episodes to run
max_steps = 100     # safety cap

# Q-learning parameters
learning_rate = 0.1 # alpha
discount_factor = 0.99 # lambda
n_actions = env.action_space.n 
n_states = env.observation_space.n
Q = np.zeros((n_states, n_actions))

def greedyPolicy(state, Q): # assuming I can index by state
  return np.argmax(Q[state])

# call using action = self.greedyPolicy()
epsilon = .05

"""
if(step % (max_steps * epsilon) == 0):
      action = env.action_space.sample()
"""

for episode in range(n_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0

    for step in range(max_steps):
        # e-greedy policy
        if(step % (max_steps * epsilon) == 0): 
          action = env.action_space.sample() # random
        else:
          action = greedyPolicy(state, Q)
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        state = next_state

        if terminated or truncated:
            break

    print(f"Episode {episode+1}: total reward = {total_reward}")

env.close()
