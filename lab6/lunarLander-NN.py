import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

env = gym.make('LunarLander-v3', render_mode='human')
n_actions = env.action_space.n # there are 4 actions possible in this env
state_dim = env.observation_space.shape[0] # state dimension is 8

gamma = 0.99 
alpha = 0.001 
epsilon = .5 
epsilon_min = 0.01 
epsilon_decay = 0.95 
num_episodes = 1000 
batch_size = 64 
replay_buffer_size = 50000 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Q-Network using a simple MLP
class QNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, 128) # 8 -> 128
        ## going to try and add another hidden layer, just for fun
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, n_actions) #(64 -> 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        # Output layer (no activation for Q-values)
        return self.fc4(x)

# Building QNetwork
q_net = QNetwork(state_dim, n_actions).to(device)
q_net_copy = QNetwork(state_dim, n_actions).to(device) # target network
optimizer = optim.Adam(q_net.parameters(), lr=alpha) # adam optimizer (improves learning rate)
loss_fn = nn.MSELoss()
replay_buffer = deque(maxlen=replay_buffer_size)

# Epsilon-greedy action selection
def epsilon_greedy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    else:
      # making state into a tensor
      state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
      with torch.no_grad(): 
          q_values = q_net(state_tensor) 
      return q_values.max(1)[1].item()
    
def train_dqn():
    """Train the DQN using experience replay."""
    if len(replay_buffer) < batch_size:
        return

    batch = random.sample(replay_buffer, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    """Should I calculate next_q_values using q_net or q_net_copy?
    I think I should use q_net_copy, since it is more stable."""
    next_q_values = q_net_copy(next_states).max(1)[0].detach()

    targets = rewards + gamma * next_q_values * (1 - dones)

    # Compute the Mean Squared Error loss between the predicted Q-values and the target Q-values
    loss = loss_fn(q_values, targets)

    # Perform backpropagation
    optimizer.zero_grad() # Clear previous gradients
    loss.backward() # Compute gradients
    optimizer.step() # Update network weights

# function to update target network
def update_target_network():
    q_net_copy.load_state_dict(q_net.state_dict()) # load the current state of the main network into the target network
    q_net_copy.eval() # set target network to evaluation mode

# Training loop
## MAIN Loop ###
rewards_dqn = []
N = 25 # Every 25 episodes, performing greedy policy
target_network_update_freq = 10 # update target network every 10 episodes

for episode in range(num_episodes):
  if episode % target_network_update_freq == 0:
    update_target_network()
  state = env.reset()[0]
  total_reward = 0
  done = False
  stepCount = 0
  while not done and stepCount < 500:
    stepCount += 1
    action = epsilon_greedy(state, epsilon)
    next_state, reward, done, _, _ = env.step(action)
    replay_buffer.append((state, action, reward, next_state,done))
    train_dqn()
    state = next_state
    total_reward += reward
  rewards_dqn.append(total_reward)
  #going to implement epsilon decay since I need to explore WAYY more than I thought I did
  print(f'Total Reward after episode {episode}: {total_reward}')
  epsilon = epsilon * epsilon_decay if epsilon >= epsilon_min else epsilon_min


