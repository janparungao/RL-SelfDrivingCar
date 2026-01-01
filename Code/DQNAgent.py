# DQN and CNN classes adapted directly from https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#training
# and https://github.com/wiitt/DQN-Car-Racing/blob/main/DQN_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple, deque

# From PyTorch tutorial
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# From PyTorch tutorial, replay memory buffer
class ReplayMemory:
    def __init__(self, capacity=10000):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# CNN-based Q-Network based on GitHub repo
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, n_actions)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # From PyTorch tutorial
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# DQN Agent, combination of both
class DQNAgent:
    def __init__(self, input_shape, n_actions, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.q_network = DQN(input_shape, n_actions).to(self.device)
        self.target_network = DQN(input_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.n_actions = n_actions
        self.gamma = 0.99 # discount factor
        self.epsilon = 1.0 # exploration rate
        self.epsilon_min = 0.1 # min exploration rate
        self.epsilon_decay = 0.999 # decay rate
        self.batch_size = 32 # training batch size
        self.target_update_freq = 10   # Update target network every 10 episodes
        
        self.memory = ReplayMemory()
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-4)
        
        self.steps = 0
        self.episodes = 0
    
    # From PyTorch tutorial
    def select_action(self, state):
        if random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)
        
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state)
            return q_values.max(1)[1].view(1, 1)
    
    def update_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), 
                                     device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        
        state_action_values = self.q_network(state_batch).gather(1, action_batch)
        
        # From PyTorch tutorial
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1)[0]
        
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()
        
        # From Gibhub repo
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()