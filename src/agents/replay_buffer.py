import torch
import numpy as np

from typing import Tuple
from random import sample
from collections import deque


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, number_actions, device):
        self.device = device
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = torch.zeros((self.memory_size, *input_shape), dtype=torch.float64, device=self.device)
        self.new_state_memory = torch.zeros((self.memory_size, *input_shape), dtype=torch.float64, device=self.device)
        self.action_memory = torch.zeros(self.memory_size, dtype=torch.int64, device=self.device)
        self.reward_memory = torch.zeros(self.memory_size, dtype=torch.float64, device=self.device)
        self.terminal_memory = torch.zeros(self.memory_size, dtype=torch.bool, device=self.device)

    def store_transition(self, state, action, reward, state_, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def sample_buffer(self, batch_size) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, batch_size, replace=False)

        states = torch.from_numpy(self.state_memory[batch]).to(self.device)
        actions = torch.from_numpy(self.action_memory[batch]).to(self.device)
        rewards = torch.from_numpy(self.reward_memory[batch]).to(self.device)
        states_ = torch.from_numpy(self.new_state_memory[batch]).to(self.device)
        terminal = torch.from_numpy(self.terminal_memory[batch]).to(self.device)

        return states, actions, rewards, states_, terminal
