import torch
import torch.nn as nn
import torch.nn.functional as fun

from src.networks.dqn import DeepQNetwork


class DuelingDeepQNetwork(DeepQNetwork):
    def __init__(self, number_actions, input_size, learning_rate, checkpoint_name, checkpoint_path, device,
                 epsilon_adam=1.5e-4):
        super().__init__(number_actions, input_size, learning_rate, checkpoint_name, checkpoint_path, device,
                         epsilon_adam)

        self.dense1 = nn.Linear(1024, 1024, device=device, dtype=torch.float64)
        self.dense2 = nn.Linear(1024, 512, device=device, dtype=torch.float64)
        self.V = nn.Linear(512, 1, device=device, dtype=torch.float64)
        self.A = nn.Linear(512, number_actions, device=device, dtype=torch.float64)

    def forward(self, state):
        output = self.convolution1(state)
        output = fun.relu(output)
        output = self.convolution2(output)
        output = fun.relu(output)
        output = self.convolution3(output)
        output = fun.relu(output)

        output = output.view(output.size()[0], -1)
        output = self.dense1(output)
        output = fun.relu(output)
        output = self.dense2(output)
        output = fun.relu(output)

        v = self.V(output)
        a = self.A(output)

        return v, a
