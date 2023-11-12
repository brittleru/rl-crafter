import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as fun

from src.networks.dqn import DeepQNetwork


class C51DQN(DeepQNetwork):
    def __init__(
            self, number_actions, input_size, num_atoms, learning_rate, checkpoint_name,
            checkpoint_path, device, epsilon_adam: float = 1e-4, hidden_units_conv: int = 16
    ):
        super().__init__(
            number_actions, input_size, learning_rate, checkpoint_name,
            checkpoint_path, device, epsilon_adam, hidden_units_conv
        )
        linear_input_size = self.infer_conv_sizes(input_size)

        self.num_atoms = num_atoms
        self.number_actions = number_actions

        self.value_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=linear_input_size, out_features=128, device=device, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_atoms, device=device, dtype=torch.float64)
        )
        self.advantage_output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=linear_input_size, out_features=128, device=device, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=number_actions * num_atoms, device=device, dtype=torch.float64)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=epsilon_adam)
        self.loss = nn.HuberLoss()
        self.supports = torch.linspace(0, 1, num_atoms, device=device, dtype=torch.float64)
        self.to(device)

    def forward(self, state):
        output = self.convolution_block_intermediate(self.convolution_block_in(state))
        v = self.value_output(output)
        a = self.advantage_output(output)
        a = a.view(-1, self.number_actions, self.num_atoms)

        q = v.unsqueeze(1) + a - a.mean(dim=1, keepdim=True)

        return fun.softmax(q, dim=2)
