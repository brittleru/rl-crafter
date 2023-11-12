import os

import torch
import torch.nn as nn
import torch.optim as optim
from torchinfo import summary


class DeepQNetwork(nn.Module):
    def __init__(
            self, number_actions, input_size, learning_rate, checkpoint_name, checkpoint_path,
            device, epsilon_adam: float = 1e-4, hidden_units_conv: int = 16
    ):
        super(DeepQNetwork, self).__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_file_path = os.path.join(self.checkpoint_path, checkpoint_name)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.hidden = hidden_units_conv

        self.convolution_block_in = nn.Sequential(
            nn.Conv2d(
                in_channels=4, out_channels=self.hidden, kernel_size=3, device=device, dtype=torch.float64
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=self.hidden, device=device, dtype=torch.float64),
            nn.Conv2d(
                in_channels=self.hidden, out_channels=self.hidden, kernel_size=3, device=device, dtype=torch.float64
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=self.hidden, device=device, dtype=torch.float64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.convolution_block_intermediate = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden, out_channels=self.hidden, kernel_size=3, device=device, dtype=torch.float64
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=self.hidden, device=device, dtype=torch.float64),
            nn.Conv2d(
                in_channels=self.hidden, out_channels=self.hidden, kernel_size=3, device=device, dtype=torch.float64
            ),
            nn.ReLU(),
            # nn.BatchNorm2d(num_features=self.hidden, device=device, dtype=torch.float64),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        linear_input_size = self.infer_conv_sizes(input_size)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=linear_input_size, out_features=128, device=device, dtype=torch.float64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=number_actions, device=device, dtype=torch.float64),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=epsilon_adam)
        self.loss = nn.HuberLoss()
        self.to(device)

    def infer_conv_sizes(self, input_size):
        state = torch.zeros(4, *input_size, device=self.device, dtype=torch.float64)
        with torch.inference_mode():
            sizes = self.convolution_block_intermediate(self.convolution_block_in(state))
        return self.hidden * sizes.shape[-2] * sizes.shape[-1]

    def save_checkpoint(self):
        print(f'{10 * "="} Saving {self.__class__.__name__} checkpoint {10 * "="}')
        torch.save(self.state_dict(), f"{self.checkpoint_file_path}.pt")

    def load_checkpoint(self):
        print(f'{10 * "="} Loading {self.__class__.__name__} checkpoint {10 * "="}')
        self.load_state_dict(torch.load(f"{self.checkpoint_file_path}.pt"))

    def view_model(self):
        summary(self)

    def forward(self, state):
        return self.classifier(self.convolution_block_intermediate(self.convolution_block_in(state)))
