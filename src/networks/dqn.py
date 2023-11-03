import os
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, number_actions, input_size, learning_rate, checkpoint_name, checkpoint_path, device,
                 epsilon_adam=1e-4):
        super(DeepQNetwork, self).__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_file_path = os.path.join(self.checkpoint_path, checkpoint_name)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.convolution1 = nn.Conv2d(4, 8, 3, device=device, dtype=torch.float64)
        self.convolution2 = nn.Conv2d(8, 8, 3, device=device, dtype=torch.float64)
        self.convolution3 = nn.Conv2d(8, 4, 3, device=device, dtype=torch.float64)
        self.convolution4 = nn.Conv2d(4, 2, 3, device=device, dtype=torch.float64)
        self.convolution5 = nn.Conv2d(2, 2, 3, device=device, dtype=torch.float64)
        self.max_pooling = nn.MaxPool2d(2, 2)
        # linear_layers_input_size = self.infer_conv_sizes(input_size)
        self.dense1 = nn.Linear(5832, 8 * 80, device=device, dtype=torch.float64)
        self.dense2 = nn.Linear(8 * 80, 32, device=device, dtype=torch.float64)
        self.dense3 = nn.Linear(32, number_actions, device=device, dtype=torch.float64)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=epsilon_adam)
        self.loss = nn.CrossEntropyLoss()
        self.to(device)

    def infer_conv_sizes(self, input_size):
        state = torch.zeros(*input_size, 3, device=self.device)
        sizes = self.convolution1(state)
        sizes = self.convolution2(sizes)
        sizes = self.convolution3(sizes)
        return int(torch.prod(sizes.size()))

    def save_checkpoint(self):
        print(f'{10 * "="} Saving {self.__class__.__name__} checkpoint {10 * "="}')
        torch.save(self.state_dict(), self.checkpoint_file_path)

    def load_checkpoint(self):
        print(f'{10 * "="} Loading {self.__class__.__name__} checkpoint {10 * "="}')
        self.load_state_dict(torch.load(self.checkpoint_file_path))

    def forward(self, state):
        output = self.convolution1(state)
        output = fun.relu(output)
        # output = self.max_pooling(output)
        output = self.convolution2(output)
        output = fun.relu(output)
        # output = self.max_pooling(output)
        output = self.convolution3(output)
        output = fun.relu(output)
        # output = self.max_pooling(output)
        output = self.convolution4(output)
        output = fun.relu(output)
        output = self.convolution5(output)
        output = fun.relu(output)

        output = output.view(output.size(0), -1)
        output = self.dense1(output)
        output = fun.relu(output)
        output = self.dense2(output)
        output = self.dense3(output)
        output = fun.relu(output)

        return output
