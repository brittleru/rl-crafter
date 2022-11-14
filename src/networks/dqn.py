import os
import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim


class DeepQNetwork(nn.Module):
    def __init__(self, number_actions, input_size, learning_rate, checkpoint_name, checkpoint_path, device):
        super(DeepQNetwork, self).__init__()
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.checkpoint_file_path = os.path.join(self.checkpoint_path, checkpoint_name)

        self.convolution1 = nn.Conv2d(input_size[0], 32, 8, stride=4, device=device)
        self.convolution2 = nn.Conv2d(32, 64, 4, stride=2, device=device)
        self.convolution3 = nn.Conv2d(64, 64, 3, stride=1, device=device)
        self.max_pooling = nn.MaxPool2d(2, 2)
        # linear_layers_input_size = self.infer_conv_sizes(input_size)
        self.dense1 = nn.Linear(3136, 512, device=device)
        self.dense2 = nn.Linear(512, number_actions, device=device)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(device)

    def infer_conv_sizes(self, input_size):
        state = torch.zeros(*input_size, 3, device=self.device)
        sizes = self.convolution1(state)
        sizes = self.convolution2(sizes)
        sizes = self.convolution3(sizes)
        return int(torch.prod(sizes.size()))

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

        output = output.view(output.size()[0], -1)
        output = self.dense1(output)
        output = fun.relu(output)
        output = self.dense2(output)

        return output

    def save_checkpoint(self):
        print(f'{10 * "="} Saving {self.__class__.__name__} checkpoint {10 * "="}')
        torch.save(self.state_dict(), self.checkpoint_file_path)

    def load_checkpoint(self):
        print(f'{10 * "="} Loading {self.__class__.__name__} checkpoint {10 * "="}')
        self.load_state_dict(torch.load(self.checkpoint_file_path))
