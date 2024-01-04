import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, cvae_input_dim):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, cvae_input_dim)
        self.fc3 = nn.Linear(cvae_input_dim, 10)


    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x

    def get_activations(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x)
        return x
