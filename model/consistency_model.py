import torch.nn as nn

class ConsistencyModel(nn.Module):
    def __init__(self, input_dim=384):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))