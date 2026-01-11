import torch.nn as nn

class ConsistencyClassifier(nn.Module):
    def __init__(self):
        super(ConsistencyClassifier, self).__init__()
        self.linear = nn.Linear(384, 1)  
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = self.linear(x)
        return self.sigmoid(x)