import torch
from torch import nn

# https://github.com/BarakOshri/ConvChess/blob/master/papers/ConvChess.pdf

class StackedConvolve(nn.Module):
    """Neural net module combining 2 convolutional layers with fully connected layers"""

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        self.layer_stack = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.Conv2d(12, 1, 3, padding=1),
            self.flatten,
            nn.Linear(8*8, 8*8),
            nn.ReLU(),
            nn.Linear(8*8, 8*8),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        return self.layer_stack(x)
