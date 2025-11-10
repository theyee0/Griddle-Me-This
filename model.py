import torch
from torch import nn

# https://github.com/BarakOshri/ConvChess/blob/master/papers/ConvChess.pdf

class StackedConvolve(nn.Module):
    """Neural net module combining 2 convolutional layers with fully connected layers"""

    def __init__(self):
        """Initialize the neural network with the 4 layers"""
        super().__init__()

        self.dropout = nn.Dropout(p=0.2)

        # Input: 12x8x8, Output: 12x8x8
        self.layer_stack_1 = nn.Sequential(
            nn.Conv2d(12, 12, 3, padding=1),
            nn.Flatten(),
            nn.Linear(8*8, 8*8),
            nn.LeakyReLU()
        )

        # Input: 12x8x8, Output: 768
        self.layer_stack_2 = nn.Sequential(
            nn.Unflatten(1, (8, 8)),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.Flatten(start_dim=0),
            nn.Linear(8*8 * 12, 8*8 * 12),
            nn.LeakyReLU()
        )

        # Input: 768, Output: 1x8x8
        self.layer_stack_3 = nn.Sequential(
            nn.Unflatten(0, (12, 8, 8)),
            nn.Conv2d(12, 1, 3, padding=1),
            nn.Flatten(),
            nn.Linear(8*8, 8*8),
            nn.LeakyReLU(),
        )

        # Input: 1x8x8, Output: 1x8x8
        self.layer_stack_4 = nn.Sequential(
            nn.Unflatten(1, (8, 8)),
            nn.Conv2d(1, 1, 3, padding=1),
            nn.Flatten(),
            nn.Linear(8*8, 8*8),
            nn.LeakyReLU()
        )

        # Input: 1x8x8, Output: 64
        self.softmax = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        """Perform inference on the network"""
        x = self.layer_stack_1(x)
        x = self.dropout(x)
        x = self.layer_stack_2(x)
        x = self.dropout(x)
        x = self.layer_stack_3(x)
        x = self.dropout(x)
        x = self.layer_stack_4(x)

        return self.softmax(x)
