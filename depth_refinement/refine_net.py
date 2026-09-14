import torch
import torch.nn as nn

class RefineNet(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        )
    

    def forward(self, x):
        return self.net(x) # delta alpha
    
