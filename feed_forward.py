import torch
import torch.nn as nn
from config import Config

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(Config.d_model, Config.d_ff)
        self.linear2 = nn.Linear(Config.d_ff, Config.d_model)
        self.dropout = nn.Dropout(Config.dropout)
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        return out * self.residual_scale
