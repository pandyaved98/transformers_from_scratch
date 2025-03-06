import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from encoder_layer import EncoderLayer
from config import Config

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(Config.num_layers)])
        self.norm = nn.LayerNorm(Config.d_model)

    def forward(self, x, mask=None):
        for layer in self.layers:
            if Config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, mask)
            else:
                x = layer(x, mask)
        return self.norm(x)
