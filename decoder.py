import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from decoder_layer import DecoderLayer
from config import Config

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(Config.num_layers)])
        self.norm = nn.LayerNorm(Config.d_model)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            if Config.use_gradient_checkpointing and self.training:
                x = checkpoint(layer, x, encoder_output, src_mask, tgt_mask)
            else:
                x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
