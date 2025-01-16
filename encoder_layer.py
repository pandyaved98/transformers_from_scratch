import torch
import torch.nn as nn
from attention import MultiHeadAttention
from feed_forward import FeedForward
from config import Config

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadAttention()
        self.feed_forward = FeedForward()
        self.norm1 = nn.LayerNorm(Config.d_model)
        self.norm2 = nn.LayerNorm(Config.d_model)
        self.dropout1 = nn.Dropout(Config.dropout)
        self.dropout2 = nn.Dropout(Config.dropout)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask)
        x = x + self.dropout1(attn_output)
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout2(ff_output)
        return x
