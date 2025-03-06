import torch
import torch.nn as nn
from attention import MultiHeadAttention, GroupedQueryAttention, MultiQueryAttention
from feed_forward import FeedForward
from config import Config

class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        if Config.attention_type == "grouped":
            self.attention = GroupedQueryAttention(Config.num_groups)
        elif Config.attention_type == "multi_query":
            self.attention = MultiQueryAttention()
        else:
            window_size = Config.sparse_window_size if Config.attention_type == "sparse" else None
            self.attention = MultiHeadAttention(window_size)
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
