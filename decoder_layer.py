import torch
import torch.nn as nn
from attention import MultiHeadAttention, GroupedQueryAttention, MultiQueryAttention
from feed_forward import FeedForward
from config import Config

class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        if Config.attention_type == "grouped":
            self.masked_attention = GroupedQueryAttention(Config.num_groups)
            self.encoder_attention = GroupedQueryAttention(Config.num_groups)
        elif Config.attention_type == "multi_query":
            self.masked_attention = MultiQueryAttention()
            self.encoder_attention = MultiQueryAttention()
        else:
            window_size = Config.sparse_window_size if Config.attention_type == "sparse" else None
            self.masked_attention = MultiHeadAttention(window_size)
            self.encoder_attention = MultiHeadAttention(window_size)
        self.feed_forward = FeedForward()
        self.norm1 = nn.LayerNorm(Config.d_model)
        self.norm2 = nn.LayerNorm(Config.d_model)
        self.norm3 = nn.LayerNorm(Config.d_model)
        self.dropout1 = nn.Dropout(Config.dropout)
        self.dropout2 = nn.Dropout(Config.dropout)
        self.dropout3 = nn.Dropout(Config.dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        attn_output, _ = self.masked_attention(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        x = x + self.dropout1(attn_output)
        attn_output, _ = self.encoder_attention(self.norm2(x), self.norm2(encoder_output), self.norm2(encoder_output), src_mask)
        x = x + self.dropout2(attn_output)
        ff_output = self.feed_forward(self.norm3(x))
        x = x + self.dropout3(ff_output)
        return x
