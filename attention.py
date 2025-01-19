import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

class SparseAttention(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, num_heads, seq_len, d_k = q.size()
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        sparse_mask = self._create_sparse_mask(seq_len).to(q.device)
        scores = scores.masked_fill(sparse_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights

    def _create_sparse_mask(self, seq_len):
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 1
        return mask

class MultiHeadAttention(nn.Module):
    def __init__(self, window_size=None):
        super().__init__()
        self.d_k = Config.d_model // Config.num_heads
        self.num_heads = Config.num_heads
        self.q_linear = nn.Linear(Config.d_model, Config.d_model)
        self.k_linear = nn.Linear(Config.d_model, Config.d_model)
        self.v_linear = nn.Linear(Config.d_model, Config.d_model)
        self.out_linear = nn.Linear(Config.d_model, Config.d_model)
        self.attention = SparseAttention(window_size) if window_size else ScaledDotProductAttention()
        self.dropout = nn.Dropout(Config.dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.q_linear(q))
        k = self.split_heads(self.k_linear(k))
        v = self.split_heads(self.v_linear(v))
        attn_output, attn_weights = self.attention(q, k, v, mask)
        attn_output = self.combine_heads(attn_output)
        output = self.out_linear(attn_output)
        return self.dropout(output), attn_weights

class GroupedQueryAttention(nn.Module):
    def __init__(self, num_groups):
        super().__init__()
        self.num_groups = num_groups
        self.d_k = Config.d_model // Config.num_heads
        self.num_heads = Config.num_heads
        self.q_linear = nn.Linear(Config.d_model, Config.d_model)
        self.k_linear = nn.Linear(Config.d_model, Config.d_model // num_groups)
        self.v_linear = nn.Linear(Config.d_model, Config.d_model // num_groups)
        self.out_linear = nn.Linear(Config.d_model, Config.d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(Config.dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.q_linear(q))
        k = self.split_heads(self.k_linear(k))
        v = self.split_heads(self.v_linear(v))
        k = k.repeat(1, self.num_groups, 1, 1)
        v = v.repeat(1, self.num_groups, 1, 1)
        attn_output, attn_weights = self.attention(q, k, v, mask)
        attn_output = self.combine_heads(attn_output)
        output = self.out_linear(attn_output)
        return self.dropout(output), attn_weights

class MultiQueryAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_k = Config.d_model // Config.num_heads
        self.num_heads = Config.num_heads
        self.q_linear = nn.Linear(Config.d_model, Config.d_model)
        self.k_linear = nn.Linear(Config.d_model, self.d_k)
        self.v_linear = nn.Linear(Config.d_model, self.d_k)
        self.out_linear = nn.Linear(Config.d_model, Config.d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(Config.dropout)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

    def forward(self, q, k, v, mask=None):
        q = self.split_heads(self.q_linear(q))
        k = self.split_heads(self.k_linear(k))
        v = self.split_heads(self.v_linear(v))
        k = k.repeat(1, self.num_heads, 1, 1)
        v = v.repeat(1, self.num_heads, 1, 1)
        attn_output, attn_weights = self.attention(q, k, v, mask)
        attn_output = self.combine_heads(attn_output)
        output = self.out_linear(attn_output)
        return self.dropout(output), attn_weights
