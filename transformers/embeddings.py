import torch
import torch.nn as nn
from config import Config

class TokenEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.vocab_size, Config.d_model, padding_idx=Config.pad_token_id)

    def forward(self, x):
        return self.embedding(x)

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(Config.max_seq_len, Config.d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        return self.embedding(positions)

class RelativePositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(2 * Config.max_seq_len - 1, Config.d_model)

    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(-seq_len + 1, seq_len, device=x.device)
        return self.embedding(positions)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

    def apply_rotary_emb(self, x):
        seq_len, dim = x.size(1), x.size(-1)
        sin_emb, cos_emb = self(seq_len).unsqueeze(0).unsqueeze(2).chunk(2, dim=-1)
        sin_emb, cos_emb = sin_emb.expand_as(x), cos_emb.expand_as(x)
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((x1 * cos_emb - x2 * sin_emb, x2 * cos_emb + x1 * sin_emb), dim=-1)

class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = TokenEmbedding()
        if Config.positional_embedding == "absolute":
            self.positional_embedding = AbsolutePositionalEmbedding()
        elif Config.positional_embedding == "relative":
            self.positional_embedding = RelativePositionalEmbedding()
        elif Config.positional_embedding == "rotary":
            self.rotary_emb = RotaryPositionalEmbedding(Config.d_model // Config.num_heads)
        else:
            raise ValueError("Invalid positional embedding type")
        self.dropout = nn.Dropout(Config.dropout)
        self.scale = torch.sqrt(torch.tensor(Config.d_model, dtype=torch.float32))

    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        if Config.positional_embedding == "rotary":
            embeddings = token_embeddings * self.scale
        else:
            positional_embeddings = self.positional_embedding(x)
            embeddings = token_embeddings * self.scale + positional_embeddings
        return self.dropout(embeddings)
