import torch
import torch.nn as nn
import torch.nn.functional as F
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

class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = TokenEmbedding()
        if Config.positional_embedding == "absolute":
            self.positional_embedding = AbsolutePositionalEmbedding()
        elif Config.positional_embedding == "relative":
            self.positional_embedding = RelativePositionalEmbedding()
        else:
            raise ValueError("Invalid positional embedding type")
        self.dropout = nn.Dropout(Config.dropout)
        self.scale = torch.sqrt(torch.tensor(Config.d_model, dtype=torch.float32))

    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        positional_embeddings = self.positional_embedding(x)
        embeddings = token_embeddings * self.scale + positional_embeddings
        return self.dropout(embeddings)
