import torch
import torch.nn as nn
from encoder import Encoder
from decoder import Decoder
from config import Config

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = nn.Linear(Config.d_model, Config.vocab_size)
        self.dropout = nn.Dropout(Config.dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_output = self.encoder(src, src_mask)
        decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        output = self.fc_out(self.dropout(decoder_output))
        return output
