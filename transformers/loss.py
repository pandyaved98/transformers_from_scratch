import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class LabelSmoothingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.confidence = 1.0 - Config.label_smoothing
        self.smoothing = Config.label_smoothing
        self.vocab_size = Config.vocab_size

    def forward(self, logits, target):
        log_probs = F.log_softmax(logits, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
