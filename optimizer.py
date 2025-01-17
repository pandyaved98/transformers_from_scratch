import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from config import Config

def get_optimizer(model):
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    return optimizer

def get_scheduler(optimizer):
    def lr_lambda(step):
        if step == 0:
            return 0.0
        return min(step ** -0.5, step * (Config.warmup_steps ** -1.5))

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler
