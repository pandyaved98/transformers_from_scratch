import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import Config

class MaskedLanguageModelingDataset(Dataset):
    def __init__(self, data, tokenizer, mask_prob=0.15):
        self.data = data
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.tokenizer.encode(self.data[idx])
        masked_tokens = []
        labels = []
        for token in tokens:
            if torch.rand(1).item() < self.mask_prob:
                masked_tokens.append(Config.mask_token_id)
                labels.append(token)
            else:
                masked_tokens.append(token)
                labels.append(-100)
        return torch.tensor(masked_tokens), torch.tensor(labels)

def pretrain_mlm(model, data_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for masked_tokens, labels in data_loader:
            masked_tokens, labels = masked_tokens.to(Config.device), labels.to(Config.device)
            optimizer.zero_grad()
            output = model(masked_tokens)
            loss = criterion(output.view(-1, Config.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")
