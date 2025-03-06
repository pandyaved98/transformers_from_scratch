import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import Config

class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = sorted(src_batch, key=lambda x: len(x), reverse=True)
    tgt_batch = sorted(tgt_batch, key=lambda x: len(x), reverse=True)
    src_batch = pad_sequence(src_batch, padding_value=Config.pad_token_id, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=Config.pad_token_id, batch_first=True)
    return src_batch, tgt_batch

def get_data_loader(src_data, tgt_data, batch_size, shuffle=True):
    dataset = TranslationDataset(src_data, tgt_data)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return data_loader
