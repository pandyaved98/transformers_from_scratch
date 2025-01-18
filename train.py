import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from config import Config
from utils import save_model, setup_logging
from data_loader import get_data_loader
from loss import LabelSmoothingLoss
from optimizer import get_optimizer, get_scheduler

def train_epoch(model, data_loader, optimizer, scheduler, criterion, scaler, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (src, tgt) in enumerate(data_loader):
        src, tgt = src.to(Config.device), tgt.to(Config.device)
        optimizer.zero_grad()

        with autocast(enabled=Config.use_fp16):
            output = model(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1, Config.vocab_size), tgt[:, 1:].contiguous().view(-1))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            logging.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")

    avg_loss = total_loss / len(data_loader)
    logging.info(f"Epoch {epoch}, Average Loss: {avg_loss}")
    return avg_loss

def train(model, train_loader, val_loader, optimizer, scheduler, criterion, num_epochs, save_dir):
    scaler = GradScaler(enabled=Config.use_fp16)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, scaler, epoch)
        val_loss = evaluate(model, val_loader, criterion)

        if val_loss < best_loss:
            best_loss = val_loss
            save_model(model, optimizer, epoch, save_dir)

        logging.info(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src, tgt = src.to(Config.device), tgt.to(Config.device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1, Config.vocab_size), tgt[:, 1:].contiguous().view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss
