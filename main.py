import torch
from data_loader import get_data_loader
from transformer import Transformer
from optimizer import get_optimizer, get_scheduler
from loss import LabelSmoothingLoss
from train import train
from evaluate import evaluate_model, evaluate_with_beam_search
from utils import setup_logging, save_model, load_model
from config import Config

def main():
    setup_logging("logs")
    train_loader = get_data_loader(train_src_data, train_tgt_data, Config.batch_size, shuffle=True)
    val_loader = get_data_loader(val_src_data, val_tgt_data, Config.batch_size, shuffle=False)
    model = Transformer().to(Config.device)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    criterion = LabelSmoothingLoss()
    train(model, train_loader, val_loader, optimizer, scheduler, criterion, Config.num_epochs, "saved_models")
    val_loss = evaluate_model(model, val_loader, criterion)
    print(f"Validation Loss: {val_loss}")
    bleu_score = evaluate_with_beam_search(model, val_loader, Config.beam_size)
    print(f"Validation BLEU Score: {bleu_score}")

if __name__ == "__main__":
    main()
