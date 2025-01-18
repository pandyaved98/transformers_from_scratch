import torch
import torch.nn.functional as F
from nltk.translate.bleu_score import sentence_bleu
from config import Config
from beam_search import beam_search

def evaluate_model(model, data_loader, criterion):
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

def generate_sequence(model, src, max_len=50):
    model.eval()
    with torch.no_grad():
        src = src.to(Config.device)
        encoder_output = model.encoder(src, None)
        tgt = torch.ones(1, 1).fill_(Config.bos_token_id).long().to(Config.device)

        for _ in range(max_len):
            output = model.decoder(tgt, encoder_output, None, None)
            next_token = output.argmax(dim=-1)[:, -1].unsqueeze(-1)
            tgt = torch.cat([tgt, next_token], dim=-1)
            if next_token.item() == Config.eos_token_id:
                break

    return tgt.squeeze(0)

def evaluate_with_beam_search(model, data_loader, beam_size=5):
    model.eval()
    total_bleu = 0
    with torch.no_grad():
        for src, tgt in data_loader:
            src = src.to(Config.device)
            tgt = tgt.to(Config.device)
            generated_seq = beam_search(model, src, beam_size)
            reference = [tgt.squeeze(0).tolist()]
            candidate = generated_seq.tolist()
            bleu_score = sentence_bleu(reference, candidate)
            total_bleu += bleu_score

    avg_bleu = total_bleu / len(data_loader)
    return avg_bleu
