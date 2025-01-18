import torch
import torch.nn.functional as F
from config import Config

def beam_search(model, src, beam_size=5, max_len=50, length_penalty=1.0):
    model.eval()
    with torch.no_grad():
        src = src.to(Config.device)
        encoder_output = model.encoder(src, None)

        beams = [(torch.ones(1, 1).fill_(Config.bos_token_id).long().to(Config.device), 0)]
        completed_beams = []

        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[0, -1].item() == Config.eos_token_id:
                    completed_beams.append((seq, score))
                    continue

                output = model.decoder(seq, encoder_output, None, None)
                log_probs = F.log_softmax(output[:, -1, :], dim=-1)
                topk_scores, topk_tokens = log_probs.topk(beam_size, dim=-1)

                for i in range(beam_size):
                    new_seq = torch.cat([seq, topk_tokens[0, i].unsqueeze(0).unsqueeze(0)], dim=-1)
                    new_score = score + topk_scores[0, i].item()
                    new_beams.append((new_seq, new_score))

            beams = sorted(new_beams, key=lambda x: x[1] / (len(x[0][0]) ** length_penalty), reverse=True)[:beam_size]

        completed_beams += beams
        best_beam = max(completed_beams, key=lambda x: x[1] / (len(x[0][0]) ** length_penalty))
        return best_beam[0]
