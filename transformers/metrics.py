from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

def calculate_bleu(reference, candidate):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference], candidate, smoothing_function=smoothie)

def calculate_rouge(reference, candidate):
    from rouge import Rouge
    rouge = Rouge()
    scores = rouge.get_scores(' '.join(map(str, candidate)), ' '.join(map(str, reference)))
    return scores[0]['rouge-l']['f']

def calculate_meteor(reference, candidate):
    from nltk.translate.meteor_score import meteor_score
    return meteor_score([reference], candidate)
