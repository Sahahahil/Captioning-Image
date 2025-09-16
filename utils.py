from torchtext.data.metrics import bleu_score

def calculate_bleu(references, candidates):
    """
    reference: list of list of tokens
    candidates: list of tokens
    """
    return bleu_score(candidates, references)
    