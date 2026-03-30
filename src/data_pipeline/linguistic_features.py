import re
from pysentiment2 import LM

_lm = None

HEDGE_WORDS = {
    "may", "might", "could", "should", "would", "possibly", "potentially",
    "approximately", "about", "likely", "expect", "believe", "assume"
}


def _get_lm():
    global _lm
    if _lm is None:
        _lm = LM()
    return _lm


def extract_lm_features(text: str) -> dict:
    lm = _get_lm()
    tokens = lm.tokenize(text)
    score  = lm.get_score(tokens)
    words  = text.lower().split()
    n      = max(len(words), 1)

    return {
        "lm_negative":     score.get("Negative", 0)    / n,
        "lm_uncertainty":  score.get("Uncertainty", 0) / n,
        "lm_litigious":    score.get("Litigious", 0)   / n,
        "lm_positive":     score.get("Positive", 0)    / n,
        "hedge_ratio":     sum(1 for w in words if w in HEDGE_WORDS) / n,
        "numeric_density": len(re.findall(r'\d+\.?\d*%?', text)) / n,
        "word_count":      n,
    }
