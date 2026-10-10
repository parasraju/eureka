import numpy as np


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def distinct_n(texts: list[str], n: int = 1) -> float:
    """Ratio of unique n-grams to total n-grams across all texts (distinct-n)."""
    all_ngrams: list[tuple[str, ...]] = []
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        words = _tokenize(text)
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i : i + n])
            all_ngrams.append(ngram)
            unique.add(ngram)
    return len(unique) / len(all_ngrams) if all_ngrams else 0.0


def self_bleu(texts: list[str]) -> float:
    """Mean BLEU of each text against all others. Higher = more repetitive."""
    if len(texts) < 2:
        return 0.0
    from sacrebleu.metrics import BLEU

    bleu = BLEU()
    scores = []
    for i, text in enumerate(texts):
        refs = [t for j, t in enumerate(texts) if j != i]
        scores.append(bleu.sentence_score(text, refs).score / 100.0)
    return float(np.mean(scores))


def count_unique(texts: list[str]) -> int:
    return len({t.strip() for t in texts})
