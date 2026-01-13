
from collections import Counter
from typing import Iterable, List, Tuple, Dict

def _tokenize(text: str) -> List[str]:
    # ultra-simple tokenizer; you can replace with your XJSON/K'uhul-aware tokenizer
    return text.strip().split()

def build_ngrams(
    corpus: Iterable[str],
    n_values: Tuple[int, ...] = (1, 2, 3),
) -> Dict[int, Counter]:
    """Build n-gram frequency tables for the given corpus of texts."""
    tables: Dict[int, Counter] = {n: Counter() for n in n_values}
    for line in corpus:
        tokens = _tokenize(line)
        for n in n_values:
            if len(tokens) < n:
                continue
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i : i + n])
                tables[n][gram] += 1
    return tables

def normalize_ngrams(tables: Dict[int, Counter]) -> Dict[int, Dict[Tuple[str, ...], float]]:
    """Convert counts into probabilities for each n."""
    norm = {}
    for n, counter in tables.items():
        total = sum(counter.values()) or 1
        norm[n] = {gram: c / total for gram, c in counter.items()}
    return norm
