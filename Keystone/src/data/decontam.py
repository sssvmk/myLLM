"""N-gram decontamination against evaluation datasets (DP-4)."""
from __future__ import annotations

import string
from typing import Iterable, List, Set, Tuple

_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def words(text: str) -> List[str]:
    return text.lower().translate(_PUNCT).split()


def ngrams(text: str, n: int) -> Set[Tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_eval_ngrams(texts: Iterable[str], n: int) -> Set[Tuple[str, ...]]:
    out: Set[Tuple[str, ...]] = set()
    for t in texts:
        out |= ngrams(t, n)
    return out


def overlap_ratio(text: str, eval_ngrams: Set[Tuple[str, ...]], n: int) -> float:
    g = ngrams(text, n)
    if not g:
        return 0.0
    return sum(1 for x in g if x in eval_ngrams) / len(g)


def is_contaminated(text: str, eval_ngrams, n: int, threshold: float) -> bool:
    return overlap_ratio(text, eval_ngrams, n) >= threshold


# --------------------------------------------------------------------------- registry-driven (DP-4)
def build_eval_ngram_set(registry, against, splits, n: int):
    """Union of word n-grams over every text field of the `against` datasets, restricted to
    `splits` (see prepared_data.decontamination.splits). A dataset with none of the splits
    contributes nothing here; static validation rejects that configuration up front.

    Returns (ngram_set, {dataset: n_items_used})."""
    out: Set[Tuple[str, ...]] = set()
    used = {}
    for name in against:
        used[name] = 0
        for split in splits:
            if not registry.has_split(name, split):
                continue
            for item in registry.items(name, split):
                for t in item.texts:
                    out |= ngrams(t, n)
                used[name] += 1
    return out, used


def filter_contaminated(items: Iterable, eval_ngrams, n: int, threshold: float, counter: dict):
    """Yield items whose overlap with the eval n-grams is below `threshold`. `counter` is filled
    with {"checked", "dropped", "no_text"}; items with no text (packed tokens) cannot be checked
    and are kept, counted under "no_text" so the manifest shows decontamination did not apply."""
    counter.update(checked=0, dropped=0, no_text=0)
    for item in items:
        text = " ".join(item.texts)
        if not text.strip():
            counter["no_text"] += 1
            yield item
            continue
        counter["checked"] += 1
        if is_contaminated(text, eval_ngrams, n, threshold):
            counter["dropped"] += 1
            continue
        yield item
