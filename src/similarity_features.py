"""Pairwise similarity feature engineering for entity resolution candidates."""

from __future__ import annotations

from typing import Sequence
import numpy as np

from src.blocking import Business


FEATURE_NAMES = [
    "exact_name_match",
    "name_jaccard",
    "name_containment",
    "name_len_diff_ratio",
    "name_char_ngram_jaccard",
    "exact_address_match",
    "address_jaccard",
    "address_containment",
    "numeric_overlap_count",
    "numeric_jaccard",
    "numeric_conflict",
    "same_country",
    "missing_s1_address",
    "missing_target_address",
    "target_is_s2",
]


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract character n-grams from normalized text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(set_a: set | frozenset, set_b: set | frozenset) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def containment(set_a: set | frozenset, set_b: set | frozenset) -> float:
    """Compute containment (overlap / min(|A|, |B|))."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    min_len = min(len(set_a), len(set_b))
    return intersection / min_len if min_len > 0 else 0.0


def compute_pair_features(s1: Business, target: Business, target_source: str = "S2") -> list[float]:
    """Compute pairwise similarity feature vector between an S1 entity and a target candidate."""
    s1_name = s1.name_norm
    t_name = target.name_norm
    s1_addr = s1.address_norm
    t_addr = target.address_norm

    exact_name = 1.0 if s1_name and s1_name == t_name else 0.0
    
    s1_name_tokens = set(s1.name_tokens)
    t_name_tokens = set(target.name_tokens)
    name_jac = jaccard(s1_name_tokens, t_name_tokens)
    name_cont = containment(s1_name_tokens, t_name_tokens)
    
    max_name_len = max(len(s1_name), len(t_name), 1)
    name_len_diff = abs(len(s1_name) - len(t_name)) / max_name_len

    s1_ngrams = char_ngrams(s1_name, 3)
    t_ngrams = char_ngrams(t_name, 3)
    name_ngram_jac = jaccard(s1_ngrams, t_ngrams)

    exact_addr = 1.0 if s1_addr and s1_addr == t_addr else 0.0
    
    s1_addr_tokens = set(s1.address_norm.split()) if s1_addr else set()
    t_addr_tokens = set(target.address_norm.split()) if t_addr else set()
    addr_jac = jaccard(s1_addr_tokens, t_addr_tokens)
    addr_cont = containment(s1_addr_tokens, t_addr_tokens)

    s1_nums = set(s1.address_numeric_tokens)
    t_nums = set(target.address_numeric_tokens)
    shared_nums = s1_nums & t_nums
    num_overlap = float(len(shared_nums))
    num_jac = jaccard(s1_nums, t_nums)

    # Conflict: both have significant numbers (length >= 3) but zero overlap
    s1_sig_nums = {n for n in s1_nums if len(n) >= 3}
    t_sig_nums = {n for n in t_nums if len(n) >= 3}
    num_conflict = 1.0 if (s1_sig_nums and t_sig_nums and not (s1_sig_nums & t_sig_nums)) else 0.0

    same_cntry = 1.0 if s1.country and s1.country == target.country else 0.0
    missing_s1_addr = 1.0 if not s1_addr else 0.0
    missing_t_addr = 1.0 if not t_addr else 0.0
    is_s2 = 1.0 if target_source == "S2" else 0.0

    return [
        exact_name,
        name_jac,
        name_cont,
        name_len_diff,
        name_ngram_jac,
        exact_addr,
        addr_jac,
        addr_cont,
        num_overlap,
        num_jac,
        num_conflict,
        same_cntry,
        missing_s1_addr,
        missing_t_addr,
        is_s2,
    ]


def compute_pair_features_array(pairs: Sequence[tuple[Business, Business, str]]) -> np.ndarray:
    """Compute feature matrix for a sequence of (s1, target, target_source) tuples."""
    if not pairs:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)
    rows = [compute_pair_features(s1, target, source) for s1, target, source in pairs]
    return np.asarray(rows, dtype=np.float32)
