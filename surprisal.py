"""Surprisal-based retrieval gate for the prompt hook.

Computes word-level surprisal against two corpora to decide whether
a user query is worth running through memory retrieval:

1. General English — high surprisal means substantive/technical content,
   low means filler ("hi how are you").
2. Personal corpus — low surprisal means familiar topic (memories likely
   exist), high means novel territory (nothing to retrieve).

Gate logic: retrieve when general surprisal is HIGH (substantive) AND
personal surprisal is LOW (familiar). Short prompts always skip.
"""

import math
import os
import re

# ln(P) for unseen words in the general corpus — ~1 in a billion
UNSEEN_GENERAL_LOG_PROB = math.log(1e-9)

# Thresholds — calibrate against reflection data
# General: above this = substantive content worth retrieving
GENERAL_THRESHOLD = float(os.environ.get("SURPRISAL_GENERAL_THRESHOLD", "6.0"))
# Personal: below this = familiar topic, memories likely exist
PERSONAL_THRESHOLD = float(os.environ.get("SURPRISAL_PERSONAL_THRESHOLD", "12.0"))


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphabetic tokens."""
    return re.findall(r"[a-z]+", text.lower())


def general_surprisal(words: list[str],
                      english_log_probs: dict[str, float]) -> float:
    """Mean surprisal of words against the general english corpus.

    Returns mean(-log_prob) across all words. Higher = more unusual content.
    english_log_probs: {word: ln(P(word))} from the database.
    """
    if not words:
        return 0.0
    total = 0.0
    for w in words:
        log_prob = english_log_probs.get(w, UNSEEN_GENERAL_LOG_PROB)
        total += -log_prob
    return total / len(words)


def personal_surprisal(words: list[str],
                       personal_counts: dict[str, int],
                       corpus_total: int,
                       vocab_size: int) -> float:
    """Mean surprisal of words against the personal corpus.

    Uses Laplace (add-1) smoothing so unseen words get high but finite
    surprisal. Returns mean(-log(P(word))). Higher = more novel topic.
    """
    if not words or corpus_total == 0:
        return float('inf')  # no personal data → can't assess familiarity
    # Laplace smoothing denominator: total + vocab_size
    # (adding 1 for each word in vocab)
    smoothed_total = corpus_total + vocab_size
    total = 0.0
    for w in words:
        count = personal_counts.get(w, 0)
        # Laplace: (count + 1) / (total + vocab_size)
        prob = (count + 1) / smoothed_total
        total += -math.log(prob)
    return total / len(words)


def should_retrieve(text: str,
                    english_log_probs: dict[str, float],
                    personal_counts: dict[str, int],
                    corpus_total: int,
                    vocab_size: int) -> dict:
    """Decide whether to fire memory retrieval for this query.

    Returns {
        "retrieve": bool,
        "reason": str,
        "general_surprisal": float,
        "personal_surprisal": float,
        "token_count": int,
    }
    """
    words = tokenize(text)

    if not words:
        return {
            "retrieve": False,
            "reason": "no_tokens",
            "general_surprisal": 0.0,
            "personal_surprisal": 0.0,
            "token_count": 0,
        }

    gen = general_surprisal(words, english_log_probs)
    pers = personal_surprisal(words, personal_counts, corpus_total, vocab_size)

    # Gate: retrieve when content is substantive AND topic is familiar
    substantive = gen >= GENERAL_THRESHOLD
    familiar = pers <= PERSONAL_THRESHOLD

    retrieve = substantive and familiar

    if not substantive:
        reason = "filler"
    elif not familiar:
        reason = "novel_topic"
    else:
        reason = "gate_passed"

    return {
        "retrieve": retrieve,
        "reason": reason,
        "general_surprisal": gen,
        "personal_surprisal": pers,
        "token_count": len(words),
    }
