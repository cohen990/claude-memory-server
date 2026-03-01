"""Tests for the surprisal-based retrieval gate."""

import math
import pytest

from surprisal import (
    tokenize,
    general_surprisal,
    personal_surprisal,
    should_retrieve,
    UNSEEN_GENERAL_LOG_PROB,
)
from graph import GraphStore


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    assert tokenize("Hello World") == ["hello", "world"]


def test_tokenize_strips_punctuation():
    assert tokenize("it's a test! right?") == ["it", "s", "a", "test", "right"]


def test_tokenize_numbers_removed():
    assert tokenize("version 3.14 release") == ["version", "release"]


def test_tokenize_empty():
    assert tokenize("") == []
    assert tokenize("123 456") == []


# ---------------------------------------------------------------------------
# general_surprisal
# ---------------------------------------------------------------------------

def test_general_surprisal_known_words():
    probs = {"the": math.log(0.05), "cat": math.log(0.001)}
    words = ["the", "cat"]
    result = general_surprisal(words, probs)
    expected = (-math.log(0.05) + -math.log(0.001)) / 2
    assert abs(result - expected) < 1e-6


def test_general_surprisal_unseen_words():
    probs = {"the": math.log(0.05)}
    words = ["the", "xyzzy"]
    result = general_surprisal(words, probs)
    expected = (-math.log(0.05) + -UNSEEN_GENERAL_LOG_PROB) / 2
    assert abs(result - expected) < 1e-6


def test_general_surprisal_empty():
    assert general_surprisal([], {}) == 0.0


def test_general_surprisal_all_unseen():
    """All unseen words should give uniformly high surprisal."""
    result = general_surprisal(["foo", "bar", "baz"], {})
    expected = -UNSEEN_GENERAL_LOG_PROB  # same for each word
    assert abs(result - expected) < 1e-6


# ---------------------------------------------------------------------------
# personal_surprisal
# ---------------------------------------------------------------------------

def test_personal_surprisal_familiar_words():
    counts = {"memory": 50, "server": 30}
    total = 1000
    vocab = 200
    result = personal_surprisal(["memory", "server"], counts, total, vocab)
    # Laplace: P(memory) = (50+1)/(1000+200), P(server) = (31)/(1200)
    expected = (
        -math.log(51 / 1200) + -math.log(31 / 1200)
    ) / 2
    assert abs(result - expected) < 1e-6


def test_personal_surprisal_unseen_word():
    counts = {"memory": 50}
    total = 1000
    vocab = 200
    result = personal_surprisal(["xyzzy"], counts, total, vocab)
    # Laplace: P(xyzzy) = 1/1200
    expected = -math.log(1 / 1200)
    assert abs(result - expected) < 1e-6


def test_personal_surprisal_empty_corpus():
    """No personal data → infinite surprisal (can't assess familiarity)."""
    result = personal_surprisal(["hello"], {}, 0, 0)
    assert result == float('inf')


def test_personal_surprisal_empty_words():
    result = personal_surprisal([], {"a": 10}, 100, 50)
    assert result == float('inf')  # no words → can't compute


# ---------------------------------------------------------------------------
# should_retrieve
# ---------------------------------------------------------------------------

def test_should_retrieve_no_tokens():
    """Empty/non-alpha input should be skipped."""
    result = should_retrieve("123 !!!", {}, {}, 0, 0)
    assert result["retrieve"] is False
    assert result["reason"] == "no_tokens"


def test_should_retrieve_filler():
    """Common everyday words should be classified as filler."""
    # All words known and very common → low general surprisal
    common_probs = {w: math.log(0.01) for w in ["how", "are", "you", "doing", "today"]}
    result = should_retrieve(
        "how are you doing today",
        common_probs, {"how": 5, "are": 5, "you": 5, "doing": 5, "today": 5},
        1000, 200,
    )
    assert result["reason"] == "filler"
    assert result["retrieve"] is False


def test_should_retrieve_novel_topic():
    """Technical content with no personal history → novel topic."""
    # High general surprisal (unseen words), no personal corpus at all
    result = should_retrieve(
        "quantum chromodynamics gluon confinement",
        {},  # no english probs → all unseen → high general surprisal
        {},  # no personal counts
        0, 0,  # empty personal corpus → inf personal surprisal
    )
    assert result["reason"] == "novel_topic"
    assert result["retrieve"] is False


def test_should_retrieve_gate_passes():
    """Substantive content about familiar topic → retrieve."""
    # High general surprisal (technical words not in common english)
    # Low personal surprisal (user talks about these a lot)
    result = should_retrieve(
        "fix the embedding model server configuration",
        # Only common words have probs → technical words get unseen floor
        {"the": math.log(0.05), "fix": math.log(0.005)},
        # User talks about these all the time
        {"fix": 20, "the": 100, "embedding": 50, "model": 40,
         "server": 60, "configuration": 15},
        10000, 500,
    )
    assert result["retrieve"] is True
    assert result["reason"] == "gate_passed"


# ---------------------------------------------------------------------------
# GraphStore word frequency integration
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    db_path = str(tmp_path / "test.db")
    return GraphStore(db_path=db_path)


def test_english_freqs_load_and_query(store):
    freqs = {"the": -2.92, "cat": -8.5, "lol": -9.1}
    store.load_english_freqs(freqs)
    assert store.english_freq_count() == 3
    assert store.get_english_log_prob("the") == pytest.approx(-2.92)
    assert store.get_english_log_prob("unknown") is None


def test_english_freqs_bulk_query(store):
    freqs = {"the": -2.92, "cat": -8.5, "dog": -8.3}
    store.load_english_freqs(freqs)
    result = store.get_english_log_probs(["the", "cat", "unknown"])
    assert result == {"the": pytest.approx(-2.92), "cat": pytest.approx(-8.5)}


def test_english_freqs_replace(store):
    store.load_english_freqs({"old": -5.0})
    store.load_english_freqs({"new": -6.0})
    assert store.get_english_log_prob("old") is None
    assert store.get_english_log_prob("new") == pytest.approx(-6.0)


def test_personal_word_counts_update(store):
    store.update_personal_word_counts(["hello", "world", "hello"])
    counts = store.get_personal_word_counts(["hello", "world", "missing"])
    assert counts == {"hello": 2, "world": 1}


def test_personal_word_counts_incremental(store):
    store.update_personal_word_counts(["hello"])
    store.update_personal_word_counts(["hello", "world"])
    counts = store.get_personal_word_counts(["hello", "world"])
    assert counts == {"hello": 2, "world": 1}


def test_personal_corpus_stats(store):
    store.update_personal_word_counts(["a", "b", "a", "c"])
    assert store.personal_corpus_total() == 4
    assert store.personal_vocab_size() == 3


def test_personal_corpus_empty(store):
    assert store.personal_corpus_total() == 0
    assert store.personal_vocab_size() == 0
