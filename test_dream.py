"""Tests for dream.py — weight boost formula, embedding blend, cosine distance.

Tests the pure math helpers extracted from cmd_reconsolidate.
"""

import numpy as np

from dream import _compute_weight_boost, _blend_embeddings, _cosine_distance


# ---------------------------------------------------------------------------
# _compute_weight_boost
# ---------------------------------------------------------------------------

def test_weight_boost_single_activation():
    """count=1, log(1)=0, so boost = 0.1 * (1+0) = 0.1 → 0.5+0.1 = 0.6."""
    result = _compute_weight_boost(0.5, count=1, boost=0.1)
    assert abs(result - 0.6) < 1e-9


def test_weight_boost_high_activation_capped():
    """Very high activation count should clamp at cap."""
    result = _compute_weight_boost(0.9, count=100, boost=0.1, cap=0.95)
    assert result == 0.95


def test_weight_boost_diminishing_returns():
    """Log curve: boost from count=2→3 should be smaller than count=1→2."""
    w1 = _compute_weight_boost(0.0, count=1, boost=0.1)
    w2 = _compute_weight_boost(0.0, count=2, boost=0.1)
    w3 = _compute_weight_boost(0.0, count=3, boost=0.1)
    delta_12 = w2 - w1
    delta_23 = w3 - w2
    assert delta_23 < delta_12


# ---------------------------------------------------------------------------
# _blend_embeddings
# ---------------------------------------------------------------------------

def test_blend_embeddings():
    """Node + 2 neighbors with known weights → verify manual calculation."""
    node = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    n1 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    n2 = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    # weights: node=1.0 (implicit), n1=0.5, n2=0.5
    # total_weight = 1.0 + 0.5 + 0.5 = 2.0
    # blended = (1,0,0) + 0.5*(0,1,0) + 0.5*(0,0,1) = (1, 0.5, 0.5) / 2 = (0.5, 0.25, 0.25)
    result = _blend_embeddings(node, [n1, n2], [0.5, 0.5])
    expected = np.array([0.5, 0.25, 0.25], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# _cosine_distance
# ---------------------------------------------------------------------------

def test_cosine_distance_identical():
    """Identical vectors → distance 0."""
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(_cosine_distance(v, v)) < 1e-6


def test_cosine_distance_orthogonal():
    """Orthogonal vectors → distance 1."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(_cosine_distance(a, b) - 1.0) < 1e-6
