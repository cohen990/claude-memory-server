"""Tests for dream.py — embedding blend, cosine distance,
and rating-weighted edge adjustment helpers.

Tests the pure math helpers extracted from cmd_reconsolidate.
"""

import numpy as np

from dream import (_blend_embeddings, _cosine_distance,
                   _cosine_sim, _apply_rating_delta,
                   RATING_VALUES, RATING_EDGE_FLOOR, RATING_EDGE_CAP)


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


# ---------------------------------------------------------------------------
# _cosine_sim
# ---------------------------------------------------------------------------

def test_cosine_sim_identical():
    """Identical vectors → similarity 1."""
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert abs(_cosine_sim(v, v) - 1.0) < 1e-6


def test_cosine_sim_orthogonal():
    """Orthogonal vectors → similarity 0."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert abs(_cosine_sim(a, b)) < 1e-6


def test_cosine_sim_zero_vector():
    """Zero vector → similarity 0."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 0.0], dtype=np.float32)
    assert _cosine_sim(a, b) == 0.0


# ---------------------------------------------------------------------------
# _apply_rating_delta
# ---------------------------------------------------------------------------

def test_apply_rating_delta_positive():
    """Positive delta increases weight."""
    result = _apply_rating_delta(0.5, 0.1)
    assert abs(result - 0.6) < 1e-9


def test_apply_rating_delta_negative():
    """Negative delta decreases weight."""
    result = _apply_rating_delta(0.5, -0.1)
    assert abs(result - 0.4) < 1e-9


def test_apply_rating_delta_floor():
    """Weight should not go below floor."""
    result = _apply_rating_delta(0.1, -0.5, floor=0.05)
    assert abs(result - 0.05) < 1e-9


def test_apply_rating_delta_cap():
    """Weight should not exceed cap."""
    result = _apply_rating_delta(0.9, 0.5, cap=0.95)
    assert abs(result - 0.95) < 1e-9


# ---------------------------------------------------------------------------
# RATING_VALUES
# ---------------------------------------------------------------------------

def test_rating_values():
    """Verify the rating scale is correct."""
    assert RATING_VALUES["U"] == 2
    assert RATING_VALUES["I"] == 1
    assert RATING_VALUES["N"] == 0
    assert RATING_VALUES["D"] == -1
    assert RATING_VALUES["M"] == -2


def test_rating_edge_bounds():
    """Floor and cap should prevent full severance or saturation."""
    assert RATING_EDGE_FLOOR > 0
    assert RATING_EDGE_CAP < 1.0
