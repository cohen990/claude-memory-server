"""Tests for dream.py — embedding blend, cosine distance,
reflection-weighted edge adjustment helpers, and synthesis prompt construction.
"""

import json
import numpy as np

from dream import (_blend_embeddings, _cosine_distance,
                   _cosine_sim, _apply_reflection_delta,
                   _build_synthesis_prompt, SYNTHESIS_SCHEMA,
                   REFLECTION_VALUES, REFLECTION_EDGE_FLOOR, REFLECTION_EDGE_CAP)


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
# _apply_reflection_delta
# ---------------------------------------------------------------------------

def test_apply_reflection_delta_positive():
    """Positive delta increases weight."""
    result = _apply_reflection_delta(0.5, 0.1)
    assert abs(result - 0.6) < 1e-9


def test_apply_reflection_delta_negative():
    """Negative delta decreases weight."""
    result = _apply_reflection_delta(0.5, -0.1)
    assert abs(result - 0.4) < 1e-9


def test_apply_reflection_delta_floor():
    """Weight should not go below floor."""
    result = _apply_reflection_delta(0.1, -0.5, floor=0.05)
    assert abs(result - 0.05) < 1e-9


def test_apply_reflection_delta_cap():
    """Weight should not exceed cap."""
    result = _apply_reflection_delta(0.9, 0.5, cap=0.95)
    assert abs(result - 0.95) < 1e-9


# ---------------------------------------------------------------------------
# REFLECTION_VALUES
# ---------------------------------------------------------------------------

def test_reflection_values():
    """Verify the reflection scale is correct."""
    assert REFLECTION_VALUES["U"] == 2
    assert REFLECTION_VALUES["I"] == 1
    assert REFLECTION_VALUES["N"] == 0
    assert REFLECTION_VALUES["D"] == -1
    assert REFLECTION_VALUES["M"] == -2


def test_reflection_edge_bounds():
    """Floor and cap should prevent full severance or saturation."""
    assert REFLECTION_EDGE_FLOOR > 0
    assert REFLECTION_EDGE_CAP < 1.0


# ---------------------------------------------------------------------------
# _build_synthesis_prompt
# ---------------------------------------------------------------------------

def test_build_synthesis_prompt_without_recalled_nodes():
    """Prompt should contain chunks but no existing memories section."""
    prompt = _build_synthesis_prompt(["chunk A", "chunk B"])
    assert "chunk A" in prompt
    assert "chunk B" in prompt
    assert "---" in prompt  # separator
    assert "Existing memories" not in prompt


def test_build_synthesis_prompt_with_recalled_nodes():
    """Prompt should include the recalled nodes section."""
    recalled = [
        {"id": "abc123", "type": "vibe", "text": "prefers functional programming"},
        {"id": "def456", "type": "detail", "text": "uses nomic embeddings"},
    ]
    prompt = _build_synthesis_prompt(["chunk 1"], recalled_nodes=recalled)

    assert "Existing memories that were recalled during these conversations:" in prompt
    assert '[0] (vibe, id=abc123) "prefers functional programming"' in prompt
    assert '[1] (detail, id=def456) "uses nomic embeddings"' in prompt
    assert "from_existing_idx" in prompt
    assert "to_existing_idx" in prompt


def test_build_synthesis_prompt_empty_recalled_nodes():
    """Empty recalled_nodes list should not add the section."""
    prompt = _build_synthesis_prompt(["chunk 1"], recalled_nodes=[])
    assert "Existing memories" not in prompt


# ---------------------------------------------------------------------------
# SYNTHESIS_SCHEMA — cross-temporal fields
# ---------------------------------------------------------------------------

def test_synthesis_schema_has_existing_idx_fields():
    """Schema should include from_existing_idx and to_existing_idx."""
    schema = json.loads(SYNTHESIS_SCHEMA)
    conn_props = schema["properties"]["connections"]["items"]["properties"]
    assert "from_existing_idx" in conn_props
    assert "to_existing_idx" in conn_props
    assert conn_props["from_existing_idx"]["type"] == "integer"
    assert conn_props["to_existing_idx"]["type"] == "integer"


def test_synthesis_schema_connections_only_require_weight():
    """Connections should only require weight, not from_idx/to_idx."""
    schema = json.loads(SYNTHESIS_SCHEMA)
    required = schema["properties"]["connections"]["items"]["required"]
    assert required == ["weight"]
