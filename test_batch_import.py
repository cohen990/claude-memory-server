"""Tests for batch_import.py — path derivation, transcript discovery, tracking persistence.

Uses tmp_path and monkeypatch for filesystem isolation.
"""

import json
from pathlib import Path
from unittest.mock import patch

import batch_import
from batch_import import derive_project_name, find_transcripts, load_tracking, save_tracking


# ---------------------------------------------------------------------------
# derive_project_name
# ---------------------------------------------------------------------------

def test_derive_standard_path():
    """Standard dash-encoded path converts back to absolute path."""
    path = Path(batch_import.CLAUDE_PROJECTS_DIR) / "-home-user-projects-myapp" / "abc.jsonl"
    result = derive_project_name(path)
    assert result == "/home/user/projects/myapp"


def test_derive_nested_dashes():
    """Every dash is treated as a separator — documents current behavior."""
    path = Path(batch_import.CLAUDE_PROJECTS_DIR) / "-home-user-my-cool-project" / "x.jsonl"
    result = derive_project_name(path)
    # All dashes become slashes — this is the known limitation
    assert result == "/home/user/my/cool/project"


def test_derive_unrelated_path():
    """Path not under CLAUDE_PROJECTS_DIR returns empty string."""
    path = Path("/tmp/random/file.jsonl")
    result = derive_project_name(path)
    assert result == ""


# ---------------------------------------------------------------------------
# find_transcripts
# ---------------------------------------------------------------------------

def _make_transcript_tree(tmp_path):
    """Create a mock ~/.claude/projects/ directory tree."""
    proj_dir = tmp_path / "projects"
    # Project A — two transcripts
    proj_a = proj_dir / "-home-user-projectA"
    proj_a.mkdir(parents=True)
    (proj_a / "session1.jsonl").write_text("{}\n")
    (proj_a / "session2.jsonl").write_text("{}\n")
    # Project A subagent
    subagent = proj_a / "subagents"
    subagent.mkdir()
    (subagent / "sub1.jsonl").write_text("{}\n")
    # Project B — one transcript
    proj_b = proj_dir / "-home-user-projectB"
    proj_b.mkdir(parents=True)
    (proj_b / "session3.jsonl").write_text("{}\n")
    return proj_dir


def test_find_transcripts_basic(tmp_path):
    proj_dir = _make_transcript_tree(tmp_path)
    with patch.object(batch_import, "CLAUDE_PROJECTS_DIR", str(proj_dir)):
        results = find_transcripts()
    # 3 transcripts (subagent excluded by default)
    assert len(results) == 3
    names = {p.name for p in results}
    assert names == {"session1.jsonl", "session2.jsonl", "session3.jsonl"}


def test_find_transcripts_excludes_subagents(tmp_path):
    proj_dir = _make_transcript_tree(tmp_path)
    with patch.object(batch_import, "CLAUDE_PROJECTS_DIR", str(proj_dir)):
        results = find_transcripts(include_subagents=False)
    names = {p.name for p in results}
    assert "sub1.jsonl" not in names


def test_find_transcripts_includes_subagents(tmp_path):
    proj_dir = _make_transcript_tree(tmp_path)
    with patch.object(batch_import, "CLAUDE_PROJECTS_DIR", str(proj_dir)):
        results = find_transcripts(include_subagents=True)
    names = {p.name for p in results}
    assert "sub1.jsonl" in names
    assert len(results) == 4


def test_find_transcripts_project_filter(tmp_path):
    proj_dir = _make_transcript_tree(tmp_path)
    with patch.object(batch_import, "CLAUDE_PROJECTS_DIR", str(proj_dir)):
        results = find_transcripts(project_filter="projectA")
    assert len(results) == 2
    for p in results:
        assert "projectA" in str(p)


# ---------------------------------------------------------------------------
# load_tracking / save_tracking
# ---------------------------------------------------------------------------

def test_tracking_roundtrip(tmp_path):
    tracking_file = str(tmp_path / "tracking.json")
    with patch.object(batch_import, "TRACKING_FILE", tracking_file):
        save_tracking({"file1.jsonl", "file2.jsonl"})
        loaded = load_tracking()
    assert loaded == {"file1.jsonl", "file2.jsonl"}


def test_tracking_missing_file(tmp_path):
    tracking_file = str(tmp_path / "nonexistent.json")
    with patch.object(batch_import, "TRACKING_FILE", tracking_file):
        loaded = load_tracking()
    assert loaded == set()
