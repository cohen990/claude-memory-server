"""Tests for ingest.py — chunker state machine, system-reminder stripping, text extraction.

All functions under test are pure (no network, no filesystem), so no fixtures needed.
"""

from ingest import _strip_system_reminders, extract_text, chunk_transcript, get_latest_turn_pair


# ---------------------------------------------------------------------------
# _strip_system_reminders
# ---------------------------------------------------------------------------

def test_strip_single_block():
    text = "Hello <system-reminder>secret</system-reminder> world"
    assert _strip_system_reminders(text) == "Hello  world"


def test_strip_multiline_block():
    text = "Before\n<system-reminder>\nline1\nline2\n</system-reminder>\nAfter"
    result = _strip_system_reminders(text)
    assert "<system-reminder>" not in result
    assert "line1" not in result
    assert "Before" in result
    assert "After" in result


def test_strip_multiple_blocks():
    text = "A <system-reminder>x</system-reminder> B <system-reminder>y</system-reminder> C"
    result = _strip_system_reminders(text)
    assert "x" not in result
    assert "y" not in result
    assert "A" in result
    assert "B" in result
    assert "C" in result


def test_strip_no_blocks_passthrough():
    text = "Just normal text, nothing to strip."
    assert _strip_system_reminders(text) == text


def test_strip_collapses_excess_whitespace():
    text = "Before\n<system-reminder>removed</system-reminder>\n\n\n\nAfter"
    result = _strip_system_reminders(text)
    # 3+ newlines should collapse to 2
    assert "\n\n\n" not in result
    assert "Before" in result
    assert "After" in result


# ---------------------------------------------------------------------------
# extract_text
# ---------------------------------------------------------------------------

def test_extract_text_string_content():
    msg = {"content": "Hello world"}
    assert extract_text(msg) == "Hello world"


def test_extract_text_list_of_text_blocks():
    msg = {"content": [
        {"type": "text", "text": "Part one"},
        {"type": "text", "text": "Part two"},
    ]}
    assert extract_text(msg) == "Part one\nPart two"


def test_extract_text_skips_tool_use_blocks():
    msg = {"content": [
        {"type": "text", "text": "Before tool"},
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
        {"type": "text", "text": "After tool"},
    ]}
    assert extract_text(msg) == "Before tool\nAfter tool"


def test_extract_text_empty_content():
    assert extract_text({}) == ""
    assert extract_text({"content": ""}) == ""
    assert extract_text({"content": []}) == ""


def test_extract_text_strips_system_reminders():
    msg = {"content": "Hello <system-reminder>secret</system-reminder> world"}
    result = extract_text(msg)
    assert "secret" not in result
    assert "Hello" in result


# ---------------------------------------------------------------------------
# chunk_transcript
# ---------------------------------------------------------------------------

def _user_msg(text, session_id="sess-1", cwd="/home/user/project",
              timestamp="2025-01-01T00:00:00Z", branch="main"):
    """Helper: build a user transcript message."""
    return {
        "type": "user",
        "message": {"role": "user", "content": text},
        "sessionId": session_id,
        "cwd": cwd,
        "timestamp": timestamp,
        "gitBranch": branch,
    }


def _assistant_msg(text):
    """Helper: build an assistant transcript message."""
    return {
        "type": "assistant",
        "message": {"role": "assistant", "content": text},
    }


def _tool_result_msg():
    """Helper: build a tool_result user message (should be skipped)."""
    return {
        "type": "user",
        "message": {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}],
        },
    }


def test_chunk_single_pair():
    messages = [_user_msg("Hello"), _assistant_msg("Hi there")]
    chunks = chunk_transcript(messages)
    assert len(chunks) == 1
    assert "User: Hello" in chunks[0]["text"]
    assert "Assistant: Hi there" in chunks[0]["text"]
    assert chunks[0]["turn_number"] == 0


def test_chunk_multiple_pairs():
    messages = [
        _user_msg("Q1"), _assistant_msg("A1"),
        _user_msg("Q2"), _assistant_msg("A2"),
        _user_msg("Q3"), _assistant_msg("A3"),
    ]
    chunks = chunk_transcript(messages)
    assert len(chunks) == 3
    assert [c["turn_number"] for c in chunks] == [0, 1, 2]


def test_chunk_skips_tool_result_messages():
    messages = [
        _user_msg("Hello"),
        _assistant_msg("Let me check"),
        _tool_result_msg(),
        _assistant_msg("Here's the answer"),
    ]
    chunks = chunk_transcript(messages)
    assert len(chunks) == 1
    assert "Here's the answer" in chunks[0]["text"]


def test_chunk_concatenates_consecutive_assistant_messages():
    messages = [
        _user_msg("Tell me a story"),
        _assistant_msg("Once upon a time"),
        _assistant_msg("the end"),
    ]
    chunks = chunk_transcript(messages)
    assert len(chunks) == 1
    assert "Once upon a time" in chunks[0]["text"]
    assert "the end" in chunks[0]["text"]


def test_chunk_truncation():
    long_text = "x" * 20000
    messages = [_user_msg("Q"), _assistant_msg(long_text)]
    chunks = chunk_transcript(messages)
    assert len(chunks) == 1
    assert len(chunks[0]["text"]) <= 16000 + len("\n\n[truncated]")
    assert chunks[0]["text"].endswith("[truncated]")


def test_chunk_skips_empty_user_text():
    messages = [_user_msg(""), _assistant_msg("response")]
    chunks = chunk_transcript(messages)
    assert len(chunks) == 0


def test_chunk_skips_non_user_role():
    messages = [{
        "type": "user",
        "message": {"role": "system", "content": "system prompt"},
    }, _assistant_msg("response")]
    chunks = chunk_transcript(messages)
    assert len(chunks) == 0


def test_chunk_metadata_extraction():
    messages = [
        _user_msg("Hi", session_id="abc-123", cwd="/home/user/myproject",
                  timestamp="2025-06-15T10:00:00Z", branch="feature-x"),
        _assistant_msg("Hello"),
    ]
    chunks = chunk_transcript(messages)
    assert len(chunks) == 1
    c = chunks[0]
    assert c["session_id"] == "abc-123"
    assert c["project"] == "/home/user/myproject"
    assert c["timestamp"] == "2025-06-15T10:00:00Z"
    assert c["branch"] == "feature-x"


# ---------------------------------------------------------------------------
# get_latest_turn_pair
# ---------------------------------------------------------------------------

def test_get_latest_turn_pair_returns_last():
    messages = [
        _user_msg("First"), _assistant_msg("R1"),
        _user_msg("Second"), _assistant_msg("R2"),
        _user_msg("Third"), _assistant_msg("R3"),
    ]
    result = get_latest_turn_pair(messages)
    assert result is not None
    assert "Third" in result["text"]
    assert "R3" in result["text"]


def test_get_latest_turn_pair_empty():
    assert get_latest_turn_pair([]) is None
