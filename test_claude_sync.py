"""Tests for claude_sync.py — tree walking, message text extraction, conversation chunking.

All functions under test are pure — no network or filesystem needed.
"""

from claude_sync import extract_active_branch, extract_message_text, conversation_to_chunks


# ---------------------------------------------------------------------------
# extract_active_branch
# ---------------------------------------------------------------------------

def _msg(uuid, parent=None, index=0, sender="human", text="hello"):
    """Helper: build a minimal claude.ai chat message."""
    return {
        "uuid": uuid,
        "parent_message_uuid": parent,
        "index": index,
        "sender": sender,
        "content": [{"type": "text", "text": text}],
    }


def test_active_branch_linear_chain():
    """A→B→C with leaf=C returns all three in order."""
    messages = [
        _msg("A", parent=None, index=0),
        _msg("B", parent="A", index=1),
        _msg("C", parent="B", index=2),
    ]
    result = extract_active_branch(messages, "C")
    assert [m["uuid"] for m in result] == ["A", "B", "C"]


def test_active_branch_with_sibling():
    """Branched tree: A→B→C and A→B→D. Leaf=C excludes D."""
    messages = [
        _msg("A", parent=None, index=0),
        _msg("B", parent="A", index=1),
        _msg("C", parent="B", index=2),
        _msg("D", parent="B", index=3),  # sibling of C
    ]
    result = extract_active_branch(messages, "C")
    uuids = [m["uuid"] for m in result]
    assert "C" in uuids
    assert "D" not in uuids
    assert uuids == ["A", "B", "C"]


def test_active_branch_single_message():
    messages = [_msg("A", parent=None, index=0)]
    result = extract_active_branch(messages, "A")
    assert len(result) == 1
    assert result[0]["uuid"] == "A"


def test_active_branch_nonexistent_leaf():
    messages = [_msg("A", parent=None, index=0)]
    result = extract_active_branch(messages, "nonexistent")
    assert result == []


# ---------------------------------------------------------------------------
# extract_message_text
# ---------------------------------------------------------------------------

def test_extract_string_content():
    msg = {"content": "Simple string"}
    assert extract_message_text(msg) == "Simple string"


def test_extract_text_blocks():
    msg = {"content": [
        {"type": "text", "text": "Part 1"},
        {"type": "text", "text": "Part 2"},
    ]}
    assert extract_message_text(msg) == "Part 1\nPart 2"


def test_extract_plain_strings_in_list():
    msg = {"content": ["Hello", "World"]}
    assert extract_message_text(msg) == "Hello\nWorld"


def test_extract_empty_content():
    assert extract_message_text({}) == ""
    assert extract_message_text({"content": []}) == ""


# ---------------------------------------------------------------------------
# conversation_to_chunks
# ---------------------------------------------------------------------------

def _conversation(messages, uuid="conv-1", leaf=None):
    """Helper: wrap messages in a conversation dict."""
    if leaf is None:
        leaf = messages[-1]["uuid"] if messages else ""
    return {
        "uuid": uuid,
        "current_leaf_message_uuid": leaf,
        "chat_messages": messages,
    }


def test_single_human_assistant_pair():
    msgs = [
        _msg("m1", parent=None, index=0, sender="human", text="Question"),
        _msg("m2", parent="m1", index=1, sender="assistant", text="Answer"),
    ]
    chunks = conversation_to_chunks(_conversation(msgs, uuid="abc-123"))
    assert len(chunks) == 1
    c = chunks[0]
    assert "User: Question" in c["text"]
    assert "Assistant: Answer" in c["text"]
    assert c["session_id"] == "claude-web-abc-123"
    assert c["project"] == "claude.ai"
    assert c["turn_number"] == 0


def test_multiple_turns():
    msgs = [
        _msg("m1", parent=None, index=0, sender="human", text="Q1"),
        _msg("m2", parent="m1", index=1, sender="assistant", text="A1"),
        _msg("m3", parent="m2", index=2, sender="human", text="Q2"),
        _msg("m4", parent="m3", index=3, sender="assistant", text="A2"),
    ]
    chunks = conversation_to_chunks(_conversation(msgs))
    assert len(chunks) == 2
    assert chunks[0]["turn_number"] == 0
    assert chunks[1]["turn_number"] == 1


def test_truncation():
    long_text = "x" * 20000
    msgs = [
        _msg("m1", parent=None, index=0, sender="human", text="Q"),
        _msg("m2", parent="m1", index=1, sender="assistant", text=long_text),
    ]
    chunks = conversation_to_chunks(_conversation(msgs))
    assert len(chunks) == 1
    assert chunks[0]["text"].endswith("[truncated]")
    assert len(chunks[0]["text"]) <= 16000 + len("\n\n[truncated]")


def test_empty_conversation():
    conv = {"uuid": "x", "current_leaf_message_uuid": "", "chat_messages": []}
    assert conversation_to_chunks(conv) == []

    conv2 = {"uuid": "x", "current_leaf_message_uuid": "missing", "chat_messages": []}
    assert conversation_to_chunks(conv2) == []
