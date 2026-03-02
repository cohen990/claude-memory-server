"""Tests for MEMORY_DISABLED=1 env var — all hooks and MCP tools bail early."""

import asyncio
import json
import os
import subprocess
import sys

import pytest


REPO_DIR = os.path.dirname(__file__)


# ---------------------------------------------------------------------------
# Hook scripts: prompt_hook.py and ingest.py should exit silently
# ---------------------------------------------------------------------------

def run_hook_script(script_name: str, stdin_data: dict, env_override: dict | None = None):
    """Run a hook script as a subprocess and return (stdout, stderr, returncode)."""
    env = os.environ.copy()
    env["MEMORY_SERVER_URL"] = "http://127.0.0.1:1"  # unreachable
    if env_override:
        env.update(env_override)

    result = subprocess.run(
        [sys.executable, os.path.join(REPO_DIR, script_name)],
        input=json.dumps(stdin_data),
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )
    return result.stdout, result.stderr, result.returncode


class TestPromptHookDisabled:
    """prompt_hook.py should produce no output when MEMORY_DISABLED=1."""

    def test_no_output_when_disabled(self):
        hook_input = {
            "prompt": "tell me about the memory server architecture",
            "session_id": "test-session-123",
        }
        stdout, stderr, rc = run_hook_script(
            "prompt_hook.py", hook_input, {"MEMORY_DISABLED": "1"}
        )
        assert stdout == ""
        assert rc == 0

    def test_no_output_when_disabled_whitespace(self):
        """MEMORY_DISABLED=' 1 ' should also work (stripped)."""
        hook_input = {"prompt": "hello", "session_id": "test-session"}
        stdout, stderr, rc = run_hook_script(
            "prompt_hook.py", hook_input, {"MEMORY_DISABLED": " 1 "}
        )
        assert stdout == ""
        assert rc == 0

    def test_not_disabled_when_zero(self):
        """MEMORY_DISABLED=0 should NOT skip — script will try to connect and
        fail silently (no output because server is unreachable), but the point
        is it doesn't exit at the guard."""
        hook_input = {"prompt": "hello", "session_id": "test-session"}
        stdout, stderr, rc = run_hook_script(
            "prompt_hook.py", hook_input, {"MEMORY_DISABLED": "0"}
        )
        # Script runs through but can't reach server — produces no stdout
        # (search_graph returns empty on connection failure, so no output)
        assert rc == 0


class TestIngestHookDisabled:
    """ingest.py should produce no output when MEMORY_DISABLED=1."""

    def test_no_output_when_disabled(self):
        hook_input = {
            "transcript_path": "/nonexistent/transcript.jsonl",
            "session_id": "test-session-123",
        }
        stdout, stderr, rc = run_hook_script(
            "ingest.py", hook_input, {"MEMORY_DISABLED": "1"}
        )
        assert stdout == ""
        assert stderr == ""
        assert rc == 0

    def test_not_disabled_without_env(self):
        """Without MEMORY_DISABLED, ingest.py should try to run (and complain
        about the nonexistent transcript)."""
        hook_input = {
            "transcript_path": "/nonexistent/transcript.jsonl",
            "session_id": "test-session-123",
        }
        stdout, stderr, rc = run_hook_script("ingest.py", hook_input)
        assert "not found" in stderr.lower() or "transcript" in stderr.lower()


# ---------------------------------------------------------------------------
# MCP bridge: all tools should return disabled message
# ---------------------------------------------------------------------------

class TestMcpBridgeDisabled:
    """When MEMORY_DISABLED=1, every MCP tool returns a disabled message."""

    @pytest.fixture(autouse=True)
    def _set_disabled(self, monkeypatch):
        monkeypatch.setenv("MEMORY_DISABLED", "1")
        # Re-import so the module-level constant picks up the new env
        import importlib
        import mcp_bridge
        importlib.reload(mcp_bridge)
        self.bridge = mcp_bridge
        yield
        # Restore
        monkeypatch.delenv("MEMORY_DISABLED", raising=False)
        importlib.reload(mcp_bridge)

    def test_search_memory_disabled(self):
        result = asyncio.run(self.bridge.search_memory("test query"))
        assert "disabled" in result.lower()

    def test_search_memory_detail_disabled(self):
        result = asyncio.run(self.bridge.search_memory_detail("test query"))
        assert "disabled" in result.lower()

    def test_search_memory_graph_disabled(self):
        result = asyncio.run(self.bridge.search_memory_graph("test query"))
        assert "disabled" in result.lower()

    def test_memory_stats_disabled(self):
        result = asyncio.run(self.bridge.memory_stats())
        assert "disabled" in result.lower()

    def test_list_recalls_disabled(self):
        result = asyncio.run(self.bridge.list_recalls("some-session"))
        assert "disabled" in result.lower()

    def test_reflect_disabled(self):
        result = asyncio.run(self.bridge.reflect("abc123:U,N,N"))
        assert "disabled" in result.lower()
