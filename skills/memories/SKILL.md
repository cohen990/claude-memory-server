---
name: memories
description: View the graph memories that were injected into this conversation and how they were rated.
user-invocable: true
---

Show the user what graph memories were injected in this session and how you rated them.

Call the `list_recalls` MCP tool with:
- `session_id`: use your current session ID (from the prompt hook context)
- `limit`: $ARGUMENTS (default to 1 if no argument given; the user can pass a number like `/memories 3`)

Format the output for the user as a readable summary. For each recall, show:
- When it was injected (timestamp)
- Each memory with its type (vibe/detail), similarity score, and rating with label:
  - U = USED (directly informed your response)
  - I = INTERESTING (added context but wasn't directly used)
  - N = NOISE (irrelevant)
  - D = DISTRACTING (irrelevant and got in the way)
  - M = MISLEADING (wrong or outdated)
- If a memory is unrated, note that

Keep the output concise. Truncate long memory texts to ~150 chars with "..." suffix.
