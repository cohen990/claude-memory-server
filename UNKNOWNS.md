# Unknowns

## AskUserQuestion transcript format

How do `AskUserQuestion` tool calls and their responses appear in the JSONL transcript? If the user's answer comes back as a `tool_result` block, the chunker will skip it — the assistant's question text (a text block before the tool_use) would be captured, but the user's answer would be lost.

Need to inspect a real transcript from a plan-mode conversation to verify.
