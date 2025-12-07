---
name: genai4test-chat
server_id: genai4test-chat-mcp
launch:
  command: ["python", "genai4test_chat_server.py"]
  env:
    PYTHONPATH: "${PYTHONPATH:-.}"
    GENAI4TEST_BASE_URL: "${GENAI4TEST_BASE_URL:-https://testassist.oraclecorp.com/vm3/}"
    GENAI4TEST_EMAIL: "${GENAI4TEST_EMAIL:-user@oracle.com}"
    GENAI4TEST_AGENT: "${GENAI4TEST_AGENT:-bug_agent_dynamic}"
    GENAI4TEST_TIMEOUT_S: "${GENAI4TEST_TIMEOUT_S:-600}"
tools:
  - name: health
    description: Server health check.
    intents: [metadata]
  - name: run_bug_test
    description: Trigger GenAI4Test bug test workflow.
    intents: [action]
  - name: tool_manifest
    description: Advertises available tools.
    intents: [metadata]
presets:
  - name: bug_test_default
    description: Run bug test with default agent/email.
    payload:
      tool: run_bug_test
      args:
        bug_id: "{bug_id}"
logs:
  paths:
    - metrics/mcp_calls.jsonl
owners:
  - name: GenAI4Test Team
    contact: genai4test@oracle.com
---

## Overview
Chat-focused GenAI4Test entrypoint for bug workflows.

## Usage
- Launch with `python genai4test_chat_server.py`.
- `run_bug_test` proxies to the GenAI4Test chat agent; requires valid credentials.

## Troubleshooting
- Verify base URL and SSO tokens; HTTP errors surface in logs.

