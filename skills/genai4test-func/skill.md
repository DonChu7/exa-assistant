---
name: genai4test-func
server_id: genai4test-mcp
launch:
  command: ["python", "genai4test_server.py"]
  env:
    PYTHONPATH: "${PYTHONPATH:-.}"
    GENAI4TEST_BASE_URL: "${GENAI4TEST_BASE_URL:-https://testassist.oraclecorp.com/vm3/}"
    GENAI4TEST_TIMEOUT_S: "${GENAI4TEST_TIMEOUT_S:-600}"
tools:
  - name: health
    description: Server health check.
    intents: [metadata]
  - name: run_bug_test
    description: Execute bug test via functional API.
    intents: [action]
  - name: list_func_test_agents
    description: List available functional test agents.
    intents: [query]
  - name: run_func_test
    description: Execute functional test with specified agent.
    intents: [action]
  - name: run_func_mem_agent
    description: Run GenAI4Test memory agent.
    intents: [action]
  - name: tool_manifest
    description: Advertises available tools.
    intents: [metadata]
presets:
  - name: run_default_agent
    description: Run default functional agent on request `{request_id}`.
    payload:
      tool: run_func_test
      args:
        request_id: "{request_id}"
        agent: "default"
logs:
  paths:
    - metrics/mcp_calls.jsonl
owners:
  - name: GenAI4Test Team
    contact: genai4test@oracle.com
---

## Overview
Functional GenAI4Test workflows (non-chat). Wraps HTTP API requests for bug and functional agents.

## Usage
- Launch with `python genai4test_server.py`.
- Provide required fields (`bug_id`, `request_id`, etc.).

## Troubleshooting
- Inspect logs for HTTP responses; adjust timeout/env for long-running operations.

