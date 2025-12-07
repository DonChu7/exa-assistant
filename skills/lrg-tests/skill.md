---
name: lrg-tests
server_id: rag-fts-mcp
launch:
  command: ["python", "lrg_test_mcp_server.py"]
  env:
    LRGTEST_DATA_ROOT: "${LRGTEST_DATA_ROOT:-/scratch/datasets/lrg}"
    LRGTEST_PIPELINE_MODEL: "${LRGTEST_PIPELINE_MODEL:-}"
    PYTHONPATH: "${PYTHONPATH:-.}"
tools:
  - name: smart_search
    description: Router→backend→writer pipeline for LRG/test questions.
    intents: [query, ask, diagnose, recommend]
  - name: query_tests
    description: Raw QueryPayload execution against the backend.
    intents: [advanced, payload]
  - name: tool_manifest
    description: Advertises available tools and capabilities.
    intents: [metadata]
presets:
  - name: list_tests_for_lrg
    description: Retrieves tests mapped to a given LRG.
    payload:
      tool: smart_search
      args:
        question: "tests for {lrg}"
        plan:
          mode: search
          text: ""
          filters:
            suite: []
            lrg: ["{lrg}"]
            setup: []
            flag: []
            doc_type: ["TEST"]
          ops:
            limit: 20
            offset: 0
            group_by: none
            sort:
              - by: score
                dir: desc
        k: 20
  - name: get_lrg_runtime_stats
    description: Summarizes recent runtime spikes.
    payload:
      tool: smart_search
      args:
        question: "runtime stats for {lrg}"
        plan:
          mode: runtime_spike
          text: "runtime stats {lrg}"
          filters:
            suite: []
            lrg: ["{lrg}"]
            setup: []
            flag: []
            doc_type: ["LRG"]
          ops:
            limit: 10
            offset: 0
            group_by: none
            sort:
              - by: runtime
                dir: asc
        k: 10
  - name: search_tests
    description: Free-form search with limit override.
    payload:
      tool: smart_search
      args:
        question: "{query}"
        plan:
          mode: search
          text: "{query}"
          filters:
            suite: []
            lrg: []
            setup: []
            flag: []
            doc_type: ["TEST"]
          ops:
            limit: "{k}"
            offset: 0
            group_by: none
            sort:
              - by: score
                dir: desc
        k: 20
logs:
  paths:
    - metrics/mcp_calls.jsonl
    - metrics/langgraph_debug.log
owners:
  - name: LRG Retrieval Team
    contact: lrg-retrieval@oracle.com
---

## Overview
`lrg_test_mcp_server.py` bundles the Router→Query→Writer pipeline for LRG metadata. `smart_search` accepts a natural language question (optionally a `plan` override) and composes final answers. `query_tests` runs the backend directly for advanced payloads.

## Usage
- Run via `python lrg_test_mcp_server.py` or let the bot spawn it using the launch command.
- `smart_search` payload: `{ "question": "...", "plan": Optional[QueryPayload], "k": limit }`.
- `query_tests` payload: any JSON compatible with `QueryPayload` schema.
- Presets map legacy Slack tools to smart-search calls; replace placeholders at runtime.

## Troubleshooting
- Ensure `PYTHONPATH` includes repo root so schemas import.
- Check `metrics/langgraph_debug.log` for pipeline traces.
- Backend errors propagate as `{"ok": false, "error": ...}`; inspect `result` field for details.
