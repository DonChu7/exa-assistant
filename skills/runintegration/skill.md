---
name: runintegration
server_id: runintegration-mcp
launch:
  command: ["python", "runintegration_server.py"]
  env:
    PYTHONPATH: "${PYTHONPATH:-.}"
    RUNTABLE_PATH: "${RUNTABLE_PATH:-/net/10.32.19.91/export/exadata_images/ImageTests/daily_runs_1/OSS_MAIN/runtable}"
    PXEQA_CONNECT: "${PXEQA_CONNECT:-/net/10.32.19.91/export/exadata_images/ImageTests/.pxeqa_connect}"
tools:
  - name: idle_envs
    description: Concurrent scan for idle RunIntegration environments.
    intents: [diagnose, list]
  - name: disabled_envs
    description: Enumerate disabled environments from runtable.
    intents: [diagnose]
  - name: status
    description: Check RunIntegration status for a rack.
    intents: [query]
  - name: pending_tests
    description: Pending tests + host map for a transaction.
    intents: [query, diagnose]
  - name: disabled_txn_status
    description: Disabled environment status per transaction.
    intents: [query]
  - name: tool_manifest
    description: Advertises available tools.
    intents: [metadata]
presets:
  - name: pending_tests_summary
    description: Summarize pending tests for txn `{txn}`.
    payload:
      tool: pending_tests
      args:
        txn: "{txn}"
logs:
  paths:
    - metrics/mcp_calls.jsonl
owners:
  - name: RunIntegration Ops
    contact: runintegration-ops@oracle.com
---

## Overview
Provides RunIntegration status helpers backed by shared runtable data and existing `runintegration_agent` functions.

## Usage
- Launch with `python runintegration_server.py` (requires network access to runtable share).
- Tools accept straightforward JSON payloads; see docstrings in server for details.
- Preset `pending_tests_summary` wraps a common support question.

## Troubleshooting
- Ensure CE/NET share is mounted at paths above; override via env if needed.
- `idle_envs` can take optional concurrency/timeouts; check server logs for SSH failures.

