---
name: realhw
server_id: realhw-mcp
launch:
  command: ["python", "realhw_mcp_server.py"]
  env:
    PYTHONPATH: "${PYTHONPATH:-.}"
    REALHW_BASE_URL: "${REALHW_BASE_URL:-https://phoenix518455.dev3sub2phx.databasede3phx.oraclevcn.com:8000/realhw}"  # placeholder
    REALHW_API_TOKEN: "${REALHW_API_TOKEN:-}"
tools:
  - name: map_lrg_to_scheduler
    description: Map LRG to scheduler.
    intents: [query]
  - name: view_status_of_sched
    description: Scheduler status overview.
    intents: [query]
  - name: reserve_hardware
    description: Reserve hardware resources.
    intents: [action]
  - name: unreserve_hardware
    description: Release hardware reservations.
    intents: [action]
  - name: flag_usage_issues
    description: Flag usage anomalies.
    intents: [diagnose]
  - name: flag_large_waitlist
    description: Detect large waitlists.
    intents: [diagnose]
  - name: get_quarantined_hardware
    description: List quarantined hardware.
    intents: [diagnose]
  - name: get_farm_job_status
    description: Query farm job status.
    intents: [query]
  - name: simulate_sched_end_time
    description: Estimate scheduler completion.
    intents: [plan]
  - name: get_functional_hardware_mapping
    description: Functional hardware mapping.
    intents: [query]
  - name: move_job_to_top
    description: Prioritize farm job.
    intents: [action]
  - name: tool_manifest
    description: Advertises available tools.
    intents: [metadata]
presets:
  - name: reserve_default
    description: Reserve hardware using defaults for `{request}`.
    payload:
      tool: reserve_hardware
      args:
        request: "{request}"
logs:
  paths:
    - metrics/mcp_calls.jsonl
owners:
  - name: Real Hardware Team
    contact: realhw-support@oracle.com
---

## Overview
Interfaces with RealHW scheduling APIs for reservations, status, and diagnostics.

## Usage
- Launch with `python realhw_mcp_server.py`.
- Ensure tokens configured; many tools expect structured payloads (see server docstrings).

## Troubleshooting
- HTTP errors typically indicate expired tokens or network restrictions.
- Review server logs and RealHW dashboards for confirmation.

