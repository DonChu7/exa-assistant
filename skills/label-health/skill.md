---
name: label-health
server_id: label_health-mcp
launch:
  command: ["python", "label_health_server.py"]
  env:
    PYTHONPATH: "${PYTHONPATH:-.}"
    LABEL_HEALTH_BASE_URL: "${LABEL_HEALTH_BASE_URL:-https://apex.oraclecorp.com/pls/apex/lrg_times/}"  # override as needed
    LABEL_HEALTH_API_TOKEN: "${LABEL_HEALTH_API_TOKEN:-}"
tools:
  - name: get_labels_from_series
    description: List labels for a series.
    intents: [query]
  - name: get_lrg_info
    description: Fetch LRG metadata.
    intents: [query]
  - name: get_lrgs_from_regress
    description: Map regress to LRGs.
    intents: [query]
  - name: find_crashes
    description: Crash search across labels.
    intents: [diagnose]
  - name: find_lrg_with_difs
    description: Locate LRGs with DIFs.
    intents: [diagnose]
  - name: find_dif_details
    description: Detailed DIF breakdown.
    intents: [diagnose]
  - name: get_my_lrgs_status
    description: Review personal LRG assignments.
    intents: [query]
  - name: find_dif_occurrence
    description: DIF occurrence timeline.
    intents: [diagnose]
  - name: find_widespread_issues
    description: Detect widespread failures.
    intents: [diagnose]
  - name: query_ai_crash_summary
    description: AI crash summarization.
    intents: [summarize]
  - name: get_se_rerun_details
    description: SE rerun analysis.
    intents: [diagnose]
  - name: get_regress_summary
    description: Regress report summary.
    intents: [summarize]
  - name: get_label_info
    description: Label metadata lookup.
    intents: [query]
  - name: get_ai_label_summary
    description: AI-generated label synopsis.
    intents: [summarize]
  - name: generate_ai_label_summary
    description: Create AI label summary email.
    intents: [generate]
  - name: get_test_info
    description: Test metadata fetch.
    intents: [query]
  - name: find_tests_with_setup_and_flag
    description: Search tests by setup/flag.
    intents: [query]
  - name: get_lrg_history
    description: Historical LRG info.
    intents: [query]
  - name: get_delta_diffs_between_labels
    description: Compare label diffs.
    intents: [diagnose]
  - name: lrg_point_of_contact
    description: Contact lookup.
    intents: [query]
  - name: get_incomplete_lrgs
    description: Identify incomplete LRGs.
    intents: [query]
  - name: draft_email_for_lrg
    description: Draft communication email.
    intents: [generate]
  - name: add_lrg_to_my_lrgs
    description: Track LRG in personal list.
    intents: [action]
  - name: add_series_to_auto_populate
    description: Auto-populate series settings.
    intents: [action]
  - name: add_label_for_se_analysis
    description: Add label for SE analysis.
    intents: [action]
  - name: add_label_for_analysis
    description: Queue label analysis.
    intents: [action]
  - name: tool_manifest
    description: Advertises available tools.
    intents: [metadata]
presets:
  - name: label_summary
    description: Summarize label `{label}` using AI.
    payload:
      tool: generate_ai_label_summary
      args:
        label: "{label}"
logs:
  paths:
    - metrics/mcp_calls.jsonl
owners:
  - name: Label Health App
    contact: dongyang.zhu@oracle.com, karan.baboota@oracle.com
---

## Overview
Large collection of label triage/reporting tools backed by internal REST services.

## Usage
- Launch with `python label_health_server.py`; ensure network access and tokens.
- Tools generally mirror Apex endpoints; pass required identifiers (series, label, regress, etc.).

## Troubleshooting
- Failures often due to expired tokens; refresh `LABEL_HEALTH_API_TOKEN`.
- Monitor server stdout for HTTP status codes and request details.

