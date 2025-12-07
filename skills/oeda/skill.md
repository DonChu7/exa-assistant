---
name: oeda
server_id: oeda-mcp
launch:
  command: ["python", "oeda_server.py"]
  env:
    PYTHONPATH: "${PYTHONPATH:-.}"
    OEDA_GENOEDAXML_ALLOWLIST: "${OEDA_GENOEDAXML_ALLOWLIST:-}"
tools:
  - name: generate_minconfig
    description: Produce Genoeda minconfig JSON for given request payload.
    intents: [generate, plan]
  - name: generate_oedaxml
    description: Generate OEDA XML and optional base64 content.
    intents: [generate]
  - name: tool_manifest
    description: Advertises available tools.
    intents: [metadata]
presets:
  - name: minconfig_from_template
    description: Runs `generate_minconfig` using stored request template `{template}`.
    payload:
      tool: generate_minconfig
      args:
        request: "{template}"
logs:
  paths:
    - metrics/mcp_calls.jsonl
owners:
  - name: OEDA Team
    contact: oeda-support@oracle.com
---

## Overview
Wraps OEDA generation helpers. Validates request payloads and optionally restricts Genoeda XML paths via `OEDA_GENOEDAXML_ALLOWLIST`.

## Usage
- Launch with `python oeda_server.py`.
- Provide JSON requests matching OEDA agent schemas.
- `generate_oedaxml` can return base64 XML if `return_xml` true.

## Troubleshooting
- Verify allowlist env includes required directories; otherwise requests are rejected.
- Inspect server stdout for validation errors.

