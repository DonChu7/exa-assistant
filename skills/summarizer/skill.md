---
name: summarizer
server_id: summarizer-mcp
launch:
  command: ["python", "summarizer_server.py"]
  env:
    PYTHONPATH: "${PYTHONPATH:-.}"
    SUMMARIZER_FILE_ALLOWLIST: "${SUMMARIZER_FILE_ALLOWLIST:-}"
    SUMMARIZER_MODEL: "${SUMMARIZER_MODEL:-oci#cohere.command}"
tools:
  - name: health
    description: Server health check.
    intents: [metadata]
  - name: summarize_text
    description: Legacy summarization pipeline.
    intents: [summarize]
  - name: summarize_pdf_file
    description: Summarize PDF on disk.
    intents: [summarize]
  - name: summarize_pdf_b64
    description: Summarize base64 PDF payload.
    intents: [summarize]
  - name: lc_health
    description: LangChain summarizer health.
    intents: [metadata]
  - name: lc_summarize_text
    description: LangChain summarization of text.
    intents: [summarize]
  - name: lc_summarize_pdf_file
    description: LangChain summarization of file path.
    intents: [summarize]
  - name: lc_summarize_pdf_b64
    description: LangChain summarization of base64 PDF.
    intents: [summarize]
  - name: tool_manifest
    description: Advertises available tools.
    intents: [metadata]
presets:
  - name: summarize_text_block
    description: Summarize provided text via LangChain tool.
    payload:
      tool: lc_summarize_text
      args:
        text: "{text}"
logs:
  paths:
    - metrics/mcp_calls.jsonl
owners:
  - name: Summarizer Team
    contact: summarizer@oracle.com
---

## Overview
Wraps both bespoke and LangChain-based summarization flows. File tools require allowlisted paths.

## Usage
- Launch with `python summarizer_server.py`.
- Provide `text` or `pdf` payloads; b64 variants expect base64 strings.

## Troubleshooting
- Ensure `SUMMARIZER_FILE_ALLOWLIST` covers directories for file-based tools.
- OCI auth issues manifest via `SUMMARIZER_MODEL`; check logs for HTTP errors.

