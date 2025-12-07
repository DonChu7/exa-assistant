---
name: exa23ai-rag
server_id: oracle23ai-rag-mcp
launch:
  command: ["python", "exa23ai_rag_server.py"]
  env:
    PYTHONPATH: "${PYTHONPATH:-.}"
    ORA_USER: "${ORA_USER:-}"
    ORA_PASSWORD: "${ORA_PASSWORD:-}"
    ORA_DSN: "${ORA_DSN:-}"
    ORA_TABLE: "${ORA_TABLE:-SLACKBOT_VECTORS}"
tools:
  - name: health
    description: Server health check.
    intents: [metadata]
  - name: rag_query
    description: Query Oracle 23ai vector store for answers.
    intents: [query, summarize]
  - name: rag_upsert_text
    description: Upsert documents into RAG index.
    intents: [ingest]
  - name: rag_delete
    description: Delete documents from index.
    intents: [ingest]
  - name: tool_manifest
    description: Advertises available tools.
    intents: [metadata]
presets:
  - name: ask_exadata
    description: Ask Exadata knowledge base question `{question}`.
    payload:
      tool: rag_query
      args:
        question: "{question}"
        k: 3
logs:
  paths:
    - metrics/mcp_calls.jsonl
owners:
  - name: 23ai RAG Team
    contact: rag-support@oracle.com
---

## Overview
Oracle 23ai RAG server leveraging OCI vector store integration.

## Usage
- Launch with `python exa23ai_rag_server.py` after setting DB credentials.
- `rag_upsert_text`/`rag_delete` manage documents; `rag_query` retrieves answers with citations.

## Troubleshooting
- Verify DB creds and network connectivity to OCI database; errors surface as SQL exceptions.
- Check ingestion script `ingest_into_23ai_vector_search.py` for bulk operations.

