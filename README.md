# Exa Assistant

Slack-first co-pilot for Exadata / 23ai teams. The bot listens in Slack, hands user questions to a LangGraph ReAct agent, and lets the agent pick from many MCP tool servers (RunIntegration, RAG, OEDA, summarizers, etc.). Answers and feedback are logged for reporting.

---

## Architecture

```
Slack Socket Mode  ─┐
                    │  Async Bolt app (`app.py`)
                    │    • Ingests Slack events + files
                    │    • Calls LangGraph prebuilt ReAct agent
LangGraph ReAct agent (`create_react_agent`)
  • uses `make_llm()` from `llm_provider.py`
  • dynamically discovers tools via MCP manifests (no router.py)
  • invokes tools through Persistent MCP clients
                    │
Persistent MCP clients (`mcp_client.py`)
  • keep stdio servers alive
  • serialize tool calls, auto-reconnect, log metrics
                    │
Multiple FastMCP servers (python entrypoints)
  • RunIntegration, OEDA, Oracle 23ai RAG,
    summarizer, label health, real hardware, GenAI4Test, etc.
```

The old `router.py` intent matcher is no longer called; the LangGraph ReAct executor uses `list_tools` from each MCP server to choose tools autonomously.

---

## Repository Layout

| Path | Purpose |
| --- | --- |
| `app.py` | Main Slack bot entrypoint (Async Bolt + Socket Mode) and Jenkins/file helpers |
| `llm_provider.py` | OCI GenAI chat client selection (Cohere Command by default) |
| `mcp_client.py` | Background MCP client with watchdog + metrics |
| `runintegration_server.py`, `oeda_server.py`, `summarizer_server.py`, `exa23ai_rag_server.py`, `label_health_server.py`, `realhw_mcp_server.py`, `genai4test_chat_server.py`, `genai4test_server.py`, `lrg_test_mcp_server.py`, `realhw_mcp_server.py` | MCP service processes exposed to the ReAct agent |
| `runintegration_agent.py`, `oeda_agent.py`, `summarizer_agent*.py`, `exa23ai_rag_agent.py`, `genai4test_chat_agent.py`, `ta_llm_access.py` | Logic behind each tool server |
| `ingest_into_23ai_vector_search.py` | Utility to load docs into Oracle 23ai vector store backing the RAG server |
| `metrics_utils.py`, `metrics/` | Feedback + MCP-call JSONL logs and helper functions |
| `weekly_feedback_report.py` | Aggregates thumbs/comments + tool usage |
| `env_check.py`, `debug.py`, `mcp_test.py` | Local diagnostics for OCI + MCP connectivity |
| `log_arbiter_agent/`, `lrg_test_tool/` | Subprojects for log triage + LRG RAG assets |
| `test/test_ta.py` | Quick smoke test for TA models via `ta_llm_access` |

---

## Prerequisites

- Python 3.11+ (LangGraph + FastMCP target CPython 3.11 in production)
- Access to:
  - Slack workspace with bot + app tokens (Socket Mode)
  - OCI Generative AI (chat + embeddings) credentials
  - Oracle DB that hosts the 23ai vector store (for RAG)
  - Internal HTTP services (GenAI4Test, TA workflow, LabelHealth, RealHW APIs)
  - Jenkins for runintegration jobs

Install dependencies:

```bash
cd exa-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-app.txt
```

Some MCP subprojects (e.g., `log_arbiter_agent`) have their own `requirements.txt`.

---

## Configuration

Create `.env` (loaded by `app.py`) with the keys you use most often.

### Slack
- `SLACK_BOT_TOKEN`
- `SLACK_APP_TOKEN`

### Jenkins
- `JENKINS_URL`
- `JENKINS_USER`
- `JENKINS_API_TOKEN`
- `JENKINS_FOLDER`, `JENKINS_JOB`, `JENKINS_VERIFY_SSL`

### LLM / OCI
- `LLM_PROVIDER=oci`
- `OCI_COMPARTMENT_ID`, `OCI_GENAI_MODEL_ID`, `OCI_GENAI_ENDPOINT`
- `OCI_AUTH_TYPE` / `OCI_CONFIG_PROFILE`
- Optional overrides for summarizer/RAG: `RAG_OCI_*`, `SUMMARIZER_FILE_ALLOWLIST`, etc.

### Oracle DB (RAG)
- `ORA_USER`, `ORA_PASSWORD`, `ORA_DSN`
- `ORA_TABLE` (defaults to `SLACKBOT_VECTORS`)

### MCP configuration via skills-book
MCP servers, launch commands, and presets are documented under `skills/`. The bot loads this catalog at startup and uses each `skill.md` to spawn Persistent MCP clients. Update the corresponding skill entry when modifying or adding tools. Legacy `*_CMD` environment overrides are still honored as fallbacks.

### Other integrations
- `GENAI4TEST_*` (base URL, endpoints, SSL)
- `RUNINTEGRATION_*` paths (`runtable`, `connect`), typically baked into the server
- `REALHW_*`, `LABELHEALTH_*` endpoints (inside their servers)
- `FEEDBACK_PATH`, `METRICS_CALLS_PATH`, `LANGGRAPH_LOG_PATH`

---

## Running the Bot

1. Ensure each MCP server can start (either let the bot spawn them or run manually for debugging).
2. Load env vars / `.env`.
3. Launch the Slack bot:

```bash
python app.py
```

The Async SocketMode handler connects and listens for messages mentioning the bot or coming via slash commands (depending on your Slack configuration). Every user message is sent to the LangGraph ReAct agent (`create_react_agent`), which:

- Streams through OCI GenAI (`make_llm()`),
- Calls `list_tools` on every connected MCP server,
- Picks tools + arguments dynamically (no handcrafted routing),
- Returns final text (plus optional attachments) which `post_with_feedback` publishes to Slack along with feedback controls.

---

## MCP Tool Servers

| Server | Tools (examples) | Notes |
| --- | --- | --- |
| `runintegration_server.py` | `idle_envs`, `disabled_envs`, `status`, `pending_tests`, `disabled_txn_status` | Reads shared runtable, needs network access to RunIntegration hosts |
| `oeda_server.py` | `generate_minconfig`, `generate_oedaxml` | Enforces live-migration defaults, optional path allowlist |
| `summarizer_server.py` | `lc_summarize_text`, `lc_summarize_pdf_b64`, manual summarizers | Supports file allowlists and base64 PDFs |
| `exa23ai_rag_server.py` | `rag_query`, `rag_upsert_text`, `rag_delete` | Uses Oracle 23ai vector store; ingestion helper in `ingest_into_23ai_vector_search.py` |
| `label_health_server.py` | `get_labels_from_series`, `get_lrg_info`, etc. | REST clients hitting apex dashboards |
| `realhw_mcp_server.py` | Scheduler detection, node lookups, file transfer helpers | Integrates with RealHW API + local utilities (`utils/guid_to_email`) |
| `genai4test_chat_server.py` / `genai4test_server.py` | Chat/regeneration workflows for QA automation | Wraps `GenAI4TestChatAgent` |
| `lrg_test_mcp_server.py` | Retrieval + FTS search over LRG metadata | Bundles FAISS/SQLite datasets inside `lrg_test_tool/` |
| `runintegration_agent.py`, `log_arbiter_agent/` | Provide Python helpers for the above tool servers |

Each server exposes a `tool_manifest()` so the LangGraph agent can discover capabilities automatically.

---

## Metrics & Feedback

- `metrics/mcp_calls.jsonl`: Every MCP invocation (tool, args keys, duration, retry flag). Written by `PersistentMCPClient`.
- `metrics/feedback.jsonl`: Slack thumbs + context. `post_with_feedback` and `record_feedback_click` append entries.
- `metrics/langgraph_debug.log`: Optional trace log written by `LangGraphFileLogger`.
- `weekly_feedback_report.py`: Collates JSONL files into weekly summaries (counts, top tools, comments).

---

## Development Tips

- Use `debug.py` or `mcp_test.py` to run a single MCP server from your shell while iterating.
- `env_check.py` lists OCI GenAI models in a compartment and prints details for `OCI_GENAI_MODEL_ID`.
- `test/test_ta.py` performs quick TA workflow smoke tests; extend with pytest for broader coverage.
- Keep `requirements-app.txt` pinned; MCP subprojects may need extra dependencies (install inside their venv if isolating).
- When adding new tools, expose them through FastMCP, implement `tool_manifest`, and add their spawn command to env defaults so the LangGraph agent can discover them automatically.

---

## Troubleshooting

- **MCP server doesn’t respond**: check `metrics/mcp_calls.jsonl` for errors, run the server directly with `python <server>.py`, confirm env variables.
- **RAG errors**: ensure Oracle wallet / creds are accessible and `ORA_*` vars are set. Use `ingest_into_23ai_vector_search.py --dry-run` to validate data sources.
- **Slack events dropped**: verify Socket Mode app token scope, and that `SLACK_APP_TOKEN` matches the workspace deployment.
- **Tool not discovered**: confirm its MCP manifest lists at least one tool; LangGraph ReAct agent relies entirely on `list_tools`.

---

## Future Work

- Expand automated tests beyond TA smoke checks (unit tests for MCP manifests, LangGraph callbacks, etc.).
- Document per-tool environment variables in a dedicated section if onboarding new operators.
- Consider packaging MCP servers as separate entrypoints for containerized deployment.
# exa-assistant
