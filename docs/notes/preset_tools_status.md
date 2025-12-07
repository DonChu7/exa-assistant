# Preset Tools Auto-Generation – Current Status

## Goal
Design a helper that reads skill presets (\`skill.md\`) and auto-generates LangChain tools so the agent (and future Slack UI) can invoke curated payloads without hand-written wrappers.

## Findings
- **Signature mismatch** – `StructuredTool` wrappers need a single input argument; our generated functions either captured `**kwargs` incorrectly or closed over loop variables, causing missing placeholder errors.
- **Async complications** – Sync paths work, but async requires `StructuredTool.from_function(... coroutine=...)`. Assigning custom `ainvoke` fails (Pydantic forbids new attributes), and some wrappers returned un-awaited coroutines.
- **Placeholder validation** – Templates referencing placeholders (e.g., `{query}`) must guard against missing fields; otherwise generic contexts raise `KeyError`.
- **Environment constraints** – Even simple decorator examples show signature/async issues in this runtime, making it hard to distinguish library behavior from our wrapper logic.

## Next Steps
- Investigate LangChain’s recommended pattern for async-compatible `StructuredTool` creation.
- Prototype outside the app until a wrapper passes both `invoke` and `ainvoke` tests.
- Once stable, integrate generated tools into `app.py`, keeping manual wrappers as fallback.
