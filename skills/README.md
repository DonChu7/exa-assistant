# Skills Catalog

This directory documents every MCP server and tool exposed by Exa Assistant. Each subdirectory contains a `skill.md` (and optional `refs/`, `assets/`) describing launch commands, environment, tools, and presets.

## Conventions
- `skill.md` begins with a YAML front matter header (`--- ... ---`) describing machine-readable metadata.
- Use `name` for the catalog entry, `server_id` for the MCP identifier (e.g., `rag-fts-mcp`).
- `launch.command` is stored as an array for direct `subprocess` invocation.
- `tools` should mirror `tool_manifest()` output (name + description + intents).
- `presets` capture opinionated invocations (plans, default params, etc.).
- Add troubleshooting, log paths, and owner contacts in the Markdown body.

## Loader Expectations
- A loader will parse the front matter to auto-start servers and expose presets to agents.
- Keep configuration (paths, env vars) synchronized between code and docs; `app.py` will rely on this catalog.

## Adding a New Skill
1. Copy `template_skill.md` into `skills/<name>/skill.md`.
2. Fill in metadata, especially launch command and tool list.
3. Document required secrets, allowlists, and presets.
4. Update automation/tests if necessary.

