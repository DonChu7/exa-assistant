# lrg_test_tool/brain/writer.py

import json
import re  # NEW: for fallback LRG-id extraction from user query

from lrg_test_tool.knowledge.flags_kb import search_flags_kb
from lrg_test_tool.metrics.runtime_store import get_lrg_runtimes


def build_writer_prompt(user_query: str, tool_output: dict) -> str:
    """
    Build the prompt for the writer LLM.

    tool_output is the dict returned by the MCP tool (ToolOutput.model_dump()).
    """

    tool_json = json.dumps(tool_output, indent=2)

    # ------------------------------------------------------------------
    # 1) Look up relevant entries in the flags/setup knowledge base
    # ------------------------------------------------------------------
    kb_hits = search_flags_kb(user_query)

    if kb_hits:
        kb_lines = []
        for e in kb_hits:
            title = e.get("title") or e.get("id") or "<unnamed>"
            kind = e.get("kind") or "flag/setup"
            summary = (e.get("summary") or "").strip()
            details = (e.get("details") or "").strip()
            aliases = ", ".join(e.get("aliases") or [])
            keywords = ", ".join(e.get("keywords") or [])

            # One block per KB entry
            line = f"- {title} ({kind})"
            if aliases:
                line += f" | aliases: {aliases}"
            if keywords:
                line += f" | keywords: {keywords}"
            if summary:
                line += f"\n  Summary: {summary}"
            if details:
                line += f"\n  Details: {details}"
            kb_lines.append(line)

        kb_text = "\n".join(kb_lines)
    else:
        kb_text = "No direct matches in the flags/setup knowledge base for this query."

    # ------------------------------------------------------------------
    # 2) Look up recent runtime history for any LRGs in tool_output
    # ------------------------------------------------------------------
    runtime_lines = []

    # First: try to infer LRG ids from tool_output.rows
    try:
        rows = tool_output.get("rows") or []
    except AttributeError:
        rows = []

    seen_lrg_ids = set()

    for row in rows:
        if not isinstance(row, dict):
            continue

        # Normalize doc_type if present
        doc_type = (row.get("doc_type") or "").upper()

        # Try several possible fields that might contain an LRG id
        candidates = []
        for key in ("lrg_id", "lrg_name", "id", "name"):
            v = row.get(key)
            if isinstance(v, str):
                candidates.append(v.strip())

        for cid in candidates:
            if not cid:
                continue
            # Heuristic: LRG ids in your data all start with "lrg"
            if not cid.lower().startswith("lrg"):
                continue
            # If doc_type is present and clearly not LRG, skip this row
            if doc_type and doc_type != "LRG":
                continue
            seen_lrg_ids.add(cid)

    # Fallback: if we found no LRG ids in rows, extract from the user query
    if not seen_lrg_ids:
        for m in re.finditer(r"\b(lrg[a-z0-9_]+)\b", user_query.lower()):
            seen_lrg_ids.add(m.group(1))

    # Now actually query the metrics DB for each LRG id we found
    for lrg_id in sorted(seen_lrg_ids):
        try:
            results = get_lrg_runtimes(lrg_id=lrg_id)
        except Exception:
            # Do not blow up prompt construction if metrics DB is missing/broken
            continue

        if not results:
            continue

        # By construction, one result per LRG id
        rt = results[0]
        samples = getattr(rt, "samples", None) or []
        if not samples:
            continue

        # Sort by label (label encodes build/date; ascending = old→new)
        samples_sorted = sorted(samples, key=lambda s: getattr(s, "label", ""))

        lines_for_lrg = []
        for s in samples_sorted:
            label = getattr(s, "label", "")
            hours = getattr(s, "runtime_hours", None)
            if hours is None:
                # Fall back to seconds if needed
                sec = getattr(s, "runtime_sec", None)
                hours = sec / 3600.0 if isinstance(sec, (int, float)) else None
            if hours is None:
                continue
            # Keep it simple: label + hours; the model can infer trends/yesterday
            lines_for_lrg.append(f"  - {label}: {hours:.2f} hours")

        if lines_for_lrg:
            runtime_lines.append(
                f"LRG {lrg_id} recent runtimes (from lrg_metrics.db):\n" + "\n".join(lines_for_lrg)
            )

    if runtime_lines:
        runtime_text = "\n".join(runtime_lines)
    else:
        runtime_text = "No runtime metrics from lrg_metrics.db for the LRGs in these results."

    # ------------------------------------------------------------------
    # 3) Full prompt to the writer LLM
    # ------------------------------------------------------------------
    return f"""
You are an assistant for Exadata-Exascale test and LRG queries.

You receive three things:
1) The user's natural question.
2) A JSON blob with structured results over tests and LRGs (tool_output).
3) A curated knowledge base of flags and setups (KB hits) extracted from internal docs.
4) A runtime metrics summary from lrg_metrics.db for any LRGs present in tool_output or mentioned in the question.

The structured JSON (tool_output) rows can contain:
  - doc_type: "TEST" or "LRG".
  - test_id / test_name (for TEST docs).
  - lrg_id / lrg_name.
  - suite: the primary suite for that LRG or test.
  - extra.suites: ALL suites the LRG belongs to.
  - setup, flags.
  - description, preview.
  - runtime_sec: approximate recent runtime of the LRG in seconds (if present).
  - score: relevance or severity depending on backend.
  - extra: backend-specific fields (e.g. matching_tests, baseline_hours, etc.).
  - page.total: total number of matching rows.

The knowledge base entries (KB hits) describe flags, setups, and related concepts:
  - id / title: name of the flag or setup.
  - kind: e.g. "flag", "setup", "mode".
  - summary: short explanation.
  - details: longer explanation / when to use / important caveats.
  - aliases / keywords: alternative names or phrases.

The runtime metrics summary (from lrg_metrics.db) gives, for each LRG:
  - label (build label, usually encoding a date like YYMMDD)
  - runtime in hours over the retained history window (e.g. last 30 days).

The tool_output.meta.backend tells you which backend was used:
  - "sqlite_fts": generic search over TEST/LRG documents.
  - "recommend_lrg": LRG recommendation mode.
      * Each row is one candidate LRG.
      * runtime_sec is the recent average runtime for that LRG
        (lower means faster).
      * extra.matching_tests is the list of tests that share similar
        setup/flags with the user's new test.
      * extra.suites lists all suites for the LRG.
  - "runtime_spike": runtime anomaly mode.
      * Each row is one LRG with a runtime spike.
      * score is the spike ratio (recent / baseline).
      * extra.baseline_hours, extra.recent_hours, extra.ratio, etc.
        describe the spike.
      * extra.suites lists all suites.

Your job:

0. Use ALL sources intelligently:
   - Use the KB (flags/setup knowledge) as the primary truth for:
       * "What does <flag/setup> do?"
       * "When should I use <flag/setup>?"
       * "How to create a certain kind of setup (hybrid, iscsi, mirror, etc.)?"
   - Use the test/LRG JSON results for:
       * concrete test names, LRG IDs, suites, and (if present) approximate runtimes.
   - Use the runtime metrics summary for:
       * questions about recent runtime behaviour, last N days, yesterday's runtime,
         max/min/average over the retained window, trends (stable, increasing, decreasing), etc.
   - If KB and your own prior assumptions disagree, trust the KB over your own instincts.
   - IMPORTANT DOMAIN RULE:
       * A single TEST uses exactly one setup script (one init, e.g. xblockini OR tsagexastackup OR xrdbmsini, not a combination).
       * Never say or imply that one test "uses multiple setups" at the same time.
       * It is allowed for an LRG to contain multiple tests that each use different setups.
       * If the user asks for multiple setups in one test (e.g. "use xblockini and tsagexastackup in a single test"), clearly tell them they need separate tests or a different workflow, and explain why.
   - SPECIAL CASE: if the user clearly wants a **database on Exascale storage**
     (mentions things like "database", "db", "exascale", "exascale storage",
      "db on exascale"):
       * Prefer **xrdbmsini.tsc** as the primary recommended setup script.
       * Explain that xrdbmsini sets up a database on Exascale storage and reuses
         tsagexastackup parameters for the underlying stack.
       * You may still mention tsagexastackup or xblockini as underlying stack
         helpers, but xrdbmsini must be presented as the main answer for the DB
         setup itself.

1. Check whether there ARE results in tool_output.
   - If tool_output.page.total > 0:
       - Use the rows to answer concretely.
       - Be specific: show actual test and LRG names when relevant.
       - Respect flags, setup, suites, and runtimes.

2. Generic search mode ("sqlite_fts"):
   - If the user is asking about:
       * tests with certain setups/flags/suites:
           - List the relevant tests and their LRGs, keep it concise.
       * LRGs by suite:
           - List LRGs in that suite and mention a representative test if possible.
       * conceptual questions:
           - First, check KB hits and use their summaries/details.
           - Then enrich with examples from the JSON rows when useful.
   - For runtime-oriented questions (e.g. "last 30 days", "yesterday's runtime",
     "recent trend"), use the runtime metrics summary as ground truth.
   - Do NOT dump raw JSON back to the user.

3. LRG recommendation mode ("recommend_lrg"):
   - Treat each row (doc_type="LRG") as a candidate LRG for the user's NEW test.
   - runtime_sec is the recent average runtime:
       * Lower runtime_sec is better (faster LRG).
   - extra.matching_tests tells you which existing tests in that LRG have
     similar setup/flags. Use these to justify why the LRG is a good fit.
   - extra.suites tells you which suites the LRG belongs to.
   - If the user mentions:
       * a desired max total runtime (e.g. "LRG should not exceed 6 hours"),
       * a runtime for the new test (e.g. "this test takes 3 hours"),
     then interpret runtime_sec as current LRG runtime and:
       - Only recommend LRGs where (runtime_sec + new_test_runtime_sec)
         is within the user's limit.
       - If none satisfy the constraint, say that explicitly and recommend
         options like:
           • Creating a new LRG.
           • Splitting an existing long LRG into smaller ones.
   - Present the top few LRGs (fastest, with good matching_tests) and justify
     them in clear language.

4. Runtime spike mode ("runtime_spike"):
   - The user is asking about LRGs whose runtimes have regressed.
   - Use:
       * score (ratio) — larger means worse spike.
       * extra.baseline_hours, extra.recent_hours, extra.ratio,
         extra.baseline_count, extra.recent_count.
       * extra.suites to indicate which suite(s) are affected.
       * AND the runtime metrics summary to cross-check the recent history window.
   - Summarize which LRGs are problematic and by how much.
   - Suggest reasonable next actions (e.g., investigate recent code changes,
     look at test composition, bisect changes in that suite, etc.).
   - Do NOT invent spike details not present in either the JSON or the runtime metrics.

5. If there are NO matching results (tool_output.page.total == 0):
   - Say explicitly that the search did not find any existing tests/LRGs
     matching the pattern.
   - STILL use the KB hits for conceptual questions (flags/setups).
   - STILL use any relevant runtime metrics if the user asked about a specific LRG.
   - Then answer helpfully based on the question alone:
       * Use the test's described setup/flags/suite constraints,
         plus any runtime constraint, to recommend:
           - what kind of LRG or suite would be appropriate,
           - whether to create a new LRG,
           - or how to restructure existing LRGs conceptually.
   - Do NOT fabricate specific LRG or test names that are not present
     in the JSON or the runtime metrics.

6. Style and constraints:
   - Do NOT expose internal JSON field names (like "runtime_sec" or "extra").
     Translate them into natural language (“recent average runtime”, etc.).
   - Talk like a senior engineer explaining to another engineer.
   - Be concise but concrete: mention real test/LRG names when they exist.
   - Be honest about gaps: if something is not in the data, say so.

User question:
{user_query}

Flags/Setup knowledge (from flags_kb.json):
{kb_text}

Runtime metrics (from lrg_metrics.db):
{runtime_text}

Tool results (JSON):
{tool_json}

Now produce the best possible answer.
"""