#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- SQLite shim ----
try:
    import sqlite3
except Exception:
    import sys
    import pysqlite3 as sqlite3
    sys.modules["sqlite3"] = sqlite3

"""
# ---------------------------------------------------------------------------
# Command-line Parameters Reference
#
# Core Query Options
#   --db PATH                : Path to the SQLite FTS database (profiles_fts.db)
#   --q TEXT                 : Natural language or keyword query text
#
# Filtering / Retrieval Behavior
#   --require-setup NAME     : Only include TESTs whose setup exactly matches NAME
#   --require-flag K=V       : Require a flag key/value pair in TEST metadata
#   --prefer-setup NAME      : Soft-boost results with matching setup
#   --prefer-flag K=V        : Soft-boost results with matching flag
#   --structured-only        : Skip FTS entirely; filter every TEST by metadata only
#   --no-expand              : Disable TEST↔LRG expansion via mappings (t2l/l2t)
#   --no-flag-values         : In FTS match, include only flag keys (omit values)
#   --require-suite NAME     : hard filter on LRG suites
#   --prefer-suite  NAME     : soft boost for LRG docs in these suites
#
# Result Limits & Ordering
#   --k N                    : Max number of TEST hits to print (default 10)
#   --k-lrgs N               : Max number of LRGs to display after aggregation
#   --pool N                 : FTS candidate pool size before filtering
#   --lrg-order runtime|count|name
#                            : How to sort LRGs — runtime ascending (default: count)
#
# Data Inputs
#   --t2l PATH               : JSON map { test_name : [lrg_names...] }
#   --l2t PATH               : JSON map { lrg_name  : [test_names...] }
#   --lrgs-json PATH         : LRG metadata JSON with runtime/platform info
#   --flags-glossary PATH    : JSON/YAML map of flag_name → human description
#
# Description & Output
#   --show-desc              : Print the formatted description text for each TEST
#   --desc-chars N           : Max characters from description to print (default 800)
#   --only-direct            : Only keep results directly mentioning given entities
#   --debug                  : Print SQL queries, MATCH expressions, etc.
#
# Example:
    python3 query_fts_refined.py \
        --db /ade/shratyag_v7/tklocal/profiles_fts.db \
        --q "xblockini iscsi" \
        --t2l /ade/shratyag_v7/tklocal/test_to_lrgs.json \
        --l2t /ade/shratyag_v7/tklocal/lrg_to_tests.json \
        --lrgs-json /ade/shratyag_v7/tklocal/lrg_map_with_suites.json \
        --require-setup xblockini \
        --require-flag  iscsi=true \
        --pool 10000000 \
        --lrg-order runtime \
        --k 20000
# ---------------------------------------------------------------------------

Refined FTS retriever (constraints-first) + LRG runtimes + suites.

"""

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---------- helpers ----------
def load_json_or_empty(p: str | None):
    if not p: return {}
    return json.loads(Path(p).read_text(encoding="utf-8"))

def extract_tokens(q: str) -> Dict[str, List[str]]:
    """Pull structured tokens from NL: tests (tsag*.tsc) and lrg ids."""
    ql = q.lower()
    tests = {
        (m if m.endswith(".tsc") else m + ".tsc")
        for m in re.findall(r"(tsag\S+?)(?:\.tsc)?\b", ql)
    }
    lrgs = set(re.findall(r"\b(lrg[a-z0-9_]+)\b", ql))
    lrgs.discard("lrg"); lrgs.discard("lrgs")
    return {"tests": sorted(tests), "lrgs": sorted(lrgs)}

def parse_kv(arg: str) -> Tuple[str, str]:
    if "=" not in arg:
        return arg.strip(), ""
    k, v = arg.split("=", 1)
    return k.strip(), v.strip()

def fts_sanitize(term: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", term.lower())

def build_structured_match(expanded_tests: list[str], expanded_lrgs: list[str]) -> str | None:
    """
    Build a tolerant MATCH for entity tokens:
    - For tests, add both base and base.tsc (in TEXT and ID).
    - For lrgs, add the id token (in TEXT and ID).
    """
    terms: list[str] = []
    for t in expanded_tests:
        t = t.lower()
        base = re.sub(r"\.tsc$", "", t)
        terms += [
            f'text:"{base}"',
            f'text:"{base}.tsc"',
        ] * 2
        terms += [
            f'id:"{base}"',
            f'id:"{base}.tsc"',
        ]
    for l in expanded_lrgs:
        l = l.lower()
        terms += [f'text:"{l}"', f'id:"{l}"'] * 3
    return " OR ".join(terms) if terms else None

def build_constraints_match(setup: str | None,
                            flags: List[Tuple[str, str]],
                            include_values: bool) -> str | None:
    """
    MATCH from constraints:
      - Always include setup in TEXT
      - Always include FLAG KEYS in TEXT
      - Include FLAG VALUES in TEXT only if include_values=True (optional)
    """
    need: List[str] = []
    if setup:
        need.append(f'text:"{fts_sanitize(setup)}"')
    for k, v in flags:
        if k:
            need.append(f'text:"{fts_sanitize(k)}"')
        if include_values and v:
            need.append(f'text:"{fts_sanitize(v)}"')
    return " AND ".join(need) if need else None

def maybe_extract_constraints_from_text(q: str) -> Tuple[str | None, List[Tuple[str, str]], List[str]]:
    """Infer simple constraints from NL query (setup/flags/suite)."""
    ql = q.lower()
    setup_name = None
    m = re.search(r"\bsetup\s+([a-z0-9_.-]+)\b", ql)
    if m:
        setup_name = m.group(1)
    flags = [(k, v) for k, v in re.findall(r"\b([a-z0-9_.-]+)\s*=\s*([a-z0-9_.-]+)\b", ql)]
    suites = [s for s in re.findall(r"\bsuite[:\s]+([a-z0-9_.-]+)\b", ql)]
    return setup_name, flags, suites

def parse_meta_json(row_val: str | None) -> Dict[str, Any]:
    try:
        return json.loads(row_val or "{}")
    except Exception:
        return {}

# --- description extraction ---
DESC_SPLIT_RE = re.compile(r"\bDescription:\s*", re.IGNORECASE)

def extract_description_from_profile_text(text: str, max_chars: int) -> str:
    """
    TEST profile text is "Test <name>. setup=.. flags=..\nDescription: <body>"
    Return body (trimmed) up to max_chars, preserving newlines.
    """
    if not text:
        return ""
    parts = DESC_SPLIT_RE.split(text, maxsplit=1)
    desc = parts[1] if len(parts) > 1 else text  # fallback: whole text
    desc = desc.strip()
    if max_chars and len(desc) > max_chars:
        desc = desc[:max_chars].rstrip() + "…"
    return desc

def norm_suites(val) -> List[str]:
    """Normalize suite field to a list of strings (no case mangling here)."""
    if not val:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, str):
        return [val.strip()] if val.strip() else []
    return []

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=10, help="Max TEST hits to display (before LRG aggregation)")
    ap.add_argument("--t2l", default=None)
    ap.add_argument("--l2t", default=None)
    ap.add_argument("--lrgs-json", default=None, help="LRG array JSON with 'lrgname','runtime','lrg_suite'")
    ap.add_argument("--doc-type", choices=["TEST","LRG"], default=None)
    ap.add_argument("--only-direct", action="store_true")
    ap.add_argument("--require-setup", default=None)
    ap.add_argument("--require-flag", action="append", default=[])
    ap.add_argument("--prefer-setup", default=None)
    ap.add_argument("--prefer-flag", action="append", default=[])
    # NEW: suite controls
    ap.add_argument("--require-suite", action="append", default=[], help="Only LRGs in these suites (repeatable)")
    ap.add_argument("--prefer-suite", action="append", default=[], help="Soft boost for LRG docs in these suites (repeatable)")
    # pool/ordering
    ap.add_argument("--pool", type=int, default=0, help="FTS candidate pool before post-filtering (default max(k*20, 500))")
    ap.add_argument("--structured-only", action="store_true", help="Bypass FTS and filter by exact metadata only")
    ap.add_argument("--no-expand", action="store_true", help="Disable expanding LRG<->TEST via mappings")
    ap.add_argument("--k-lrgs", type=int, default=50, help="Max LRGs to display after aggregation")
    ap.add_argument("--lrg-order", choices=["runtime","count","name"], default="count", help="Sort LRGs by runtime asc, test count desc, or name")
    ap.add_argument("--no-flag-values", action="store_true", help="Do not include flag VALUES in FTS MATCH (keys only)")
    ap.add_argument("--show-desc", action="store_true", help="Print description for TEST hits")
    ap.add_argument("--desc-chars", type=int, default=800, help="Max description characters to print")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Load mappings
    t2l = load_json_or_empty(args.t2l)
    l2t = load_json_or_empty(args.l2t)
    lrg_catalog = set(l2t.keys()) if l2t else set()

    # Build LRG info (runtime, platform, flags, suites)
    lrg_info: Dict[str, Dict[str, Any]] = {}
    if args.lrgs_json:
        lrgs_arr = json.loads(Path(args.lrgs_json).read_text(encoding="utf-8"))
        if isinstance(lrgs_arr, list):
            for rec in lrgs_arr:
                name = (rec.get("lrgname") or "").strip()
                if not name: continue
                lrg_info[name] = {
                    "runtime": rec.get("runtime"),
                    "platform": rec.get("Platform"),
                    "flags": rec.get("flags"),
                    "tests_count": len(rec.get("tests") or []),
                    "suites": norm_suites(rec.get("lrg_suite")),
                }

    # Structured tokens from NL
    tokens = extract_tokens(args.q)
    if lrg_catalog:
        tokens["lrgs"] = [l for l in tokens["lrgs"] if l in lrg_catalog]

    expanded_tests = set(tokens["tests"])
    expanded_lrgs  = set(tokens["lrgs"])

    # Optional expansion via mappings
    if not args.no_expand:
        for t in list(expanded_tests):
            expanded_lrgs.update(t2l.get(t, []))
        for l in list(expanded_lrgs):
            expanded_tests.update(l2t.get(l, []))

    # Constraints
    req_setup = (args.require_setup or "").strip() or None
    req_flags = [parse_kv(s) for s in (args.require_flag or [])]

    # Merge inferred constraints from NL (explicit wins)
    inf_setup, inf_flags, inf_suites = maybe_extract_constraints_from_text(args.q)
    if not req_setup and inf_setup:
        req_setup = inf_setup
    seen = {k for k,_ in req_flags}
    for k, v in inf_flags:
        if k not in seen:
            req_flags.append((k, v))
            seen.add(k)

    # Suites required/preferred (as written)
    req_suites = [s.strip() for s in (args.require_suite or []) if s and s.strip()]
    pref_suites = [s.strip() for s in (args.prefer_suite or []) if s and s.strip()]
    # If user wrote "suite perf" in query and didn't pass --require-suite, treat it as require
    if not req_suites and inf_suites:
        req_suites = list({*inf_suites})

    # If structured-only: full scan of TEST docs and exact filter
    if args.structured_only:
        if args.debug:
            print("MATCH: (STRUCTURED-ONLY: no FTS)")
        conn = sqlite3.connect(args.db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("""
          SELECT id, doc_type, text, meta_json, bm25(docs_fts) AS score
          FROM docs_fts
          WHERE doc_type='TEST'
        """).fetchall()
        conn.close()

        def pass_require_meta(meta: Dict[str, Any]) -> bool:
            if req_setup and (meta.get("setup") or "").lower() != req_setup.lower():
                return False
            flags = meta.get("flags") or {}
            if not isinstance(flags, dict): return False
            def is_truthy(val: Any) -> bool:
                if val is None:
                    return False
                s = str(val).strip().lower()
                return s in {"1", "true", "yes", "on"}

            for k, v in req_flags:
                if k not in flags:
                    return False
                val = flags.get(k)
                if v.lower() == "true":
                    if not is_truthy(val):
                        return False
                else:
                    if str(val).strip('"').lower() != v.lower():
                        return False
            return True

        hits: List[Dict[str, Any]] = []
        for r in rows:
            meta = parse_meta_json(r["meta_json"])
            if not pass_require_meta(meta):
                continue
            rid = r["id"]
            test_id = meta.get("id") or (rid.split("::")[1] if "::" in rid else None)
            hits.append({
                "id": rid, "doc_type": "TEST", "meta": meta,
                "bm25": float(r["score"]), "rank_score": float(r["score"]),
                "test_id": test_id, "lrg_id": None,
                "text": r["text"] or "",
            })

        # Truncate tests to --k for display
        hits = hits[:max(args.k, 10)]

        print(f"\nQuery: {args.q}")
        print("Direct tokens:",
              f'tests={sorted(expanded_tests) if expanded_tests else "()"}',
              f'lrgs={sorted(expanded_lrgs) if expanded_lrgs else "()"}')

        if not hits:
            print("\n(no hits)")
            return

        print("\nMatched TESTs (exact constraints applied):")
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            print(f"{i:2d}. {h['test_id']}  setup={meta.get('setup')}  flags={meta.get('flags')}")
            if args.show_desc:
                desc = extract_description_from_profile_text(h["text"], args.desc_chars)
                if desc:
                    print("    --- description ---")
                    for line in desc.splitlines():
                        print("    " + line)

        # Aggregate to LRGs
        if t2l:
            lrg_counts: Dict[str, List[str]] = {}
            for h in hits:
                tname = h["test_id"]
                for l in t2l.get(tname, []):
                    lrg_counts.setdefault(l, []).append(tname)

            if lrg_counts:
                # Sorting function
                def lrg_sort_key(item):
                    lrg, tlist = item
                    info = lrg_info.get(lrg) or {}
                    rt = info.get("runtime")
                    if args.lrg_order == "runtime":
                        return (1 if rt is None else 0, (rt if isinstance(rt,(int,float)) else 1e12), -len(tlist), lrg)
                    elif args.lrg_order == "name":
                        return (lrg,)
                    else:  # count
                        return (-len(tlist), lrg)

                print("\nCorresponding LRGs (by matched tests):")
                shown = 0
                for lrg, tlist in sorted(lrg_counts.items(), key=lrg_sort_key):
                    info = lrg_info.get(lrg) or {}
                    suites = info.get("suites") or []
                    # hard filter by suite (if requested)
                    if req_suites and not (set(s.lower() for s in suites) & set(s.lower() for s in req_suites)):
                        continue
                    rt = info.get("runtime")
                    rt_str = f"{rt} hours" if isinstance(rt, (int, float)) else "unknown"
                    print(f"- {lrg}: {len(tlist)} tests  runtime={rt_str}  suites={suites or '[]'}  e.g., {', '.join(tlist[:8])}")
                    shown += 1
                    if shown >= args.k_lrgs:
                        break
        return

    # ---------- FTS path ----------
    constraints_match = build_constraints_match(req_setup, req_flags, include_values=(not args.no_flag_values))
    structured_match  = build_structured_match(sorted(expanded_tests), sorted(expanded_lrgs))
    match = constraints_match or structured_match or None
    if args.debug:
        print("MATCH:", match if match is not None else "(FULL SCAN)")

    # doc_type decision: if constraints exist and not specified, force TESTs
    doc_type = args.doc_type
    if (req_setup or req_flags) and not doc_type:
        doc_type = "TEST"

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    base_sql = """
    SELECT id, doc_type, text, meta_json, bm25(docs_fts) AS score
    FROM docs_fts
    """
    where_clauses: List[str] = []
    params: List[Any] = []

    if match:
        where_clauses.append("docs_fts MATCH ?")
        params.append(match)
    if doc_type:
        where_clauses.append("doc_type = ?")
        params.append(doc_type)

    sql = base_sql
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    pool = args.pool if args.pool > 0 else max(args.k * 20, 500)
    sql += " ORDER BY score LIMIT ?"
    params.append(pool)

    if args.debug:
        print("SQL:", sql)
        print("PARAMS:", params)

    rows = conn.execute(sql, params).fetchall()
    conn.close()

    hits: List[Dict[str, Any]] = []
    for r in rows:
        meta = parse_meta_json(r["meta_json"])
        rid  = r["id"]
        dtype = r["doc_type"]
        test_id = lrg_id = None
        if dtype == "TEST":
            test_id = meta.get("id") or (rid.split("::")[1] if "::" in rid else None)
        else:
            lrg_id  = meta.get("id") or (rid.split("::")[1] if "::" in rid else None)

        hits.append({
            "id": rid,
            "doc_type": dtype,
            "meta": meta,
            "bm25": float(r["score"]),
            "test_id": test_id,
            "lrg_id": lrg_id,
            "text": r["text"] or "",
        })

    # Optional: restrict to direct entity mentions
    if args.only_direct and (expanded_tests or expanded_lrgs):
        def keep(h):
            if h["doc_type"] == "TEST":
                return (h["test_id"] in expanded_tests) or any(l in expanded_lrgs for l in t2l.get(h["test_id"] or "", []))
            else:
                return (h["lrg_id"] in expanded_lrgs) or any(t in expanded_tests for t in l2t.get(h["lrg_id"] or "", []))
        hits = [h for h in hits if keep(h)]

    # Hard filters (exact) if constraints present
    def pass_require(h) -> bool:
        meta = h["meta"]
        if req_setup and (meta.get("setup") or "").lower() != req_setup.lower():
            return False
        flags = meta.get("flags") or {}
        if not isinstance(flags, dict): return False
        for k, v in req_flags:
            if k not in flags: return False
            if v and str(flags.get(k)).strip('"').lower() != v.lower():
                return False
        return True

    if req_setup or req_flags:
        hits = [h for h in hits if h["doc_type"] == "TEST"]
        hits = [h for h in hits if pass_require(h)]

    # Soft preferences
    pref_setup = (args.prefer_setup or "").strip() or None
    pref_flags = [parse_kv(s) for s in (args.prefer_flag or [])]
    pref_suites_lower = {s.lower() for s in (args.prefer_suite or [])}

    def bonus(h) -> float:
        b = 0.0
        meta = h["meta"]
        if pref_setup and (meta.get("setup") or "").lower() == pref_setup.lower():
            b += 2.0
        if pref_flags:
            flags = meta.get("flags") or {}
            if isinstance(flags, dict):
                for k, v in pref_flags:
                    if k in flags and (not v or str(flags.get(k)).strip('"').lower() == v.lower()):
                        b += 1.0
        # add small bonus for LRG docs in preferred suites
        if h["doc_type"] == "LRG" and pref_suites_lower:
            suites = (lrg_info.get(h["lrg_id"] or "") or {}).get("suites") or []
            if any((s or "").lower() in pref_suites_lower for s in suites):
                b += 1.0
        return b

    for h in hits:
        h["rank_score"] = h["bm25"] - bonus(h)  # lower is better

    hits.sort(key=lambda x: x["rank_score"])
    hits = hits[:max(args.k, 10)]

    # ---------- Output ----------
    print(f"\nQuery: {args.q}")
    print("Direct tokens:",
          f'tests={sorted(expanded_tests) if expanded_tests else "()"}',
          f'lrgs={sorted(expanded_lrgs) if expanded_lrgs else "()"}')

    if not hits:
        print("\n(no hits)")
        return

    if req_setup or req_flags:
        print("\nMatched TESTs (exact constraints applied):")
        for i, h in enumerate(hits, 1):
            meta = h["meta"]
            print(f"{i:2d}. {h['test_id']}  setup={meta.get('setup')}  flags={meta.get('flags')}")
            if args.show_desc:
                desc = extract_description_from_profile_text(h["text"], args.desc_chars)
                if desc:
                    print("    --- description ---")
                    for line in desc.splitlines():
                        print("    " + line)

        if t2l:
            lrg_counts: Dict[str, List[str]] = {}
            for h in hits:
                tname = h["test_id"]
                for l in t2l.get(tname, []):
                    lrg_counts.setdefault(l, []).append(tname)

            if lrg_counts:
                # Sorting function
                def lrg_sort_key(item):
                    lrg, tlist = item
                    info = lrg_info.get(lrg) or {}
                    rt = info.get("runtime")
                    if args.lrg_order == "runtime":
                        return (1 if rt is None else 0, (rt if isinstance(rt,(int,float)) else 1e12), -len(tlist), lrg)
                    elif args.lrg_order == "name":
                        return (lrg,)
                    else:  # count
                        return (-len(tlist), lrg)

                print("\nCorresponding LRGs (by matched tests):")
                shown = 0
                for lrg, tlist in sorted(lrg_counts.items(), key=lrg_sort_key):
                    info = lrg_info.get(lrg) or {}
                    suites = info.get("suites") or []
                    # hard filter by suite (if requested)
                    if req_suites and not (set(s.lower() for s in suites) & set(s.lower() for s in req_suites)):
                        continue
                    rt = info.get("runtime")
                    rt_str = f"{rt} hours" if isinstance(rt, (int, float)) else "unknown"
                    print(f"- {lrg}: {len(tlist)} tests  runtime={rt_str}  suites={suites or '[]'}  e.g., {', '.join(tlist[:8])}")
                    shown += 1
                    if shown >= args.k_lrgs:
                        break
        return

    # Mixed hits mode (no hard constraints)
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        if h["doc_type"] == "TEST":
            print(f"    setup={meta.get('setup')}  flags={meta.get('flags')}")
            if args.show_desc:
                desc = extract_description_from_profile_text(h["text"], args.desc_chars)
                if desc:
                    print("    --- description ---")
                    for line in desc.splitlines():
                        print("    " + line)
            if t2l and h["test_id"]:
                lrgs = t2l.get(h["test_id"], [])
                if lrgs:
                    print(f"    mapped LRGs: {', '.join(lrgs)}")
        else:
            info = lrg_info.get(h["lrg_id"] or "") or {}
            rt = info.get("runtime")
            rts = f"{rt} hours" if isinstance(rt, (int, float)) else "unknown"
            suites = info.get("suites") or []
            print(f"{i:2d}. LRG  {h['lrg_id']}")
            print(f"    platform={meta.get('platform')} tests_count={meta.get('tests_count')} flags={meta.get('flags')}")
            print(f"    runtime={rts}  suites={suites or '[]'}")
            if l2t and h["lrg_id"]:
                tests = l2t.get(h["lrg_id"], [])
                if tests:
                    print(f"    Tests: {', '.join(tests[:12])}")

if __name__ == "__main__":
    main()