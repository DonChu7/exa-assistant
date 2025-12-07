#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
parse_lrg_map.py  (tolerant headers + nested if/endif inside sections)

Supports section headers (case/space tolerant, optional 'then' and trailing comments):
  1) if (( section == NAME ) OR ( section == ALL ))     # 'OR' or '||'
  2) if (section == NAME)
  3) if (( section == NAME ))                           # double-paren simple

Section body may contain nested 'if ... endif' blocks (e.g., feature switches).
We keep a nesting counter and only close the LRG section when an 'endif'
arrives at depth==0.

Inside a section, tests are collected from ANY of:
  - 'runtest TOKEN', 'run TOKEN', or 'include TOKEN'
    * TOKEN may be raw or quoted or <angle-bracketed>
    * '.tsc' is appended if missing
  - Flags from 'let KEY VALUE' (last assignment wins)


python3 parse_lrg_map.py \
  /ade/shratyag_v7/oss/test/tsage/src/tsagexacldsuite.tsc \
  --start-line 880 \
  --platform Exascale \
  --runtime null \
  --out-json /ade/shratyag_v7/tklocal/lrg_map.json \
  --print-first \
  --debug-unmatched-if /ade/shratyag_v7/tklocal/unmatched_if.log

Output JSON (array):
[
  {
    "lrgname": "lrg<NAME>",
    "tests":   ["<file>.tsc", ...],   # deduped, in order found
    "flags":   {"flag":"value", ...},
    "Platform":"Exascale",            # from CLI
    "runtime": null                   # from CLI
  },
  ...
]
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional

# ---------- Section header patterns ----------
IF_WITH_ALL_RE = re.compile(
    r"""
    ^\s*if\s*
    \(\s*\(\s*section\s*==\s*([A-Za-z0-9_]+)\s*\)\s*
    (?:\|\||or)\s*
    \(\s*section\s*==\s*ALL\s*\)\s*\)\s*
    (?:then\b)?\s*
    (?:[#;].*)?$
    """, re.IGNORECASE | re.VERBOSE,
)

IF_SIMPLE_RE = re.compile(
    r"""
    ^\s*if\s*
    \(\s*section\s*==\s*([A-Za-z0-9_]+)\s*\)\s*
    (?:then\b)?\s*
    (?:[#;].*)?$
    """, re.IGNORECASE | re.VERBOSE,
)

IF_DOUBLE_SIMPLE_RE = re.compile(
    r"""
    ^\s*if\s*
    \(\s*\(\s*section\s*==\s*([A-Za-z0-9_]+)\s*\)\s*\)\s*
    (?:then\b)?\s*
    (?:[#;].*)?$
    """, re.IGNORECASE | re.VERBOSE,
)

# ---------- Generic IF/ENDIF for nesting ----------
IF_ANY_RE   = re.compile(r"^\s*if\b", re.IGNORECASE)
ENDIF_RE    = re.compile(r"^\s*endif\b.*$", re.IGNORECASE)

# ---------- Test & flag lines ----------
TEST_LINE_RE = re.compile(
    r"""
    ^\s*(?:runtest|run|include)\s*
    (?:<\s*([^>\s]+)\s*>\s*|   # group(1): <token>
       "([^"\s]+)"\s*|         # group(2): "token"
       '([^'\s]+)'\s*|         # group(3): 'token'
       ([^\s#;]+)              # group(4): bare token
    )
    """, re.IGNORECASE | re.VERBOSE,
)

LET_RE = re.compile(r"^\s*let\s+([A-Za-z0-9_.\-]+)\s+(.*?)\s*$", re.IGNORECASE)

# For diagnostics
IF_LINE_GUESS_RE = re.compile(r"^\s*if\s*\(", re.IGNORECASE)

# ---------- Helpers ----------

def add_lrg_prefix(name: str) -> str:
    return name if name.lower().startswith("lrg") else "lrg" + name

def match_section_header(line: str) -> Optional[str]:
    m = IF_WITH_ALL_RE.match(line)
    if m: return m.group(1)
    m = IF_DOUBLE_SIMPLE_RE.match(line)   # check before simple
    if m: return m.group(1)
    m = IF_SIMPLE_RE.match(line)
    if m: return m.group(1)
    return None

def normalize_token_to_testname(tok: str) -> Optional[str]:
    if not tok: return None
    base = Path(tok.strip()).name
    if not base: return None
    if not base.lower().endswith(".tsc"):
        base += ".tsc"
    return base

# ---------- Core parsing ----------

def parse_sections(lines: List[str], start_line: int, debug_unmatched_if: Optional[Path]) -> List[Dict]:
    results: List[Dict] = []
    i = max(0, start_line - 1)
    n = len(lines)

    ferr = debug_unmatched_if.open("w", encoding="utf-8") if debug_unmatched_if else None

    while i < n:
        line = lines[i]
        i += 1

        name = match_section_header(line)
        if name is None:
            if ferr and IF_LINE_GUESS_RE.match(line):
                ferr.write(f"[UNMATCHED_IF] L{i}: {line.rstrip()}\n")
            continue

        lrgname = add_lrg_prefix(name.strip())
        tests: List[str] = []
        flags: Dict[str, str] = {}

        # --- read the section body with nested if/endif handling ---
        depth = 0  # nesting depth for inner if/endif (not section headers)
        while i < n:
            ln = lines[i]

            # Close the section only when we hit ENDIF at depth==0
            if ENDIF_RE.match(ln):
                if depth == 0:
                    i += 1  # consume this ENDIF and close the section
                    break
                else:
                    depth -= 1
                    i += 1
                    continue

            # Increase depth for any 'if' line (that isn't a new section header)
            if IF_ANY_RE.match(ln) and match_section_header(ln) is None:
                depth += 1
                i += 1
                continue

            # Collect flags
            fm = LET_RE.match(ln)
            if fm:
                flag = fm.group(1).strip()
                value = fm.group(2).strip()
                if flag:
                    flags[flag] = value
                i += 1
                continue

            # Collect tests from runtest/run/include
            tm = TEST_LINE_RE.match(ln)
            if tm:
                tok = tm.group(1) or tm.group(2) or tm.group(3) or tm.group(4)
                test_name = normalize_token_to_testname(tok)
                if test_name and test_name not in tests:
                    tests.append(test_name)
                i += 1
                continue

            # Default advance
            i += 1

        # Append the parsed section
        results.append({
            "lrgname": lrgname,
            "tests": tests,
            "flags": flags,
            "Platform": None,
            "runtime": None
        })

    if ferr:
        ferr.close()

    return results

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Parse LRG→tests mapping from a control file (nested if/endif supported).")
    ap.add_argument("file", type=str, help="Mapping file path")
    ap.add_argument("--start-line", type=int, default=880, help="Start parsing AFTER this line (1-indexed)")
    ap.add_argument("--platform", default="Exascale", help="Platform value written into each record")
    ap.add_argument("--runtime", default="null", help="Number/string or 'null' for JSON null")
    ap.add_argument("--out-json", default="lrg_map.json", help="Output JSON filename (array)")
    ap.add_argument("--print-first", action="store_true", help="Print first few extracted sections")
    ap.add_argument("--debug-unmatched-if", default=None, help="Write unmatched 'if' header lines here")
    args = ap.parse_args()

    # Normalize runtime for JSON
    if isinstance(args.runtime, str) and args.runtime.lower() == "null":
        runtime_val = None
    else:
        try:
            runtime_val = float(args.runtime) if "." in str(args.runtime) else int(args.runtime)
        except (ValueError, TypeError):
            runtime_val = args.runtime

    text = Path(args.file).read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    debug_path = Path(args.debug_unmatched_if) if args.debug_unmatched_if else None
    records = parse_sections(lines, start_line=args.start_line, debug_unmatched_if=debug_path)

    # Inject CLI defaults
    for rec in records:
        rec["Platform"] = args.platform
        rec["runtime"]  = runtime_val

    Path(args.out_json).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.print_first:
        for rec in records[:5]:
            print("\n--- LRG ---")
            print(json.dumps(rec, ensure_ascii=False, indent=2))

    print(f"Parsed sections: {len(records)}")
    print(f"Wrote: {args.out_json}")
    if debug_path:
        print(f"Unmatched 'if' headers (if any) → {debug_path}")

if __name__ == "__main__":
    main()