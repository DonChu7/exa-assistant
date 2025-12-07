#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_profiles_fast.py (instrumented)

Adds:
- Progress logs while processing tests/LRGs
- --limit-tests / --limit-lrgs for quick subset runs
- Timings for build and write phases

# ---------------------------------------------------------------------------
# CLI PARAMETERS (type • default • description)
#
# Required inputs (files)
#   --tests PATH            • str  • default="tests_extracted.json"
#       JSON array of test objects with fields: test_name, setup, flags, description.
#
#   --lrgs PATH             • str  • default="lrg_map.json"
#       JSON array of LRG objects with fields: lrgname, tests[], flags{}, Platform, runtime (minutes or null).
#
# Output
#   --out PATH              • str  • default="profiles.json"
#       Output profiles file (JSON array or JSONL when --jsonl).
#
# Chunking controls for TEST descriptions
#   --no-chunk              • bool • default=False
#       If set, do not chunk long descriptions (one profile per test).
#
#   --fast                  • bool • default=False
#       Use fast fixed-width chunker (no overlap) instead of newline-aware one.
#
#   --max-chars-per-chunk   • int  • default=3000
#       Target max characters per chunk (TEST descriptions).
#
#   --overlap-chars         • int  • default=200
#       Overlap between consecutive chunks (only for slow/newline-aware chunker).
#
#   --max-desc-chars        • int  • default=20000
#       Cap description length per test before chunking (protects runaway inputs).
#
#   --max-tests-preview     • int  • default=8
#       Number of test names to preview in LRG text.
#
# Output formatting
#   --jsonl                 • bool • default=False
#       Write one JSON object per line (JSON Lines) instead of a single JSON array.
#
#   --pretty                • bool • default=False
#       Pretty-print JSON output (indentation) when not using JSONL.
#
# Limits & verification
#   --limit-tests N         • int  • default=0
#       Only process first N test objects (0 means no limit).
#
#   --limit-lrgs N          • int  • default=0
#       Only process first N LRG objects (0 means no limit).
#
#   --verify                • bool • default=False
#       Validate the built profiles (counts, empty texts, bad types, duplicate IDs).
#
#   --report-out PATH       • str  • default=None
#       If set, write the verification report to this path (JSON).
#
#   --fail-on-error         • bool • default=False
#       Exit with non-zero code if verification finds any errors.
#
#   --print-examples N      • int  • default=2
#       Print the first N profile texts to stdout after writing.
# ---------------------------------------------------------------------------

Example runs:

python3.11 build_profiles_fast.py \
  --tests ../tests_extracted.json \
  --lrgs  ../lrg_map_with_suites.json \
  --out   ../profiles.json \
  --verify --report-out ../profiles_report.json \
  --print-examples 3

python3 build_profiles_fast.py \
  --tests /ade/shratyag_v7/tklocal/tests_filtered.json \
  --lrgs  /ade/shratyag_v7/tklocal/lrg_map_with_runtimes.json \
  --out   /ade/shratyag_v7/tklocal/profiles.json

python3 build_profiles_fast.py \
  --tests /ade/shratyag_v7/tklocal/tests_filtered.json \
  --lrgs  /ade/shratyag_v7/tklocal/lrg_map_with_suites.json \
  --out   /ade/shratyag_v7/tklocal/profiles.json \
  --verify --report-out /ade/shratyag_v7/tklocal/profiles_report.json \
  --print-examples 3

python3 build_profiles_fast.py \
  --tests /home/kbaboota/scripts/label_health_copilot/shray/jsons/tests_filtered.json \
  --lrgs  /home/kbaboota/scripts/label_health_copilot/shray/jsons/lrg_map_with_runtimes.json \
  --out   /home/kbaboota/scripts/label_health_copilot/shray/profiles.json

"""

import argparse, json, sys, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
from collections import Counter
# from lrg_test_tool.brain.constants import normalize_setup

# ---------- Defaults ----------
MAX_CHARS_PER_CHUNK_DEFAULT = 3000
OVERLAP_CHARS_DEFAULT       = 200
MAX_DESC_CHARS_DEFAULT      = 20000
MAX_TESTS_PREVIEW_DEFAULT   = 8
TEST_PROGRESS_EVERY         = 200
LRG_PROGRESS_EVERY          = 50
# ------------------------------

try:
    import orjson as _fastjson
    def dumps(obj, pretty: bool = False) -> str:
        opt = _fastjson.OPT_NON_STR_KEYS
        if pretty: opt |= _fastjson.OPT_INDENT_2
        return _fastjson.dumps(obj, option=opt).decode("utf-8")
except Exception:
    def dumps(obj, pretty: bool = False) -> str:
        return json.dumps(obj, ensure_ascii=False, indent=(2 if pretty else None))

def load_array(path: str) -> List[Dict[str, Any]]:
    t0 = time.time()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"{path} must be a JSON ARRAY of objects.")
    print(f"Loaded {path} ({len(data)} rows) in {time.time()-t0:.2f}s")
    return data

def fmt_flags(flags: Any) -> str:
    if not isinstance(flags, dict) or not flags: return "(none)"
    return ", ".join(f"{k}={v}" for k, v in flags.items())

def sanitize_flags(flags: Any) -> Dict[str, Any]:
    return flags if isinstance(flags, dict) else {}

def sanitize_tests_list(lst: Any) -> List[str]:
    if not isinstance(lst, list): return []
    out: List[str] = []
    for x in lst:
        if isinstance(x, str):
            s = x.strip()
            if s: out.append(s)
    return out

def cap_text(s: str, max_chars: int) -> str:
    s = (s or "").strip()
    if max_chars and len(s) > max_chars: return s[:max_chars]
    return s

def chunk_text_slow(text: str, max_chars: int, overlap: int) -> List[str]:
    """
    Nicer chunking: try to cut at newlines near the window end, with overlap.
    Guaranteed forward progress even if no newline is found at a reasonable spot.
    """
    text = text or ""
    n = len(text)
    if n <= max_chars:
        return [text] if text else []

    chunks: List[str] = []
    i = 0
    # minimum advance to avoid infinite loops (at least 1 char)
    MIN_ADVANCE = max(1, max_chars // 2)

    while i < n:
        j = min(i + max_chars, n)

        # Try to cut at a newline within the last ~500 chars of the window
        search_start = max(i, j - 500)
        cut = text.rfind("\n", search_start, j)

        # If no newline found, or it would not advance, fall back to j
        if cut == -1 or cut <= i:
            cut = j

        piece = text[i:cut].strip()
        if piece:
            chunks.append(piece)

        # Compute next start with overlap, but ensure forward progress
        next_i = max(cut - overlap, i + MIN_ADVANCE)
        if next_i <= i:  # hard guarantee
            next_i = i + MIN_ADVANCE

        i = next_i

    return chunks

def chunk_text_fast(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars: return [text] if text else []
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]

def test_profile_chunks(t: Dict[str, Any],
                        max_chunk: int,
                        overlap: int,
                        no_chunk: bool,
                        fast: bool,
                        max_desc_chars: int) -> List[Tuple[str, Dict[str, Any], str]]:
    name = (t.get("test_name") or "").strip()
    if not name: return []
    setup = t.get("setup")
    flags = sanitize_flags(t.get("flags"))
    desc  = cap_text((t.get("description") or ""), max_desc_chars)
    header = f"Test {name}. setup={setup}. flags: {fmt_flags(flags)}."
    if no_chunk:
        chunks = [desc] if desc else ["(empty)"]
    else:
        text_norm = desc.strip()
        if not text_norm:
            chunks = ["(empty)"]
        else:
            chunks = chunk_text_fast(text_norm, max_chunk) if fast else \
                     chunk_text_slow(text_norm, max_chunk, overlap)
            if not chunks:
                chunks = ["(empty)"]
    out: List[Tuple[str, Dict[str, Any], str]] = []
    for ci, chunk in enumerate(chunks):
        text = f"{header}\nDescription: {chunk}"
        # meta = {"doc_type":"TEST","id":name,"chunk_id":ci,"setup":setup,"flags":flags}
        raw_setup = t.get("setup")
        setup = normalize_setup(raw_setup)

        meta = {
            "doc_type": "TEST",
            "id": name,
            "chunk_id": ci,
            "setup": setup,
            "flags": flags,
        }
        out.append((text, meta, f"chunk{ci}"))
    return out

def lrg_profile(L: Dict[str, Any], max_tests_preview: int) -> Tuple[str, Dict[str, Any]]:
    lrg = (L.get("lrgname") or "").strip()
    if not lrg: return ("", {})
    platform = (L.get("Platform") or "Unknown")
    flags    = sanitize_flags(L.get("flags"))
    tests    = sanitize_tests_list(L.get("tests"))
    runtime  = L.get("runtime")              # minutes or None
    suites   = sanitize_suites(L.get("lrg_suite"))  # <— NEW

    preview  = ", ".join(tests[:max_tests_preview]) if tests else "(none)"
    runtime_line = f"Runtime: {runtime} minutes." if runtime is not None else "Runtime: (unknown)."
    suites_line  = f"Suites: {', '.join(suites)}." if suites else "Suites: (none)."

    text = (
        f"LRG {lrg} on {platform}.\n"
        f"{runtime_line}\n"
        f"{suites_line}\n"                      # <— NEW (searchable by FTS)
        f"Contains tests: {preview}.\n"
        f"Flags: {fmt_flags(flags)}."
    )

    meta = {
        "doc_type": "LRG",
        "id": lrg,
        "platform": platform,
        "flags": flags,
        "tests_count": len(tests),
        "runtime": runtime,
        "lrg_suite": suites,                    # <— NEW (visible in query output)
    }
    return text, meta

def verify_profiles(profiles: List[Dict[str, Any]]) -> Dict[str, Any]:
    report: Dict[str, Any] = {"errors": [], "counts": {}, "samples": profiles[:3]}
    report["counts"]["profiles"] = len(profiles)
    by_type = Counter(p.get("doc_type") for p in profiles)
    report["counts"]["by_type"] = dict(by_type)
    ids = [p.get("id") for p in profiles]
    dup_ids = [k for k, c in Counter(ids).items() if c > 1]
    if dup_ids: report["errors"].append({"duplicate_ids": dup_ids[:10], "count": len(dup_ids)})
    empties = [p.get("id") for p in profiles if not (p.get("text") or "").strip()]
    if empties: report["errors"].append({"empty_texts": empties[:10], "count": len(empties)})
    bad_types = [p.get("id") for p in profiles if p.get("doc_type") not in ("TEST","LRG")]
    if bad_types: report["errors"].append({"bad_doc_type": bad_types[:10], "count": len(bad_types)})
    return report

def sanitize_suites(x):
    # Accepts "lrg_suite": "suiteA" or ["suiteA","suiteB"]
    if x is None: return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, list):
        out = []
        for v in x:
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
        return out
    return []

def main():
    ap = argparse.ArgumentParser(description="Fast, instrumented profile builder.")
    ap.add_argument("--tests", default="tests_extracted.json")
    ap.add_argument("--lrgs",  default="lrg_map.json")
    ap.add_argument("--out",   default="profiles.json")
    # knobs
    ap.add_argument("--no-chunk", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--max-chars-per-chunk", type=int, default=MAX_CHARS_PER_CHUNK_DEFAULT)
    ap.add_argument("--overlap-chars", type=int, default=OVERLAP_CHARS_DEFAULT)
    ap.add_argument("--max-desc-chars", type=int, default=MAX_DESC_CHARS_DEFAULT)
    ap.add_argument("--max-tests-preview", type=int, default=MAX_TESTS_PREVIEW_DEFAULT)
    # output
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--pretty", action="store_true")
    # limits & verify
    ap.add_argument("--limit-tests", type=int, default=0, help="Process only first N tests")
    ap.add_argument("--limit-lrgs", type=int, default=0, help="Process only first N lrgs")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--fail-on-error", action="store_true")
    ap.add_argument("--print-examples", type=int, default=2)
    args = ap.parse_args()

    t_load = time.time()
    tests = load_array(args.tests)
    lrgs  = load_array(args.lrgs)
    if args.limit_tests: tests = tests[:args.limit_tests]
    if args.limit_lrgs:  lrgs  = lrgs[:args.limit_lrgs]
    print(f"Limits: tests={len(tests)}, lrgs={len(lrgs)}")

    profiles: List[Dict[str, Any]] = []
    skipped_tests = skipped_lrgs = 0

    # Build tests
    t0 = time.time()
    for idx, t in enumerate(tests, 1):
        try:
            items = test_profile_chunks(
                t,
                max_chunk=args.max_chars_per_chunk,
                overlap=args.overlap_chars,
                no_chunk=args.no_chunk,
                fast=args.fast,
                max_desc_chars=args.max_desc_chars
            )
        except Exception as e:
            skipped_tests += 1
            if skipped_tests <= 3:
                print(f"[WARN] skipping test idx={idx}: {e}", file=sys.stderr)
            continue

        if not items:
            skipped_tests += 1
            continue

        for text, meta, suffix in items:
            profiles.append({"id": f"TEST::{meta['id']}::{suffix}",
                             "doc_type":"TEST", "text": text, "meta": meta})

        if idx % TEST_PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            print(f"[tests] processed {idx}/{len(tests)} in {elapsed:.1f}s "
                  f"(profiles so far: {len(profiles)})")
            sys.stdout.flush()

    # Build lrgs
    t1 = time.time()
    for idx, L in enumerate(lrgs, 1):
        try:
            text, meta = lrg_profile(L, max_tests_preview=args.max_tests_preview)
        except Exception as e:
            skipped_lrgs += 1
            if skipped_lrgs <= 3:
                print(f"[WARN] skipping lrg idx={idx}: {e}", file=sys.stderr)
            continue
        if not text or not meta:
            skipped_lrgs += 1
            continue
        profiles.append({"id": f"LRG::{meta['id']}",
                         "doc_type":"LRG", "text": text, "meta": meta})
        if idx % LRG_PROGRESS_EVERY == 0:
            elapsed = time.time() - t1
            print(f"[lrgs]  processed {idx}/{len(lrgs)} in {elapsed:.1f}s "
                  f"(profiles so far: {len(profiles)})")
            sys.stdout.flush()

    build_elapsed = time.time() - t0
    print(f"Built {len(profiles)} profiles "
          f"(skipped tests={skipped_tests}, lrgs={skipped_lrgs}) in {build_elapsed:.1f}s")

    # Write output
    t2 = time.time()
    out_path = Path(args.out)
    if args.jsonl:
        with out_path.open("w", encoding="utf-8") as f:
            for p in profiles:
                f.write(dumps(p, pretty=False)); f.write("\n")
    else:
        out_path.write_text(dumps(profiles, pretty=args.pretty), encoding="utf-8")
    write_elapsed = time.time() - t2
    print(f"Wrote {args.out} in {write_elapsed:.1f}s (total {time.time()-t_load:.1f}s)")

    # Examples
    if args.print_examples > 0 and profiles:
        print("\n--- Sample profile texts ---")
        for p in profiles[:args.print_examples]:
            print(f"\nID: {p['id']} (type={p['doc_type']})")
            print(p["text"])

    # Verify
    if args.verify or args.report_out:
        report = verify_profiles(profiles)
        if args.report_out:
            Path(args.report_out).write_text(dumps(report, pretty=True), encoding="utf-8")
            print(f"Verification report -> {args.report_out}")
        errs = report.get("errors", [])
        print(f"Verification: profiles={report['counts']['profiles']}, "
              f"by_type={report['counts']['by_type']}, errors={len(errs)}")
        if args.fail_on_error and errs: sys.exit(1)

if __name__ == "__main__":
    main()