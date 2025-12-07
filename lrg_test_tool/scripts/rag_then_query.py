#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run RAG to get candidate tests from a natural-language description,
then forward those tests to your existing query_fts_refined.py.

Usage:
  python3 rag_then_query.py \
    --db /ade/shratyag_v7/tklocalprofiles_fts.db \
    --t2l /ade/shratyag_v7/tklocaltest_to_lrgs.json \
    --l2t /ade/shratyag_v7/tklocal/lrg_to_tests.json \
    --lrgs-json /ade/shratyag_v7/tklocal/lrg_map_with_suites.json \
    --rag-index /ade/shratyag_v7/tklocal/rag_tests.faiss \
    --rag-meta  /ade/shratyag_v7/tklocal/rag_tests.jsonl \
    --q "triple failure test in egs sandbox" \
    --k 30 \
    [--require-setup tsagexastackup] [--require-flag iscsi=true] \
    [--extra "--structured-only --lrg-order runtime --show-desc"]

Notes:
- We pass the found test names as a space-joined query into query_fts_refined.py
  and use --only-direct so it prints exactly those tests/LRGs.
"""
import argparse, subprocess, shlex, sys, json
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--t2l", required=True)
    ap.add_argument("--l2t", required=True)
    ap.add_argument("--lrgs-json", required=True)
    ap.add_argument("--rag-index", required=True)
    ap.add_argument("--rag-meta",  required=True)
    ap.add_argument("--q", required=True)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--require-setup", default=None)
    ap.add_argument("--require-flag", action="append", default=[])
    ap.add_argument("--extra", default="", help="extra flags passed to query_fts_refined.py verbatim")
    ap.add_argument("--rag-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--scripts-dir", default=str(Path(__file__).resolve().parent))
    args = ap.parse_args()

    rag_cmd = [
        sys.executable, str(Path(args.scripts_dir) / "rag_find_tests.py"),
        "--index", args.rag_index, "--meta", args.rag_meta, "--q", args.q,
        "--k", str(args.k), "--model", args.rag_model
    ]
    if args.require_setup:
        rag_cmd += ["--require-setup", args.require_setup]
    for rf in (args.require_flag or []):
        rag_cmd += ["--require-flag", rf]

    # 1) get tests (newline)
    tests_txt = subprocess.check_output(rag_cmd, text=True)
    tests = [t.strip() for t in tests_txt.splitlines() if t.strip()]
    if not tests:
        print("(RAG) No tests matched the description.", file=sys.stderr)
        return 2

    # 2) forward to your existing query tool
    #    pass them as space-joined terms, and use --only-direct so only those tests appear
    q_str = " ".join(tests)
    query_cmd = [
        sys.executable, str(Path(args.scripts_dir) / "query_fts_refined.py"),
        "--db", args.db, "--q", q_str,
        "--t2l", args.t2l, "--l2t", args.l2t, "--lrgs-json", args.lrgs_json,
        "--only-direct", "--show-desc"
    ]
    if args.require_setup:
        query_cmd += ["--require-setup", args.require_setup]
    for rf in (args.require_flag or []):
        query_cmd += ["--require-flag", rf]
    if args.extra:
        query_cmd += shlex.split(args.extra)

    return subprocess.call(query_cmd)

if __name__ == "__main__":
    raise SystemExit(main())