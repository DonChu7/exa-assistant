#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic query over TEST descriptions (one vector per TEST).
Ranks by FAISS cosine similarity fused with BM25 using Reciprocal Rank Fusion (RRF).

Usage:
python3 rag_query_tests.py \
  --index /ade/shratyag_v7/tklocal/rag_tests.faiss \
  --meta  /ade/shratyag_v7/tklocal/rag_tests.jsonl \
  --q "Blockstore tests" \
  --k 10 \
  --t2l /ade/shratyag_v7/tklocal/test_to_lrgs.json \
  --lrgs-json /ade/shratyag_v7/tklocal/lrg_map_with_suites.json \
  --pool-mult 8


python3 rag_query_tests.py \
  --index /ade/shratyag_v8/tklocal/rag_tests.faiss \
  --meta  /ade/shratyag_v8/tklocal/rag_tests.jsonl \
  --q "Blockstore tests" \
  --k 10 \
  --t2l /ade/shratyag_v8/tklocal/test_to_lrgs.json \
  --lrgs-json /ade/shratyag_v8/tklocal/lrg_map_with_suites.json \
  --pool-mult 8

Notes:
- pool-mult controls how many FAISS candidates to re-rank (k * pool-mult).
- rrf-k is the standard constant in RRF (60 is common).
"""

import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ALNUM = re.compile(r"[a-z0-9]+")

def load_meta(meta_path: str) -> List[Dict[str, Any]]:
    docs = []
    with Path(meta_path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            docs.append(json.loads(line))
    return docs

def load_json_or_empty(p: str | None):
    if not p: return {}
    return json.loads(Path(p).read_text(encoding="utf-8"))

def tokenize(s: str) -> List[str]:
    if not s: return []
    return ALNUM.findall(s.lower())

def description_only(text: str) -> str:
    """Trim TEST text to only the Description body if present."""
    if not text: return ""
    m = re.split(r"\bDescription:\s*", text, maxsplit=1, flags=re.IGNORECASE)
    return (m[1] if len(m) > 1 else text).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="FAISS index built by build_rag_index.py")
    ap.add_argument("--meta",  required=True, help="jsonl lines aligned with index rows")
    ap.add_argument("--q",     required=True, help="natural language description")
    ap.add_argument("--k", type=int, default=10, help="top-k to print")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--pool-mult", type=int, default=8, help="rerank pool multiplier (k * pool-mult)")
    ap.add_argument("--rrf-k", type=float, default=60.0, help="RRF constant (bigger = smoother)")
    ap.add_argument("--t2l", default=None)
    ap.add_argument("--lrgs-json", default=None)
    args = ap.parse_args()

    # Load FAISS & meta
    index = faiss.read_index(args.index)
    docs  = load_meta(args.meta)
    if len(docs) != index.ntotal:
        raise SystemExit(f"Meta/docs count ({len(docs)}) != index.ntotal ({index.ntotal})")

    # Optional map for printing LRG info
    t2l = load_json_or_empty(args.t2l)
    lrg_info: Dict[str, Dict[str, Any]] = {}
    if args.lrgs_json:
        arr = json.loads(Path(args.lrgs_json).read_text(encoding="utf-8"))
        if isinstance(arr, list):
            for rec in arr:
                name = (rec.get("lrgname") or "").strip()
                if not name: continue
                lrg_info[name] = {
                    "runtime": rec.get("runtime"),
                    "suites": rec.get("lrg_suite") or rec.get("suites") or [],
                    "platform": rec.get("Platform"),
                }

    # -------- Embed query & FAISS ANN --------
    model = SentenceTransformer(args.model)
    q_vec = model.encode([args.q], convert_to_numpy=True, normalize_embeddings=True)
    pool_n = max(args.k * max(args.pool_mult, 1), args.k)
    sims, idxs = index.search(q_vec.astype("float32"), pool_n)
    sims, idxs = sims[0], idxs[0]

    # -------- BM25 over description-only text --------
    desc_corpus = [description_only(d.get("text") or "") for d in docs]
    bm25 = BM25Okapi([tokenize(t) for t in desc_corpus])
    bm25_scores = bm25.get_scores(tokenize(args.q))

    # -------- RRF fusion (embedding rank + bm25 rank) --------
    # ranks by embedding (0 is best) among the FAISS pool
    embed_rank = {i: r for r, i in enumerate(idxs)}
    # ranks by bm25 over whole corpus (desc)
    bm25_sorted = np.argsort(-bm25_scores)  # descending
    bm25_rank = {i: r for r, i in enumerate(bm25_sorted)}

    scored: List[Tuple[float, int]] = []
    K = float(args.rrf_k)
    for i, sim in zip(idxs, sims):
        r1 = embed_rank.get(i, 10_000)
        r2 = bm25_rank.get(i, 10_000)
        rrf = 1.0/(K + r1) + 1.0/(K + r2)
        # small tie-breaker with cosine similarity
        final = rrf + 0.01*float(sim) + 0.002*bm25_scores[i]  # <-- additive BM25
        scored.append((final, i))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:args.k]

    # -------- Print results --------
    print(f"\nQuery: {args.q}")
    for rank, (score, i) in enumerate(top, 1):
        d = docs[i]
        test  = d.get("test") or "(unknown)"
        setup = d.get("setup")
        flags = d.get("flags") or {}
        # print(f"{rank:2d}. {test}  score={score:.3f}")
        print(f"{rank:2d}. {test} ")
        print(f"    setup={setup}  flags={flags}")
        if t2l:
            lrgs = t2l.get(test, [])
            if lrgs:
                preview = []
                for l in lrgs[:6]:
                    info = lrg_info.get(l) or {}
                    suites = info.get("suites") or []
                    rt = info.get("runtime")
                    frag = l
                    if suites: frag += f" [suites={','.join(suites)}]"
                    if isinstance(rt, (int, float)): frag += f" (~{rt} hours)"
                    preview.append(frag)
                print(f"    LRGs: {', '.join(preview)}")
        
        # full description body (no truncation)
        desc = description_only(d.get("text") or "").strip()
        if desc:
            print("    --- description ---")
            for line in desc.splitlines():
                print("    " + line)
            print("    --------------------")

if __name__ == "__main__":
    main()