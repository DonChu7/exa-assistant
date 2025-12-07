#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a FAISS index over TEST descriptions in profiles.json.
(Embeds *only* the Description body to improve semantic signal.)

Usage:
python3 build_rag_index.py \
    --profiles /ade/shratyag_v7/tklocal/profiles.json \
    --index    /ade/shratyag_v7/tklocal/rag_tests.faiss \
    --meta     /ade/shratyag_v7/tklocal/rag_tests.jsonl \
    --model sentence-transformers/all-MiniLM-L6-v2

Dependencies:
  pip install faiss-cpu sentence-transformers
"""
import argparse, json, re
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

def load_profiles(p: str) -> List[Dict[str, Any]]:
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("profiles.json must be a JSON array.")
    return data

def extract_docs(profiles):
    docs = []
    for row in profiles:
        if row.get("doc_type") == "TEST" and (row.get("text") or "").strip():
            meta = row.get("meta") or {}
            tid  = meta.get("id")
            if not tid: continue
            docs.append({
                "test": tid,
                "text": row["text"],        # full text kept in meta file
                "setup": meta.get("setup"),
                "flags": meta.get("flags") or {}
            })
    return docs

SPLIT = re.compile(r"\bDescription:\s*", re.IGNORECASE)
def desc_body(text: str) -> str:
    """Return only the description body after the 'Description:' marker."""
    if not text: return ""
    parts = SPLIT.split(text, maxsplit=1)
    return (parts[1] if len(parts) > 1 else text).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True, help="profiles.json created by build_profiles_fast.py")
    ap.add_argument("--index",    required=True, help="output FAISS index path")
    ap.add_argument("--meta",     required=True, help="output JSONL metadata aligned with index rows")
    ap.add_argument("--model",    default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    import faiss

    profiles = load_profiles(args.profiles)
    docs = extract_docs(profiles)
    if not docs:
        raise SystemExit("No TEST docs with text found.")

    # Use only the description body for embeddings
    texts = [desc_body(d["text"]) for d in docs]

    model = SentenceTransformer(args.model)
    embs = model.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    # cosine via normalized inner product
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, args.index)

    
    with Path(args.meta).open("w", encoding="utf-8") as f:
        for d in docs:
            d_out = dict(d)
            d_out["desc"] = desc_body(d["text"])  # <-- precompute and persist
            # Optional: if you don't need full text at query time, you can remove it:
            # del d_out["text"]
            f.write(json.dumps(d_out, ensure_ascii=False) + "\n")

    print(f"Wrote: {args.index} and {args.meta} (docs={len(docs)})")

if __name__ == "__main__":
    main()






