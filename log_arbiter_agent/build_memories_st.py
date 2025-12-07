#!/usr/bin/env python3
import argparse, os, pickle, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
from utils import dump_pickle, normalize_log_window

EMB_PATH_KNOWN = "data/.known_miniLM.pkl"
EMB_PATH_IGNORE = "data/.ignore_miniLM.pkl"

def make_model(model_name: str):
    # Small, fast, CPU-friendly sentence embedding model
    return SentenceTransformer(model_name)

def embed_texts(model, texts):
    embs = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    return np.asarray(embs, dtype=np.float32)

def main(known_path: str, ignore_path: str, model_name: str):
    os.makedirs("data", exist_ok=True)

    model = make_model(model_name)

    # Known bugs
    kb = pd.read_csv(known_path).fillna("")
    kb_texts = kb['context_window'].astype(str).apply(normalize_log_window).tolist()
    kb_embs = embed_texts(model, kb_texts)
    dump_pickle(dict(df=kb, model_name=model_name, embs=kb_embs), EMB_PATH_KNOWN)
    print(f"Built known-bugs MiniLM index: {EMB_PATH_KNOWN} (n={kb_embs.shape[0]})")

    # Ignorable
    ig = pd.read_csv(ignore_path).fillna("")
    ig_texts = ig['context_window'].astype(str).apply(normalize_log_window).tolist()
    ig_embs = embed_texts(model, ig_texts)
    dump_pickle(dict(df=ig, model_name=model_name, embs=ig_embs), EMB_PATH_IGNORE)
    print(f"Built ignorable MiniLM index: {EMB_PATH_IGNORE} (n={ig_embs.shape[0]})")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", required=True)
    ap.add_argument("--ignore", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    main(args.known, args.ignore, args.model)
