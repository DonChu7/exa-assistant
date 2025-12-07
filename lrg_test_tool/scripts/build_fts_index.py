#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---- SQLite shim (loads vendored SQLite if stdlib fails) ----
try:
    import sqlite3  # try stdlib first
except Exception:
    import sys
    import pysqlite3 as sqlite3
    sys.modules["sqlite3"] = sqlite3

"""
Build a SQLite FTS5 index from profiles.json / profiles.jsonl.

python3.11 build_fts_index.py \
  --profiles ../profiles.json \
  --db       ../profiles_fts.db


python3 build_fts_index.py \
  --profiles /home/kbaboota/scripts/label_health_copilot/shray/profiles.json \
  --db       /home/kbaboota/scripts/label_health_copilot/shray/profiles_fts.db
"""

import argparse, json, time
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter  # NEW: used to report input duplicates

def load_profiles(path: str) -> List[Dict[str, Any]]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
        return data if isinstance(data, list) else [data]
    except Exception:
        out = []
        for ln in raw.splitlines():
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
        return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True)
    ap.add_argument("--db", required=True)
    ap.add_argument("--wipe", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    profiles = load_profiles(args.profiles)
    print(f"Loaded {len(profiles)} profiles")

    # Report duplicates present inside the input itself (same id repeated).
    ids = [p.get("id") for p in profiles if p.get("id")]
    dup_ids = [k for k, c in Counter(ids).items() if c > 1]
    if dup_ids:
        print(f"[WARN] {len(dup_ids)} duplicate id(s) detected in input; duplicates will be SKIPPED.")
        # Optional: show a small sample
        for s in dup_ids[:10]:
            print("  -", s)
        if len(dup_ids) > 10:
            print(f"  ... and {len(dup_ids) - 10} more")

    conn = sqlite3.connect(args.db)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    if args.wipe:
        conn.execute("DROP TABLE IF EXISTS docs;")
        conn.execute("DROP TABLE IF EXISTS docs_fts;")

    conn.execute("""
      CREATE TABLE IF NOT EXISTS docs (
        id TEXT PRIMARY KEY,
        doc_type TEXT,
        text TEXT,
        meta_json TEXT
      )
    """)
    conn.execute("""
      CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts
      USING fts5(
        id, doc_type, text, meta_json,
        content='docs', content_rowid='rowid'
      );
    """)

    cur = conn.execute("SELECT COUNT(*) FROM docs;")
    count = cur.fetchone()[0]
    if count == 0:
        print("Inserting profiles (skipping duplicates by id)...")
        rows = [
            (
                p.get("id"),
                p.get("doc_type"),
                p.get("text") or "",
                json.dumps(p.get("meta", {}), ensure_ascii=False)
            )
            for p in profiles if p.get("id")
        ]

        before = conn.execute("SELECT COUNT(*) FROM docs;").fetchone()[0]
        # KEY CHANGE: INSERT OR IGNORE => duplicates (same id) are skipped
        conn.executemany(
            "INSERT OR IGNORE INTO docs(id, doc_type, text, meta_json) VALUES (?,?,?,?)",
            rows
        )
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM docs;").fetchone()[0]
        inserted = after - before
        skipped = len(rows) - inserted
        print(f"Inserted: {inserted}  Skipped (dups): {skipped}")

        print("Populating FTS index...")
        conn.execute("INSERT INTO docs_fts(docs_fts) VALUES('rebuild');")
        conn.commit()
    else:
        print(f"DB already has {count} docs. Use --wipe for a fresh build.")

    conn.execute("ANALYZE;")
    conn.commit()
    conn.close()
    print(f"Built FTS DB at {args.db} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()