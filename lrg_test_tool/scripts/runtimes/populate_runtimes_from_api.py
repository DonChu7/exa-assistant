#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
populate_runtimes_from_api.py

Goal:
  - Read LRG IDs from a mapping JSON (e.g. lrg_to_tests_with_suites.json).
  - For each LRG, call the APEX runtime API.
  - Parse dates out of labels and keep ONLY the last 30 days of data.
  - Store (lrg_id, label, runtime_hours) into lrg_metrics.db.

Usage example (from repo root or from scripts/runtimes/ with adjusted paths):

  python3.11 scripts/runtimes/populate_runtimes_from_api.py \
    --lrgs-json lrg_to_tests_with_suites.json \
    --days 30

  python3.11 populate_runtimes_from_api.py \
    --lrgs-json ../../lrg_to_tests_with_suites.json \
    --days 30

If you omit --limit-lrgs, it processes ALL LRGs from the JSON.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import re
import sqlite3
import time
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
from lrg_test_tool.metrics.runtime_store import DB_PATH as RUNTIME_DB_PATH
import requests

BASE_URL = "https://apex.oraclecorp.com/pls/apex/lrg_times/MAIN/lrg/{lrg_id}"
DAYS_TO_KEEP = 30


def load_lrg_ids(path: Path) -> Dict[str, Any]:
    """
    Accept either:
      1) A JSON OBJECT mapping lrg_id -> info, e.g.:
         { "lrgsaexacldbsfail18": { ... }, "lrgc16sacc2qm": { ... } }

      2) A JSON ARRAY of LRG records like lrg_map_with_suites.json, e.g.:
         [
           { "lrgname": "lrgdbcsaexcpdbcarousel", "tests": [...], "lrg_suite": [...] },
           { "lrgname": "lrgdbcsaexcpdbcarousel2", ... }
         ]

    We normalize both into a dict: lrg_id -> record.
    """
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        out: Dict[str, Any] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            lrg_id = (item.get("lrgname") or item.get("id") or "").strip()
            if not lrg_id:
                continue
            if lrg_id not in out:
                out[lrg_id] = item
        if not out:
            raise SystemExit(f"{path} list did not contain any objects with 'lrgname' or 'id'")
        return out

    raise SystemExit(
        f"{path} must be either:\n"
        f"  - a JSON OBJECT mapping lrg_id -> info, or\n"
        f"  - a JSON ARRAY of objects with 'lrgname'."
    )


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(RUNTIME_DB_PATH)
    conn.execute("""
      CREATE TABLE IF NOT EXISTS lrg_runtime_history (
        lrg_id        TEXT NOT NULL,
        label         TEXT NOT NULL,
        runtime_hours REAL NOT NULL,
        PRIMARY KEY (lrg_id, label)
      )
    """)
    return conn


_label_date_re = re.compile(r".*?(\d{6})(?:\.\d+)?$")


def parse_label_date(label: str) -> Optional[date]:
    m = _label_date_re.match(label)
    if not m:
        return None

    yymmdd = m.group(1)
    yy = int(yymmdd[0:2])
    mm = int(yymmdd[2:4])
    dd = int(yymmdd[4:6])

    year = 2000 + yy
    try:
        return date(year, mm, dd)
    except ValueError:
        return None


def fetch_lrg_runtimes(lrg_id: str) -> List[dict]:
    url = BASE_URL.format(lrg_id=lrg_id)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    items = data.get("items")
    if items is None:
        if isinstance(data, list):
            return data
        return []
    if not isinstance(items, list):
        return []
    return items


def main():
    ap = argparse.ArgumentParser(description="Populate lrg_metrics.db from APEX API (last N days only).")
    ap.add_argument("--lrgs-json", required=True, help="Path to lrg_to_tests_with_suites.json or lrg_map_with_suites.json")
    ap.add_argument("--limit-lrgs", type=int, default=0, help="Only process first N LRGs (0 = all)")
    ap.add_argument("--days", type=int, default=DAYS_TO_KEEP,
                    help=f"Number of days of history to keep (default {DAYS_TO_KEEP})")
    args = ap.parse_args()

    lrg_map_path = Path(args.lrgs_json)

    lrg_map = load_lrg_ids(lrg_map_path)
    lrg_ids = sorted(lrg_map.keys())
    if args.limit_lrgs:
        lrg_ids = lrg_ids[:args.limit_lrgs]

    cutoff_date = date.today() - timedelta(days=args.days)
    print(f"Will fetch runtimes for {len(lrg_ids)} LRGs")
    print(f"Keeping only labels with date >= {cutoff_date.isoformat()} (last {args.days} days)")
    print(f"Writing to metrics DB: {RUNTIME_DB_PATH}")

    conn = get_conn()
    cur = conn.cursor()

    total_inserted = 0

    for idx, lrg_id in enumerate(lrg_ids, 1):
        try:
            items = fetch_lrg_runtimes(lrg_id)
        except Exception as e:
            print(f"[WARN] Failed to fetch {lrg_id}: {e}")
            continue

        if not items:
            print(f"[INFO] No items for {lrg_id}")
            cur.execute("DELETE FROM lrg_runtime_history WHERE lrg_id = ?", (lrg_id,))
            conn.commit()
            continue

        rows = []
        kept_labels: List[str] = []

        for it in items:
            label = str(it.get("label") or "").strip()
            if not label:
                continue

            dt = parse_label_date(label)
            if not dt:
                continue
            if dt < cutoff_date:
                continue

            t_min = it.get("time")
            if not isinstance(t_min, (int, float)):
                continue
            runtime_hours = float(t_min) / 60.0

            rows.append((lrg_id, label, runtime_hours))
            kept_labels.append(label)

        if not rows:
            cur.execute("DELETE FROM lrg_runtime_history WHERE lrg_id = ?", (lrg_id,))
            conn.commit()
            print(f"[INFO] No recent (last {args.days} days) items for {lrg_id}, cleared any old rows.")
        else:
            cur.executemany(
                """
                INSERT OR REPLACE INTO lrg_runtime_history(lrg_id, label, runtime_hours)
                VALUES (?, ?, ?)
                """,
                rows,
            )
            conn.commit()
            total_inserted += len(rows)

            placeholders = ",".join(["?"] * len(kept_labels))
            sql_delete = f"""
              DELETE FROM lrg_runtime_history
              WHERE lrg_id = ?
                AND label NOT IN ({placeholders})
            """
            cur.execute(sql_delete, (lrg_id, *kept_labels))
            conn.commit()

            print(f"[{idx}/{len(lrg_ids)}] {lrg_id}: kept {len(rows)} recent rows (total inserted {total_inserted})")

        if idx % 10 == 0:
            time.sleep(0.2)

    conn.close()
    print(f"Done. Total recent rows inserted/updated: {total_inserted}")
    print(f"DB: {RUNTIME_DB_PATH}")


if __name__ == "__main__":
    main()