#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_runtimes_into_lrg.py

Merge per-LRG runtimes (minutes) into lrg_map.json.

Usage:
  python3 merge_runtimes_into_lrg.py \
    --lrgs /ade/shratyag_v7/tklocal/lrg_map.json \
    --rt   /ade/shratyag_v7/tklocal/runtimes_avg_last3.json \
    --out  /ade/shratyag_v7/tklocal/lrg_map_with_runtimes.json \
    --write-hours
"""
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List

def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    ap = argparse.ArgumentParser(description="Populate runtimes into lrg_map.json")
    ap.add_argument("--lrgs", required=True, help="Path to lrg_map.json (array)")
    ap.add_argument("--rt",   required=True, help="Path to runtimes map JSON {lrg: minutes}")
    ap.add_argument("--out",  required=True, help="Output path")
    ap.add_argument("--write-hours", dest="write_hours", action="store_true",
                    help="Also write hours into 'runtime' (overwriting existing)")
    args = ap.parse_args()

    lrgs = load_json(args.lrgs)
    if not isinstance(lrgs, list):
        print("ERROR: --lrgs must be a JSON array", file=sys.stderr)
        sys.exit(2)

    runtimes = load_json(args.rt)
    if not isinstance(runtimes, dict):
        print("ERROR: --rt must be a JSON object {lrg: minutes}", file=sys.stderr)
        sys.exit(2)

    updated = 0
    missing = 0
    for rec in lrgs:
        if not isinstance(rec, dict):
            continue
        name = (rec.get("lrgname") or "").strip()
        if not name:
            continue
        if name in runtimes:
            minutes = runtimes[name]
            try:
                minutes_f = float(minutes)
            except Exception:
                missing += 1
                continue

            rec["runtime_minutes"] = int(round(minutes_f))
            if args.write_hours:
                rec["runtime"] = round(minutes_f / 60.0, 3)
            updated += 1
        else:
            missing += 1

    Path(args.out).write_text(json.dumps(lrgs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} | updated={updated} | no_runtime={missing}")

if __name__ == "__main__":
    main()