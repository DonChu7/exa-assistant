#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Removes tests with empty LRG lists from test_to_lrgs.json.

Example:
python3 clean_empty_tests.py \
  --in /ade/shratyag_v7/tklocal/test_to_lrgs.json \
  --out /ade/shratyag_v7/tklocal/test_to_lrgs.json
"""

import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Clean empty test entries from test_to_lrgs.json")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSON file (test_to_lrgs.json)")
    ap.add_argument("--out", dest="out", required=True, help="Output cleaned JSON file")
    args = ap.parse_args()

    data = json.loads(Path(args.inp).read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} tests from {args.inp}")

    # Filter out entries where list is empty or null
    cleaned = {k: v for k, v in data.items() if isinstance(v, list) and len(v) > 0}
    removed = len(data) - len(cleaned)

    Path(args.out).write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Removed {removed} empty entries; wrote {len(cleaned)} tests → {args.out}")

if __name__ == "__main__":
    main()