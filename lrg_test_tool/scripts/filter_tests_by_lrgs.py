#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Filter tests_extracted.json using test_to_lrgs.json mapping.

Generates:
  - tests_filtered.json                → Tests that have valid LRG mappings
  - tests_removed.json                 → Tests that were removed (no LRGs)
  - tests_missing_from_extracted.json  → Tests that appear in mapping but not in extracted file

Usage:
  python3 filter_tests_by_lrgs.py \
    --tests /ade/shratyag_v7/tklocal/tests_extracted.json \
    --map /ade/shratyag_v7/tklocal/test_to_lrgs.json \
    --out-dir /ade/shratyag_v7/tklocal/
"""

import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser(description="Filter tests_extracted.json using test_to_lrgs.json.")
    ap.add_argument("--tests", required=True, help="Path to tests_extracted.json")
    ap.add_argument("--map", required=True, help="Path to test_to_lrgs.json mapping")
    ap.add_argument("--out-dir", required=True, help="Directory to write output JSONs")
    args = ap.parse_args()

    tests_path = Path(args.tests)
    map_path = Path(args.map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tests = json.loads(tests_path.read_text(encoding="utf-8"))
    mapping = json.loads(map_path.read_text(encoding="utf-8"))

    print(f"Loaded {len(tests)} tests from {tests_path}")
    print(f"Loaded {len(mapping)} mapping entries from {map_path}")

    test_names = {t["test_name"] for t in tests}
    mapping_keys = set(mapping.keys())

    # 1️⃣ Tests that have valid LRGs
    filtered = [t for t in tests if mapping.get(t["test_name"])]

    # 2️⃣ Tests with empty or missing LRGs
    removed = [t for t in tests if not mapping.get(t["test_name"])]

    # 3️⃣ Tests present in mapping but missing from extracted tests
    missing_from_extracted = sorted(list(mapping_keys - test_names))

    print(f"\nSummary:")
    print(f"  ✅ Tests retained: {len(filtered)}")
    print(f"  ❌ Tests removed: {len(removed)} (no LRGs)")
    print(f"  ⚠️ Tests missing from extracted: {len(missing_from_extracted)}")

    # Write all outputs
    (out_dir / "tests_filtered.json").write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "tests_removed.json").write_text(json.dumps(removed, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "tests_missing_from_extracted.json").write_text(json.dumps(missing_from_extracted, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWritten to directory: {out_dir}")

if __name__ == "__main__":
    main()