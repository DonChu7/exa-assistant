#!/usr/bin/env python3
import json
from pathlib import Path
import sys

"""
inject_suites_into_lrg_map.py

Usage:
    python3 inject_suites_into_lrg_map.py <suite_folder>
    cd lrg_test_tool/scripts
    python3.11 inject_suites_into_lrg_map.py ../../suite_lists

Example:
    python3 inject_suites_into_lrg_map.py ../suite_lists/

Inputs:
  - lrg_to_tests.json (must be in same folder as script or adjust path)
  - suite_<suite>.txt in <suite_folder>

Output:
  - lrg_to_tests_with_suites.json
"""

ROOT = Path(__file__).resolve().parents[2]   # lrg_test_tool/
INPUT = ROOT / "lrg_to_test.json"
OUTPUT = ROOT / "lrg_to_tests_with_suites.json"


def load_suite_files(folder: Path):
    suite_map = {}

    for path in folder.glob("suite_*.txt"):
        suite_name = path.stem.replace("suite_", "").strip().upper()
        if not suite_name:
            continue

        for line in path.read_text(encoding="utf-8").splitlines():
            lrg = line.strip()
            if not lrg:
                continue

            suite_map.setdefault(lrg, []).append(suite_name)

    # Deduplicate + sort
    for lrg in suite_map:
        suite_map[lrg] = sorted(set(suite_map[lrg]))

    return suite_map


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 inject_suites_into_lrg_map.py <folder_with_suite_txt_files>")
        sys.exit(1)

    suite_folder = Path(sys.argv[1]).resolve()
    if not suite_folder.exists():
        print(f"Error: Suite folder does not exist: {suite_folder}")
        sys.exit(1)

    if not INPUT.exists():
        print(f"Error: Missing input mapping: {INPUT}")
        sys.exit(1)

    lrg_map = json.loads(INPUT.read_text(encoding="utf-8"))
    suite_map = load_suite_files(suite_folder)

    new_map = {}

    for lrg, tests in lrg_map.items():
        suites = suite_map.get(lrg, [])
        new_map[lrg] = {
            "tests": tests,
            "suites": suites
        }

    OUTPUT.write_text(json.dumps(new_map, indent=2), encoding="utf-8")
    print(f"Updated map saved: {OUTPUT}")


if __name__ == "__main__":
    main()