#!/usr/bin/env python3

"""
merge_test_descriptions.py
--------------------------
Merge or append test descriptions into an existing JSON file (e.g. tests_filtered.json).

  Purpose:
    - Takes a text file containing test descriptions in the format:
          tsagexcsandbox_lastmirror.tsc -> Triple failure test in EGS Sandbox env
          tsagexclostdisk.tsc -> Verify data redundancy after cell failure
    - For each test:
          • If 'description' is empty/null → replaces it with new description.
          • If 'description' already exists → appends the new one as "Additional:".
    - Overwrites the existing JSON file in place (no new file created).

  Usage:
    python3 merge_test_descriptions.py <tests_json> <descriptions_txt>

    Example:
    python3 merge_test_descriptions.py \
        /ade/shratyag_v7/tklocal/tests_filtered.json \
        /ade/shratyag_v7/tklocal/scripts/description/test_descriptions.txt

  Input:
    - tests_json: JSON array, each element like:
        { "test_name": "tsagexcsandbox_lastmirror.tsc", "description": null, ... }
    - descriptions_txt: Plain text file with "test_name -> description" lines

  Output:
    - Overwrites the same JSON file with updated description fields.
"""

import json
import sys
from pathlib import Path

def load_descriptions(desc_path):
    """Load test_name -> description mapping from text file."""
    mapping = {}
    with open(desc_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or "->" not in line:
                continue
            name, desc = line.split("->", 1)
            name, desc = name.strip(), desc.strip()
            if name and desc:
                mapping[name] = desc
    print(f"Loaded {len(mapping)} new descriptions from {desc_path}")
    return mapping

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 merge_test_descriptions.py <tests_json> <descriptions_txt>")
        sys.exit(1)

    tests_json = Path(sys.argv[1])
    desc_file = Path(sys.argv[2])

    tests = json.loads(tests_json.read_text(encoding="utf-8"))
    desc_map = load_descriptions(desc_file)

    updated = 0
    appended = 0

    for t in tests:
        name = t.get("test_name")
        if not name or name not in desc_map:
            continue

        new_desc = desc_map[name]
        old_desc = (t.get("description") or "").strip()

        if not old_desc:
            t["description"] = new_desc
            updated += 1
        elif new_desc not in old_desc:
            t["description"] = f"{old_desc}\n\nAdditional: {new_desc}"
            appended += 1

    tests_json.write_text(json.dumps(tests, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Done. Updated {updated} tests, appended to {appended} tests.")
    print(f"Overwritten file: {tests_json}")

if __name__ == "__main__":
    main()