#!/usr/bin/env python3
"""
python3 convert_dict_lrgs_to_list.py ../lrg_to_tests_with_suites.json ../lrg_map_with_suites.json

"""

import json, sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python3 convert_lrg_list_to_object.py input.json output.json")
    sys.exit(1)

inp = Path(sys.argv[1])
out = Path(sys.argv[2])

raw = json.loads(inp.read_text())

if not isinstance(raw, list):
    print("Input is already an object. Nothing to convert.")
    out.write_text(json.dumps(raw, indent=2))
    sys.exit(0)

merged = {}

for item in raw:
    if isinstance(item, dict):
        for k, v in item.items():
            if k not in merged:
                merged[k] = v
            else:
                # merge duplicates
                merged[k] = sorted(set(merged[k] + v))

out.write_text(json.dumps(merged, indent=2))
print(f"Converted list → object. Wrote {out}")