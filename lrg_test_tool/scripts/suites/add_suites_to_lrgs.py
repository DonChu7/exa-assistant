#!/usr/bin/env python3
import json, sys
from pathlib import Path

"""
Usage: 

python3 add_suites_to_lrgs.py \
  /home/shratyag/ai_tool/exa-assistant/lrg_test_tool/lrg_to_test.json \
  /home/shratyag/ai_tool/exa-assistant/lrg_test_tool/suites/ \
  /home/shratyag/ai_tool/exa-assistant/lrg_test_tool/lrg_map.json

"""

if len(sys.argv) < 3:
    print("Usage: python3 add_suites_to_lrgs.py lrg_map_with_runtimes.json suites_dir/ [out.json]")
    sys.exit(1)

lrgs_json = Path(sys.argv[1])
suites_dir = Path(sys.argv[2])
out_json = Path(sys.argv[3]) if len(sys.argv) > 3 else lrgs_json

# Load main LRG JSON
data = json.loads(lrgs_json.read_text())

# Build mapping: lrgname → [suites]
lrg_to_suites = {}

for txt_file in suites_dir.glob("suite_*.txt"):
    suite_name = txt_file.stem.replace("suite_", "")
    for line in txt_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lrg_to_suites.setdefault(line, []).append(suite_name)

# Merge into JSON
for rec in data:
    name = rec.get("lrgname")
    rec["lrg_suite"] = sorted(lrg_to_suites.get(name, []))

# Write out
out_json.write_text(json.dumps(data, indent=2))
print(f"Updated {len(data)} LRG records → {out_json}")