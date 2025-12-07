# python3 build_mappings.py /ade/shratyag_v8/tklocal/tests_extracted.json /ade/shratyag_v8/tklocal/lrg_map.json

"""
python3 build_mappings.py \
  --tests /ade/shratyag_v7/tklocal/tests_extracted.json \
  --lrgs /ade/shratyag_v7/tklocal/lrg_map.json 


python3 /ade/shratyag_v7/tklocal/scripts/build_mappings.py \
  /ade/shratyag_v7/tklocal/tests_extracted.json \
  /ade/shratyag_v7/tklocal/lrg_map.json \
  /ade/shratyag_v7/tklocal/test_to_lrgs.json \
  /ade/shratyag_v7/tklocal/lrg_to_tests.json

"""

#!/usr/bin/env python3
import json, sys
from pathlib import Path
from collections import defaultdict

def main(tests_json="tests_extracted.json", lrgs_json="lrg_map.json",
         out_test_to_lrgs="test_to_lrgs.json", out_lrg_to_tests="lrg_to_tests.json"):
    tests = json.loads(Path(tests_json).read_text(encoding="utf-8"))
    lrgs  = json.loads(Path(lrgs_json).read_text(encoding="utf-8"))

    # Normalize names as they appear (keep original casing for display)
    test_to_lrgs = defaultdict(list)
    lrg_to_tests = {}

    for L in lrgs:
        lrg = L["lrgname"]
        lst = []
        for t in L.get("tests", []):
            tname = t.strip()
            if tname and tname not in test_to_lrgs:
                pass
            if tname not in lst:
                lst.append(tname)
            if lrg not in test_to_lrgs[tname]:
                test_to_lrgs[tname].append(lrg)
        lrg_to_tests[lrg] = lst

    # Optional: ensure all tests appear (even if not in any lrg)
    test_names = {t["test_name"] for t in tests}
    for t in sorted(test_names):
        test_to_lrgs.setdefault(t, [])

    Path(out_test_to_lrgs).write_text(json.dumps(test_to_lrgs, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(out_lrg_to_tests).write_text(json.dumps(lrg_to_tests, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {out_test_to_lrgs} and {out_lrg_to_tests}")

if __name__ == "__main__":
    # Usage: python3 build_mappings.py tests.json lrgs.json
    args = sys.argv[1:]
    main(*args) if args else main()
