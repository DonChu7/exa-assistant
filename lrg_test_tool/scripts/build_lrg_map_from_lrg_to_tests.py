#!/usr/bin/env python3
"""
build_lrg_map_from_lrg_to_tests.py

Convert:

  lrg_to_tests_with_suites.json  (dict)
    {
      "lrg1": { "tests": [...], "suites": [...] },
      ...
    }

into:

  lrg_map_with_suites.json  (array)
    [
      {
        "lrgname": "lrg1",
        "lrg_suite": [...],
        "tests": [...],
        "runtime": null
      },
      ...
    ]

so that build_profiles_fast.py can consume it via --lrgs.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # .../lrg_test_tool
SRC  = ROOT / "lrg_to_tests_with_suites.json"
DST  = ROOT / "lrg_map_with_suites.json"

def main():
    if not SRC.exists():
        raise SystemExit(f"Missing {SRC}")

    data = json.loads(SRC.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{SRC} must be a JSON object mapping lrg -> {{tests, suites}}")

    out = []
    for lrg, info in data.items():
        if not isinstance(info, dict):
            continue

        tests  = info.get("tests")  or []
        suites = info.get("suites") or []

        # normalize types
        if not isinstance(tests, list):
            tests = []
        tests = [t.strip() for t in tests if isinstance(t, str) and t.strip()]

        if isinstance(suites, str):
            suites = [suites]
        elif not isinstance(suites, list):
            suites = []
        suites = [s.strip() for s in suites if isinstance(s, str) and s.strip()]

        out.append(
            {
                "lrgname": lrg,
                "lrg_suite": suites,
                "tests": tests,
                # "runtime": None,  # runtime comes later from metrics DB / API
            }
        )

    DST.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {DST} with {len(out)} LRG entries")

if __name__ == "__main__":
    main()